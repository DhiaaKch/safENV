"""
ldap_loader.py - VERSION CORRIGÉE
===================================
Charge les fichiers LDAP mensuels → snapshot RH enrichi par user par slice.

Corrections :
  - tenure_days : utilise pd.Timestamp pour que .dt.days fonctionne
  - compute_changes : compare snapshot actif vs snapshot LDAP précédent
  - detect_if_static : détecte si le LDAP est statique sur tout le dataset
    et logge un avertissement pour le feature engineering
  - _to_naive_dt : normalise tous les types datetime pour éviter TypeError

Lit  : data/raw/LDAP/YYYY-MM.csv  (cfg.data.ldap_folder)
Écrit: data/03_features/ldap/slice_NNN/ldap.parquet  (cfg.output.features_dir)

Colonnes produites :
  user_id, employee_name, email, role, business_unit, functional_unit,
  department, team, supervisor,
  is_new_employee,    ← apparaît pour la 1ère fois dans le LDAP
  changed_department, ← changement vs snapshot LDAP précédent
  changed_role,       ← changement vs snapshot LDAP précédent
  changed_supervisor, ← changement vs snapshot LDAP précédent
  tenure_days,        ← ancienneté en jours depuis 1ère apparition LDAP
  ldap_is_static      ← True si LDAP statique (changed_* à exclure du ML)
"""

import pandas as pd
from pathlib import Path
import yaml, logging, sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(logs_dir: str) -> logging.Logger:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)s │ %(message)s",
        handlers=[
            logging.FileHandler(
                Path(logs_dir) / f"ldap_loader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("LDAPLoader")


def _to_naive_dt(t) -> datetime:
    """Normalise pd.Timestamp / datetime / str → datetime naive Python."""
    if isinstance(t, pd.Timestamp):
        return t.to_pydatetime().replace(tzinfo=None)
    if isinstance(t, datetime):
        return t.replace(tzinfo=None)
    return pd.Timestamp(t).to_pydatetime().replace(tzinfo=None)


def load_all_snapshots(ldap_dir: Path, log) -> pd.DataFrame:
    """Charge tous les YYYY-MM.csv et ajoute snapshot_month (datetime naive)."""
    files = sorted(ldap_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier CSV dans {ldap_dir}")

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, dtype=str).fillna("")
            df["snapshot_month"] = datetime.strptime(fp.stem, "%Y-%m")
            frames.append(df)
        except Exception as e:
            log.warning(f"  Impossible de charger {fp.name}: {e}")

    if not frames:
        raise ValueError("Aucun snapshot LDAP chargé")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["user_id"] = all_df["user_id"].str.upper().str.strip()
    log.info(f"  LDAP : {len(files)} fichiers │ {all_df['user_id'].nunique()} users distincts")
    return all_df


def detect_if_static(all_df: pd.DataFrame, log) -> bool:
    """
    Vérifie si le LDAP est statique (aucun changement entre tous les snapshots).
    Si oui, loggue un avertissement → changed_* à exclure du feature engineering.
    """
    months = sorted(all_df["snapshot_month"].unique())
    if len(months) < 2:
        return True

    total_changes = 0
    for i in range(1, len(months)):
        curr = all_df[all_df["snapshot_month"] == months[i]]
        prev = all_df[all_df["snapshot_month"] == months[i - 1]]
        merged = curr.merge(
            prev[["user_id", "department", "role", "supervisor"]],
            on="user_id", how="inner", suffixes=("", "_prev")
        )
        for col in ["department", "role", "supervisor"]:
            total_changes += (
                merged[col].str.strip() != merged[f"{col}_prev"].str.strip()
            ).sum()

    is_static = (total_changes == 0)
    if is_static:
        log.warning("  ⚠️  LDAP STATIQUE : aucun changement org. détecté sur 18 mois.")
        log.warning("      changed_department / changed_role / changed_supervisor")
        log.warning("      → variance nulle, À EXCLURE du feature engineering.")
    else:
        log.info(f"  LDAP dynamique : {total_changes} changements org. détectés")
    return is_static


def get_snapshot_le(all_df: pd.DataFrame, target: datetime) -> pd.DataFrame:
    """Retourne le snapshot le plus récent ≤ target."""
    target = _to_naive_dt(target)
    months = sorted(all_df["snapshot_month"].unique())
    valid  = [m for m in months if m <= target]
    chosen = max(valid) if valid else min(months)
    return all_df[all_df["snapshot_month"] == chosen].copy()


def compute_changes(all_df: pd.DataFrame, target: datetime) -> pd.DataFrame:
    """
    Compare le snapshot actif (≤ target) avec le snapshot LDAP qui le précède.
    Retourne les flags de changement par user.
    """
    target = _to_naive_dt(target)
    months = sorted(m for m in all_df["snapshot_month"].unique() if m <= target)

    if len(months) < 2:
        # Premier snapshot : tous marqués nouveaux, aucun changement
        snap = get_snapshot_le(all_df, target).copy()
        snap["changed_department"] = False
        snap["changed_role"]       = False
        snap["changed_supervisor"] = False
        snap["is_new_employee"]    = True
        return snap[["user_id", "changed_department", "changed_role",
                     "changed_supervisor", "is_new_employee"]]

    curr_month = months[-1]
    prev_month = months[-2]

    curr = all_df[all_df["snapshot_month"] == curr_month].copy()
    prev = all_df[all_df["snapshot_month"] == prev_month]

    merged = curr.merge(
        prev[["user_id", "department", "role", "supervisor"]],
        on="user_id", how="left", suffixes=("", "_prev")
    )

    for col in ["department", "role", "supervisor"]:
        prev_col = f"{col}_prev"
        merged[f"changed_{col}"] = (
            merged[prev_col].notna() &
            (merged[col].str.strip() != merged[prev_col].fillna("").str.strip())
        )

    prev_users = set(prev["user_id"])
    merged["is_new_employee"] = ~merged["user_id"].isin(prev_users)

    return merged[["user_id", "changed_department", "changed_role",
                   "changed_supervisor", "is_new_employee"]]


def build_for_slice(all_df: pd.DataFrame, first_seen: pd.DataFrame,
                    slice_start, is_static: bool) -> pd.DataFrame:
    slice_start_dt = _to_naive_dt(slice_start)
    slice_start_ts = pd.Timestamp(slice_start_dt)  # pour arithmetic .dt

    snap    = get_snapshot_le(all_df, slice_start_dt)
    changes = compute_changes(all_df, slice_start_dt)

    result = snap.merge(changes, on="user_id", how="left")
    result = result.merge(first_seen, on="user_id", how="left")

    # tenure_days : soustraction entre pd.Timestamp et colonne datetime
    result["tenure_days"] = (
        slice_start_ts - pd.to_datetime(result["first_seen_month"])
    ).dt.days.clip(lower=0)

    result["ldap_is_static"] = is_static

    drop = ["snapshot_month", "first_seen_month",
            "department_prev", "role_prev", "supervisor_prev"]
    result = result.drop(columns=[c for c in drop if c in result.columns])

    for col in ["changed_department", "changed_role", "changed_supervisor", "is_new_employee"]:
        result[col] = result[col].fillna(False)
    result["tenure_days"] = result["tenure_days"].fillna(0).astype(int)

    return result


def run(cfg: dict, log) -> None:
    ldap_dir   = Path(cfg["data"]["ldap_folder"])
    slices_dir = Path(cfg["output"]["slices_dir"])
    out_base   = Path(cfg["output"]["features_dir"]) / "ldap"

    log.info("=" * 60)
    log.info("LDAP LOADER — Enrichissement RH par slice")
    log.info("=" * 60)
    log.info(f"LDAP dir   : {ldap_dir}")
    log.info(f"Output dir : {out_base}")

    all_df    = load_all_snapshots(ldap_dir, log)
    is_static = detect_if_static(all_df, log)

    first_seen = (
        all_df.groupby("user_id")["snapshot_month"].min()
        .reset_index().rename(columns={"snapshot_month": "first_seen_month"})
    )

    slice_dirs = sorted(p for p in Path(slices_dir).iterdir()
                        if p.is_dir() and p.name.startswith("slice_"))
    if not slice_dirs:
        raise FileNotFoundError(f"Aucune slice dans {slices_dir}")

    log.info(f"Slices : {len(slice_dirs)}\n{'─'*60}")

    for slice_dir in slice_dirs:
        meta_path = slice_dir / "meta.parquet"
        if not meta_path.exists():
            log.warning(f"  {slice_dir.name} : meta.parquet absent, ignoré")
            continue

        meta        = pd.read_parquet(meta_path)
        slice_start = pd.to_datetime(meta["start"].iloc[0])
        slice_id    = int(meta["slice_id"].iloc[0])

        ldap_slice = build_for_slice(all_df, first_seen, slice_start, is_static)

        out_dir = out_base / slice_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        ldap_slice.to_parquet(out_dir / "ldap.parquet", index=False)

        n_new      = int(ldap_slice["is_new_employee"].sum())
        n_chg      = int((ldap_slice["changed_department"] |
                          ldap_slice["changed_role"] |
                          ldap_slice["changed_supervisor"]).sum())
        tenure_med = int(ldap_slice["tenure_days"].median())

        log.info(f"  ✓ slice_{slice_id:03d} │ {len(ldap_slice):>5,} users │ "
                 f"{n_new:>3} nouveaux │ {n_chg:>3} changements │ "
                 f"ancienneté médiane : {tenure_med} jours")

    if is_static:
        log.info("\n  ─── Rappel feature engineering ───────────────────────")
        log.info("  Exclure : changed_department, changed_role, changed_supervisor")
        log.info("  Garder  : role, department, business_unit, functional_unit,")
        log.info("            team, tenure_days  (+ is_new_employee si variance > 0)")
        log.info("  ──────────────────────────────────────────────────────")

    log.info(f"\n{'='*60}\n✅ LDAP Loader terminé\n{'='*60}")


if __name__ == "__main__":
    try:
        cfg = load_config()
        log = get_logger(cfg["output"]["logs_dir"])
        run(cfg, log)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}"); sys.exit(1)
    except Exception as e:
        import traceback; traceback.print_exc(); sys.exit(1)

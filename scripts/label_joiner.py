"""
label_joiner.py - VERSION FINALE
==================================
Joint insiders.csv aux slices pour l'entraînement supervisé.

Lit  : data/labels/insiders.csv  (cfg.data.answers_file)
       data/01_slices/slice_NNN/*.parquet  (pour récupérer tous les users)
Écrit: data/03_features/labels/slice_NNN/labels.parquet

Format insiders.csv : dataset, scenario, details, user, start, end
  → start/end au format %m/%d/%Y %H:%M:%S

Logique : un user est insider dans une slice si sa fenêtre malveillante
          [start, end] chevauche [slice_start, slice_end[


"""

import pandas as pd
import duckdb
from pathlib import Path
import yaml, logging, sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSIDER_TS_FMT = "%m/%d/%Y %H:%M:%S"


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
                Path(logs_dir) / f"label_joiner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("LabelJoiner")


def load_insiders(labels_path: Path, log) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip().str.lower()
    df["user_id"] = df["user"].astype(str).str.upper().str.strip()

    start_raw = df["start"].astype(str).str.strip()
    end_raw = df["end"].astype(str).str.strip()
    df["malicious_start"] = pd.to_datetime(
        start_raw, format=INSIDER_TS_FMT, errors="coerce"
    )
    df["malicious_end"] = pd.to_datetime(
        end_raw, format=INSIDER_TS_FMT, errors="coerce"
    )
    df["scenario"] = pd.to_numeric(df["scenario"], errors="coerce").astype("Int64")

    invalid = (
        df["user_id"].eq("") |
        df["malicious_start"].isna() |
        df["malicious_end"].isna() |
        df["scenario"].isna()
    )
    n_invalid = int(invalid.sum())
    if n_invalid:
        log.warning(f"  insiders.csv : {n_invalid} lignes invalides ignorées")
        df = df[~invalid].copy()
    df["scenario"] = df["scenario"].astype(int)

    log.info(f"  insiders.csv : {len(df)} entrées │ "
             f"{df['user_id'].nunique()} users │ "
             f"scénarios {sorted(df['scenario'].unique())}")
    return df[["user_id", "scenario", "malicious_start", "malicious_end"]]


def get_all_users(slice_dir: Path, sources: list) -> set:
    """Tous les user_id présents dans une slice (toutes sources)."""
    users = set()
    con = duckdb.connect()
    for src in sources:
        fp = slice_dir / f"{src}.parquet"
        if not fp.exists(): continue
        try:
            rows = con.execute(
                f"SELECT DISTINCT user_id FROM read_parquet('{fp.as_posix()}') "
                "WHERE user_id IS NOT NULL"
            ).fetchall()
            users.update(r[0] for r in rows)
        except Exception:
            pass
    con.close()
    return users


def build_labels(insiders: pd.DataFrame, all_users: set,
                 slice_start: pd.Timestamp, slice_end: pd.Timestamp) -> pd.DataFrame:
    """
    Label par user pour cette slice.
    Chevauchement : malicious_start < slice_end AND malicious_end >= slice_start
    """
    active = insiders[
        (insiders["malicious_start"] < slice_end) &
        (insiders["malicious_end"]   > slice_start)   # > strict : exclut fin exactement = début slice
    ].copy()

    active["is_start_slice"] = active["malicious_start"] >= slice_start
    active["is_end_slice"]   = active["malicious_end"]   <  slice_end

    # Si un user a plusieurs scénarios actifs → garder le plus élevé (plus grave)
    active = (active.sort_values("scenario", ascending=False)
                    .drop_duplicates(subset="user_id", keep="first"))

    result = pd.DataFrame({"user_id": sorted(all_users)})
    result = result.merge(
        active[["user_id", "scenario", "malicious_start", "malicious_end",
                "is_start_slice", "is_end_slice"]],
        on="user_id", how="left"
    )
    result["is_insider"]    = result["user_id"].isin(set(active["user_id"])).astype(int)
    result["is_start_slice"] = result["is_start_slice"].fillna(False)
    result["is_end_slice"]   = result["is_end_slice"].fillna(False)
    result["scenario"]       = result["scenario"].astype("Int64")  # nullable int
    return result


def run(cfg: dict, log) -> None:
    labels_path = Path(cfg["data"]["answers_file"])
    slices_dir  = Path(cfg["output"]["slices_dir"])
    out_base    = Path(cfg["output"]["features_dir"]) / "labels"
    sources     = cfg["data"]["sources"]

    log.info("=" * 60)
    log.info("LABEL JOINER — Annotation insiders par slice")
    log.info("=" * 60)
    log.info(f"Labels     : {labels_path}")
    log.info(f"Output dir : {out_base}")

    if not labels_path.exists():
        raise FileNotFoundError(f"insiders.csv introuvable : {labels_path}")

    insiders = load_insiders(labels_path, log)

    slice_dirs = sorted(p for p in slices_dir.iterdir()
                        if p.is_dir() and p.name.startswith("slice_"))
    if not slice_dirs:
        raise FileNotFoundError(f"Aucune slice dans {slices_dir}")

    log.info(f"Slices : {len(slice_dirs)}\n{'─'*60}")

    total_insider_user_slices = 0

    for slice_dir in slice_dirs:
        meta_path = slice_dir / "meta.parquet"
        if not meta_path.exists():
            log.warning(f"  {slice_dir.name} : meta.parquet absent, ignoré"); continue

        meta        = pd.read_parquet(meta_path)
        slice_start = pd.to_datetime(meta["start"].iloc[0])
        slice_end   = pd.to_datetime(meta["end"].iloc[0])
        slice_id    = int(meta["slice_id"].iloc[0])

        all_users = get_all_users(slice_dir, sources)
        labels    = build_labels(insiders, all_users, slice_start, slice_end)

        n_insiders = int(labels["is_insider"].sum())
        n_normal   = len(labels) - n_insiders
        total_insider_user_slices += n_insiders

        out_dir = out_base / slice_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(out_dir / "labels.parquet", index=False)

        scenarios = sorted(labels[labels["is_insider"]==1]["scenario"].dropna().unique().tolist())
        log.info(f"  ✓ slice_{slice_id:03d} │ {len(labels):>5,} users │ "
                 f"{n_insiders:>3} insiders │ {n_normal:>5,} normaux │ "
                 f"scénarios: {scenarios if scenarios else '—'}")

    total_users_all_slices = sum(
        len(get_all_users(sd, sources)) for sd in slice_dirs
        if (sd / "meta.parquet").exists()
    )
    ratio = total_insider_user_slices / max(1, total_users_all_slices)

    log.info(f"\n{'='*60}\n Label Joiner terminé\n{'='*60}")
    log.info(f"Total (user×slice) insiders : {total_insider_user_slices}")
    log.info(f"Ratio déséquilibre approx.  : ~1:{int(1/ratio) if ratio > 0 else '∞'}")
    log.info(f" Prévoir class_weight='balanced' ou SMOTE (ratio={cfg['pipeline'].get('smote_ratio', 0.3)})")


if __name__ == "__main__":
    try:
        cfg = load_config()
        log = get_logger(cfg["output"]["logs_dir"])
        run(cfg, log)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}"); sys.exit(1)
    except Exception as e:
        import traceback; traceback.print_exc(); sys.exit(1)

"""
splicer_ldap.py
================
Slicer LDAP autonome.

- LDAP déjà découpé par mois (YYYY-MM.csv)
- Utilise meta.parquet de chaque slice
- Produit ldap.parquet dans CHAQUE slice
- Indépendant de config.yaml
"""

import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import datetime
import logging
import sys

# ─────────────────────────────────────────────────────────────
# Config hardcodée (simple & claire)
# ─────────────────────────────────────────────────────────────

RAW_DIR  = Path("data/00_raw/r4.2")
LDAP_DIR = RAW_DIR / "LDAP"
SLICES_DIR = Path("data/01_slices")
LOGS_DIR   = Path("logs")

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def get_logger():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"splicer_ldap_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)s │ %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("LDAP-Slicer")

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def months_between(start, end):
    """Retourne ['YYYY-MM', ...] couvrant [start, end]."""
    cur = start.replace(day=1)
    end = end.replace(day=1)
    months = []

    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        cur += relativedelta(months=1)

    return months


# ─────────────────────────────────────────────────────────────
# Core LDAP slicing
# ─────────────────────────────────────────────────────────────

def slice_ldap_for_slice(slice_dir: Path, log) -> int:
    meta_fp = slice_dir / "meta.parquet"

    if not meta_fp.exists():
        log.warning(f"{slice_dir.name}: meta.parquet manquant, skip")
        return 0

    meta = pd.read_parquet(meta_fp).iloc[0]
    start = pd.to_datetime(meta["start"])
    end   = pd.to_datetime(meta["end"])

    months = months_between(start, end)
    files = [LDAP_DIR / f"{m}.csv" for m in months if (LDAP_DIR / f"{m}.csv").exists()]

    if not files:
        log.info(f"{slice_dir.name}: ldap → 0 événement")
        return 0

    log.info(f"{slice_dir.name}: ldap → {len(files)} fichiers")

    dfs = []
    total = 0

    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
            dfs.append(df)
            total += len(df)
        except Exception as e:
            log.error(f"{slice_dir.name}: erreur lecture {fp.name} - {e}")

    if dfs:
        out = slice_dir / "ldap.parquet"
        final = pd.concat(dfs, ignore_index=True)
        final.to_parquet(out, compression="snappy", index=False)
        log.info(f"{slice_dir.name}: ldap ✓ {total:,} événements")
        return total

    return 0


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

def main():
    log = get_logger()

    log.info("=" * 60)
    log.info("LDAP SLICER AUTONOME")
    log.info("=" * 60)

    if not LDAP_DIR.exists():
        log.error(f"Dossier LDAP introuvable: {LDAP_DIR}")
        return

    slices = sorted(p for p in SLICES_DIR.iterdir() if p.is_dir() and p.name.startswith("slice_"))

    if not slices:
        log.error(f"Aucune slice trouvée dans {SLICES_DIR}")
        return

    total_global = 0

    for slice_dir in slices:
        n = slice_ldap_for_slice(slice_dir, log)
        total_global += n

    log.info("=" * 60)
    log.info(f"TOTAL LDAP (toutes slices): {total_global:,} événements")
    log.info("✅ LDAP slicing terminé")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

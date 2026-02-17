"""
01_slicer.py - FIXED VERSION (HTTP 13GB support)
============
Découpe chaque CSV source en tranches temporelles de N mois.

CORRECTIONS APPLIQUÉES:
- Utilise read_csv_auto() pour détecter colonnes automatiquement
- Détecte fichiers volumineux (>1GB) et traite en chunks
- http.csv (13.8GB) traité par chunks Pandas (évite timeout DuckDB)
- Logging amélioré avec progression chunks

Input:  data/00_raw/r4.2/*.csv
Output: data/01_slices/slice_NNN/{source}.parquet
"""

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yaml
import logging
import sys

# Fix UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')


# ── Config & Logging ────────────────────────────────────────────────

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_logger(logs_dir):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"01_slicer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)s │ %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("Slicer")


# ── Core ─────────────────────────────────────────────────────────────

def get_date_range(con, raw_dir: Path, ts_format: str) -> tuple:
    """Lit min/max date depuis logon.csv sans tout charger."""
    fp = raw_dir / "logon.csv"

    try:
        row = con.execute(f"""
            SELECT
                MIN(try_strptime(CAST(date AS VARCHAR), '{ts_format}')) AS mn,
                MAX(try_strptime(CAST(date AS VARCHAR), '{ts_format}')) AS mx
            FROM read_csv_auto('{fp}', 
                              header=true, 
                              ignore_errors=true,
                              sample_size=100000)
            WHERE date IS NOT NULL
        """).fetchone()

        return row[0], row[1]
    except Exception as e:
        print(f"❌ Erreur lecture période: {e}")
        return None, None


def build_periods(start, end, n_months: int) -> list:
    """Construit liste de (slice_id, start, end)."""
    periods, cur, idx = [], start, 1

    while cur < end:
        nxt = min(cur + relativedelta(months=n_months), end)
        periods.append((idx, cur, nxt))
        cur = nxt
        idx += 1

    return periods


def slice_source_chunked(fp: Path, out: Path, start, end, ts_format: str, log, source: str) -> int:
    """
    Traite gros fichiers (http.csv = 13.8GB) par chunks avec Pandas.
    Évite surcharge mémoire/timeout DuckDB.
    
    Stratégie:
    1. Lit CSV par morceaux de 500K lignes
    2. Parse dates et filtre période
    3. Écrit chunks temporaires en Parquet
    4. Fusionne tous les chunks
    5. Nettoie fichiers temporaires
    """
    from datetime import datetime as dt
    
    CHUNK_SIZE = 500_000  # 500K lignes par chunk (~200MB RAM)
    
    start_dt = dt.fromisoformat(str(start))
    end_dt = dt.fromisoformat(str(end))
    
    total_events = 0
    chunks_written = []
    
    try:
        # Lire par chunks
        chunk_iter = pd.read_csv(fp, chunksize=CHUNK_SIZE, low_memory=False)
        
        for i, chunk in enumerate(chunk_iter):
            # Parser dates
            chunk['ts'] = pd.to_datetime(chunk['date'], format=ts_format, errors='coerce')
            
            # Filtrer période
            mask = (chunk['ts'] >= start_dt) & (chunk['ts'] <= end_dt) & (chunk['ts'].notna())
            filtered = chunk[mask].drop(columns=['ts'])
            
            if len(filtered) > 0:
                # Écrire chunk temporaire
                temp_out = out.parent / f"{out.stem}_chunk_{i}.parquet"
                filtered.to_parquet(temp_out, compression='snappy', index=False)
                chunks_written.append(temp_out)
                total_events += len(filtered)
            
            # Log progression tous les 10 chunks
            if (i + 1) % 10 == 0:
                log.info(f"      ... chunk {i+1} traité ({total_events:,} événements jusqu'ici)")
        
        # Fusionner tous les chunks
        if chunks_written:
            log.info(f"      Fusion de {len(chunks_written)} chunks...")
            
            all_chunks = [pd.read_parquet(c) for c in chunks_written]
            final = pd.concat(all_chunks, ignore_index=True)
            final.to_parquet(out, compression='snappy', index=False)
            
            # Nettoyer chunks temporaires
            for temp in chunks_written:
                temp.unlink()
            
            log.info(f"    ✓ {source:8s}: {total_events:>10,} événements")
        else:
            log.info(f"    ○ {source:8s}: 0 événements")
        
        return total_events
    
    except Exception as e:
        log.error(f"    {source:8s}: Erreur chunked - {e}")
        
        # Nettoyer chunks temporaires en cas d'erreur
        for temp in chunks_written:
            if temp.exists():
                temp.unlink()
        
        return 0


def slice_source(con, source: str, raw_dir: Path, out_dir: Path,
                 start, end, ts_format: str, log) -> int:
    """
    Extrait événements d'une source pour la période [start, end].
    
    STRATÉGIE AUTOMATIQUE:
    - Fichiers < 1GB: DuckDB (rapide, en mémoire)
    - Fichiers > 1GB: Pandas chunks (économe mémoire)
    """

    fp = raw_dir / f"{source}.csv"
    out = out_dir / f"{source}.parquet"

    if not fp.exists():
        log.warning(f"    {source:8s}: Fichier absent, skip")
        return 0

    # Vérifier taille fichier (en MB)
    file_size_mb = fp.stat().st_size / 1024 / 1024
    
    # Si fichier > 1GB, traiter par chunks avec Pandas
    if file_size_mb > 1000:
        log.info(f"    {source:8s}: Fichier volumineux ({file_size_mb:.0f} MB), traitement chunked...")
        return slice_source_chunked(fp, out, start, end, ts_format, log, source)
    
    # Sinon, méthode DuckDB standard (rapide pour fichiers normaux)
    try:
        con.execute(f"""
            COPY (
                SELECT *
                FROM (
                    SELECT *,
                        try_strptime(CAST(date AS VARCHAR), '{ts_format}') AS ts
                    FROM read_csv_auto(
                        '{fp}',
                        header=true,
                        ignore_errors=true,
                        all_varchar=false,
                        sample_size=100000
                    )
                )
                WHERE ts IS NOT NULL
                  AND ts >= TIMESTAMP '{start}'
                  AND ts <= TIMESTAMP '{end}'
            )
            TO '{out}'
            (FORMAT PARQUET, COMPRESSION 'snappy')
        """)

        # Compter événements écrits
        n = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{out}')"
        ).fetchone()[0]

        return n

    except Exception as e:
        log.error(f"    {source:8s}: Erreur - {e}")
        return 0


def run(cfg: dict, log):
    raw_dir    = Path(cfg["data"]["raw_dir"])
    slices_dir = Path(cfg["output"]["slices_dir"])
    ts_format  = cfg["data"]["timestamp_format"]
    sources    = cfg["data"]["sources"]
    n_months   = cfg["pipeline"]["slice_months"]

    log.info("=" * 60)
    log.info("SLICER - Découpage temporel des sources")
    log.info("=" * 60)
    log.info(f"Raw dir   : {raw_dir}")
    log.info(f"Slice size: {n_months} mois")

    # Vérifier raw_dir existe
    if not raw_dir.exists():
        log.error(f"❌ Dossier raw inexistant: {raw_dir}")
        return []

    # Connexion unique
    con = duckdb.connect()

    # Période globale
    log.info("Détection période globale...")
    mn, mx = get_date_range(con, raw_dir, ts_format)

    if mn is None or mx is None:
        log.error("❌ Impossible de déterminer la période (dates invalides).")
        con.close()
        return []

    log.info(f"Période   : {mn.date()} → {mx.date()}")

    periods = build_periods(mn, mx, n_months)
    log.info(f"Tranches  : {len(periods)}")

    # Traiter chaque slice
    for sid, start, end in periods:
        out_dir = slices_dir / f"slice_{sid:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("")
        log.info(f"{'─' * 60}")
        log.info(f"Slice {sid:03d} │ {start.date()} → {end.date()}")
        log.info(f"{'─' * 60}")

        # Métadonnées slice
        pd.DataFrame([{
            "slice_id": sid,
            "start": start.isoformat(),
            "end": end.isoformat()
        }]).to_parquet(out_dir / "meta.parquet", index=False)

        # Traiter chaque source
        total_events = 0
        for src in sources:
            n = slice_source(con, src, raw_dir, out_dir,
                           start, end, ts_format, log)
            
            if n > 0:
                if src != 'http':  # http log déjà géré dans slice_source_chunked
                    log.info(f"    ✓ {src:8s}: {n:>10,} événements")
                total_events += n
            else:
                if src != 'http':
                    log.info(f"    ○ {src:8s}: 0 événements")

        log.info(f"    {'─' * 40}")
        log.info(f"    TOTAL: {total_events:>10,} événements")

    con.close()

    log.info("")
    log.info("=" * 60)
    log.info("✅ Slicing terminé avec succès")
    log.info("=" * 60)
    
    return periods


if __name__ == "__main__":
    try:
        cfg = load_config()
        log = get_logger(cfg["output"]["logs_dir"])
        run(cfg, log)
    except FileNotFoundError:
        print("❌ Fichier config.yaml introuvable!")
        print("   Créez configs/config.yaml avec la structure requise")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
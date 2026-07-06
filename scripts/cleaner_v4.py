"""
02_cleaner.py - FIXED VERSION
=============
Nettoie les parquets d'une tranche temporelle.
Chaque source a ses règles spécifiques (CERT r4.2).


Input:  data/01_slices/slice_NNN/{source}.parquet  (bruts)
Output: même fichiers écrasés, nettoyés
"""

import duckdb
from pathlib import Path
import yaml
import logging
import sys
from datetime import datetime


# ── Config & Logging ────────────────────────────────────────────────

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def get_logger(logs_dir):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"02_cleaner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)s │ %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("Cleaner")


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _time_filter_sql(start_ts: str | None, end_ts: str | None, col: str = "timestamp") -> str:
    clauses = []
    if start_ts:
        clauses.append(f"{col} >= TIMESTAMP '{_sql_escape(start_ts)}'")
    if end_ts:
        clauses.append(f"{col} < TIMESTAMP '{_sql_escape(end_ts)}'")
    if not clauses:
        return ""
    return " AND " + " AND ".join(clauses)


def _load_slice_bounds(con, slice_dir: Path, log) -> tuple[str | None, str | None]:
    meta_path = slice_dir / "meta.parquet"
    if not meta_path.exists():
        log.warning("    meta.parquet absent: no time-bound filter for this slice")
        return None, None
    try:
        row = con.execute(
            f"""
            SELECT
                CAST(start AS VARCHAR) AS start_ts,
                CAST("end" AS VARCHAR) AS end_ts
            FROM read_parquet('{meta_path.as_posix()}')
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None, None
        return row[0], row[1]
    except Exception as e:
        log.warning(f"    impossible de lire meta.parquet ({e}); no time-bound filter")
        return None, None


# ── Nettoyeurs par source ────────────────────────────────────────────

JOB_SITES = [
    "indeed.com", "monster.com", "linkedin.com", "careerbuilder.com",
    "simplyhired.com", "jobhuntersbible.com", "job-hunt.org",
    "glassdoor.com", "dice.com", "ziprecruiter.com",
    "aol.com/jobs", "yahoo.com/hotjobs"
]

SENSITIVE_KEYWORDS = [
    'classified', 'confidential', 'restricted', 'top-secret', 
    'proprietary', 'spy', 'sabotage', 'insider', 'leak'
]


def clean_logon(con, path: Path, ts_fmt: str, log, start_ts: str | None = None, end_ts: str | None = None) -> int:
    """Nettoie fichier logon: normalise user/pc, filtre activités valides."""
    if not path.exists():
        log.warning(f"      logon.parquet absent")
        return 0
    
    try:
        time_filter = _time_filter_sql(start_ts, end_ts)
        con.execute(f"""
            CREATE OR REPLACE TABLE _tmp AS
            SELECT DISTINCT
                id,
                timestamp,
                user_id,
                pc,
                activity,
                source
            FROM (
                SELECT
                    id,
                    try_strptime(date, '{ts_fmt}') AS timestamp,
                    UPPER(TRIM(user)) AS user_id,
                    UPPER(TRIM(pc)) AS pc,
                    TRIM(activity) AS activity,
                    'logon' AS source
                FROM read_parquet('{path}')
                WHERE date IS NOT NULL
                  AND user IS NOT NULL
                  AND TRIM(user) != ''
                  AND TRIM(activity) IN ('Logon', 'Logoff')
            ) t
            WHERE timestamp IS NOT NULL {time_filter}
        """)
        
        count = con.execute("SELECT COUNT(*) FROM _tmp").fetchone()[0]
        
        if count > 0:
            con.execute(f"COPY _tmp TO '{path}' (FORMAT PARQUET, COMPRESSION 'snappy')")
        
        return count
        
    except Exception as e:
        log.error(f"      Erreur logon: {e}")
        return 0


def clean_device(con, path: Path, ts_fmt: str, log, start_ts: str | None = None, end_ts: str | None = None) -> int:
    """Nettoie fichier device: normalise user/pc, filtre Connect/Disconnect."""
    if not path.exists():
        log.warning(f"      device.parquet absent")
        return 0
    
    try:
        time_filter = _time_filter_sql(start_ts, end_ts)
        con.execute(f"""
            CREATE OR REPLACE TABLE _tmp AS
            SELECT DISTINCT
                id,
                timestamp,
                user_id,
                pc,
                activity,
                source
            FROM (
                SELECT
                    id,
                    try_strptime(date, '{ts_fmt}') AS timestamp,
                    UPPER(TRIM(user)) AS user_id,
                    UPPER(TRIM(pc)) AS pc,
                    TRIM(activity) AS activity,
                    'device' AS source
                FROM read_parquet('{path}')
                WHERE date IS NOT NULL
                  AND user IS NOT NULL
                  AND TRIM(user) != ''
                  AND TRIM(activity) IN ('Connect', 'Disconnect')
            ) t
            WHERE timestamp IS NOT NULL {time_filter}
        """)
        
        count = con.execute("SELECT COUNT(*) FROM _tmp").fetchone()[0]
        
        if count > 0:
            con.execute(f"COPY _tmp TO '{path}' (FORMAT PARQUET, COMPRESSION 'snappy')")
        
        return count
        
    except Exception as e:
        log.error(f"      Erreur device: {e}")
        return 0


def clean_file(con, path: Path, ts_fmt: str, log, start_ts: str | None = None, end_ts: str | None = None) -> int:
    """Nettoie fichier file: extrait extension, normalise noms."""
    if not path.exists():
        log.warning(f"      file.parquet absent")
        return 0
    
    try:
        time_filter = _time_filter_sql(start_ts, end_ts)
        con.execute(f"""
            CREATE OR REPLACE TABLE _tmp AS
            SELECT DISTINCT
                id,
                timestamp,
                user_id,
                pc,
                filename,
                file_extension,
                source
            FROM (
                SELECT
                    id,
                    try_strptime(date, '{ts_fmt}') AS timestamp,
                    UPPER(TRIM(user)) AS user_id,
                    UPPER(TRIM(pc)) AS pc,
                    TRIM(filename) AS filename,
                    LOWER(regexp_extract(TRIM(filename), '\\.([^.]+)$', 1)) AS file_extension,
                    'file' AS source
                FROM read_parquet('{path}')
                WHERE date IS NOT NULL
                  AND user IS NOT NULL
                  AND TRIM(user) != ''
                  AND filename IS NOT NULL
                  AND TRIM(filename) != ''
            ) t
            WHERE timestamp IS NOT NULL {time_filter}
        """)
        
        count = con.execute("SELECT COUNT(*) FROM _tmp").fetchone()[0]
        
        if count > 0:
            con.execute(f"COPY _tmp TO '{path}' (FORMAT PARQUET, COMPRESSION 'snappy')")
        
        return count
        
    except Exception as e:
        log.error(f"      Erreur file: {e}")
        return 0


def clean_http(con, path: Path, ts_fmt: str, log, start_ts: str | None = None, end_ts: str | None = None) -> int:
    """Nettoie fichier http: extrait domaine, détecte job sites, keywords."""
    if not path.exists():
        log.warning(f"      http.parquet absent")
        return 0
    
    try:
        # Construire condition job sites
        job_conditions = " OR ".join(f"LOWER(url) LIKE '%{site}%'" for site in JOB_SITES)
        
        # Construire détection keywords (sur content si disponible)
        keyword_conditions = " OR ".join(
            f"LOWER(COALESCE(content, '')) LIKE '%{kw}%'" 
            for kw in SENSITIVE_KEYWORDS
        )
        
        # Vérifier si colonne content existe
        cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        col_names = [c[0] for c in cols]
        has_content = 'content' in col_names
        
        content_select = "TRIM(content) AS content," if has_content else "NULL AS content,"
        keyword_select = f"CASE WHEN {keyword_conditions} THEN true ELSE false END AS has_sensitive_keywords," if has_content else "false AS has_sensitive_keywords,"
        
        time_filter = _time_filter_sql(start_ts, end_ts)
        con.execute(f"""
            CREATE OR REPLACE TABLE _tmp AS
            SELECT DISTINCT
                id,
                timestamp,
                user_id,
                pc,
                url,
                domain,
                is_job_search,
                is_wikileaks,
                is_file_sharing,
                content,
                has_sensitive_keywords,
                source
            FROM (
                SELECT
                    id,
                    try_strptime(date, '{ts_fmt}') AS timestamp,
                    UPPER(TRIM(user)) AS user_id,
                    UPPER(TRIM(pc)) AS pc,
                    TRIM(url) AS url,
                    regexp_extract(LOWER(TRIM(url)), '(?:https?://)?([^/]+)', 1) AS domain,
                    CASE WHEN {job_conditions} THEN true ELSE false END AS is_job_search,
                    CASE WHEN LOWER(url) LIKE '%wikileaks%' THEN true ELSE false END AS is_wikileaks,
                    CASE WHEN LOWER(url) LIKE '%dropbox%' OR LOWER(url) LIKE '%wetransfer%' THEN true ELSE false END AS is_file_sharing,
                    {content_select}
                    {keyword_select}
                    'http' AS source
                FROM read_parquet('{path}')
                WHERE date IS NOT NULL
                  AND user IS NOT NULL
                  AND TRIM(user) != ''
                  AND url IS NOT NULL
                  AND TRIM(url) != ''
            ) t
            WHERE timestamp IS NOT NULL {time_filter}
        """)
        
        count = con.execute("SELECT COUNT(*) FROM _tmp").fetchone()[0]
        
        if count > 0:
            con.execute(f"COPY _tmp TO '{path}' (FORMAT PARQUET, COMPRESSION 'snappy')")
        
        return count
        
    except Exception as e:
        log.error(f"      Erreur http: {e}")
        return 0


def clean_email(con, path: Path, ts_fmt: str, log, start_ts: str | None = None, end_ts: str | None = None) -> int:
    """Nettoie fichier email: normalise addresses, détecte externes."""
    if not path.exists():
        log.warning(f"      email.parquet absent")
        return 0
    
    try:
        # Vérifier colonnes disponibles
        cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        col_names = [c[0] for c in cols]
        
        # Gérer colonne 'attachments' qui peut s'appeler différemment
        has_attachments = 'attachments' in col_names
        has_attachment_count = 'attachment_count' in col_names
        
        if has_attachments:
            attach_select = "COALESCE(TRY_CAST(attachments AS SMALLINT), 0) AS attachments,"
        elif has_attachment_count:
            attach_select = "COALESCE(TRY_CAST(attachment_count AS SMALLINT), 0) AS attachments,"
        else:
            attach_select = "0 AS attachments,"
        
        time_filter = _time_filter_sql(start_ts, end_ts)
        con.execute(f"""
            CREATE OR REPLACE TABLE _tmp AS
            SELECT DISTINCT
                id,
                timestamp,
                user_id,
                pc,
                to_addr,
                cc,
                bcc,
                from_addr,
                to_domain,
                is_external,
                size,
                attachments,
                source
            FROM (
                SELECT
                    id,
                    try_strptime(date, '{ts_fmt}') AS timestamp,
                    UPPER(TRIM(user)) AS user_id,
                    UPPER(TRIM(pc)) AS pc,
                    LOWER(TRIM("to")) AS to_addr,
                    COALESCE(LOWER(TRIM(cc)), '') AS cc,
                    COALESCE(LOWER(TRIM(bcc)), '') AS bcc,
                    LOWER(TRIM("from")) AS from_addr,
                    regexp_extract(LOWER(TRIM("to")), '@([^@]+)$', 1) AS to_domain,
                    CASE
                        WHEN LOWER(TRIM("to")) LIKE '%@dtaa.com%' THEN false
                        ELSE true
                    END AS is_external,
                    COALESCE(TRY_CAST(size AS INTEGER), 0) AS size,
                    {attach_select}
                    'email' AS source
                FROM read_parquet('{path}')
                WHERE date IS NOT NULL
                  AND user IS NOT NULL
                  AND TRIM(user) != ''
                  AND "to" IS NOT NULL
            ) t
            WHERE timestamp IS NOT NULL {time_filter}
        """)
        
        count = con.execute("SELECT COUNT(*) FROM _tmp").fetchone()[0]
        
        if count > 0:
            con.execute(f"COPY _tmp TO '{path}' (FORMAT PARQUET, COMPRESSION 'snappy')")
        
        return count
        
    except Exception as e:
        log.error(f"      Erreur email: {e}")
        return 0


# ── Dispatcher ───────────────────────────────────────────────────────

CLEANERS = {
    "logon":  clean_logon,
    "device": clean_device,
    "file":   clean_file,
    "http":   clean_http,
    "email":  clean_email,
}


def clean_slice(slice_id: int, cfg: dict, log) -> dict:
    """
    Nettoie tous les parquets d'une slice.
    Retourne statistiques par source.
    """
    slices_dir = Path(cfg["output"]["slices_dir"])
    ts_fmt     = cfg["data"]["timestamp_format"]
    sources    = cfg["data"]["sources"]
    slice_dir  = slices_dir / f"slice_{slice_id:03d}"

    if not slice_dir.exists():
        log.warning(f"  Slice {slice_id:03d} inexistante, skip")
        return {}

    log.info(f"")
    log.info(f"  {'─' * 60}")
    log.info(f"  🧹 Cleaning Slice {slice_id:03d}")
    log.info(f"  {'─' * 60}")

    con = duckdb.connect()
    stats = {}
    start_ts, end_ts = _load_slice_bounds(con, slice_dir, log)
    if start_ts or end_ts:
        log.info(f"    time bounds: start={start_ts} end={end_ts} (end exclusive)")
    
    for src in sources:
        path = slice_dir / f"{src}.parquet"
        fn = CLEANERS.get(src)
        
        if fn:
            n_before = 0
            if path.exists():
                try:
                    n_before = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
                except:
                    n_before = 0
            
            n_after = fn(con, path, ts_fmt, log, start_ts=start_ts, end_ts=end_ts)
            
            if n_after > 0:
                pct_kept = (n_after / n_before * 100) if n_before > 0 else 100
                log.info(f"    ✓ {src:8s}: {n_after:>10,} lignes ({pct_kept:>5.1f}% conservées)")
            else:
                log.info(f"    ○ {src:8s}: 0 lignes")
            
            stats[src] = {'before': n_before, 'after': n_after}
    
    con.close()
    
    total_after = sum(s['after'] for s in stats.values())
    log.info(f"  {'─' * 60}")
    log.info(f"  TOTAL: {total_after:>10,} lignes")
    
    return stats


def run(cfg: dict, log, slice_ids: list = None) -> None:
    """Nettoie toutes les slices spécifiées (ou toutes si None)."""
    slices_dir = Path(cfg["output"]["slices_dir"])

    # Si pas de liste → traiter toutes les slices existantes
    if slice_ids is None:
        slice_ids = sorted(
            int(p.name.split("_")[1])
            for p in slices_dir.iterdir()
            if p.is_dir() and p.name.startswith("slice_")
        )

    log.info("=" * 60)
    log.info("CLEANER - Nettoyage des slices")
    log.info("=" * 60)
    log.info(f"Slices à traiter: {slice_ids}")
    
    all_stats = {}
    for sid in slice_ids:
        stats = clean_slice(sid, cfg, log)
        all_stats[sid] = stats

    log.info("")
    log.info("=" * 60)
    log.info("✅ Cleaning terminé avec succès")
    log.info("=" * 60)
    
    # Résumé global
    total_before = sum(
        sum(s['before'] for s in stats.values())
        for stats in all_stats.values()
    )
    total_after = sum(
        sum(s['after'] for s in stats.values())
        for stats in all_stats.values()
    )
    
    if total_before > 0:
        pct = total_after / total_before * 100
        log.info(f"Global: {total_after:,} / {total_before:,} lignes ({pct:.1f}% conservées)")


if __name__ == "__main__":
    try:
        cfg = load_config()
        log = get_logger(cfg["output"]["logs_dir"])
        run(cfg, log)
    except FileNotFoundError:
        print("❌ Fichier config.yaml introuvable!")
    except Exception as e:
        print(f"erreur: {e}")
        import traceback
        traceback.print_exc()

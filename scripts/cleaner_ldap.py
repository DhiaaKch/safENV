import duckdb
from pathlib import Path
import logging
import sys
from datetime import datetime

def get_logger(logs_dir: str):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"cleaner_ldap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)s │ %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("CleanerLDAP")


def clean_ldap(con, path: Path, log) -> int:
    """Nettoie ldap.parquet en normalisant les colonnes existantes."""
    if not path.exists():
        log.warning(f"      ldap.parquet absent dans {path.parent.name}")
        return 0

    try:
        # Lire les colonnes existantes
        cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        col_names = [c[0] for c in cols]

        select_parts = []

        # Ajouter colonnes standard si elles existent
        if 'id' in col_names:
            select_parts.append("id")
        if 'date' in col_names:
            select_parts.append("strptime(date, '%Y-%m-%d %H:%M:%S') AS timestamp")
        if 'user' in col_names:
            select_parts.append("UPPER(TRIM(user)) AS user_id")
        if 'pc' in col_names:
            select_parts.append("UPPER(TRIM(pc)) AS pc")
        if 'activity' in col_names:
            select_parts.append("TRIM(activity) AS activity")

        # Toujours ajouter source
        select_parts.append("'ldap' AS source")

        # Si aucune colonne standard n’existe, sélectionner toutes les colonnes existantes
        if len(select_parts) == 1:  # seul 'source' a été ajouté
            select_parts = [f'"{c}"' for c in col_names] + ["'ldap' AS source"]

        query = f"""
            CREATE OR REPLACE TABLE _tmp AS
            SELECT {', '.join(select_parts)}
            FROM read_parquet('{path}')
        """

        # Optionnel: filtrer lignes vides sur user/activity si présentes
        where_conditions = []
        if 'user' in col_names:
            where_conditions.append("user IS NOT NULL AND TRIM(user) != ''")
        if 'activity' in col_names:
            where_conditions.append("activity IS NOT NULL AND TRIM(activity) != ''")
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)

        con.execute(query)

        count = con.execute("SELECT COUNT(*) FROM _tmp").fetchone()[0]

        if count > 0:
            con.execute(f"COPY _tmp TO '{path}' (FORMAT PARQUET, COMPRESSION 'snappy')")

        return count

    except Exception as e:
        log.error(f"      Erreur ldap: {e}")
        return 0


def run(slices_dir: str, log_dir: str):
    slices_path = Path(slices_dir)
    log = get_logger(log_dir)
    con = duckdb.connect()

    slice_dirs = sorted(p for p in slices_path.iterdir() if p.is_dir() and p.name.startswith("slice_"))
    log.info(f"Slices détectées: {[p.name for p in slice_dirs]}")

    total_before = 0
    total_after = 0

    for slice_dir in slice_dirs:
        path = slice_dir / "ldap.parquet"
        n_before = 0
        if path.exists():
            try:
                n_before = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
            except:
                n_before = 0

        n_after = clean_ldap(con, path, log)
        pct = (n_after / n_before * 100) if n_before > 0 else 100
        log.info(f"  {slice_dir.name}: {n_after} lignes nettoyées ({pct:.1f}% conservées)")

        total_before += n_before
        total_after += n_after

    log.info("="*60)
    log.info(f"TOTAL: {total_after} / {total_before} lignes ({(total_after/total_before*100) if total_before else 100:.1f}% conservées)")
    log.info("="*60)
    con.close()


if __name__ == "__main__":
    SLICES_DIR = "data/01_slices"
    LOGS_DIR = "logs"
    run(SLICES_DIR, LOGS_DIR)

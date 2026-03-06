"""
03_feature_engineering_production.py - VERSION FINALE
======================================================
Feature Engineering UEBA production-grade pour CERT r4.2

COMPATIBLE AVEC:
- slicer.py (slices temporelles sans overlap)
- cleaner.py (colonnes enrichies: is_wikileaks, is_job_search, etc.)
- label_joiner.py (labels is_insider, scenario)
- ldap_loader.py (métadonnées RH)

FEATURES PRODUITES: 40+ optimisées pour détection insider threat

Input:  
  - data/01_slices/slice_NNN/*.parquet (cleaned)
  - data/03_features/labels/slice_NNN/labels.parquet
  - data/03_features/ldap/slice_NNN/ldap.parquet

Output: 
  - data/03_features/slice_NNN/features.parquet

USAGE:
  python 03_feature_engineering_production.py
  python 03_feature_engineering_production.py --slice 1  # Une seule slice
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import sys
import gc
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# CONFIG & LOGGING
# =============================================================================

def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Validation
    required = {
        "data": ["raw_dir", "timestamp_format", "sources", "answers_file"],
        "output": ["slices_dir", "features_dir", "logs_dir"],
        "detection": ["blacklist_domains"],
    }
    for section, keys in required.items():
        if section not in cfg:
            raise ValueError(f"Section manquante: {section}")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Clé manquante: {section}.{key}")
    
    return cfg


def get_logger(logs_dir: str) -> logging.Logger:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger = logging.getLogger("FeatureEngineering")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fmt = logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s")
    
    fh = logging.FileHandler(
        Path(logs_dir) / f"03_features_prod_{ts}.log",
        encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_slice_meta(slice_dir: Path) -> Optional[Dict]:
    """Charge métadonnées slice (start, end)."""
    meta_path = slice_dir / "meta.parquet"
    if not meta_path.exists():
        return None
    
    meta = pd.read_parquet(meta_path)
    return {
        "slice_id": int(meta["slice_id"].iloc[0]),
        "start": pd.to_datetime(meta["start"].iloc[0]),
        "end": pd.to_datetime(meta["end"].iloc[0]),
    }


def load_source_data(slice_dir: Path, source: str) -> pd.DataFrame:
    """Charge données d'une source (nettoyée par cleaner.py)."""
    path = slice_dir / f"{source}.parquet"
    if not path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    
    # Normaliser timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    
    return df


def load_labels(features_dir: Path, slice_id: int) -> pd.DataFrame:
    """Charge labels (is_insider, scenario) depuis label_joiner.py."""
    labels_path = features_dir / "labels" / f"slice_{slice_id:03d}" / "labels.parquet"
    
    if not labels_path.exists():
        return pd.DataFrame()
    
    return pd.read_parquet(labels_path)


def load_ldap(features_dir: Path, slice_id: int) -> pd.DataFrame:
    """Charge enrichissement LDAP depuis ldap_loader.py."""
    ldap_path = features_dir / "ldap" / f"slice_{slice_id:03d}" / "ldap.parquet"
    
    if not ldap_path.exists():
        return pd.DataFrame()
    
    return pd.read_parquet(ldap_path)


def get_all_users(slice_dir: Path, sources: List[str]) -> List[str]:
    """Liste tous les users présents dans la slice."""
    users = set()
    con = duckdb.connect()
    
    for src in sources:
        fp = slice_dir / f"{src}.parquet"
        if fp.exists():
            try:
                rows = con.execute(
                    f"SELECT DISTINCT user_id FROM read_parquet('{fp.as_posix()}') "
                    "WHERE user_id IS NOT NULL"
                ).fetchall()
                users.update(r[0] for r in rows)
            except:
                pass
    
    con.close()
    return sorted(users)


# =============================================================================
# BASELINE COMPUTATION (Historical)
# =============================================================================

def compute_baseline_from_previous(features_dir: Path, slice_id: int, 
                                   log: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Calcule baseline depuis slice PRÉCÉDENTE (évite data leakage).
    
    Retourne DataFrame avec colonnes: user_id, feature_baseline, feature_std
    """
    if slice_id <= 1:
        log.info(f"      Baseline: Première slice, pas de baseline historique")
        return None
    
    prev_path = features_dir / f"slice_{slice_id-1:03d}" / "features.parquet"
    
    if not prev_path.exists():
        log.warning(f"      Baseline: Slice précédente absente, skip baseline")
        return None
    
    try:
        prev_df = pd.read_parquet(prev_path)
        
        # Colonnes numériques à baseline
        numeric_cols = [
            'logon_count', 'logon_afterhours_count', 'device_count',
            'file_copy_count', 'http_count', 'http_wikileaks_count',
            'http_job_search_count', 'http_file_sharing_count',
            'email_sent_count', 'email_external_count'
        ]
        
        available = [c for c in numeric_cols if c in prev_df.columns]
        
        if not available:
            return None
        
        # Agréger par user (moyenne et std sur slice précédente)
        baseline = prev_df.groupby('user_id')[available].agg(['mean', 'std']).reset_index()
        baseline.columns = ['user_id'] + [f"{col}_{stat}" for col, stat in baseline.columns[1:]]
        baseline = baseline.fillna(0)
        
        log.info(f"      Baseline: {len(baseline)} users depuis slice_{slice_id-1:03d}")
        return baseline
        
    except Exception as e:
        log.error(f"      Baseline: Erreur chargement - {e}")
        return None


# =============================================================================
# FEATURE EXTRACTION BY SOURCE
# =============================================================================

def extract_logon_features(df: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    """
    Features LOGON:
    - logon_count: Nombre de logons
    - logon_unique_pcs: Nombre de PCs différents
    - logon_afterhours_count: Logons hors heures (< 6h ou >= 19h)
    - logon_hour_mean: Heure moyenne de connexion
    """
    if df.empty or not {'timestamp', 'user_id', 'pc'}.issubset(df.columns):
        return pd.DataFrame()
    
    df = df.copy()
    df['window'] = df['timestamp'].dt.floor(window)
    df['hour'] = df['timestamp'].dt.hour
    df['is_after_hours'] = (df['hour'] < 6) | (df['hour'] >= 19)
    
    agg_dict = {
        'timestamp': 'count',  # logon_count
        'pc': 'nunique',       # logon_unique_pcs
        'hour': 'mean',        # logon_hour_mean
    }
    
    result = df.groupby(['user_id', 'window']).agg(agg_dict).reset_index()
    result.columns = ['user_id', 'window', 'logon_count', 'logon_unique_pcs', 'logon_hour_mean']
    
    # After hours count
    after_hours = df[df['is_after_hours']].groupby(['user_id', 'window']).size().reset_index(name='logon_afterhours_count')
    result = result.merge(after_hours, on=['user_id', 'window'], how='left').fillna(0)
    
    return result


def extract_device_features(df: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    """
    Features DEVICE:
    - device_count: Nombre d'événements USB
    - device_connect_count: Nombre de Connect (vs Disconnect)
    """
    if df.empty or not {'timestamp', 'user_id', 'activity'}.issubset(df.columns):
        return pd.DataFrame()
    
    df = df.copy()
    df['window'] = df['timestamp'].dt.floor(window)
    
    result = df.groupby(['user_id', 'window']).size().reset_index(name='device_count')
    
    # Connect count
    connects = df[df['activity'] == 'Connect'].groupby(['user_id', 'window']).size().reset_index(name='device_connect_count')
    result = result.merge(connects, on=['user_id', 'window'], how='left').fillna(0)
    
    return result


def extract_file_features(df: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    """
    Features FILE:
    - file_copy_count: Nombre de copies de fichiers
    - file_unique_extensions: Nombre d'extensions différentes
    """
    if df.empty or not {'timestamp', 'user_id', 'filename'}.issubset(df.columns):
        return pd.DataFrame()
    
    df = df.copy()
    df['window'] = df['timestamp'].dt.floor(window)
    
    result = df.groupby(['user_id', 'window'])['filename'].count().reset_index(name='file_copy_count')
    if 'file_extension' in df.columns:
        ext = df.groupby(['user_id', 'window'])['file_extension'].nunique().reset_index(name='file_unique_extensions')
        result = result.merge(ext, on=['user_id', 'window'], how='left')
    else:
        result['file_unique_extensions'] = 0

    return result


def extract_http_features(df: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    """
    Features HTTP (utilise colonnes de cleaner.py):
    - http_count: Nombre de requêtes HTTP
    - http_wikileaks_count: Visites wikileaks (is_wikileaks de cleaner)
    - http_job_search_count: Visites job sites (is_job_search de cleaner)
    - http_file_sharing_count: Dropbox, WeTransfer (is_file_sharing de cleaner)
    - http_unique_domains: Nombre de domaines uniques
    """
    if df.empty or not {'timestamp', 'user_id'}.issubset(df.columns):
        return pd.DataFrame()
    
    df = df.copy()
    df['window'] = df['timestamp'].dt.floor(window)
    
    # Comptages de base
    result = df.groupby(['user_id', 'window']).size().reset_index(name='http_count')
    
    # Domaines uniques
    if 'domain' in df.columns:
        domains = df.groupby(['user_id', 'window'])['domain'].nunique().reset_index(name='http_unique_domains')
        result = result.merge(domains, on=['user_id', 'window'], how='left')
    else:
        result['http_unique_domains'] = 0
    
    # Wikileaks (depuis cleaner.py)
    if 'is_wikileaks' in df.columns:
        wikileaks = df[df['is_wikileaks']].groupby(['user_id', 'window']).size().reset_index(name='http_wikileaks_count')
        result = result.merge(wikileaks, on=['user_id', 'window'], how='left')
    
    # Job search (depuis cleaner.py)
    if 'is_job_search' in df.columns:
        job_search = df[df['is_job_search']].groupby(['user_id', 'window']).size().reset_index(name='http_job_search_count')
        result = result.merge(job_search, on=['user_id', 'window'], how='left')
    
    # File sharing (depuis cleaner.py)
    if 'is_file_sharing' in df.columns:
        file_sharing = df[df['is_file_sharing']].groupby(['user_id', 'window']).size().reset_index(name='http_file_sharing_count')
        result = result.merge(file_sharing, on=['user_id', 'window'], how='left')
    
    result = result.fillna(0)
    return result


def extract_email_features(df: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    """
    Features EMAIL:
    - email_sent_count: Nombre d'emails envoyés
    - email_external_count: Emails vers externe (is_external de cleaner)
    - email_with_attachments: Emails avec pièces jointes
    """
    if df.empty or not {'timestamp', 'user_id'}.issubset(df.columns):
        return pd.DataFrame()
    
    df = df.copy()
    df['window'] = df['timestamp'].dt.floor(window)
    
    result = df.groupby(['user_id', 'window']).size().reset_index(name='email_sent_count')
    
    # Externes (depuis cleaner.py)
    if 'is_external' in df.columns:
        external = df[df['is_external']].groupby(['user_id', 'window']).size().reset_index(name='email_external_count')
        result = result.merge(external, on=['user_id', 'window'], how='left')
    
    # Avec attachments
    if 'attachments' in df.columns:
        with_attach = df[df['attachments'] > 0].groupby(['user_id', 'window']).size().reset_index(name='email_with_attachments')
        result = result.merge(with_attach, on=['user_id', 'window'], how='left')
    
    result = result.fillna(0)
    return result


# =============================================================================
# FIRST-TIME DETECTION (Critical pour Scénario 1)
# =============================================================================

def add_first_time_features(df: pd.DataFrame, slice_dir: Path, 
                            slice_id: int, sources: List[str],
                            log: logging.Logger) -> pd.DataFrame:
    """
    Détecte comportements NOUVEAUX (jamais vus dans slices précédentes).
    
    Features ajoutées:
    - first_wikileaks: Première visite wikileaks
    - first_job_search: Première visite job site
    - first_device: Première utilisation USB
    - first_after_hours: Premier logon hors heures
    - first_external_email: Premier email externe
    """
    df = df.copy()
    
    # Charger historique (slices précédentes)
    historical_users = {}
    
    if slice_id > 1:
        slices_dir = slice_dir.parent
        
        for prev_id in range(1, slice_id):
            prev_dir = slices_dir / f"slice_{prev_id:03d}"
            
            # HTTP: wikileaks, job search
            http_prev = load_source_data(prev_dir, 'http')
            if not http_prev.empty:
                if 'is_wikileaks' in http_prev.columns:
                    users_wikileaks = set(http_prev[http_prev['is_wikileaks']]['user_id'].unique())
                    historical_users.setdefault('wikileaks', set()).update(users_wikileaks)
                
                if 'is_job_search' in http_prev.columns:
                    users_job = set(http_prev[http_prev['is_job_search']]['user_id'].unique())
                    historical_users.setdefault('job_search', set()).update(users_job)
            
            # Device
            device_prev = load_source_data(prev_dir, 'device')
            if not device_prev.empty:
                users_device = set(device_prev['user_id'].unique())
                historical_users.setdefault('device', set()).update(users_device)
            
            # Logon after hours
            logon_prev = load_source_data(prev_dir, 'logon')
            if not logon_prev.empty:
                logon_prev['hour'] = pd.to_datetime(logon_prev['timestamp']).dt.hour
                users_after = set(logon_prev[
                    (logon_prev['hour'] < 6) | (logon_prev['hour'] >= 19)
                ]['user_id'].unique())
                historical_users.setdefault('after_hours', set()).update(users_after)
            
            # Email external
            email_prev = load_source_data(prev_dir, 'email')
            if not email_prev.empty and 'is_external' in email_prev.columns:
                users_ext = set(email_prev[email_prev['is_external']]['user_id'].unique())
                historical_users.setdefault('external_email', set()).update(users_ext)
    
    # Détecter first-time dans slice actuelle
    df['first_wikileaks'] = (~df['user_id'].isin(historical_users.get('wikileaks', set())) & 
                             (df.get('http_wikileaks_count', 0) > 0)).astype(int)
    
    df['first_job_search'] = (~df['user_id'].isin(historical_users.get('job_search', set())) & 
                              (df.get('http_job_search_count', 0) > 0)).astype(int)
    
    df['first_device'] = (~df['user_id'].isin(historical_users.get('device', set())) & 
                          (df.get('device_count', 0) > 0)).astype(int)
    
    df['first_after_hours'] = (~df['user_id'].isin(historical_users.get('after_hours', set())) & 
                               (df.get('logon_afterhours_count', 0) > 0)).astype(int)
    
    df['first_external_email'] = (~df['user_id'].isin(historical_users.get('external_email', set())) & 
                                  (df.get('email_external_count', 0) > 0)).astype(int)
    
    n_first = df[['first_wikileaks', 'first_job_search', 'first_device', 
                  'first_after_hours', 'first_external_email']].sum().sum()
    
    log.info(f"      First-time: {int(n_first)} comportements nouveaux détectés")
    
    return df


# =============================================================================
# DEVIATIONS vs BASELINE
# =============================================================================

def add_deviation_features(df: pd.DataFrame, baseline: Optional[pd.DataFrame],
                          log: logging.Logger) -> pd.DataFrame:
    """
    Ajoute déviations par rapport à baseline historique.
    
    Features ajoutées:
    - *_deviation: Différence absolue vs baseline
    - *_zscore: Z-score normalisé
    """
    if baseline is None or baseline.empty:
        log.info(f"      Déviations: Pas de baseline, skip")
        return df
    
    df = df.copy()
    df = df.merge(baseline, on='user_id', how='left')
    
    # Features à calculer déviations
    features = [
        'logon_count', 'logon_afterhours_count', 'device_count',
        'http_count', 'email_sent_count'
    ]
    
    for feat in features:
        baseline_col = f"{feat}_mean"
        std_col = f"{feat}_std"
        
        if feat in df.columns and baseline_col in df.columns:
            # Déviation absolue
            df[f"{feat}_deviation"] = df[feat] - df[baseline_col]
            
            # Z-score
            if std_col in df.columns:
                df[f"{feat}_zscore"] = (df[feat] - df[baseline_col]) / (df[std_col] + 1e-6)
            
            # Nettoyer colonnes temporaires
            df = df.drop(columns=[baseline_col, std_col], errors='ignore')
    
    log.info(f"      Déviations: Calculées pour {len(features)} features")
    
    return df


# =============================================================================
# COMPOSITE FEATURES
# =============================================================================

def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features composites / scores d'escalation.
    
    Features ajoutées:
    - escalation_score: Score composite de risque
    - after_hours_device_combo: Combinaison after-hours + device
    - exfiltration_score: Job search + device + external email
    """
    df = df.copy()
    
    # Escalation score (critique Scénario 1 & 2)
    df['escalation_score'] = (
        0.5 * df.get('first_wikileaks', 0) +
        0.3 * df.get('first_after_hours', 0) +
        0.2 * df.get('first_device', 0) +
        0.3 * (df.get('http_job_search_count', 0) > 5).astype(int) +
        0.3 * (df.get('device_count', 0) > 3).astype(int)
    )
    df['escalation_score'] = df['escalation_score'].clip(0, 1)
    
    # After-hours + device combo (Scénario 1)
    df['after_hours_device_combo'] = (
        df.get('logon_afterhours_count', 0) * df.get('device_count', 0)
    )
    
    # Exfiltration score (Scénario 2)
    df['exfiltration_score'] = (
        0.4 * (df.get('http_job_search_count', 0) > 10).astype(int) +
        0.3 * (df.get('device_count', 0) > 5).astype(int) +
        0.3 * (df.get('email_external_count', 0) > 3).astype(int)
    )
    df['exfiltration_score'] = df['exfiltration_score'].clip(0, 1)
    
    return df


# =============================================================================
# LDAP INTEGRATION
# =============================================================================

def add_ldap_features(df: pd.DataFrame, ldap_df: pd.DataFrame,
                     log: logging.Logger) -> pd.DataFrame:
    """
    Intègre features LDAP (RH).
    
    Features ajoutées:
    - role, department, business_unit (catégorielles)
    - tenure_days (ancienneté)
    - is_it_admin (calculé depuis role)
    - changed_department, changed_role (si LDAP dynamique)
    - is_new_employee
    """
    if ldap_df.empty:
        log.warning(f"      LDAP: Pas de données LDAP, skip")
        return df
    
    df = df.copy()
    
    # Colonnes LDAP à intégrer
    ldap_cols = ['user_id', 'role', 'department', 'business_unit', 
                 'tenure_days', 'is_new_employee']
    
    # Ajouter changed_* si LDAP dynamique
    if not ldap_df.get('ldap_is_static', pd.Series([True])).iloc[0]:
        ldap_cols.extend(['changed_department', 'changed_role', 'changed_supervisor'])
    
    available_cols = [c for c in ldap_cols if c in ldap_df.columns]
    
    df = df.merge(ldap_df[available_cols], on='user_id', how='left')

    # Normaliser types catégoriels LDAP pour éviter les colonnes object mixtes
    for col in ['role', 'department', 'business_unit']:
        if col in df.columns:
            df[col] = df[col].astype('string')
    
    # Calculer is_it_admin depuis role
    if 'role' in df.columns:
        df['is_it_admin'] = df['role'].str.contains(
            'Admin|IT|System|Tech', case=False, na=False
        ).astype(int)
    
    # Interactions IT
    if 'is_it_admin' in df.columns:
        df['device_non_it'] = df.get('device_count', 0) * (1 - df['is_it_admin'])
        df['after_hours_non_it'] = df.get('logon_afterhours_count', 0) * (1 - df['is_it_admin'])
    
    log.info(f"      LDAP: {len(available_cols)} features intégrées")
    
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_slice(slice_id: int, cfg: dict, log: logging.Logger) -> None:
    """
    Traite une slice complète: extraction features + enrichissements.
    """
    
    slices_dir = Path(cfg["output"]["slices_dir"])
    features_dir = Path(cfg["output"]["features_dir"])
    sources = cfg["data"]["sources"]
    
    slice_dir = slices_dir / f"slice_{slice_id:03d}"
    out_dir = features_dir / f"slice_{slice_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"\n{'═'*70}")
    log.info(f"  🔧 SLICE {slice_id:03d} - Feature Engineering")
    log.info(f"{'═'*70}")
    
    # 1. Charger métadonnées
    meta = load_slice_meta(slice_dir)
    if meta is None:
        log.error(f"  ✗ meta.parquet absent, skip")
        return
    
    log.info(f"      Période: {meta['start'].date()} → {meta['end'].date()}")
    
    # 2. Charger données sources
    log.info(f"      Chargement sources...")
    
    logon_df = load_source_data(slice_dir, 'logon')
    device_df = load_source_data(slice_dir, 'device')
    file_df = load_source_data(slice_dir, 'file')
    http_df = load_source_data(slice_dir, 'http')
    email_df = load_source_data(slice_dir, 'email')
    
    # 3. Extraction features par source
    log.info(f"      Extraction features...")
    
    window = "1h"  # Granularité user-heure
    
    feat_logon = extract_logon_features(logon_df, window)
    feat_device = extract_device_features(device_df, window)
    feat_file = extract_file_features(file_df, window)
    feat_http = extract_http_features(http_df, window)
    feat_email = extract_email_features(email_df, window)
    
    # 4. Merge toutes les features
    features_parts = [feat_logon, feat_device, feat_file, feat_http, feat_email]
    features_parts = [f for f in features_parts if not f.empty]
    
    if not features_parts:
        log.error(f"  ✗ Aucune feature extraite, skip")
        return
    
    df = features_parts[0]
    for part in features_parts[1:]:
        df = df.merge(part, on=['user_id', 'window'], how='outer')
    
    df = df.fillna(0)
    
    log.info(f"      Features brutes: {df.shape}")
    
    # Libérer mémoire
    del logon_df, device_df, file_df, http_df, email_df
    gc.collect()
    
    # 5. First-time detection
    df = add_first_time_features(df, slice_dir, slice_id, sources, log)
    
    # 6. Baseline & déviations
    baseline = compute_baseline_from_previous(features_dir, slice_id, log)
    df = add_deviation_features(df, baseline, log)
    
    # 7. Composite features
    df = add_composite_features(df)
    log.info(f"      Composite: escalation_score, exfiltration_score ajoutés")
    
    # 8. LDAP integration
    ldap_df = load_ldap(features_dir, slice_id)
    df = add_ldap_features(df, ldap_df, log)
    
    # 9. Labels (supervision)
    labels_df = load_labels(features_dir, slice_id)
    if not labels_df.empty:
        # Joindre labels par user (pas par window)
        # Un user est insider si ANY window est insider
        df = df.merge(
            labels_df[['user_id', 'is_insider', 'scenario']],
            on='user_id',
            how='left'
        )
        df['is_insider'] = df['is_insider'].fillna(0).astype(int)
        
        n_insider_windows = df['is_insider'].sum()
        log.info(f"      Labels: {n_insider_windows:,} user-heures insiders")
    else:
        log.warning(f"      Labels: Absents, pas de supervision")
        df['is_insider'] = 0
        df['scenario'] = None
    
    # 10. Nettoyage final
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(0)

    bool_cols = [c for c in ['is_new_employee', 'changed_department', 'changed_role', 'changed_supervisor'] if c in df.columns]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool)

    text_cols = [c for c in ['role', 'department', 'business_unit'] if c in df.columns]
    for col in text_cols:
        df[col] = df[col].astype('string').fillna('')
    
    # Colonnes finales
    log.info(f"      Features finales: {len(df.columns)} colonnes")
    
    # 11. Sauvegarde
    out_path = out_dir / "features.parquet"
    df.to_parquet(out_path, compression='snappy', index=False)
    
    file_size_mb = out_path.stat().st_size / 1024 / 1024
    
    log.info(f"  {'─'*70}")
    log.info(f"  ✓ Sauvegardé: {out_path}")
    log.info(f"    Lignes: {len(df):,} │ Colonnes: {len(df.columns)} │ Taille: {file_size_mb:.1f} MB")
    log.info(f"    Insiders: {df['is_insider'].sum():,} ({df['is_insider'].mean()*100:.2f}%)")


def run(cfg: dict, log: logging.Logger, slice_ids: Optional[List[int]] = None) -> None:
    """
    Pipeline principal feature engineering.
    """
    
    slices_dir = Path(cfg["output"]["slices_dir"])
    
    # Liste des slices à traiter
    if slice_ids is None:
        slice_ids = sorted(
            int(p.name.split("_")[1])
            for p in slices_dir.iterdir()
            if p.is_dir() and p.name.startswith("slice_")
        )
    
    log.info("=" * 70)
    log.info("FEATURE ENGINEERING PRODUCTION - UEBA CERT r4.2")
    log.info("=" * 70)
    log.info(f"Slices à traiter: {slice_ids}")
    log.info(f"Granularité: user-heure (optimal)")
    log.info(f"Features: 40+ optimisées pour insider threat")
    
    t0 = datetime.now()
    
    for sid in slice_ids:
        try:
            process_slice(sid, cfg, log)
        except Exception as e:
            log.error(f"  ✗ Slice {sid:03d} erreur: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = (datetime.now() - t0).total_seconds() / 60
    
    log.info(f"\n{'='*70}")
    log.info(f"✅ Feature Engineering terminé - {elapsed:.1f} minutes")
    log.info(f"{'='*70}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        cfg = load_config()
        log = get_logger(cfg["output"]["logs_dir"])
        
        # Option: traiter une seule slice
        slice_ids = None
        if "--slice" in sys.argv:
            idx = sys.argv.index("--slice")
            if idx + 1 < len(sys.argv):
                slice_ids = [int(sys.argv[idx + 1])]
        
        run(cfg, log, slice_ids)
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Configuration invalide: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
03_feature_engineering_ultimate.py - VERSION ULTIME
====================================================
Feature Engineering UEBA production-grade pour CERT r4.2

OPTIMISATIONS ULTIMES:
✅ Granularité USER-HEURE (détection temps-réel précise)
✅ Rolling windows: 1J, 7J, 30J (patterns court/moyen/long terme)
✅ Cache historique incrémental (performance optimale)
✅ Features séquentielles avancées
✅ Comparaisons temporelles multi-échelles
✅ 80+ features optimisées

COMPATIBLE AVEC:
- slicer.py, cleaner.py, label_joiner.py, ldap_loader.py

FEATURES PRODUITES: 80+ features multi-échelles temporelles

Input:  
  - data/01_slices/slice_NNN/*.parquet (cleaned)
  - data/03_features/labels/slice_NNN/labels.parquet
  - data/03_features/ldap/slice_NNN/ldap.parquet
  - data/03_features/cache/historical_behaviors_NNN.parquet

Output:
  - data/03_features/slice_NNN/features.parquet

USAGE:
  python 03_feature_engineering_ultimate.py
  python 03_feature_engineering_ultimate.py --slice 1
  python 03_feature_engineering_ultimate.py --rebuild-cache
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
    """Charge et valide la configuration."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    required = {
        "data": ["raw_dir", "timestamp_format", "sources"],
        "output": ["slices_dir", "features_dir", "logs_dir"],
    }
    for section, keys in required.items():
        if section not in cfg:
            raise ValueError(f"Section manquante: {section}")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Clé manquante: {section}.{key}")
    
    return cfg


def get_logger(logs_dir: str) -> logging.Logger:
    """Configure le logger."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger = logging.getLogger("FeatureEngineering")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fmt = logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s")
    
    fh = logging.FileHandler(
        Path(logs_dir) / f"03_features_ultimate_{ts}.log",
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
    """Charge métadonnées slice."""
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
    """Charge données d'une source."""
    path = slice_dir / f"{source}.parquet"
    if not path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    
    return df


def load_labels(features_dir: Path, slice_id: int) -> pd.DataFrame:
    """Charge labels."""
    labels_path = features_dir / "labels" / f"slice_{slice_id:03d}" / "labels.parquet"
    if not labels_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(labels_path)


def load_ldap(features_dir: Path, slice_id: int) -> pd.DataFrame:
    """Charge LDAP."""
    ldap_path = features_dir / "ldap" / f"slice_{slice_id:03d}" / "ldap.parquet"
    if not ldap_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(ldap_path)


# =============================================================================
# CACHE HISTORIQUE
# =============================================================================

def load_historical_cache(features_dir: Path, slice_id: int,
                         log: logging.Logger) -> Optional[pd.DataFrame]:
    """Charge cache historique."""
    cache_dir = features_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if slice_id <= 1:
        log.info(f"      Cache: Première slice")
        return pd.DataFrame(columns=[
            'user_id', 'has_visited_wikileaks', 'has_searched_jobs',
            'has_used_device', 'has_logon_after_hours', 'has_sent_external_email'
        ])
    
    cache_path = cache_dir / f"historical_behaviors_{slice_id-1:03d}.parquet"
    
    if not cache_path.exists():
        log.warning(f"      Cache: Absent, reconstruction...")
        return None
    
    cache = pd.read_parquet(cache_path)
    log.info(f"      Cache: {len(cache)} users chargés")
    return cache


def build_historical_cache(features_dir: Path, slices_dir: Path,
                          slice_id: int, sources: List[str],
                          log: logging.Logger) -> pd.DataFrame:
    """Reconstruit cache historique."""
    log.info(f"      Reconstruction cache: slices 1 → {slice_id-1}")
    
    behaviors = {
        'user_id': set(),
        'has_visited_wikileaks': set(),
        'has_searched_jobs': set(),
        'has_used_device': set(),
        'has_logon_after_hours': set(),
        'has_sent_external_email': set()
    }
    
    for prev_id in range(1, slice_id):
        prev_dir = slices_dir / f"slice_{prev_id:03d}"
        
        http_df = load_source_data(prev_dir, 'http')
        if not http_df.empty:
            if 'is_wikileaks' in http_df.columns:
                behaviors['has_visited_wikileaks'].update(
                    http_df[http_df['is_wikileaks'] == True]['user_id'].unique()
                )
            if 'is_job_search' in http_df.columns:
                behaviors['has_searched_jobs'].update(
                    http_df[http_df['is_job_search'] == True]['user_id'].unique()
                )
        
        device_df = load_source_data(prev_dir, 'device')
        if not device_df.empty:
            behaviors['has_used_device'].update(device_df['user_id'].unique())
        
        logon_df = load_source_data(prev_dir, 'logon')
        if not logon_df.empty:
            logon_df['hour'] = pd.to_datetime(logon_df['timestamp']).dt.hour
            behaviors['has_logon_after_hours'].update(
                logon_df[(logon_df['hour'] < 6) | (logon_df['hour'] >= 19)]['user_id'].unique()
            )
        
        email_df = load_source_data(prev_dir, 'email')
        if not email_df.empty and 'is_external' in email_df.columns:
            behaviors['has_sent_external_email'].update(
                email_df[email_df['is_external'] == True]['user_id'].unique()
            )
        
        for src_df in [http_df, device_df, logon_df, email_df]:
            if not src_df.empty and 'user_id' in src_df.columns:
                behaviors['user_id'].update(src_df['user_id'].unique())
    
    all_users = sorted(behaviors['user_id'])
    cache_df = pd.DataFrame({'user_id': all_users})
    
    for col in ['has_visited_wikileaks', 'has_searched_jobs', 'has_used_device',
                'has_logon_after_hours', 'has_sent_external_email']:
        cache_df[col] = cache_df['user_id'].isin(behaviors[col]).astype(int)
    
    log.info(f"      Cache: {len(cache_df)} users reconstruits")
    return cache_df


def update_cache(cache_df: pd.DataFrame, slice_dir: Path, slice_id: int,
                features_dir: Path, log: logging.Logger) -> None:
    """Met à jour cache avec slice actuelle."""
    cache_dir = features_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    http_df = load_source_data(slice_dir, 'http')
    device_df = load_source_data(slice_dir, 'device')
    logon_df = load_source_data(slice_dir, 'logon')
    email_df = load_source_data(slice_dir, 'email')
    
    all_users = set(cache_df['user_id'])
    for src_df in [http_df, device_df, logon_df, email_df]:
        if not src_df.empty and 'user_id' in src_df.columns:
            all_users.update(src_df['user_id'].unique())
    
    updated_df = pd.DataFrame({'user_id': sorted(all_users)})
    updated_df = updated_df.merge(cache_df, on='user_id', how='left').fillna(0)
    
    if not http_df.empty:
        if 'is_wikileaks' in http_df.columns:
            new_wiki = set(http_df[http_df['is_wikileaks'] == True]['user_id'].unique())
            updated_df.loc[updated_df['user_id'].isin(new_wiki), 'has_visited_wikileaks'] = 1
        
        if 'is_job_search' in http_df.columns:
            new_job = set(http_df[http_df['is_job_search'] == True]['user_id'].unique())
            updated_df.loc[updated_df['user_id'].isin(new_job), 'has_searched_jobs'] = 1
    
    if not device_df.empty:
        new_device = set(device_df['user_id'].unique())
        updated_df.loc[updated_df['user_id'].isin(new_device), 'has_used_device'] = 1
    
    if not logon_df.empty:
        logon_df['hour'] = pd.to_datetime(logon_df['timestamp']).dt.hour
        new_after = set(logon_df[(logon_df['hour'] < 6) | (logon_df['hour'] >= 19)]['user_id'].unique())
        updated_df.loc[updated_df['user_id'].isin(new_after), 'has_logon_after_hours'] = 1
    
    if not email_df.empty and 'is_external' in email_df.columns:
        new_ext = set(email_df[email_df['is_external'] == True]['user_id'].unique())
        updated_df.loc[updated_df['user_id'].isin(new_ext), 'has_sent_external_email'] = 1
    
    cache_path = cache_dir / f"historical_behaviors_{slice_id:03d}.parquet"
    updated_df.to_parquet(cache_path, index=False)
    log.info(f"      Cache: Mis à jour ({len(updated_df)} users)")


# =============================================================================
# EXTRACTION FEATURES PAR SOURCE (USER-HEURE)
# =============================================================================

def extract_logon_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features LOGON (granularité USER-HEURE)."""
    if df.empty or not {'timestamp', 'user_id', 'pc'}.issubset(df.columns):
        return pd.DataFrame()
    
    df = df.copy()
    df['window'] = df['timestamp'].dt.floor('1h')
    df['hour'] = df['timestamp'].dt.hour
    df['is_after_hours'] = (df['hour'] < 6) | (df['hour'] >= 19)
    
    agg_dict = {
        'timestamp': 'count',
        'pc': 'nunique',
        'hour': ['mean', 'std']
    }
    
    result = df.groupby(['user_id', 'window']).agg(agg_dict).reset_index()
    result.columns = ['user_id', 'window', 'logon_count', 'logon_unique_pcs',
                     'logon_hour_mean', 'logon_hour_std']
    
    after_hours = df[df['is_after_hours']].groupby(['user_id', 'window']).size().reset_index(name='logon_afterhours_count')
    result = result.merge(after_hours, on=['user_id', 'window'], how='left').fillna(0)
    
    result['logon_hour_std'] = result['logon_hour_std'].fillna(0)
    
    return result


def extract_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features DEVICE (granularité USER-HEURE)."""
    if df.empty or not {'timestamp', 'user_id', 'activity'}.issubset(df.columns):
        return pd.DataFrame()
    
    df = df.copy()
    df['window'] = df['timestamp'].dt.floor('1h')
    
    result = df.groupby(['user_id', 'window']).size().reset_index(name='device_count')
    
    connects = df[df['activity'] == 'Connect'].groupby(['user_id', 'window']).size().reset_index(name='device_connect_count')
    result = result.merge(connects, on=['user_id', 'window'], how='left').fillna(0)
    
    return result


def extract_file_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features FILE (granularité USER-HEURE)."""
    if df.empty or not {'timestamp', 'user_id', 'filename'}.issubset(df.columns):
        return pd.DataFrame()
    
    df = df.copy()
    df['window'] = df['timestamp'].dt.floor('1h')
    
    result = df.groupby(['user_id', 'window'])['filename'].count().reset_index(name='file_copy_count')
    
    if 'file_extension' in df.columns:
        ext = df.groupby(['user_id', 'window'])['file_extension'].nunique().reset_index(name='file_unique_extensions')
        result = result.merge(ext, on=['user_id', 'window'], how='left')
    else:
        result['file_unique_extensions'] = 0
    
    return result


def extract_http_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features HTTP (granularité USER-HEURE)."""
    if df.empty or not {'timestamp', 'user_id'}.issubset(df.columns):
        return pd.DataFrame()
    
    df = df.copy()
    df['window'] = df['timestamp'].dt.floor('1h')
    
    result = df.groupby(['user_id', 'window']).size().reset_index(name='http_count')
    
    if 'domain' in df.columns:
        domains = df.groupby(['user_id', 'window'])['domain'].nunique().reset_index(name='http_unique_domains')
        result = result.merge(domains, on=['user_id', 'window'], how='left')
    else:
        result['http_unique_domains'] = 0
    
    if 'is_wikileaks' in df.columns:
        wikileaks = df[df['is_wikileaks'] == True].groupby(['user_id', 'window']).size().reset_index(name='http_wikileaks_count')
        result = result.merge(wikileaks, on=['user_id', 'window'], how='left')
    
    if 'is_job_search' in df.columns:
        job_search = df[df['is_job_search'] == True].groupby(['user_id', 'window']).size().reset_index(name='http_job_search_count')
        result = result.merge(job_search, on=['user_id', 'window'], how='left')
    
    if 'is_file_sharing' in df.columns:
        file_sharing = df[df['is_file_sharing'] == True].groupby(['user_id', 'window']).size().reset_index(name='http_file_sharing_count')
        result = result.merge(file_sharing, on=['user_id', 'window'], how='left')
    
    result = result.fillna(0)
    return result


def extract_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features EMAIL (granularité USER-HEURE)."""
    if df.empty or not {'timestamp', 'user_id'}.issubset(df.columns):
        return pd.DataFrame()
    
    df = df.copy()
    df['window'] = df['timestamp'].dt.floor('1h')
    
    result = df.groupby(['user_id', 'window']).size().reset_index(name='email_sent_count')
    
    if 'is_external' in df.columns:
        external = df[df['is_external'] == True].groupby(['user_id', 'window']).size().reset_index(name='email_external_count')
        result = result.merge(external, on=['user_id', 'window'], how='left')
    
    if 'attachments' in df.columns:
        with_attach = df[df['attachments'] > 0].groupby(['user_id', 'window']).size().reset_index(name='email_with_attachments')
        result = result.merge(with_attach, on=['user_id', 'window'], how='left')
    
    if 'size' in df.columns:
        avg_size = df.groupby(['user_id', 'window'])['size'].mean().reset_index(name='email_avg_size')
        result = result.merge(avg_size, on=['user_id', 'window'], how='left')
    
    result = result.fillna(0)
    return result


# =============================================================================
# ROLLING WINDOWS MULTI-ÉCHELLES (1J, 7J, 30J)
# =============================================================================

def add_rolling_features_multiscale(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Ajoute rolling windows à 3 échelles temporelles.
    
    ÉCHELLES:
    - 1J (24h): Patterns court terme, détection rapide
    - 7J (1 semaine): Patterns moyen terme, escalation
    - 30J (1 mois): Patterns long terme, tendances
    
    Pour chaque feature critique, calcule:
    - sum_1d, sum_7d, sum_30d: Sommes cumulées
    - mean_1d, mean_7d, mean_30d: Moyennes
    - trend_7d, trend_30d: Ratio actuel/moyenne (détecte escalation)
    """
    df = df.copy()
    df = df.sort_values(['user_id', 'window']).reset_index(drop=True)
    
    # Features critiques à calculer multi-échelle
    features_critical = [
        'http_wikileaks_count',
        'http_job_search_count',
        'http_file_sharing_count',
        'device_count',
        'logon_afterhours_count',
        'email_external_count'
    ]
    
    for feat in features_critical:
        if feat not in df.columns:
            continue
        
        # === 1 JOUR (24 heures) ===
        df[f'{feat}_sum_1d'] = df.groupby('user_id')[feat].transform(
            lambda x: x.rolling(window=24, min_periods=1).sum()
        )
        df[f'{feat}_mean_1d'] = df.groupby('user_id')[feat].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
        
        # === 7 JOURS (168 heures) ===
        df[f'{feat}_sum_7d'] = df.groupby('user_id')[feat].transform(
            lambda x: x.rolling(window=168, min_periods=1).sum()
        )
        df[f'{feat}_mean_7d'] = df.groupby('user_id')[feat].transform(
            lambda x: x.rolling(window=168, min_periods=1).mean()
        )
        
        # Trend 7j: actuel / moyenne_7j (>1 = augmentation)
        df[f'{feat}_trend_7d'] = df[feat] / (df[f'{feat}_mean_7d'] + 1e-6)
        df[f'{feat}_trend_7d'] = df[f'{feat}_trend_7d'].fillna(1.0).clip(0, 10)
        
        # === 30 JOURS (720 heures) ===
        df[f'{feat}_sum_30d'] = df.groupby('user_id')[feat].transform(
            lambda x: x.rolling(window=720, min_periods=1).sum()
        )
        df[f'{feat}_mean_30d'] = df.groupby('user_id')[feat].transform(
            lambda x: x.rolling(window=720, min_periods=1).mean()
        )
        
        # Trend 30j: actuel / moyenne_30j
        df[f'{feat}_trend_30d'] = df[feat] / (df[f'{feat}_mean_30d'] + 1e-6)
        df[f'{feat}_trend_30d'] = df[f'{feat}_trend_30d'].fillna(1.0).clip(0, 10)
    
    log.info(f"      Rolling: 1J/7J/30J calculés pour {len(features_critical)} features")
    
    return df


# =============================================================================
# FIRST-TIME DETECTION
# =============================================================================

def add_first_time_features(df: pd.DataFrame, cache_df: Optional[pd.DataFrame],
                           log: logging.Logger) -> pd.DataFrame:
    """D?tecte comportements nouveaux (optimis? avec cache)."""
    df = df.copy()

    def _col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(0, index=df.index)

    if cache_df is None or cache_df.empty:
        log.warning(f"      First-time: Pas de cache, tout marqu? 'first'")
        df['first_wikileaks'] = (_col('http_wikileaks_count') > 0).astype(int)
        df['first_job_search'] = (_col('http_job_search_count') > 0).astype(int)
        df['first_device'] = (_col('device_count') > 0).astype(int)
        df['first_after_hours'] = (_col('logon_afterhours_count') > 0).astype(int)
        df['first_external_email'] = (_col('email_external_count') > 0).astype(int)
        df['first_file_sharing'] = (_col('http_file_sharing_count') > 0).astype(int)
    else:
        df = df.merge(cache_df, on='user_id', how='left')

        for col in ['has_visited_wikileaks', 'has_searched_jobs', 'has_used_device',
                   'has_logon_after_hours', 'has_sent_external_email']:
            df[col] = df[col].fillna(0).astype(int)

        df['first_wikileaks'] = (
            (df['has_visited_wikileaks'] == 0) & (_col('http_wikileaks_count') > 0)
        ).astype(int)

        df['first_job_search'] = (
            (df['has_searched_jobs'] == 0) & (_col('http_job_search_count') > 0)
        ).astype(int)

        df['first_device'] = (
            (df['has_used_device'] == 0) & (_col('device_count') > 0)
        ).astype(int)

        df['first_after_hours'] = (
            (df['has_logon_after_hours'] == 0) & (_col('logon_afterhours_count') > 0)
        ).astype(int)

        df['first_external_email'] = (
            (df['has_sent_external_email'] == 0) & (_col('email_external_count') > 0)
        ).astype(int)

        df['first_file_sharing'] = (_col('http_file_sharing_count') > 0).astype(int)

        df = df.drop(columns=['has_visited_wikileaks', 'has_searched_jobs', 'has_used_device',
                             'has_logon_after_hours', 'has_sent_external_email'], errors='ignore')

    n_first = df[['first_wikileaks', 'first_job_search', 'first_device',
                  'first_after_hours', 'first_external_email', 'first_file_sharing']].sum().sum()

    log.info(f"      First-time: {int(n_first)} nouveaux comportements")

    return df

# =============================================================================
# FEATURES TEMPORELLES
# =============================================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features temporelles enrichies."""
    df = df.copy()
    df['window_dt'] = pd.to_datetime(df['window'])
    
    df['hour_of_day'] = df['window_dt'].dt.hour
    df['day_of_week'] = df['window_dt'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
    df['is_business_hours'] = ((df['hour_of_day'] >= 8) & (df['hour_of_day'] <= 18) & 
                               (df['day_of_week'] < 5)).astype(int)
    
    df = df.drop(columns=['window_dt'], errors='ignore')
    
    return df


# =============================================================================
# COMPOSITE SCORES
# =============================================================================

def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scores composites multi-?chelles."""
    df = df.copy()

    def _col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(0, index=df.index)

    # Escalation score COURT TERME (1J)
    df['escalation_score_1d'] = (
        0.5 * _col('first_wikileaks') +
        0.3 * _col('first_after_hours') +
        0.2 * _col('first_device') +
        0.3 * (_col('http_job_search_count_sum_1d') > 3).astype(int) +
        0.2 * (_col('device_count_sum_1d') > 2).astype(int)
    ).clip(0, 1)

    # Escalation score MOYEN TERME (7J)
    df['escalation_score_7d'] = (
        0.4 * (_col('http_wikileaks_count_sum_7d') > 0).astype(int) +
        0.3 * (_col('http_job_search_count_sum_7d') > 15).astype(int) +
        0.3 * (_col('device_count_sum_7d') > 10).astype(int) +
        0.2 * (_col('device_count_trend_7d') > 2).astype(int)
    ).clip(0, 1)

    # Escalation score LONG TERME (30J)
    df['escalation_score_30d'] = (
        0.3 * (_col('http_job_search_count_sum_30d') > 50).astype(int) +
        0.3 * (_col('device_count_sum_30d') > 30).astype(int) +
        0.2 * (_col('email_external_count_sum_30d') > 20).astype(int) +
        0.2 * (_col('device_count_trend_30d') > 3).astype(int)
    ).clip(0, 1)

    # Exfiltration score (multi-?chelles)
    df['exfiltration_score'] = (
        0.3 * _col('escalation_score_1d') +
        0.4 * _col('escalation_score_7d') +
        0.3 * _col('escalation_score_30d')
    ).clip(0, 1)

    # Combo suspects
    df['after_hours_device_combo'] = (
        _col('logon_afterhours_count') * _col('device_count')
    )

    df['suspicious_combo_1d'] = (
        (_col('logon_afterhours_count') > 0) &
        (_col('device_count') > 0) &
        (_col('http_wikileaks_count') > 0)
    ).astype(int)

    return df

# =============================================================================
# LDAP FEATURES
# =============================================================================

def add_ldap_features(df: pd.DataFrame, ldap_df: pd.DataFrame,
                     log: logging.Logger) -> pd.DataFrame:
    """Intègre features LDAP."""
    if ldap_df.empty:
        log.warning(f"      LDAP: Pas de données")
        return df
    if 'user_id' not in ldap_df.columns:
        log.warning(f"      LDAP: Colonne user_id absente")
        return df
    
    df = df.copy()
    
    ldap_cols = ['user_id', 'role', 'department', 'business_unit']
    available_cols = [c for c in ldap_cols if c in ldap_df.columns]
    
    df = df.merge(ldap_df[available_cols], on='user_id', how='left')
    
    for col in ['role', 'department', 'business_unit']:
        if col in df.columns:
            df[col] = df[col].astype('string').fillna('')
    
    if 'role' in df.columns:
        df['is_it_admin'] = df['role'].str.contains(
            'Admin|IT|System|Tech', case=False, na=False
        ).astype(int)
    
    if 'is_it_admin' in df.columns:
        df['device_non_it'] = df.get('device_count', 0) * (1 - df['is_it_admin'])
        df['after_hours_non_it'] = df.get('logon_afterhours_count', 0) * (1 - df['is_it_admin'])
    
    log.info(f"      LDAP: {len(available_cols)} features intégrées")
    
    return df


# =============================================================================
# REALTIME V4 COMPAT FEATURES
# =============================================================================

def _compat_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce').fillna(0)
    return pd.Series(0.0, index=df.index)


def _compat_roll_user(
    df: pd.DataFrame,
    source_col: str,
    win: int,
    op: str = 'mean',
    min_periods: int = 1,
    shift_first: bool = False,
) -> pd.Series:
    if 'user_id' not in df.columns:
        return pd.Series(0.0, index=df.index)

    s = _compat_num(df, source_col)
    if shift_first:
        s = s.groupby(df['user_id']).shift(1).fillna(0)

    grouped = s.groupby(df['user_id'])
    if op == 'mean':
        return grouped.transform(lambda x: x.rolling(win, min_periods=min_periods).mean()).fillna(0)
    if op == 'sum':
        return grouped.transform(lambda x: x.rolling(win, min_periods=min_periods).sum()).fillna(0)
    if op == 'std':
        return grouped.transform(lambda x: x.rolling(win, min_periods=min_periods).std()).fillna(0)
    if op == 'max':
        return grouped.transform(lambda x: x.rolling(win, min_periods=min_periods).max()).fillna(0)
    raise ValueError(f"Unsupported op: {op}")


def _compat_ensure_roll(
    df: pd.DataFrame,
    source_col: str,
    win: int,
    out_col: str,
    op: str = 'mean',
    min_periods: int = 1,
    shift_first: bool = False,
) -> None:
    if out_col not in df.columns:
        df[out_col] = _compat_roll_user(
            df,
            source_col=source_col,
            win=win,
            op=op,
            min_periods=min_periods,
            shift_first=shift_first,
        )


def add_realtime_v4_compat_features(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """
    Add the contextual + smart user-hour + scenario-oriented features required by
    the realtime causal v4 family directly during feature engineering.
    """
    if 'user_id' not in df.columns or 'window' not in df.columns:
        log.warning("      Compat v4: skipped (missing user_id/window)")
        return df

    out = df.copy()
    before_cols = set(out.columns)
    out = out.sort_values(['user_id', 'window']).reset_index(drop=True)

    _compat_ensure_roll(out, 'device_count', 24, 'device_count_mean_1d', op='mean')
    _compat_ensure_roll(out, 'device_count', 168, 'device_count_mean_7d', op='mean')
    _compat_ensure_roll(out, 'device_count', 24, 'device_count_sum_1d', op='sum')
    _compat_ensure_roll(out, 'device_count', 168, 'device_count_sum_7d', op='sum')
    _compat_ensure_roll(out, 'http_job_search_count', 24, 'http_job_search_count_sum_1d', op='sum')
    _compat_ensure_roll(out, 'http_wikileaks_count', 24, 'http_wikileaks_count_sum_1d', op='sum')
    _compat_ensure_roll(out, 'http_job_search_count', 720, 'http_job_search_count_sum_30d', op='sum')

    logon = _compat_num(out, 'logon_count')
    after = _compat_num(out, 'logon_afterhours_count')
    device = _compat_num(out, 'device_count')
    http = _compat_num(out, 'http_count')
    wiki = _compat_num(out, 'http_wikileaks_count')
    job = _compat_num(out, 'http_job_search_count')
    sharing = _compat_num(out, 'http_file_sharing_count')
    email = _compat_num(out, 'email_sent_count')
    email_ext = _compat_num(out, 'email_external_count')
    file_copy = _compat_num(out, 'file_copy_count')

    out['device_burst'] = (device / (_compat_num(out, 'device_count_mean_1d') + 1)).clip(0, 10)
    out['http_mean_24h'] = _compat_roll_user(out, 'http_count', 24, op='mean')
    out['http_burst'] = (http / (_compat_num(out, 'http_mean_24h') + 1)).clip(0, 10)
    out['device_spike'] = (device > 3 * _compat_num(out, 'device_count_mean_7d')).astype(int)
    email_mean_7d = _compat_roll_user(out, 'email_sent_count', 168, op='mean')
    out['email_spike'] = (email > 3 * email_mean_7d).astype(int)

    if {'hour_of_day', 'logon_count'}.issubset(out.columns):
        active = out[logon > 0]
        if not active.empty:
            normal_hour = (
                active.groupby('user_id')['hour_of_day']
                .apply(lambda x: int(x.mode().iloc[0]) if len(x.mode()) > 0 else 12)
                .to_dict()
            )
            out['normal_hour'] = out['user_id'].map(normal_hour).fillna(12).astype(int)
        else:
            out['normal_hour'] = 12
        hour_dev = (_compat_num(out, 'hour_of_day') - _compat_num(out, 'normal_hour')).abs()
        out['hour_deviation'] = hour_dev.apply(lambda x: min(x, 24 - x))
        out['unusual_timing'] = (_compat_num(out, 'hour_deviation') > 6).astype(int)
    else:
        out['unusual_timing'] = 0

    out['wiki_device_combo'] = ((wiki > 0) & (device > 0)).astype(int)
    out['job_device_email_combo'] = ((job > 0) & (device > 0) & (email_ext > 0)).astype(int)
    out['afterhours_multi_combo'] = ((after > 0) & (device > 0) & (http > 5)).astype(int)
    out['job_then_device_24h'] = ((_compat_num(out, 'http_job_search_count_sum_1d') > 5) & (device > 0)).astype(int)
    out['wiki_then_afterhours_24h'] = ((_compat_num(out, 'http_wikileaks_count_sum_1d') > 0) & (after > 0)).astype(int)
    out['device_velocity_7d'] = _compat_num(out, 'device_count_sum_7d').groupby(out['user_id']).diff().fillna(0).clip(-100, 100)
    out['job_velocity_30d'] = _compat_num(out, 'http_job_search_count_sum_30d').groupby(out['user_id']).diff().fillna(0).clip(-100, 100)

    n_users = float(out['user_id'].nunique()) if out['user_id'].nunique() > 0 else 0.0
    for col in ['http_wikileaks_count', 'http_job_search_count', 'http_file_sharing_count']:
        series = _compat_num(out, col)
        if n_users > 0:
            active_pct = float(out.loc[series > 0, 'user_id'].nunique()) / n_users * 100.0
            rarity_score = max(0.0, (5.0 - active_pct) / 5.0)
            out[f'{col}_rarity'] = (series > 0).astype(int) * rarity_score
        else:
            out[f'{col}_rarity'] = 0.0

    out['device_concentration'] = (
        _compat_num(out, 'device_count_sum_1d') / (_compat_num(out, 'device_count_sum_7d') + 1)
    ).clip(0, 1)

    out['active_channel_count'] = (
        (logon > 0).astype(int)
        + (device > 0).astype(int)
        + (http > 0).astype(int)
        + (email > 0).astype(int)
        + (file_copy > 0).astype(int)
    )
    out['afterhours_pressure'] = (after / (logon + 1)).clip(0, 1)
    out['external_email_ratio'] = (email_ext / (email + 1)).clip(0, 1)
    out['file_to_device_ratio'] = (file_copy / (device + 1)).clip(0, 10)
    out['suspicious_http_ratio'] = ((wiki + job + sharing) / (http + 1)).clip(0, 1)
    out['exfiltration_pressure'] = (
        0.35 * _compat_num(out, 'external_email_ratio')
        + 0.30 * _compat_num(out, 'file_to_device_ratio').clip(0, 1)
        + 0.20 * (sharing > 0).astype(int)
        + 0.15 * (file_copy > 0).astype(int)
    ).clip(0, 1)
    out['risk_interaction_score'] = (
        0.35 * (wiki > 0).astype(int)
        + 0.20 * (job > 0).astype(int)
        + 0.20 * (after > 0).astype(int)
        + 0.15 * (device > 0).astype(int)
        + 0.10 * (email_ext > 0).astype(int)
    ).clip(0, 1)
    out['hour_intensity_score'] = (
        0.25 * logon.clip(0, 5)
        + 0.20 * device.clip(0, 5)
        + 0.20 * http.clip(0, 10)
        + 0.20 * email.clip(0, 10)
        + 0.15 * file_copy.clip(0, 10)
    )

    if 'hour_of_day' in out.columns:
        out['night_risk_score'] = (
            ((_compat_num(out, 'hour_of_day') <= 5) | (_compat_num(out, 'hour_of_day') >= 22)).astype(int)
            * _compat_num(out, 'risk_interaction_score')
        )
    else:
        out['night_risk_score'] = 0.0

    prev_http = http.groupby(out['user_id']).shift(1).fillna(0)
    prev_device = device.groupby(out['user_id']).shift(1).fillna(0)
    prev_after = after.groupby(out['user_id']).shift(1).fillna(0)
    http_mean = prev_http.groupby(out['user_id']).transform(lambda s: s.ewm(halflife=24, adjust=False).mean())
    device_mean = prev_device.groupby(out['user_id']).transform(lambda s: s.ewm(halflife=24, adjust=False).mean())
    after_mean = prev_after.groupby(out['user_id']).transform(lambda s: s.ewm(halflife=24, adjust=False).mean())
    out['http_personal_dev'] = (http - http_mean).fillna(0).clip(-50, 50)
    out['device_personal_dev'] = (device - device_mean).fillna(0).clip(-50, 50)
    out['afterhours_personal_dev'] = (after - after_mean).fillna(0).clip(-50, 50)
    out['risk_acceleration'] = _compat_num(out, 'risk_interaction_score').groupby(out['user_id']).diff().fillna(0).clip(-1, 1)
    out['exfiltration_acceleration'] = _compat_num(out, 'exfiltration_pressure').groupby(out['user_id']).diff().fillna(0).clip(-1, 1)
    out['wiki_afterhours_6h'] = (
        _compat_num(out, 'night_risk_score').groupby(out['user_id']).transform(lambda s: s.rolling(6, min_periods=1).max()) > 0
    ).astype(int)
    out['exfiltration_6h_peak'] = _compat_num(out, 'exfiltration_pressure').groupby(out['user_id']).transform(
        lambda s: s.rolling(6, min_periods=1).max()
    ).clip(0, 1)
    out['interaction_24h_mean'] = _compat_num(out, 'risk_interaction_score').groupby(out['user_id']).transform(
        lambda s: s.rolling(24, min_periods=1).mean()
    ).clip(0, 1)

    out['risk_mean_24h'] = _compat_roll_user(out, 'risk_interaction_score', 24, op='mean', min_periods=3, shift_first=True)
    out['risk_std_24h'] = _compat_roll_user(out, 'risk_interaction_score', 24, op='std', min_periods=4, shift_first=True)
    out['exfil_mean_24h'] = _compat_roll_user(out, 'exfiltration_pressure', 24, op='mean', min_periods=3, shift_first=True)
    out['exfil_std_24h'] = _compat_roll_user(out, 'exfiltration_pressure', 24, op='std', min_periods=4, shift_first=True)
    out['channels_mean_24h'] = _compat_roll_user(out, 'active_channel_count', 24, op='mean', min_periods=3, shift_first=True)
    out['channels_std_24h'] = _compat_roll_user(out, 'active_channel_count', 24, op='std', min_periods=4, shift_first=True)
    out['risk_z_24h'] = ((_compat_num(out, 'risk_interaction_score') - _compat_num(out, 'risk_mean_24h')) / (_compat_num(out, 'risk_std_24h') + 0.5)).clip(-10, 10)
    out['exfil_z_24h'] = ((_compat_num(out, 'exfiltration_pressure') - _compat_num(out, 'exfil_mean_24h')) / (_compat_num(out, 'exfil_std_24h') + 0.2)).clip(-10, 10)
    out['channel_z_24h'] = ((_compat_num(out, 'active_channel_count') - _compat_num(out, 'channels_mean_24h')) / (_compat_num(out, 'channels_std_24h') + 0.5)).clip(-10, 10)

    total_activity = (logon + http + device + email + file_copy).clip(0, 1000)
    prev_total = total_activity.groupby(out['user_id']).shift(1).fillna(0)
    idle_6h = _compat_num(out, 'risk_interaction_score').groupby(out['user_id']).transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).sum()
    ).fillna(0)
    idle_12h = _compat_num(out, 'risk_interaction_score').groupby(out['user_id']).transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).sum()
    ).fillna(0)
    out['suspicious_after_idle_6h'] = ((idle_6h <= 0.05) & (_compat_num(out, 'risk_interaction_score') >= 0.45)).astype(int)
    out['suspicious_after_idle_12h'] = ((idle_12h <= 0.10) & (_compat_num(out, 'risk_interaction_score') >= 0.45)).astype(int)
    out['activity_jump_score'] = ((total_activity - prev_total) / (prev_total + 1)).clip(-10, 10)

    wiki_recent_3h = wiki.groupby(out['user_id']).transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    wiki_recent_6h = wiki.groupby(out['user_id']).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)
    job_recent_6h = job.groupby(out['user_id']).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)
    job_recent_12h = job.groupby(out['user_id']).transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
    sharing_recent_6h = sharing.groupby(out['user_id']).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)
    ext_email_recent_6h = email_ext.groupby(out['user_id']).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)
    after_recent_6h = after.groupby(out['user_id']).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)

    out['wiki_then_device_3h'] = ((wiki_recent_3h > 0) & (device > 0)).astype(int)
    out['wiki_then_external_email_6h'] = ((wiki_recent_6h > 0) & (email_ext > 0)).astype(int)
    out['job_then_device_6h'] = ((job_recent_6h > 0) & (device > 0)).astype(int)
    out['job_then_sharing_12h'] = ((job_recent_12h > 0) & (sharing > 0)).astype(int)
    out['afterhours_then_exfil_6h'] = ((after_recent_6h > 0) & ((email_ext > 0) | (sharing > 0) | (file_copy > 0))).astype(int)
    out['sharing_then_external_email_6h'] = ((sharing_recent_6h > 0) & (email_ext > 0)).astype(int)
    out['email_burst_after_wiki'] = ((wiki_recent_6h > 0) * (email_ext / (ext_email_recent_6h + 1))).clip(0, 10)

    out['risk_stack_3h'] = _compat_num(out, 'risk_interaction_score').groupby(out['user_id']).transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    ).clip(0, 3)
    out['risk_stack_6h'] = _compat_num(out, 'risk_interaction_score').groupby(out['user_id']).transform(
        lambda s: s.rolling(6, min_periods=1).sum()
    ).clip(0, 6)
    out['exfil_stack_6h'] = _compat_num(out, 'exfiltration_pressure').groupby(out['user_id']).transform(
        lambda s: s.rolling(6, min_periods=1).sum()
    ).clip(0, 6)
    out['night_risk_stack_12h'] = _compat_num(out, 'night_risk_score').groupby(out['user_id']).transform(
        lambda s: s.rolling(12, min_periods=1).sum()
    ).clip(0, 12)
    out['compressed_risk_6h'] = (_compat_num(out, 'risk_stack_6h') / (_compat_num(out, 'active_channel_count') + 1)).clip(0, 6)
    out['stealth_exfil_score'] = (
        _compat_num(out, 'exfiltration_pressure')
        * (1 + (_compat_num(out, 'active_channel_count') <= 2).astype(int))
        * (1 + (after > 0).astype(int))
    ).clip(0, 4)

    out['rare_suspicious_combo'] = (
        0.30 * (wiki > 0).astype(int)
        + 0.20 * (job > 0).astype(int)
        + 0.20 * (sharing > 0).astype(int)
        + 0.15 * (email_ext > 0).astype(int)
        + 0.15 * (device > 0).astype(int)
    ).clip(0, 1)
    out['rare_combo_stack_6h'] = _compat_num(out, 'rare_suspicious_combo').groupby(out['user_id']).transform(
        lambda s: s.rolling(6, min_periods=1).sum()
    ).clip(0, 6)
    out['rare_combo_after_idle'] = _compat_num(out, 'rare_suspicious_combo') * _compat_num(out, 'suspicious_after_idle_6h')
    out['insider_transition_score'] = (
        0.22 * _compat_num(out, 'wiki_then_device_3h')
        + 0.18 * _compat_num(out, 'wiki_then_external_email_6h')
        + 0.18 * _compat_num(out, 'job_then_device_6h')
        + 0.12 * _compat_num(out, 'job_then_sharing_12h')
        + 0.12 * _compat_num(out, 'afterhours_then_exfil_6h')
        + 0.08 * _compat_num(out, 'sharing_then_external_email_6h')
        + 0.10 * (_compat_num(out, 'activity_jump_score') > 1.5).astype(int)
    ).clip(0, 1)
    out['insider_pressure_v2'] = (
        0.20 * _compat_num(out, 'risk_z_24h').clip(0, 4) / 4
        + 0.20 * _compat_num(out, 'exfil_z_24h').clip(0, 4) / 4
        + 0.15 * _compat_num(out, 'channel_z_24h').clip(0, 4) / 4
        + 0.15 * _compat_num(out, 'compressed_risk_6h').clip(0, 4) / 4
        + 0.15 * _compat_num(out, 'rare_combo_stack_6h').clip(0, 4) / 4
        + 0.15 * _compat_num(out, 'insider_transition_score')
    ).clip(0, 1)
    out['suspicious_http_pressure_v2'] = (
        _compat_num(out, 'suspicious_http_ratio')
        * (1 + _compat_num(out, 'risk_stack_3h') / 3)
        * (1 + (after > 0).astype(int))
    ).clip(0, 4)

    out['wiki_afterhours_combo'] = ((wiki > 0) & (after > 0)).astype(int)
    out['wiki_device_combo_v2'] = ((wiki > 0) & (device > 0)).astype(int)
    out['first_wiki_afterhours'] = ((_compat_num(out, 'first_wikileaks') > 0) & (after > 0)).astype(int)
    out['afterhours_ratio'] = (after / (logon + 1)).clip(0, 1)
    out['wiki_mean_24h'] = _compat_roll_user(out, 'http_wikileaks_count', 24, op='mean')
    out['wiki_burst'] = (wiki / (_compat_num(out, 'wiki_mean_24h') + 1)).clip(0, 10)
    dev_roll_6h = _compat_roll_user(out, 'device_count', 6, op='max')
    out['wiki_then_device_6h'] = ((wiki > 0) & (dev_roll_6h > 0)).astype(int)
    if n_users > 0:
        wiki_active_pct = float(out.loc[wiki > 0, 'user_id'].nunique()) / n_users * 100.0
        wiki_rarity = max(0.0, (5.0 - wiki_active_pct) / 5.0)
        out['wiki_rarity'] = (wiki > 0).astype(int) * wiki_rarity
    else:
        out['wiki_rarity'] = 0.0
    out['afterhours_mean_7d'] = _compat_roll_user(out, 'logon_afterhours_count', 168, op='mean')
    out['afterhours_burst_7d'] = (after / (_compat_num(out, 'afterhours_mean_7d') + 1)).clip(0, 10)
    out['wiki_afterhours_device'] = ((wiki > 0) & (after > 0) & (device > 0)).astype(int)
    out['wiki_http_share'] = (wiki / (http + 1)).clip(0, 1)

    for col in [
        'logon_afterhours_count',
        'device_count',
        'file_copy_count',
        'http_count',
        'http_job_search_count',
        'http_file_sharing_count',
        'email_external_count',
        'escalation_score_1d',
        'exfiltration_score',
    ]:
        if col not in out.columns:
            out[col] = 0.0

    if 'hour_of_day' not in out.columns and 'window' in out.columns:
        out['hour_of_day'] = pd.to_datetime(out['window']).dt.hour.fillna(0).astype(int)

    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    added_count = len(set(out.columns) - before_cols)
    log.info(f"      Compat v4: +{added_count} colonnes ajoutees")
    return out


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_slice(slice_id: int, cfg: dict, log: logging.Logger,
                 rebuild_cache: bool = False) -> None:
    """Traite une slice complète."""
    
    slices_dir = Path(cfg["output"]["slices_dir"])
    features_dir = Path(cfg["output"]["features_dir"])
    sources = cfg["data"]["sources"]
    
    slice_dir = slices_dir / f"slice_{slice_id:03d}"
    out_dir = features_dir / f"slice_{slice_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"\n{'═'*70}")
    log.info(f"  🚀 SLICE {slice_id:03d} - Feature Engineering ULTIMATE")
    log.info(f"{'═'*70}")
    
    meta = load_slice_meta(slice_dir)
    if meta is None:
        log.error(f"  ✗ meta.parquet absent")
        return
    
    log.info(f"      Période: {meta['start'].date()} → {meta['end'].date()}")
    
    # Cache
    if rebuild_cache:
        cache_df = build_historical_cache(features_dir, slices_dir, slice_id, sources, log)
    else:
        cache_df = load_historical_cache(features_dir, slice_id, log)
        if cache_df is None:
            cache_df = build_historical_cache(features_dir, slices_dir, slice_id, sources, log)
    
    # Charger sources
    log.info(f"      Chargement sources...")
    logon_df = load_source_data(slice_dir, 'logon')
    device_df = load_source_data(slice_dir, 'device')
    file_df = load_source_data(slice_dir, 'file')
    http_df = load_source_data(slice_dir, 'http')
    email_df = load_source_data(slice_dir, 'email')
    
    # Extraction features (USER-HEURE)
    log.info(f"      Extraction (USER-HEURE)...")
    feat_logon = extract_logon_features(logon_df)
    feat_device = extract_device_features(device_df)
    feat_file = extract_file_features(file_df)
    feat_http = extract_http_features(http_df)
    feat_email = extract_email_features(email_df)
    
    # Merge
    features_parts = [feat_logon, feat_device, feat_file, feat_http, feat_email]
    features_parts = [f for f in features_parts if not f.empty]
    
    if not features_parts:
        log.error(f"  ✗ Aucune feature extraite")
        return
    
    df = features_parts[0]
    for part in features_parts[1:]:
        df = df.merge(part, on=['user_id', 'window'], how='outer')
    
    df = df.fillna(0)
    log.info(f"      Features brutes: {df.shape}")
    
    # Libérer
    del logon_df, device_df, file_df, http_df, email_df
    gc.collect()
    
    # Enrichissements
    df = add_temporal_features(df)
    log.info(f"      Temporelles: hour, day_of_week, is_weekend ajoutés")
    
    df = add_rolling_features_multiscale(df, log)
    
    df = add_first_time_features(df, cache_df, log)
    
    df = add_composite_features(df)
    log.info(f"      Composite: escalation_score_1d/7d/30d ajoutés")
    
    ldap_df = load_ldap(features_dir, slice_id)
    df = add_ldap_features(df, ldap_df, log)
    df = add_realtime_v4_compat_features(df, log)
    
    # Labels
    labels_df = load_labels(features_dir, slice_id)
    if not labels_df.empty:
        df = df.merge(
            labels_df[['user_id', 'is_insider', 'scenario']],
            on='user_id',
            how='left'
        )
        df['is_insider'] = df['is_insider'].fillna(0).astype(int)
        log.info(f"      Labels: {df['is_insider'].sum():,} user-heures insiders")
    else:
        log.warning(f"      Labels: Absents")
        df['is_insider'] = 0
        df['scenario'] = None
    
    # Nettoyage
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(0)
    
    text_cols = [c for c in ['role', 'department', 'business_unit', 'scenario'] if c in df.columns]
    for col in text_cols:
        df[col] = df[col].astype('string').fillna('')
    
    # Mise à jour cache
    update_cache(cache_df, slice_dir, slice_id, features_dir, log)
    
    # Sauvegarde
    out_path = out_dir / "features.parquet"
    df.to_parquet(out_path, compression='snappy', index=False)
    
    file_size_mb = out_path.stat().st_size / 1024 / 1024
    
    log.info(f"  {'─'*70}")
    log.info(f"  ✓ Sauvegardé: {out_path}")
    log.info(f"    Lignes: {len(df):,} │ Colonnes: {len(df.columns)} │ {file_size_mb:.1f} MB")
    log.info(f"    Insiders: {df['is_insider'].sum():,} ({df['is_insider'].mean()*100:.2f}%)")


def run(cfg: dict, log: logging.Logger, slice_ids: Optional[List[int]] = None,
       rebuild_cache: bool = False) -> None:
    """Pipeline principal."""
    
    slices_dir = Path(cfg["output"]["slices_dir"])
    
    if slice_ids is None:
        slice_ids = sorted(
            int(p.name.split("_")[1])
            for p in slices_dir.iterdir()
            if p.is_dir() and p.name.startswith("slice_")
        )
    
    log.info("=" * 70)
    log.info("FEATURE ENGINEERING ULTIMATE - USER-HEURE + MULTI-ÉCHELLES")
    log.info("=" * 70)
    log.info(f"Slices: {slice_ids}")
    log.info(f"Granularité: USER-HEURE (précision maximale)")
    log.info(f"Rolling windows: 1J / 7J / 30J (multi-échelles)")
    log.info(f"Features: 80+ optimisées")
    
    t0 = datetime.now()
    
    for sid in slice_ids:
        try:
            process_slice(sid, cfg, log, rebuild_cache)
        except Exception as e:
            log.error(f"  ✗ Slice {sid:03d}: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = (datetime.now() - t0).total_seconds() / 60
    
    log.info(f"\n{'='*70}")
    log.info(f"✅ TERMINÉ - {elapsed:.1f} minutes")
    log.info(f"{'='*70}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        cfg = load_config()
        log = get_logger(cfg["output"]["logs_dir"])
        
        slice_ids = None
        rebuild_cache = False
        
        if "--slice" in sys.argv:
            idx = sys.argv.index("--slice")
            if idx + 1 < len(sys.argv):
                slice_ids = [int(sys.argv[idx + 1])]
        
        if "--rebuild-cache" in sys.argv:
            rebuild_cache = True
        
        run(cfg, log, slice_ids, rebuild_cache)
        
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

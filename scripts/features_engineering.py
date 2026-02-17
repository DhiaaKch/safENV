"""
03_feature_engineering.py
UEBA Feature Engineering — CERT r4.2
Compatible avec config.yaml fourni
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import sys
import gc
from datetime import datetime
from typing import Optional

# =============================================================================
# CONFIG & LOGGING
# =============================================================================

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def get_logger(logs_dir):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("Features")
    logger.setLevel(logging.INFO)

    fmt = "%(asctime)s | %(levelname)s | %(message)s"


    fh = logging.FileHandler(
        Path(logs_dir) / f"03_features_{ts}.log",
        encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    return logger

# =============================================================================
# LOADERS
# =============================================================================

def load_parquet_for_users(path: Path, users: list) -> pd.DataFrame:
    if not path.exists() or not users:
        return pd.DataFrame()

    con = duckdb.connect()
    user_sql = ", ".join(f"'{u}'" for u in users)
    df = con.execute(
        f"SELECT * FROM read_parquet('{path}') WHERE user_id IN ({user_sql})"
    ).df()
    con.close()

    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df

def get_all_users(slice_dir: Path, sources: list):
    users = set()
    con = duckdb.connect()

    for src in sources:
        fp = slice_dir / f"{src}.parquet"
        if fp.exists():
            try:
                rows = con.execute(
                    f"SELECT DISTINCT user_id FROM read_parquet('{fp}')"
                ).df()["user_id"].dropna().tolist()
                users.update(rows)
            except:
                pass

    con.close()
    return sorted(users)

def load_answers(raw_dir: Path, ts_format: str):
    fp = raw_dir / "answers.csv"
    if not fp.exists():
        return pd.DataFrame()

    df = pd.read_csv(fp)
    df["user"] = df["user"].str.strip().str.upper()
    df["start"] = pd.to_datetime(df["start"], format=ts_format, errors="coerce")
    df["end"]   = pd.to_datetime(df["end"],   format=ts_format, errors="coerce")
    return df

# =============================================================================
# RAW FEATURES
# =============================================================================

def feat_logon(df, window):
    if df.empty:
        return pd.DataFrame()

    df["window"] = df["timestamp"].dt.floor(window)
    df["hour"] = df["timestamp"].dt.hour
    df["is_after"] = (df["hour"] < 6) | (df["hour"] >= 19)

    g = df.groupby(["user_id","window"])

    out = g.size().reset_index(name="logon_count")
    out["logon_unique_pcs"] = g["pc"].nunique().values

    after = df[df["is_after"]].groupby(["user_id","window"]).size()
    out["logon_afterhours_count"] = after.reindex(
        out.set_index(["user_id","window"]).index
    ).fillna(0).values

    return out.fillna(0)

def feat_device(df, window):
    if df.empty:
        return pd.DataFrame()

    df["window"] = df["timestamp"].dt.floor(window)
    g = df.groupby(["user_id","window"])

    out = g.size().reset_index(name="usb_count")
    out["usb_connect_count"] = g["activity"].apply(
        lambda x: (x=="Connect").sum()
    ).values

    return out.fillna(0)

def feat_file(df, window):
    if df.empty:
        return pd.DataFrame()

    df["window"] = df["timestamp"].dt.floor(window)
    g = df.groupby(["user_id","window"])

    out = g.size().reset_index(name="file_copy_count")
    return out.fillna(0)

def feat_http(df, window, blacklist):
    if df.empty:
        return pd.DataFrame()

    df["window"] = df["timestamp"].dt.floor(window)
    domain = df["domain"].str.lower().fillna("")
    df["is_black"] = domain.apply(lambda d: any(b in d for b in blacklist))

    g = df.groupby(["user_id","window"])
    out = g.size().reset_index(name="http_count")

    black = df[df["is_black"]].groupby(["user_id","window"]).size()
    out["http_blacklist_count"] = black.reindex(
        out.set_index(["user_id","window"]).index
    ).fillna(0).values

    return out.fillna(0)

def feat_email(df, window):
    if df.empty:
        return pd.DataFrame()

    df["window"] = df["timestamp"].dt.floor(window)
    g = df.groupby(["user_id","window"])
    out = g.size().reset_index(name="email_sent_count")
    return out.fillna(0)

# =============================================================================
# CROSS FEATURES
# =============================================================================

def add_cross_features(df):
    df = df.copy()

    df["cross_afterhours_score"] = (
        df.get("logon_afterhours_count",0) +
        df.get("usb_count",0)
    )

    df["cross_risk_combo"] = (
        2*df.get("http_blacklist_count",0) +
        1.5*df.get("cross_afterhours_score",0)
    )

    return df

# =============================================================================
# LDAP INTERACTION (FIXED POSITION)
# =============================================================================

def add_ldap_interaction_features(df):
    df = df.copy()

    if "is_it_admin" in df.columns:
        is_it = df["is_it_admin"] == 1
    else:
        is_it = pd.Series(False, index=df.index)

    df["usb_non_it"] = df.get("usb_connect_count",0) * (~is_it)
    df["afterhours_non_it"] = df.get("logon_afterhours_count",0) * (~is_it)

    return df

# =============================================================================
# BASELINE & ZSCORE
# =============================================================================

def compute_baseline(df):
    numeric = df.select_dtypes(include=np.number)
    bl = numeric.groupby(df["user_id"]).agg(["mean","std"])
    bl.columns = ["_".join(c) for c in bl.columns]
    return bl.reset_index().fillna(0)

def add_zscores(df, baseline):
    df = df.merge(baseline, on="user_id", how="left")
    for col in df.columns:
        if col.endswith("_mean"):
            base = col.replace("_mean","")
            std = base+"_std"
            if std in df.columns:
                df[base+"_z"] = (df[base]-df[col])/(df[std]+1e-6)
    drop = [c for c in df.columns if c.endswith(("_mean","_std"))]
    return df.drop(columns=drop)

# =============================================================================
# PIPELINE
# =============================================================================

def process_slice(slice_id, cfg, log, baseline, answers):

    slice_dir = Path(cfg["output"]["slices_dir"]) / f"slice_{slice_id:03d}"
    out_dir   = Path(cfg["output"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    window = "1h"
    blacklist = cfg["detection"]["blacklist_domains"]

    log.info(f"\n  🔧 Features slice {slice_id:03d}")

    users = get_all_users(slice_dir, cfg["data"]["sources"])
    log.info(f"    Users: {len(users)}")

    parts = []
    for src, fn in [
        ("logon", feat_logon),
        ("device", feat_device),
        ("file", feat_file),
        ("http", feat_http),
        ("email", feat_email)
    ]:
        raw = load_parquet_for_users(slice_dir / f"{src}.parquet", users)

        if src == "http":
            out = fn(raw, window, blacklist)
        else:
            out = fn(raw, window)

        if not out.empty:
            parts.append(out)

    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on=["user_id","window"], how="outer")

    df = df.fillna(0)
    log.info(f"    Shape brut: {df.shape}")

    df = add_cross_features(df)
    df = add_ldap_interaction_features(df)

    if baseline is None:
        baseline = compute_baseline(df)
    else:
        baseline = compute_baseline(df)

    df = add_zscores(df, baseline)

    out_path = out_dir / f"slice_{slice_id:03d}_features.parquet"
    df.to_parquet(out_path, compression="snappy", index=False)

    log.info(f"    💾 Saved {out_path}")

    return baseline

def run(cfg, log):
    slices_dir = Path(cfg["output"]["slices_dir"])
    slice_ids = sorted(
        int(p.name.split("_")[1])
        for p in slices_dir.iterdir()
        if p.is_dir() and p.name.startswith("slice_")
    )

    answers = load_answers(
        Path(cfg["data"]["raw_dir"]),
        cfg["data"]["timestamp_format"]
    )

    baseline = None
    for sid in slice_ids:
        baseline = process_slice(sid, cfg, log, baseline, answers)

    log.info("\n✅ FEATURE ENGINEERING TERMINÉ")

# =============================================================================

if __name__ == "__main__":
    cfg = load_config()
    log = get_logger(cfg["output"]["logs_dir"])
    run(cfg, log)

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


REQUESTED_FEATURES = [
    "logon_unique_pcs", "logon_count", "exfiltration_acceleration", "after_hours_non_it",
    "wiki_afterhours_6h", "device_velocity_7d", "risk_stack_3h", "day_of_week",
    "hour_deviation", "activity_jump_score", "logon_hour_mean", "http_count",
    "device_burst", "hour_of_day", "rare_combo_stack_6h", "first_device",
    "http_burst", "http_job_search_count_sum_1d", "device_count_trend_7d",
    "risk_stack_6h", "email_external_count_mean_1d", "hour_intensity_score",
    "insider_pressure_v2", "device_count_trend_30d", "email_external_count_sum_1d",
    "escalation_score_1d", "compressed_risk_6h", "http_job_search_count_mean_1d",
    "exfil_stack_6h", "channel_z_24h", "exfiltration_6h_peak",
    "http_file_sharing_count_sum_7d", "http_file_sharing_count_mean_7d", "risk_z_24h",
    "is_it_admin", "escalation_score_7d", "http_personal_dev",
    "http_file_sharing_count_mean_30d", "exfil_z_24h", "http_file_sharing_count_sum_30d",
    "logon_afterhours_count_sum_1d", "logon_afterhours_count_mean_1d", "exfiltration_score",
    "escalation_score_30d", "http_wikileaks_count_sum_7d", "http_mean_24h",
    "channels_mean_24h", "channels_std_24h", "interaction_24h_mean", "normal_hour",
    "http_wikileaks_count_sum_30d", "afterhours_personal_dev",
    "http_wikileaks_count_mean_7d", "risk_mean_24h", "risk_std_24h",
    "night_risk_stack_12h", "logon_afterhours_count_sum_30d",
    "http_wikileaks_count_mean_30d", "logon_afterhours_count_sum_7d",
    "email_external_count_mean_7d", "http_job_search_count_sum_7d", "exfil_mean_24h",
    "device_count_sum_1d", "http_job_search_count_mean_7d", "exfil_std_24h",
    "email_external_count_sum_7d", "afterhours_mean_7d", "device_personal_dev",
    "device_count_mean_1d", "http_job_search_count_mean_30d", "device_concentration",
    "logon_afterhours_count_mean_7d", "http_job_search_count_sum_30d",
    "email_external_count_sum_30d", "email_external_count_mean_30d",
    "logon_afterhours_count_mean_30d", "device_count_sum_30d", "device_count_sum_7d",
    "device_count_mean_7d", "device_count_mean_30d",
]

JOB_DOMAINS = {"linkedin.com", "indeed.com", "glassdoor.com"}
FILE_SHARING_DOMAINS = {"drive.google.com", "dropbox.com", "mega.nz", "we-transfer.com"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UEBA features from Elasticsearch.")
    parser.add_argument("--es-url", required=True)
    parser.add_argument("--index", default="ueba-events-*")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--scroll", default="2m")
    parser.add_argument("--output", required=True)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def parse_ts(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def get_logger() -> logging.Logger:
    logger = logging.getLogger("elk_feature_builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


class ElasticsearchClient:
    def __init__(self, base_url: str, username: Optional[str], password: Optional[str], timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.auth_header = None
        if username:
            token = base64.b64encode(f"{username}:{password or ''}".encode("utf-8")).decode("ascii")
            self.auth_header = f"Basic {token}"

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        if self.auth_header:
            req.add_header("Authorization", self.auth_header)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}

    def search_scroll(self, index: str, query: dict, size: int, scroll: str) -> List[dict]:
        encoded = urllib.parse.quote(index, safe="*,-_")
        first = self._request(
            "POST",
            f"/{encoded}/_search?scroll={scroll}",
            {"size": size, "sort": [{"@timestamp": "asc"}], "_source": True, "query": query},
        )
        scroll_id = first.get("_scroll_id")
        hits = [hit.get("_source", {}) for hit in first.get("hits", {}).get("hits", [])]
        while scroll_id:
            page = self._request("POST", "/_search/scroll", {"scroll": scroll, "scroll_id": scroll_id})
            page_hits = page.get("hits", {}).get("hits", [])
            if not page_hits:
                break
            hits.extend(hit.get("_source", {}) for hit in page_hits)
            scroll_id = page.get("_scroll_id", scroll_id)
        if scroll_id:
            try:
                self._request("DELETE", "/_search/scroll", {"scroll_id": [scroll_id]})
            except Exception:
                pass
        return hits


def first_non_null(frame: pd.DataFrame, candidates: List[str], default=None) -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            series = frame[col]
            if series.notna().any():
                return series
    return pd.Series(default, index=frame.index)


def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def roll_user(df: pd.DataFrame, source_col: str, win: int, op: str, min_periods: int = 1, shift_first: bool = False) -> pd.Series:
    s = safe_num(df, source_col)
    if shift_first:
        s = s.groupby(df["user_id"]).shift(1).fillna(0)
    grouped = s.groupby(df["user_id"])
    if op == "sum":
        return grouped.transform(lambda x: x.rolling(win, min_periods=min_periods).sum()).fillna(0)
    if op == "mean":
        return grouped.transform(lambda x: x.rolling(win, min_periods=min_periods).mean()).fillna(0)
    if op == "std":
        return grouped.transform(lambda x: x.rolling(win, min_periods=min_periods).std()).fillna(0)
    if op == "max":
        return grouped.transform(lambda x: x.rolling(win, min_periods=min_periods).max()).fillna(0)
    raise ValueError(f"Unsupported rolling op: {op}")


def normalize_events(events: List[dict], log: logging.Logger) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    df = pd.json_normalize(events, sep=".")
    df["event_ts"] = pd.to_datetime(first_non_null(df, ["@timestamp", "timestamp", "event.created"], None), utc=True, errors="coerce")
    df = df[df["event_ts"].notna()].copy()
    if df.empty:
        return df

    df["user_id"] = first_non_null(df, ["user.name", "user.id", "user_id"], "").fillna("").astype(str)
    df["pc"] = first_non_null(df, ["host.name", "host", "pc"], "").fillna("").astype(str)
    df["department"] = first_non_null(df, ["organization.department", "department"], "").fillna("").astype(str)
    df["role"] = first_non_null(df, ["role", "user.role", "user.roles", "title"], "").fillna("").astype(str)
    df["event_category"] = first_non_null(df, ["event.category", "event_type"], "").fillna("").astype(str).str.lower()
    df["event_action"] = first_non_null(df, ["event.action", "action"], "").fillna("").astype(str).str.lower()
    df["event_outcome"] = first_non_null(df, ["event.outcome"], "").fillna("").astype(str).str.lower()
    df["window"] = df["event_ts"].dt.floor("1h")
    df["hour"] = df["event_ts"].dt.hour

    if "logon.is_afterhours" in df.columns:
        df["is_afterhours"] = df["logon.is_afterhours"].fillna(False).astype(bool)
    else:
        df["is_afterhours"] = (df["hour"] < 6) | (df["hour"] >= 19)

    domain = first_non_null(df, ["url.domain", "domain"], "").fillna("").astype(str).str.lower()
    df["http_wikileaks_flag"] = np.where(safe_num(df, "ueba.http_wikileaks_flag") > 0, 1, domain.eq("wikileaks.org").astype(int))
    df["http_job_search_flag"] = np.where(safe_num(df, "ueba.http_job_search_flag") > 0, 1, domain.isin(JOB_DOMAINS).astype(int))
    df["http_file_sharing_flag"] = np.where(safe_num(df, "ueba.http_file_sharing_flag") > 0, 1, domain.isin(FILE_SHARING_DOMAINS).astype(int))
    df["email_external_flag"] = np.where(
        safe_num(df, "ueba.email_external_flag") > 0,
        1,
        first_non_null(df, ["email.is_external", "is_external"], False).fillna(False).astype(bool).astype(int),
    )
    df["file_copy_flag"] = np.where(
        safe_num(df, "ueba.file_copy_flag") > 0,
        1,
        df["event_action"].isin(["copy", "file_copy"]).astype(int),
    )

    df["is_logon"] = ((df["event_category"] == "authentication") | df["event_action"].isin(["logon", "login"])).astype(int)
    df["is_logon_success"] = ((df["is_logon"] == 1) & ((df["event_outcome"] == "") | (df["event_outcome"] == "success"))).astype(int)
    df["is_device"] = ((df["event_category"] == "device") & df["event_action"].isin(["connect", "device_connect", "mount", "usb_connect"])).astype(int)
    df["is_http"] = ((df["event_category"] == "network") & (df["event_action"] == "http_request")).astype(int)
    df["is_email"] = ((df["event_category"] == "email") & df["event_action"].isin(["send", "email_send"])).astype(int)
    df["is_file"] = (df["event_category"] == "file").astype(int)

    df = df[df["user_id"] != ""].copy()
    log.info("Normalized %d events across %d users", len(df), df["user_id"].nunique())
    return df


def aggregate_hourly(events_df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    users = sorted(events_df["user_id"].unique())
    windows = pd.date_range(events_df["window"].min(), events_df["window"].max(), freq="1h", tz="UTC")
    grid = pd.MultiIndex.from_product([users, windows], names=["user_id", "window"]).to_frame(index=False)
    user_static = events_df.sort_values("event_ts").groupby("user_id", as_index=False).agg({"department": "last", "role": "last"})

    logon = events_df[events_df["is_logon"] == 1]
    logon_agg = logon.groupby(["user_id", "window"], as_index=False).agg(
        logon_count=("is_logon_success", "sum"),
        logon_unique_pcs=("pc", lambda s: s[s != ""].nunique()),
        logon_hour_mean=("hour", "mean"),
        logon_afterhours_count=("is_afterhours", "sum"),
    )
    device_agg = events_df[events_df["is_device"] == 1].groupby(["user_id", "window"], as_index=False).size().rename(columns={"size": "device_count"})
    http_agg = events_df[events_df["is_http"] == 1].groupby(["user_id", "window"], as_index=False).agg(
        http_count=("is_http", "sum"),
        http_wikileaks_count=("http_wikileaks_flag", "sum"),
        http_job_search_count=("http_job_search_flag", "sum"),
        http_file_sharing_count=("http_file_sharing_flag", "sum"),
    )
    email_agg = events_df[events_df["is_email"] == 1].groupby(["user_id", "window"], as_index=False).agg(
        email_sent_count=("is_email", "sum"),
        email_external_count=("email_external_flag", "sum"),
    )
    file_agg = events_df[events_df["is_file"] == 1].groupby(["user_id", "window"], as_index=False).agg(file_copy_count=("file_copy_flag", "sum"))

    out = grid.merge(user_static, on="user_id", how="left")
    for frame in [logon_agg, device_agg, http_agg, email_agg, file_agg]:
        out = out.merge(frame, on=["user_id", "window"], how="left")

    for col in [
        "logon_count", "logon_unique_pcs", "logon_hour_mean", "logon_afterhours_count",
        "device_count", "http_count", "http_wikileaks_count", "http_job_search_count",
        "http_file_sharing_count", "email_sent_count", "email_external_count", "file_copy_count",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["department"] = out["department"].fillna("")
    out["role"] = out["role"].fillna("")
    out["logon_hour_mean"] = np.where(out["logon_count"] > 0, out["logon_hour_mean"], 0.0)
    out = out.sort_values(["user_id", "window"]).reset_index(drop=True)
    log.info("Built hourly grid: %d rows", len(out))
    return out


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_of_day"] = out["window"].dt.hour.astype(int)
    out["day_of_week"] = out["window"].dt.dayofweek.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_night"] = ((out["hour_of_day"] >= 22) | (out["hour_of_day"] <= 5)).astype(int)
    out["is_business_hours"] = ((out["hour_of_day"] >= 8) & (out["hour_of_day"] <= 18) & (out["day_of_week"] < 5)).astype(int)
    return out


def add_first_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for source, target in [
        ("http_wikileaks_count", "first_wikileaks"),
        ("http_job_search_count", "first_job_search"),
        ("device_count", "first_device"),
        ("logon_afterhours_count", "first_after_hours"),
        ("email_external_count", "first_external_email"),
    ]:
        prior = safe_num(out, source).groupby(out["user_id"]).cumsum().groupby(out["user_id"]).shift(1).fillna(0)
        out[target] = ((safe_num(out, source) > 0) & (prior <= 0)).astype(int)
    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feat in [
        "http_wikileaks_count",
        "http_job_search_count",
        "http_file_sharing_count",
        "device_count",
        "logon_afterhours_count",
        "email_external_count",
    ]:
        out[f"{feat}_sum_1d"] = roll_user(out, feat, 24, "sum")
        out[f"{feat}_mean_1d"] = roll_user(out, feat, 24, "mean")
        out[f"{feat}_sum_7d"] = roll_user(out, feat, 168, "sum")
        out[f"{feat}_mean_7d"] = roll_user(out, feat, 168, "mean")
        out[f"{feat}_trend_7d"] = (safe_num(out, feat) / (safe_num(out, f"{feat}_mean_7d") + 1e-6)).fillna(1.0).clip(0, 10)
        out[f"{feat}_sum_30d"] = roll_user(out, feat, 720, "sum")
        out[f"{feat}_mean_30d"] = roll_user(out, feat, 720, "mean")
        out[f"{feat}_trend_30d"] = (safe_num(out, feat) / (safe_num(out, f"{feat}_mean_30d") + 1e-6)).fillna(1.0).clip(0, 10)
    return out


def add_org_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dept = out["department"].astype(str).str.lower()
    role = out["role"].astype(str).str.lower()
    out["is_it_admin"] = (dept.str.contains("it") | role.str.contains("admin|it|system|tech")).astype(int)
    out["after_hours_non_it"] = safe_num(out, "logon_afterhours_count") * (1 - out["is_it_admin"])
    return out


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["escalation_score_1d"] = (
        0.5 * safe_num(out, "first_wikileaks")
        + 0.3 * safe_num(out, "first_after_hours")
        + 0.2 * safe_num(out, "first_device")
        + 0.3 * (safe_num(out, "http_job_search_count_sum_1d") > 3).astype(int)
        + 0.2 * (safe_num(out, "device_count_sum_1d") > 2).astype(int)
    ).clip(0, 1)
    out["escalation_score_7d"] = (
        0.4 * (safe_num(out, "http_wikileaks_count_sum_7d") > 0).astype(int)
        + 0.3 * (safe_num(out, "http_job_search_count_sum_7d") > 15).astype(int)
        + 0.3 * (safe_num(out, "device_count_sum_7d") > 10).astype(int)
        + 0.2 * (safe_num(out, "device_count_trend_7d") > 2).astype(int)
    ).clip(0, 1)
    out["escalation_score_30d"] = (
        0.3 * (safe_num(out, "http_job_search_count_sum_30d") > 50).astype(int)
        + 0.3 * (safe_num(out, "device_count_sum_30d") > 30).astype(int)
        + 0.2 * (safe_num(out, "email_external_count_sum_30d") > 20).astype(int)
        + 0.2 * (safe_num(out, "device_count_trend_30d") > 3).astype(int)
    ).clip(0, 1)
    out["exfiltration_score"] = (
        0.3 * safe_num(out, "escalation_score_1d")
        + 0.4 * safe_num(out, "escalation_score_7d")
        + 0.3 * safe_num(out, "escalation_score_30d")
    ).clip(0, 1)
    return out


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    logon = safe_num(out, "logon_count")
    after = safe_num(out, "logon_afterhours_count")
    device = safe_num(out, "device_count")
    http = safe_num(out, "http_count")
    wiki = safe_num(out, "http_wikileaks_count")
    job = safe_num(out, "http_job_search_count")
    sharing = safe_num(out, "http_file_sharing_count")
    email = safe_num(out, "email_sent_count")
    email_ext = safe_num(out, "email_external_count")
    file_copy = safe_num(out, "file_copy_count")

    out["device_burst"] = (device / (safe_num(out, "device_count_mean_1d") + 1)).clip(0, 10)
    out["http_mean_24h"] = roll_user(out, "http_count", 24, "mean")
    out["http_burst"] = (http / (safe_num(out, "http_mean_24h") + 1)).clip(0, 10)
    out["device_velocity_7d"] = safe_num(out, "device_count_sum_7d").groupby(out["user_id"]).diff().fillna(0).clip(-100, 100)

    active = out[logon > 0]
    if not active.empty:
        normal_hour_map = active.groupby("user_id")["hour_of_day"].apply(lambda x: int(x.mode().iloc[0]) if not x.mode().empty else 12).to_dict()
        out["normal_hour"] = out["user_id"].map(normal_hour_map).fillna(12).astype(int)
    else:
        out["normal_hour"] = 12
    hour_dev = (safe_num(out, "hour_of_day") - safe_num(out, "normal_hour")).abs()
    out["hour_deviation"] = hour_dev.apply(lambda x: min(x, 24 - x))

    out["device_concentration"] = (safe_num(out, "device_count_sum_1d") / (safe_num(out, "device_count_sum_7d") + 1)).clip(0, 1)
    out["active_channel_count"] = ((logon > 0).astype(int) + (device > 0).astype(int) + (http > 0).astype(int) + (email > 0).astype(int) + (file_copy > 0).astype(int))
    out["external_email_ratio"] = (email_ext / (email + 1)).clip(0, 1)
    out["file_to_device_ratio"] = (file_copy / (device + 1)).clip(0, 10)
    out["exfiltration_pressure"] = (
        0.35 * safe_num(out, "external_email_ratio")
        + 0.30 * safe_num(out, "file_to_device_ratio").clip(0, 1)
        + 0.20 * (sharing > 0).astype(int)
        + 0.15 * (file_copy > 0).astype(int)
    ).clip(0, 1)
    out["risk_interaction_score"] = (
        0.35 * (wiki > 0).astype(int)
        + 0.20 * (job > 0).astype(int)
        + 0.20 * (after > 0).astype(int)
        + 0.15 * (device > 0).astype(int)
        + 0.10 * (email_ext > 0).astype(int)
    ).clip(0, 1)
    out["hour_intensity_score"] = (0.25 * logon.clip(0, 5) + 0.20 * device.clip(0, 5) + 0.20 * http.clip(0, 10) + 0.20 * email.clip(0, 10) + 0.15 * file_copy.clip(0, 10))
    out["night_risk_score"] = (((safe_num(out, "hour_of_day") <= 5) | (safe_num(out, "hour_of_day") >= 22)).astype(int) * safe_num(out, "risk_interaction_score"))

    prev_http = http.groupby(out["user_id"]).shift(1).fillna(0)
    prev_device = device.groupby(out["user_id"]).shift(1).fillna(0)
    prev_after = after.groupby(out["user_id"]).shift(1).fillna(0)
    http_mean = prev_http.groupby(out["user_id"]).transform(lambda s: s.ewm(halflife=24, adjust=False).mean())
    device_mean = prev_device.groupby(out["user_id"]).transform(lambda s: s.ewm(halflife=24, adjust=False).mean())
    after_mean = prev_after.groupby(out["user_id"]).transform(lambda s: s.ewm(halflife=24, adjust=False).mean())
    out["http_personal_dev"] = (http - http_mean).fillna(0).clip(-50, 50)
    out["device_personal_dev"] = (device - device_mean).fillna(0).clip(-50, 50)
    out["afterhours_personal_dev"] = (after - after_mean).fillna(0).clip(-50, 50)
    out["exfiltration_acceleration"] = safe_num(out, "exfiltration_pressure").groupby(out["user_id"]).diff().fillna(0).clip(-1, 1)
    out["wiki_afterhours_6h"] = (safe_num(out, "night_risk_score").groupby(out["user_id"]).transform(lambda s: s.rolling(6, min_periods=1).max()) > 0).astype(int)
    out["exfiltration_6h_peak"] = safe_num(out, "exfiltration_pressure").groupby(out["user_id"]).transform(lambda s: s.rolling(6, min_periods=1).max()).clip(0, 1)
    out["interaction_24h_mean"] = safe_num(out, "risk_interaction_score").groupby(out["user_id"]).transform(lambda s: s.rolling(24, min_periods=1).mean()).clip(0, 1)

    out["risk_mean_24h"] = roll_user(out, "risk_interaction_score", 24, "mean", min_periods=3, shift_first=True)
    out["risk_std_24h"] = roll_user(out, "risk_interaction_score", 24, "std", min_periods=4, shift_first=True)
    out["exfil_mean_24h"] = roll_user(out, "exfiltration_pressure", 24, "mean", min_periods=3, shift_first=True)
    out["exfil_std_24h"] = roll_user(out, "exfiltration_pressure", 24, "std", min_periods=4, shift_first=True)
    out["channels_mean_24h"] = roll_user(out, "active_channel_count", 24, "mean", min_periods=3, shift_first=True)
    out["channels_std_24h"] = roll_user(out, "active_channel_count", 24, "std", min_periods=4, shift_first=True)
    out["risk_z_24h"] = ((safe_num(out, "risk_interaction_score") - safe_num(out, "risk_mean_24h")) / (safe_num(out, "risk_std_24h") + 0.5)).clip(-10, 10)
    out["exfil_z_24h"] = ((safe_num(out, "exfiltration_pressure") - safe_num(out, "exfil_mean_24h")) / (safe_num(out, "exfil_std_24h") + 0.2)).clip(-10, 10)
    out["channel_z_24h"] = ((safe_num(out, "active_channel_count") - safe_num(out, "channels_mean_24h")) / (safe_num(out, "channels_std_24h") + 0.5)).clip(-10, 10)

    total_activity = (logon + http + device + email + file_copy).clip(0, 1000)
    prev_total = total_activity.groupby(out["user_id"]).shift(1).fillna(0)
    out["activity_jump_score"] = ((total_activity - prev_total) / (prev_total + 1)).clip(-10, 10)

    wiki_recent_3h = wiki.groupby(out["user_id"]).transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    wiki_recent_6h = wiki.groupby(out["user_id"]).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)
    job_recent_6h = job.groupby(out["user_id"]).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)
    job_recent_12h = job.groupby(out["user_id"]).transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
    sharing_recent_6h = sharing.groupby(out["user_id"]).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)
    email_recent_6h = email_ext.groupby(out["user_id"]).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)
    after_recent_6h = after.groupby(out["user_id"]).transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)

    out["risk_stack_3h"] = safe_num(out, "risk_interaction_score").groupby(out["user_id"]).transform(lambda s: s.rolling(3, min_periods=1).sum()).clip(0, 3)
    out["risk_stack_6h"] = safe_num(out, "risk_interaction_score").groupby(out["user_id"]).transform(lambda s: s.rolling(6, min_periods=1).sum()).clip(0, 6)
    out["exfil_stack_6h"] = safe_num(out, "exfiltration_pressure").groupby(out["user_id"]).transform(lambda s: s.rolling(6, min_periods=1).sum()).clip(0, 6)
    out["night_risk_stack_12h"] = safe_num(out, "night_risk_score").groupby(out["user_id"]).transform(lambda s: s.rolling(12, min_periods=1).sum()).clip(0, 12)
    out["compressed_risk_6h"] = (safe_num(out, "risk_stack_6h") / (safe_num(out, "active_channel_count") + 1)).clip(0, 6)

    out["rare_suspicious_combo"] = (0.30 * (wiki > 0).astype(int) + 0.20 * (job > 0).astype(int) + 0.20 * (sharing > 0).astype(int) + 0.15 * (email_ext > 0).astype(int) + 0.15 * (device > 0).astype(int)).clip(0, 1)
    out["rare_combo_stack_6h"] = safe_num(out, "rare_suspicious_combo").groupby(out["user_id"]).transform(lambda s: s.rolling(6, min_periods=1).sum()).clip(0, 6)
    transition = (
        0.22 * ((wiki_recent_3h > 0) & (device > 0)).astype(int)
        + 0.18 * ((wiki_recent_6h > 0) & (email_ext > 0)).astype(int)
        + 0.18 * ((job_recent_6h > 0) & (device > 0)).astype(int)
        + 0.12 * ((job_recent_12h > 0) & (sharing > 0)).astype(int)
        + 0.12 * ((after_recent_6h > 0) & ((email_ext > 0) | (sharing > 0) | (file_copy > 0))).astype(int)
        + 0.08 * ((sharing_recent_6h > 0) & (email_ext > 0)).astype(int)
        + 0.10 * (safe_num(out, "activity_jump_score") > 1.5).astype(int)
    ).clip(0, 1)
    out["insider_pressure_v2"] = (
        0.20 * safe_num(out, "risk_z_24h").clip(0, 4) / 4
        + 0.20 * safe_num(out, "exfil_z_24h").clip(0, 4) / 4
        + 0.15 * safe_num(out, "channel_z_24h").clip(0, 4) / 4
        + 0.15 * safe_num(out, "compressed_risk_6h").clip(0, 4) / 4
        + 0.15 * safe_num(out, "rare_combo_stack_6h").clip(0, 4) / 4
        + 0.15 * transition
    ).clip(0, 1)
    out["afterhours_mean_7d"] = roll_user(out, "logon_afterhours_count", 168, "mean")
    _ = email_recent_6h
    return out


def build_features(events_df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    features = aggregate_hourly(events_df, log)
    features = add_temporal_features(features)
    features = add_first_time_features(features)
    features = add_rolling_features(features)
    features = add_org_features(features)
    features = add_composite_features(features)
    features = add_context_features(features)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
        return
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        return
    raise ValueError("Output must end with .parquet or .csv")


def main() -> None:
    args = parse_args()
    log = get_logger()

    output_start = parse_ts(args.start)
    output_end = parse_ts(args.end)
    fetch_start = output_start - timedelta(days=args.lookback_days)

    client = ElasticsearchClient(args.es_url, args.username, args.password, args.timeout)
    query = {"bool": {"filter": [{"range": {"@timestamp": {"gte": fetch_start.isoformat(), "lte": output_end.isoformat()}}}]}}

    log.info("Fetching events from %s between %s and %s", args.index, fetch_start.isoformat(), output_end.isoformat())
    events = client.search_scroll(args.index, query, size=args.batch_size, scroll=args.scroll)
    if not events:
        raise RuntimeError("No events returned from Elasticsearch for the requested range")

    normalized = normalize_events(events, log)
    if normalized.empty:
        raise RuntimeError("Events were fetched but no usable normalized records were found")

    features = build_features(normalized, log)
    mask = (features["window"] >= pd.Timestamp(output_start)) & (features["window"] <= pd.Timestamp(output_end))
    features = features.loc[mask].copy()

    missing = [name for name in REQUESTED_FEATURES if name not in features.columns]
    if missing:
        raise RuntimeError(f"Missing expected features after build: {missing}")

    final_df = features[["user_id", "window"] + REQUESTED_FEATURES].sort_values(["user_id", "window"]).reset_index(drop=True)
    save_output(final_df, Path(args.output))
    log.info("Saved %d rows with %d requested features to %s", len(final_df), len(REQUESTED_FEATURES), args.output)


if __name__ == "__main__":
    main()
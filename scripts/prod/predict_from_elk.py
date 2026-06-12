from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd

# ============================================================================
# CONFIGURATION PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [PROJECT_ROOT, PROJECT_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ============================================================================
# IMPORTS PROJET
# ============================================================================
import model_3
import build_elk_features as elk_feat


# ============================================================================
# UTILITAIRES
# ============================================================================

def _strip_tz(ts) -> pd.Timestamp:
    """Retourne un Timestamp naïf (sans timezone) pour comparaison uniforme."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        return t.tz_convert("UTC").tz_localize(None)
    return t


def _normalize_window_col(series: pd.Series) -> pd.Series:
    """Normalise une colonne de timestamps vers naïf UTC."""
    if hasattr(series.dtype, "tz") and series.dtype.tz is not None:
        return series.dt.tz_convert("UTC").dt.tz_localize(None)
    try:
        if series.dt.tz is not None:
            return series.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    return series


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build features from ELK and score them with ensemble model."
    )
    parser.add_argument("--es-url",      required=True,  help="Elasticsearch URL (ex: http://localhost:9200)")
    parser.add_argument("--index",       default="ueba-events-*", help="Index Elasticsearch source")
    parser.add_argument("--start",       required=True,  help="Date début ISO8601 (ex: 2024-01-01T00:00:00)")
    parser.add_argument("--end",         required=True,  help="Date fin ISO8601")
    parser.add_argument("--lookback-days",   type=int, default=30,   help="Jours historique pour rolling features")
    parser.add_argument("--batch-size",      type=int, default=2000, help="Taille batch scroll ES")
    parser.add_argument("--scroll",      default="2m",   help="Timeout scroll ES")
    parser.add_argument(
        "--models-path",
        default=str(PROJECT_ROOT / "data" / "09_ensemble_hybride" / "models.pkl"),
        help="Chemin modèle entraîné (.pkl)",
    )
    parser.add_argument("--output",      required=True,  help="Fichier sortie prédictions (.json, .parquet ou .csv)")
    parser.add_argument("--username",    default=None,   help="Username ES (si auth activée)")
    parser.add_argument("--password",    default=None,   help="Password ES")
    parser.add_argument("--timeout",     type=int, default=30, help="Timeout requêtes HTTP (secondes)")
    parser.add_argument(
        "--predictions-index",
        default=None,
        help="Index ES optionnel pour réinjecter prédictions (ex: ueba-predictions)",
    )
    parser.add_argument("--model-version", default=None, help="Tag version modèle (metadata)")
    return parser.parse_args()


# ============================================================================
# SAUVEGARDE
# ============================================================================

def save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Sauvegarde DataFrame en JSON, Parquet ou CSV selon l'extension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext == ".parquet":
        df.to_parquet(output_path, index=False)
    elif ext == ".csv":
        df.to_csv(output_path, index=False)
    elif ext == ".json":
        df.to_json(output_path, orient="records", indent=2, default_handler=str)
    else:
        raise ValueError(f"Extension non supportée : {ext}. Utilise .json, .parquet ou .csv")


# ============================================================================
# RÉINJECTION ELASTICSEARCH
# ============================================================================

def post_bulk_predictions(
    es_url: str,
    index: str,
    rows: List[dict],
    username: Optional[str],
    password: Optional[str],
    timeout: int,
) -> None:
    """Envoie prédictions vers Elasticsearch via _bulk API."""
    if not rows:
        return
    import urllib.request

    lines: List[str] = []
    for row in rows:
        lines.append(json.dumps({"index": {"_index": index}}, ensure_ascii=True))
        lines.append(json.dumps(row, ensure_ascii=True, default=str))
    body = ("\n".join(lines) + "\n").encode("utf-8")

    req = urllib.request.Request(
        url=f"{es_url.rstrip('/')}/_bulk",
        data=body,
        headers={"Content-Type": "application/x-ndjson"},
        method="POST",
    )
    if username:
        token = base64.b64encode(
            f"{username}:{password or ''}".encode("utf-8")
        ).decode("ascii")
        req.add_header("Authorization", f"Basic {token}")

    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
        if payload.get("errors"):
            raise RuntimeError("Elasticsearch bulk indexing returned errors")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()
    log  = elk_feat.get_logger()

    # ── Parsing dates ────────────────────────────────────────────────────────
    # On normalise immédiatement en UTC naïf pour une cohérence totale
    start_dt = elk_feat.parse_ts(args.start)
    end_dt   = elk_feat.parse_ts(args.end)
    start_naive = _strip_tz(start_dt)
    end_naive   = _strip_tz(end_dt)
    fetch_start_dt = start_dt - pd.Timedelta(days=args.lookback_days)

    log.info("Fenêtre scoring  : %s → %s", start_naive, end_naive)
    log.info("Fenêtre fetch ES : %s → %s", fetch_start_dt, end_dt)

    # ── Connexion Elasticsearch ───────────────────────────────────────────────
    client = elk_feat.ElasticsearchClient(
        args.es_url, args.username, args.password, args.timeout
    )
    query = {
        "bool": {
            "filter": [
                {
                    "range": {
                        "@timestamp": {
                            "gte": fetch_start_dt.isoformat(),
                            "lte": end_dt.isoformat(),
                        }
                    }
                }
            ]
        }
    }

    # ── Récupération événements ───────────────────────────────────────────────
    log.info("Fetching ELK events for scoring")
    events = client.search_scroll(
        args.index, query, size=args.batch_size, scroll=args.scroll
    )
    if not events:
        raise RuntimeError(
            "No events returned from Elasticsearch for the requested range. "
            f"Vérifie que l'index '{args.index}' existe : "
            f"curl http://localhost:9200/_cat/indices?v"
        )

    # ── Normalisation ─────────────────────────────────────────────────────────
    normalized = elk_feat.normalize_events(events, log)
    if normalized.empty:
        raise RuntimeError("No usable normalized events returned from Elasticsearch")
    log.info("Normalized %d events across %d users",
             len(normalized), normalized["user_id"].nunique())

    # ── Feature engineering ───────────────────────────────────────────────────
    features = elk_feat.build_features(normalized, log)
    log.info("Built %d feature rows before window filter", len(features))

    # ── FIX TIMEZONE : normalise la colonne window vers naïf UTC ─────────────
    win_col      = _normalize_window_col(features["window"])

    log.info(
        "Window features dispo : %s → %s",
        win_col.min(), win_col.max()
    )
    log.info(
        "Window filtre appliqué: %s → %s",
        start_naive, end_naive
    )

    mask     = (win_col >= start_naive) & (win_col <= end_naive)
    features = features.loc[mask].copy()

    if features.empty:
        raise RuntimeError(
            f"No feature rows available in the requested output window "
            f"({start_naive} → {end_naive}). "
            f"Plage features disponible : {win_col.min()} → {win_col.max()}. "
            f"Ajuste --start / --end pour correspondre à cette plage."
        )

    log.info("Feature rows après filtre : %d", len(features))

    # ── Chargement modèle ─────────────────────────────────────────────────────
    models_path = Path(args.models_path)
    if not models_path.exists():
        raise FileNotFoundError(f"Model file not found: {models_path}")

    artifact          = joblib.load(models_path)
    selected_features = artifact.get("selected_features")
    threshold         = float(artifact.get("threshold", 0.5))

    if not selected_features:
        raise RuntimeError("Model artifact does not contain 'selected_features'")

    model_entries = {
        k: v
        for k, v in artifact.items()
        if isinstance(v, dict) and "model" in v and "type" in v
    }
    if not model_entries:
        raise RuntimeError("No model entries found in artifact")

    log.info(
        "Modèle chargé : %d sous-modèles | %d features | threshold=%.3f",
        len(model_entries), len(selected_features), threshold
    )

    # ── Préparation features pour scoring ─────────────────────────────────────
    X = features.copy()
    missing_cols = [col for col in selected_features if col not in X.columns]
    if missing_cols:
        log.warning("%d features manquantes (remplies à 0) : %s",
                    len(missing_cols), missing_cols[:5])
    for col in missing_cols:
        X[col] = 0.0
    X = X[selected_features].copy()

    # ── Scoring ───────────────────────────────────────────────────────────────
    if model_3 is None:
        raise RuntimeError(
            "Le module ensemble_ml est introuvable. "
            "Vérifie que pipelines/ensemble_ml.py existe et est importable."
        )

    log.info("Scoring %d rows avec %d features…", len(X), len(selected_features))
    probs = model_3.ensemble_predict(model_entries, X)
    preds = (probs >= threshold).astype(int)

    n_alerts = int(preds.sum())
    log.info(
        "Scoring terminé : %d alertes / %d rows (%.2f%%)",
        n_alerts, len(preds), 100 * n_alerts / max(len(preds), 1)
    )

    # ── Métadonnées modèle ────────────────────────────────────────────────────
    model_version = args.model_version or (
        f"{models_path.stem}_"
        f"{datetime.fromtimestamp(models_path.stat().st_mtime).strftime('%Y%m%d_%H%M%S')}"
    )

    # ── Construction résultats ────────────────────────────────────────────────
    result = features[["user_id", "window"]].copy()
    result["@timestamp"]           = result["window"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    result["risk_score"]           = probs
    result["is_insider_pred"]      = preds
    result["alert"]                = preds.astype(bool)
    result["threshold"]            = threshold
    result["model_version"]        = model_version
    result["missing_feature_count"] = len(missing_cols)
    result["missing_feature_names"] = ",".join(missing_cols)

    # ── Sauvegarde locale ─────────────────────────────────────────────────────
    output_path = Path(args.output)
    save_output(result, output_path)
    log.info("Saved %d predictions → %s", len(result), output_path)

    # ── Réinjection ES (optionnel) ────────────────────────────────────────────
    if args.predictions_index:
        rows = result.to_dict(orient="records")
        post_bulk_predictions(
            es_url=args.es_url,
            index=args.predictions_index,
            rows=rows,
            username=args.username,
            password=args.password,
            timeout=args.timeout,
        )
        log.info(
            "Indexed %d predictions → ES index '%s'",
            len(rows), args.predictions_index
        )

    # ── Résumé final ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RÉSUMÉ")
    log.info("  Rows scorées   : %d", len(result))
    log.info("  Alertes        : %d (%.2f%%)",
             n_alerts, 100 * n_alerts / max(len(result), 1))
    log.info("  Risk score max : %.4f", float(probs.max()))
    log.info("  Risk score moy : %.4f", float(probs.mean()))
    log.info("  Output         : %s", output_path)
    if args.predictions_index:
        log.info("  ES index       : %s", args.predictions_index)
    log.info("=" * 60)


if __name__ == "__main__":
    main()

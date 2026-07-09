"""
pipelinev4.py - PIPELINE ML COMPLET
===========================================
Pipeline UEBA production-grade pour CERT r4.2
Version baseline historique (reference F1 ~0.48).

ÉTAPES:
  Étape 1 → Seuil corrélation > 0.85  (supprime features redondantes)
  Étape 2 → SMOTE                      (gère déséquilibre is_insider)
  Étape 3 → LightGBM + feature import  (garde top 30 features)
  Étape 4 → Évaluation finale           (AUC-ROC, Precision-Recall, F1)

SORTIES:
  - data/04_model/model.pkl
  - data/04_model/selected_features.json
  - data/04_model/evaluation_report.json
  - data/04_model/plots/  (ROC, PR curve, feature importance, confusion matrix)

USAGE:
  python scripts/pipelinev4.py
  python scripts/pipelinev4.py --slice 3
  python scripts/pipelinev4.py --top-features 20
  python scripts/pipelinev4.py --no-smote
"""

import gc
import argparse
import json
import logging
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(encoding="utf-8")
Apikey = "334dDFSSDHDHJHKKfghdd452"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# CONSTANTES
# =============================================================================

CORRELATION_THRESHOLD = 0.85
TOP_N_FEATURES = 30
RANDOM_STATE = 42
N_SPLITS_CV = 5
CORRELATION_MAX_ROWS = 200_000

# Colonnes à exclure de la modélisation
EXCLUDE_COLS = {
    "user_id", "window", "is_insider", "scenario",
    "role", "department", "business_unit",
}

# Hyperparamètres LightGBM optimisés UEBA (déséquilibre de classes)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


# =============================================================================
# CONFIG & LOGGING
# =============================================================================

def load_config(path: Optional[Path] = None) -> dict:
    """Charge et valide la configuration."""
    cfg_path = path or PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = {
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

    logger = logging.getLogger("ModelPipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s")

    fh = logging.FileHandler(
        Path(logs_dir) / f"pipelinev4_{ts}.log", encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_features(
    features_dir: Path,
    slice_ids: Optional[List[int]],
    log: logging.Logger,
) -> pd.DataFrame:
    """Charge et concatène toutes les features de toutes les slices."""
    if not features_dir.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {features_dir}")

    if slice_ids is None:
        slice_ids = sorted(
            int(p.name.split("_")[1])
            for p in features_dir.iterdir()
            if p.is_dir() and p.name.startswith("slice_")
        )

    dfs = []
    for sid in slice_ids:
        path = features_dir / f"slice_{sid:03d}" / "features.parquet"
        if not path.exists():
            log.warning(f"  Slice {sid:03d}: features.parquet absent — ignoré")
            continue
        df = pd.read_parquet(path)
        df["slice_id"] = sid
        dfs.append(df)
        log.info(f"  Slice {sid:03d}: {len(df):,} lignes chargées")

    if not dfs:
        raise FileNotFoundError("Aucun fichier features.parquet trouvé.")

    full = pd.concat(dfs, ignore_index=True)
    log.info(f"  Total: {len(full):,} lignes │ {len(full.columns)} colonnes")
    return full


def optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Réduit l'empreinte mémoire des colonnes numériques."""
    if df.empty:
        return df

    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns

    if len(float_cols) > 0:
        df.loc[:, float_cols] = df.loc[:, float_cols].astype(np.float32)
    if len(int_cols) > 0:
        df.loc[:, int_cols] = df.loc[:, int_cols].apply(pd.to_numeric, downcast="integer")
    return df


def prepare_X_y(df: pd.DataFrame, log: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Prépare X (features) et y (cible) depuis le DataFrame brut."""
    if "is_insider" not in df.columns:
        raise ValueError("Colonne 'is_insider' absente du DataFrame.")

    y = df["is_insider"].astype(int)

    exclude = EXCLUDE_COLS | {"slice_id"}
    feature_cols = [c for c in df.columns if c not in exclude]
    if not feature_cols:
        raise ValueError("Aucune colonne candidate pour les features.")

    # Garder uniquement les colonnes numériques
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    if X.empty:
        raise ValueError("Aucune feature numérique disponible après filtrage.")

    # Remplir NaN résiduels
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = optimize_numeric_dtypes(X)

    log.info(f"  X: {X.shape} │ y: insiders={y.sum():,} ({y.mean() * 100:.3f}%)")
    return X, y


# =============================================================================
# ÉTAPE 1 — SUPPRESSION CORRÉLATION > 0.85
# =============================================================================

def remove_correlated_features(
    X: pd.DataFrame,
    threshold: float,
    log: logging.Logger,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Supprime les features avec corrélation Pearson > threshold.

    Stratégie : pour chaque paire corrélée, supprimer la feature
    avec la plus faible variance (moins informative).
    """
    log.info(f"\n{'─' * 60}")
    log.info(f"  ÉTAPE 1 — Suppression corrélation > {threshold}")
    log.info(f"{'─' * 60}")
    log.info(f"  Features initiales: {X.shape[1]}")

    if X.shape[1] < 2:
        log.info("  Pas assez de features pour calculer la corrélation.")
        return X, []

    # Supprimer les colonnes constantes avant la corrélation.
    nunique = X.nunique(dropna=False)
    constant_cols = sorted(nunique[nunique <= 1].index.tolist())
    if constant_cols:
        X = X.drop(columns=constant_cols)
        log.info(f"  Colonnes constantes supprimées: {len(constant_cols)}")
        if X.shape[1] < 2:
            return X, constant_cols

    corr_source = X
    if len(X) > CORRELATION_MAX_ROWS:
        corr_source = X.sample(n=CORRELATION_MAX_ROWS, random_state=RANDOM_STATE)
        log.info(f"  Corrélation calculée sur un échantillon de {len(corr_source):,} lignes")

    corr_matrix = corr_source.corr(method="pearson").abs()

    # Triangle supérieur uniquement (évite les doublons)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Identifier les colonnes à supprimer
    # Pour chaque paire corrélée → supprimer celle avec la plus faible variance
    to_drop = set()
    variances = X.var()

    for col in upper.columns:
        if col in to_drop:
            continue
        # Partenaires fortement corrélés
        partners = upper.index[upper[col] > threshold].tolist()
        for partner in partners:
            if partner in to_drop:
                continue
            # Supprimer celui avec la plus faible variance
            if variances[col] >= variances[partner]:
                to_drop.add(partner)
            else:
                to_drop.add(col)
                break  # col est supprimé, passer au suivant

    to_drop_list = sorted(to_drop)
    X_filtered = X.drop(columns=to_drop_list)
    dropped_total = sorted(set(to_drop_list) | set(constant_cols))

    log.info(f"  Features supprimées: {len(dropped_total)}")
    if dropped_total:
        for f in dropped_total:
            log.info(f"    ✗ {f}")
    log.info(f"  Features restantes: {X_filtered.shape[1]}")

    return X_filtered, dropped_total


# =============================================================================
# ÉTAPE 2 — SMOTE (GESTION DÉSÉQUILIBRE)
# =============================================================================

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool,
    log: logging.Logger,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applique SMOTE sur le jeu d'entra?nement uniquement.

    Note : SMOTE appliqu? SEULEMENT sur train, jamais sur test/val.
    """
    log.info(f"\n{'?' * 60}")
    log.info(f"  ?TAPE 2 ? Gestion d?s?quilibre")
    log.info(f"{'?' * 60}")

    n_majority = int((y_train == 0).sum())
    n_minority = int((y_train == 1).sum())
    ratio = (n_minority / n_majority * 100) if n_majority > 0 else 0.0

    log.info(f"  Avant SMOTE: majority={n_majority:,} ? minority={n_minority:,} ({ratio:.3f}%)")

    if not use_smote:
        log.info("  SMOTE d?sactiv? ? utilisation de class_weight dans LightGBM")
        return X_train, y_train

    if n_majority == 0 or n_minority == 0:
        log.warning("  SMOTE: une seule classe pr?sente ? SMOTE ignor?")
        return X_train, y_train

    if n_minority < 6:
        log.warning(f"  SMOTE: trop peu d'insiders ({n_minority}) ? d?sactiv?, class_weight utilis?")
        return X_train, y_train

    k_neighbors = min(5, n_minority - 1)

    smote = SMOTE(
        sampling_strategy="auto",
        k_neighbors=k_neighbors,
        random_state=RANDOM_STATE,
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name="is_insider")

    n_new_minority = int((y_resampled == 1).sum())
    n_new_majority = int((y_resampled == 0).sum())
    log.info(f"  Apr?s SMOTE : majority={n_new_majority:,} ? minority={n_new_minority:,}")

    return X_resampled, y_resampled

def compute_scale_pos_weight(y: pd.Series) -> float:
    """Calcule scale_pos_weight pour LightGBM (ratio majority/minority)."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / n_pos if n_pos > 0 else 1.0


def train_lightgbm_cv(
    X: pd.DataFrame,
    y: pd.Series,
    use_smote: bool,
    log: logging.Logger,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Entra?ne LightGBM avec validation crois?e stratifi?e.

    Retourne:
    - model final entra?n? sur toutes les donn?es
    - DataFrame feature importances
    - dict m?triques CV
    """
    log.info(f"\n{'?' * 60}")
    log.info(f"  ?TAPE 3 ? LightGBM + Feature Importance")
    log.info(f"{'?' * 60}")

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Impossible d'entra?ner: une seule classe pr?sente dans y.")
    n_splits = min(N_SPLITS_CV, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError(
            f"Pas assez d'exemples dans la classe minoritaire pour la CV (min_class={int(class_counts.min())})."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    cv_auc_roc = []
    cv_auc_pr = []
    cv_f1 = []
    importance_folds = []

    params = LGBM_PARAMS.copy()

    if not use_smote:
        spw = compute_scale_pos_weight(y)
        params["scale_pos_weight"] = spw
        log.info(f"  scale_pos_weight = {spw:.2f}")

    log.info(f"  Cross-validation: {n_splits} folds stratifi?s")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_tr, y_tr = apply_smote(X_tr, y_tr, use_smote, log) if fold == 1 else _smote_silent(X_tr, y_tr, use_smote)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fold_auc = roc_auc_score(y_val, y_prob) if y_val.nunique() > 1 else np.nan
        fold_pr = average_precision_score(y_val, y_prob)
        fold_f1 = f1_score(y_val, y_pred, zero_division=0)

        cv_auc_roc.append(fold_auc)
        cv_auc_pr.append(fold_pr)
        cv_f1.append(fold_f1)

        imp = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
            "fold": fold,
        })
        importance_folds.append(imp)

        log.info(
            f"  Fold {fold}: AUC-ROC={fold_auc:.4f} ? AUC-PR={fold_pr:.4f} ? F1={fold_f1:.4f}"
        )
        del model, X_tr, X_val, y_tr, y_val, y_prob, y_pred, imp
        gc.collect()

    cv_metrics = {
        "auc_roc_mean": float(np.nanmean(cv_auc_roc)),
        "auc_roc_std": float(np.nanstd(cv_auc_roc)),
        "auc_pr_mean": float(np.nanmean(cv_auc_pr)),
        "auc_pr_std": float(np.nanstd(cv_auc_pr)),
        "f1_mean": float(np.nanmean(cv_f1)),
        "f1_std": float(np.nanstd(cv_f1)),
    }

    log.info(f"\n  CV Results:")
    log.info(f"  AUC-ROC : {cv_metrics['auc_roc_mean']:.4f} ? {cv_metrics['auc_roc_std']:.4f}")
    log.info(f"  AUC-PR  : {cv_metrics['auc_pr_mean']:.4f} ? {cv_metrics['auc_pr_std']:.4f}")
    log.info(f"  F1      : {cv_metrics['f1_mean']:.4f} ? {cv_metrics['f1_std']:.4f}")

    all_imp = pd.concat(importance_folds, ignore_index=True)
    mean_imp = (
        all_imp.groupby("feature")["importance"]
        .mean()
        .reset_index()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return mean_imp, cv_metrics

def _smote_silent(
    X: pd.DataFrame, y: pd.Series, use_smote: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    """SMOTE sans logging (pour les folds 2+)."""
    if not use_smote:
        return X, y
    n_minority = int((y == 1).sum())
    n_majority = int((y == 0).sum())
    if n_majority == 0 or n_minority < 6:
        return X, y
    k_neighbors = min(5, n_minority - 1)
    smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    X_r, y_r = smote.fit_resample(X, y)
    return pd.DataFrame(X_r, columns=X.columns), pd.Series(y_r, name="is_insider")

def select_top_features(
    X: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int,
    log: logging.Logger,
) -> Tuple[pd.DataFrame, List[str]]:
    """Sélectionne les top N features par importance LightGBM."""
    top_features = importance_df.head(top_n)["feature"].tolist()
    # Garder seulement celles présentes dans X
    top_features = [f for f in top_features if f in X.columns]

    log.info(f"\n  Top {top_n} features sélectionnées:")
    for i, feat in enumerate(top_features, 1):
        imp = importance_df.loc[importance_df["feature"] == feat, "importance"].values
        imp_val = imp[0] if len(imp) > 0 else 0
        log.info(f"    {i:2d}. {feat:<45} {imp_val:.1f}")

    return X[top_features], top_features


# =============================================================================
# ÉTAPE 4 — ÉVALUATION FINALE
# =============================================================================

def evaluate_final_model(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cv_metrics: Dict,
    plots_dir: Path,
    log: logging.Logger,
) -> Dict:
    """
    ?valuation compl?te sur tout le dataset (+ CV metrics d?j? calcul?es).
    G?n?re ROC, PR curve, confusion matrix, feature importance plot.
    """
    log.info(f"\n{'?' * 60}")
    log.info(f"  ?TAPE 4 ? ?valuation finale")
    log.info(f"{'?' * 60}")

    plots_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X)[:, 1]

    thresholds = np.arange(0.1, 0.95, 0.01)
    f1_scores = [
        f1_score(y, (y_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_threshold = float(thresholds[np.argmax(f1_scores)])
    y_pred = (y_prob >= best_threshold).astype(int)

    log.info(f"  Seuil optimal (F1 max): {best_threshold:.2f}")

    report = classification_report(
        y,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Insider"],
        output_dict=True,
        zero_division=0,
    )
    auc_roc = roc_auc_score(y, y_prob) if y.nunique() > 1 else float('nan')
    auc_pr = average_precision_score(y, y_prob)
    f1 = f1_score(y, y_pred, zero_division=0)

    evaluation = {
        "best_threshold": best_threshold,
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "f1_score": float(f1),
        "classification_report": report,
        "cv_metrics": cv_metrics,
        "n_samples": int(len(y)),
        "n_insiders": int(y.sum()),
        "insider_rate_pct": float(y.mean() * 100),
    }

    log.info(f"  AUC-ROC  : {auc_roc:.4f}")
    log.info(f"  AUC-PR   : {auc_pr:.4f}")
    log.info(f"  F1-Score : {f1:.4f}")
    log.info(f"  Precision Insider: {report['Insider']['precision']:.4f}")
    log.info(f"  Recall Insider   : {report['Insider']['recall']:.4f}")

    _plot_roc_curve(y, y_prob, auc_roc, plots_dir)
    _plot_pr_curve(y, y_prob, auc_pr, plots_dir)
    _plot_confusion_matrix(y, y_pred, best_threshold, plots_dir)
    _plot_feature_importance(model, X.columns.tolist(), plots_dir)
    _plot_threshold_analysis(y, y_prob, thresholds, f1_scores, best_threshold, plots_dir)

    log.info(f"  ? Plots sauvegard?s dans: {plots_dir}")

    return evaluation

def _plot_roc_curve(y, y_prob, auc_roc, plots_dir):
    if pd.Series(y).nunique() < 2:
        return
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC-ROC = {auc_roc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve ? Insider Threat Detection", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_pr_curve(y, y_prob, auc_pr, plots_dir):
    precision, recall, _ = precision_recall_curve(y, y_prob)
    baseline = y.mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#3498db", lw=2, label=f"AUC-PR = {auc_pr:.4f}")
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1.5,
               label=f"Baseline (random) = {baseline:.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color="#3498db")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Insider Threat Detection", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(y, y_pred, threshold, plots_dir):
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Insider"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix (seuil={threshold:.2f})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(model, feature_names, plots_dir, top_n=30):
    importances = model.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp_df)))
    bars = ax.barh(imp_df["feature"], imp_df["importance"], color=colors)

    # Valeurs sur les barres
    for bar, val in zip(bars, imp_df["importance"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=8)

    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — LightGBM", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_analysis(y, y_prob, thresholds, f1_scores, best_threshold, plots_dir):
    """Courbe F1 / Précision / Recall en fonction du seuil."""
    precisions = []
    recalls = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        from sklearn.metrics import precision_score, recall_score
        precisions.append(precision_score(y, y_pred_t, zero_division=0))
        recalls.append(recall_score(y, y_pred_t, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, f1_scores, label="F1-Score", color="#e74c3c", lw=2)
    ax.plot(thresholds, precisions, label="Precision", color="#3498db", lw=2, linestyle="--")
    ax.plot(thresholds, recalls, label="Recall", color="#2ecc71", lw=2, linestyle="--")
    ax.axvline(x=best_threshold, color="black", linestyle=":", lw=1.5,
               label=f"Seuil optimal = {best_threshold:.2f}")
    ax.set_xlabel("Seuil de décision", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Analyse du seuil de décision", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(plots_dir / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SAUVEGARDE
# =============================================================================

def save_artifacts(
    model: lgb.LGBMClassifier,
    selected_features: List[str],
    dropped_features: List[str],
    evaluation: Dict,
    importance_df: pd.DataFrame,
    model_dir: Path,
    log: logging.Logger,
) -> None:
    """Sauvegarde tous les artefacts du pipeline."""
    model_dir.mkdir(parents=True, exist_ok=True)

    # Modèle
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    log.info(f"  ✓ Modèle: {model_path}")

    # Features sélectionnées
    features_path = model_dir / "selected_features.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_features": selected_features,
                "dropped_correlated": dropped_features,
                "n_selected": len(selected_features),
                "n_dropped": len(dropped_features),
            },
            f, indent=2, ensure_ascii=False,
        )
    log.info(f"  ✓ Features: {features_path}")

    # Rapport évaluation
    eval_path = model_dir / "evaluation_report.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    log.info(f"  ✓ Évaluation: {eval_path}")

    # Feature importance CSV
    imp_path = model_dir / "feature_importance.csv"
    importance_df.to_csv(imp_path, index=False)
    log.info(f"  ✓ Importances: {imp_path}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_pipeline(
    cfg: dict,
    log: logging.Logger,
    slice_ids: Optional[List[int]] = None,
    top_n_features: int = TOP_N_FEATURES,
    use_smote: bool = True,
) -> None:
    """Pipeline ML complet : Corrélation → SMOTE → LightGBM → Évaluation."""
    if top_n_features < 1:
        raise ValueError("--top-features doit être >= 1")

    features_dir = Path(cfg["output"]["features_dir"])
    model_dir = features_dir.parent / "04_model"
    plots_dir = model_dir / "plots"

    log.info("=" * 70)
    log.info("PIPELINE ML UEBA — CERT r4.2")
    log.info("=" * 70)
    log.info(f"Corrélation seuil : {CORRELATION_THRESHOLD}")
    log.info(f"SMOTE             : {'Activé' if use_smote else 'Désactivé (class_weight)'}")
    log.info(f"Top features      : {top_n_features}")
    log.info(f"CV folds          : {N_SPLITS_CV}")
    log.info("=" * 70)

    t0 = datetime.now()

    # --- Chargement ---
    log.info("\n  Chargement des features...")
    df = load_all_features(features_dir, slice_ids, log)
    X, y = prepare_X_y(df, log)
    del df
    gc.collect()

    # --- Étape 1 : Corrélation ---
    X_filtered, dropped = remove_correlated_features(X, CORRELATION_THRESHOLD, log)
    del X
    gc.collect()
    if X_filtered.shape[1] == 0:
        raise ValueError("Aucune feature restante après filtrage de corrélation.")

    # --- Étape 2+3 : SMOTE + LightGBM CV ---
    importance_df, cv_metrics = train_lightgbm_cv(X_filtered, y, use_smote, log)

    # --- Sélection top N features ---
    top_n_features = min(top_n_features, X_filtered.shape[1])
    X_top, selected_features = select_top_features(X_filtered, importance_df, top_n_features, log)
    del X_filtered
    gc.collect()

    # --- Ré-entraînement final sur top features uniquement ---
    log.info(f"\n  Ré-entraînement final sur top {top_n_features} features...")
    X_top_final, y_final = _smote_silent(X_top, y, use_smote)

    params = LGBM_PARAMS.copy()
    if not use_smote:
        params["scale_pos_weight"] = compute_scale_pos_weight(y)

    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_top_final, y_final, callbacks=[lgb.log_evaluation(-1)])

    # --- Étape 4 : Évaluation ---
    evaluation = evaluate_final_model(final_model, X_top, y, cv_metrics, plots_dir, log)

    # --- Sauvegarde ---
    log.info(f"\n{'─' * 60}")
    log.info(f"  Sauvegarde des artefacts...")
    save_artifacts(
        model=final_model,
        selected_features=selected_features,
        dropped_features=dropped,
        evaluation=evaluation,
        importance_df=importance_df,
        model_dir=model_dir,
        log=log,
    )

    elapsed = (datetime.now() - t0).total_seconds() / 60

    log.info(f"\n{'=' * 70}")
    log.info(f"✅ PIPELINE TERMINÉ — {elapsed:.1f} minutes")
    log.info(f"{'=' * 70}")
    log.info(f"  AUC-ROC  (CV) : {cv_metrics['auc_roc_mean']:.4f} ± {cv_metrics['auc_roc_std']:.4f}")
    log.info(f"  AUC-PR   (CV) : {cv_metrics['auc_pr_mean']:.4f} ± {cv_metrics['auc_pr_std']:.4f}")
    log.info(f"  F1       (CV) : {cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}")
    log.info(f"  AUC-ROC final : {evaluation['auc_roc']:.4f}")
    log.info(f"  AUC-PR  final : {evaluation['auc_pr']:.4f}")
    log.info(f"  F1      final : {evaluation['f1_score']:.4f}")
    log.info(f"{'=' * 70}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(description="Pipeline ML UEBA (CERT r4.2)")
    parser.add_argument("--slice", type=int, default=None, help="ID d'une slice unique")
    parser.add_argument(
        "--top-features",
        type=int,
        default=TOP_N_FEATURES,
        help="Nombre de features à conserver après importance LightGBM",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Désactive SMOTE et utilise scale_pos_weight",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        args = parse_args(sys.argv[1:])
        cfg = load_config()
        log = get_logger(cfg["output"]["logs_dir"])

        slice_ids = [args.slice] if args.slice is not None else None
        run_pipeline(cfg, log, slice_ids, args.top_features, not args.no_smote)

    except Exception as e:
        print(f"❌ Erreur: {e}")
        traceback.print_exc()
        sys.exit(1)

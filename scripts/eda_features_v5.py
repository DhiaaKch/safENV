"""
eda_features_v5.py - EDA UEBA resource-aware
======================================================================

AMÉLIORATIONS:
✅ Matrices de corrélation GRANDES (carrés visibles)
✅ VALEURS affichées dans chaque cellule
✅ Taille adaptative selon nombre de features
✅ Pearson + Spearman en haute résolution
✅ Couleurs optimisées pour lisibilité

Usage:
  python scripts/eda_features_v5.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_VIF = True
except Exception:
    HAS_VIF = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
ANALYSIS_EXCLUDE = {"is_insider", "scenario", "slice_id"}

MAX_CORR_PLOT_FEATURES = 50
MAX_VIF_FEATURES = 25
MAX_DISTRIBUTION_FEATURES = 6
MAX_CLASS_FEATURES = 6
MAX_TIME_FEATURES = 6
MAX_PAIRPLOT_FEATURES = 5
MAX_PAIRPLOT_ROWS = 800
MAX_DIST_ROWS = 10000
MAX_CLASS_ROWS_PER_CLASS = 2000
MAX_PROXY_IMPORTANCE_ROWS = 8000
MAX_PROXY_IMPORTANCE_FEATURES = 40
MAX_PCA_ROWS = 5000
MAX_PCA_FEATURES = 15
KMEANS_CLUSTERS = 3


def get_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("EDAFeaturesV5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(logs_dir / f"eda_features_v5_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_feature_files(features_dir: Path) -> list[Path]:
    return sorted(
        p for p in features_dir.glob("slice_*/features.parquet") if p.is_file()
    )


def ensure_dirs(reports_dir: Path) -> tuple[Path, Path]:
    out_dir = reports_dir / "features_eda_v5"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, plots_dir


def quote_paths(paths: Iterable[Path]) -> str:
    return ", ".join(f"'{p.as_posix()}'" for p in paths)


def load_features_df(files: list[Path]) -> pd.DataFrame:
    con = duckdb.connect()
    rel = quote_paths(files)
    df = con.execute(f"SELECT * FROM read_parquet([{rel}])").df()
    con.close()
    if "window" in df.columns:
        df["window"] = pd.to_datetime(df["window"], errors="coerce", utc=True)
    return df


def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number, "bool"]).copy()
    for c in num.columns:
        if str(num[c].dtype) == "bool":
            num[c] = num[c].astype(int)
    return num


def duplicate_key_count(df: pd.DataFrame) -> int:
    if "user_id" not in df.columns or "window" not in df.columns:
        return 0
    return int(df.duplicated(subset=["user_id", "window"]).sum())


def feature_columns(
    num_df: pd.DataFrame,
    exclude: Iterable[str] = ANALYSIS_EXCLUDE,
    min_unique: int = 2,
) -> list[str]:
    excluded = set(exclude)
    cols: list[str] = []
    for c in num_df.columns:
        if c in excluded:
            continue
        if num_df[c].nunique(dropna=True) < min_unique:
            continue
        cols.append(c)
    return cols


def continuous_feature_columns(num_df: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(num_df) if num_df[c].nunique(dropna=True) > 5]


def variance_ranked_features(
    num_df: pd.DataFrame,
    cols: Sequence[str] | None = None,
) -> list[str]:
    if cols is None:
        cols = feature_columns(num_df)
    cols = [c for c in cols if c in num_df.columns]
    if not cols:
        return []
    return num_df[cols].var().sort_values(ascending=False).index.tolist()


def sample_dataframe(
    df: pd.DataFrame,
    max_rows: int,
    stratify_col: str | None = None,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    if stratify_col and stratify_col in df.columns and df[stratify_col].nunique(dropna=True) > 1:
        parts: list[pd.DataFrame] = []
        total = max(len(df), 1)
        for _, grp in df.groupby(stratify_col, dropna=False):
            n = max(1, int(round(len(grp) / total * max_rows)))
            n = min(n, len(grp))
            parts.append(grp.sample(n=n, random_state=RANDOM_STATE))
        out = pd.concat(parts, axis=0)
        if len(out) > max_rows:
            out = out.sample(n=max_rows, random_state=RANDOM_STATE)
        return out.sort_index()
    return df.sample(n=max_rows, random_state=RANDOM_STATE).sort_index()


def sample_per_class(df: pd.DataFrame, target_col: str, max_rows_per_class: int) -> pd.DataFrame:
    if target_col not in df.columns:
        return sample_dataframe(df, max_rows=max_rows_per_class)
    parts: list[pd.DataFrame] = []
    for _, grp in df.groupby(target_col, dropna=False):
        n = min(len(grp), max_rows_per_class)
        if len(grp) > n:
            parts.append(grp.sample(n=n, random_state=RANDOM_STATE))
        else:
            parts.append(grp.copy())
    return pd.concat(parts, axis=0).sort_index()


def prepare_numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df.replace([np.inf, -np.inf], np.nan).copy()
    medians = x.median(numeric_only=True).fillna(0.0)
    return x.fillna(medians).astype(float)


def target_label_series(s: pd.Series) -> pd.Series:
    return s.astype("Int64").map({0: "normal", 1: "insider"}).fillna("unknown")


def merge_ranked_features(
    num_df: pd.DataFrame,
    ranked_frames: Sequence[pd.DataFrame],
    limit: int,
    continuous_only: bool = False,
) -> list[str]:
    allowed = set(
        continuous_feature_columns(num_df) if continuous_only else feature_columns(num_df)
    )
    ranked: list[str] = []
    for frame in ranked_frames:
        if frame is None or frame.empty or "feature" not in frame.columns:
            continue
        for c in frame["feature"].tolist():
            if c in allowed and c not in ranked:
                ranked.append(c)
    for c in variance_ranked_features(num_df, cols=list(allowed)):
        if c not in ranked:
            ranked.append(c)
    return ranked[:limit]


# ================================================================
# ✅ MATRICE DE CORRÉLATION OPTIMISÉE
# ================================================================

def correlation_outputs_optimized(
    num_df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Génère matrices de corrélation GRANDES avec valeurs dans cellules.
    """
    
    if num_df.empty:
        return pd.DataFrame()
    
    logger.info("Computing correlation matrices...")
    
    pearson = num_df.corr(method="pearson", numeric_only=True)
    spearman = num_df.corr(method="spearman", numeric_only=True)
    
    # Sauvegarder CSV
    pearson.to_csv(out_dir / "correlation_matrix_pearson.csv")
    spearman.to_csv(out_dir / "correlation_matrix_spearman.csv")
    logger.info("  CSV saved")

    if not HAS_PLOT:
        logger.warning("  Matplotlib unavailable - skipping plots")
        return pd.DataFrame()
    
    n_features = len(pearson.columns)
    logger.info(f"  Plotting correlation for {n_features} features...")
    
    # ✅ Taille adaptative
    if n_features <= 10:
        figsize = (16, 14)
        annot_fontsize = 11
        label_fontsize = 12
    elif n_features <= 20:
        figsize = (24, 22)
        annot_fontsize = 9
        label_fontsize = 10
    elif n_features <= 30:
        figsize = (32, 30)
        annot_fontsize = 7
        label_fontsize = 9
    elif n_features <= 50:
        figsize = (42, 40)
        annot_fontsize = 6
        label_fontsize = 8
    else:
        # Trop de features: prendre top 50
        logger.info(f"  Too many features ({n_features}), selecting top 50 by variance")
        variances = num_df.var().sort_values(ascending=False)
        top_50 = variances.head(50).index.tolist()
        pearson = pearson.loc[top_50, top_50]
        spearman = spearman.loc[top_50, top_50]
        n_features = 50
        figsize = (42, 40)
        annot_fontsize = 6
        label_fontsize = 8
    
    # ================================================================
    # HEATMAP PEARSON
    # ================================================================
    
    fig = plt.figure(figsize=figsize)
    
    if HAS_SNS:
        logger.info("  Creating Pearson heatmap with seaborn...")
        
        # ✅ Heatmap avec VALEURS dans cellules
        ax = sns.heatmap(
            pearson,
            annot=True,              # ← AFFICHE LES VALEURS ✅
            fmt=".2f",               # ← 2 décimales
            cmap="RdBu_r",           # ← Rouge-Blanc-Bleu inversé
            center=0.0,
            vmin=-1,
            vmax=1,
            linewidths=0.8,          # ← Lignes épaisses pour séparer cellules
            linecolor='gray',
            cbar_kws={
                'label': 'Corrélation de Pearson',
                'shrink': 0.85,
                'aspect': 30
            },
            annot_kws={'size': annot_fontsize, 'weight': 'bold'},  # ← Police en gras
            square=True              # ← Cellules CARRÉES (grandes) ✅
        )
        
        # Rotation labels
        plt.xticks(rotation=45, ha='right', fontsize=label_fontsize)
        plt.yticks(rotation=0, fontsize=label_fontsize)
        
    else:
        logger.info("  Creating Pearson heatmap (fallback mode)...")
        
        # Fallback sans seaborn
        plt.imshow(pearson.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        cbar = plt.colorbar(shrink=0.85)
        cbar.set_label('Corrélation de Pearson', fontsize=14)
        
        plt.xticks(range(n_features), pearson.columns, rotation=45, ha='right', fontsize=label_fontsize)
        plt.yticks(range(n_features), pearson.index, fontsize=label_fontsize)
        
        # ✅ Ajouter valeurs manuellement
        for i in range(n_features):
            for j in range(n_features):
                val = pearson.iloc[i, j]
                # Couleur texte selon fond
                color = 'white' if abs(val) > 0.5 else 'black'
                plt.text(j, i, f'{val:.2f}',
                        ha='center', va='center',
                        color=color, fontsize=annot_fontsize,
                        weight='bold')
    
    plt.title(
        "Matrice de Corrélation de Pearson - CERT r4.2 UEBA",
        fontsize=18,
        fontweight='bold',
        pad=25
    )
    
    plt.tight_layout()
    
    output_path = plots_dir / "correlation_heatmap_pearson_LARGE.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"  ✓ Pearson heatmap saved: {output_path}")
    
    # ================================================================
    # HEATMAP SPEARMAN (si pas trop de features)
    # ================================================================
    
    if n_features <= 40:
        fig2 = plt.figure(figsize=figsize)
        
        if HAS_SNS:
            logger.info("  Creating Spearman heatmap...")
            
            ax2 = sns.heatmap(
                spearman,
                annot=True,
                fmt=".2f",
                cmap="PuOr_r",       # ← Violet-Orange
                center=0.0,
                vmin=-1,
                vmax=1,
                linewidths=0.8,
                linecolor='gray',
                cbar_kws={
                    'label': 'Corrélation de Spearman',
                    'shrink': 0.85,
                    'aspect': 30
                },
                annot_kws={'size': annot_fontsize, 'weight': 'bold'},
                square=True
            )
            
            plt.xticks(rotation=45, ha='right', fontsize=label_fontsize)
            plt.yticks(rotation=0, fontsize=label_fontsize)
        
        plt.title(
            "Matrice de Corrélation de Spearman - CERT r4.2 UEBA",
            fontsize=18,
            fontweight='bold',
            pad=25
        )
        
        plt.tight_layout()
        
        output_path2 = plots_dir / "correlation_heatmap_spearman_LARGE.png"
        fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        logger.info(f"  ✓ Spearman heatmap saved: {output_path2}")
    
    # ================================================================
    # TOP PAIRES CORRÉLÉES
    # ================================================================
    
    upper = pearson.where(np.triu(np.ones(pearson.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_x", "level_1": "feature_y", 0: "corr"})
    )
    pairs["abs_corr"] = pairs["corr"].abs()
    pairs = pairs.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    pairs.to_csv(out_dir / "top_feature_correlations.csv", index=False)
    
    logger.info(f"  ✓ Top correlations saved")
    
    return pairs


# ================================================================
# AUTRES ANALYSES (inchangées)
# ================================================================

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().mul(100).sort_values(ascending=False)
    out = miss.reset_index()
    out.columns = ["feature", "missing_pct"]
    return out


def target_correlation(num_df: pd.DataFrame, target_col: str = "is_insider") -> pd.DataFrame:
    if target_col not in num_df.columns:
        return pd.DataFrame()
    target = num_df[target_col]
    rows = []
    for c in num_df.columns:
        if c == target_col:
            continue
        s = num_df[c]
        if s.nunique(dropna=True) <= 1:
            corr = 0.0
        else:
            corr = float(pd.Series(s).corr(target))
            if np.isnan(corr):
                corr = 0.0
        rows.append({"feature": c, "corr_with_is_insider": corr, "abs_corr": abs(corr)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_corr", ascending=False).reset_index(drop=True)


def descriptive_stats(num_df: pd.DataFrame) -> pd.DataFrame:
    if num_df.empty:
        return pd.DataFrame()
    q = num_df.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).T
    q.columns = ["p01", "p05", "p50", "p95", "p99"]
    out = pd.DataFrame(
        {
            "feature": num_df.columns,
            "count": num_df.count().values,
            "mean": num_df.mean().values,
            "std": num_df.std(ddof=1).values,
            "min": num_df.min().values,
            "max": num_df.max().values,
            "skew": num_df.skew(numeric_only=True).values,
            "kurtosis": num_df.kurt(numeric_only=True).values,
            "n_unique": num_df.nunique(dropna=True).values,
            "zero_pct": (num_df.eq(0).mean() * 100).values,
        }
    )
    out = out.merge(q.reset_index().rename(columns={"index": "feature"}), on="feature", how="left")
    out["abs_skew"] = out["skew"].abs()
    return out.sort_values("feature").reset_index(drop=True)


def compute_vif(num_df: pd.DataFrame, max_features: int = MAX_VIF_FEATURES) -> pd.DataFrame:
    if not HAS_VIF or num_df.empty:
        return pd.DataFrame()
    keep = feature_columns(num_df)
    if not keep:
        return pd.DataFrame()
    ranked = num_df[keep].var().sort_values(ascending=False).index.tolist()[:max_features]
    x = prepare_numeric_matrix(num_df[ranked])
    rows = []
    for i, c in enumerate(x.columns):
        try:
            vif = float(variance_inflation_factor(x.values, i))
        except Exception:
            vif = np.nan
        rows.append({"feature": c, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def outlier_summary(num_df: pd.DataFrame) -> pd.DataFrame:
    cols = continuous_feature_columns(num_df)
    if not cols:
        return pd.DataFrame()

    work = num_df[cols].replace([np.inf, -np.inf], np.nan)
    q1 = work.quantile(0.25)
    q3 = work.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_mask = work.lt(lower, axis=1) | work.gt(upper, axis=1)

    mean = work.mean()
    std = work.std(ddof=0).replace(0, np.nan)
    z_mask = work.sub(mean, axis=1).div(std, axis=1).abs().gt(3.0)

    count = work.count()
    out = pd.DataFrame(
        {
            "feature": cols,
            "count": count.reindex(cols).values,
            "q1": q1.reindex(cols).values,
            "q3": q3.reindex(cols).values,
            "iqr": iqr.reindex(cols).values,
            "iqr_lower": lower.reindex(cols).values,
            "iqr_upper": upper.reindex(cols).values,
            "iqr_outlier_count": iqr_mask.sum().reindex(cols).values,
            "zscore_outlier_count": z_mask.sum().reindex(cols).values,
            "mean": mean.reindex(cols).values,
            "std": std.reindex(cols).values,
            "skew": work.skew(numeric_only=True).reindex(cols).values,
        }
    )
    out["iqr_outlier_pct"] = out["iqr_outlier_count"] / out["count"].clip(lower=1) * 100.0
    out["zscore_outlier_pct"] = out["zscore_outlier_count"] / out["count"].clip(lower=1) * 100.0
    out["abs_skew"] = out["skew"].abs()
    out["outlier_priority"] = (
        out["iqr_outlier_pct"].fillna(0.0)
        + out["zscore_outlier_pct"].fillna(0.0)
        + out["abs_skew"].fillna(0.0) * 10.0
    )
    return out.sort_values(
        ["outlier_priority", "iqr_outlier_pct", "zscore_outlier_pct", "abs_skew"],
        ascending=False,
    ).reset_index(drop=True)


def column_profile_summary(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = max(len(df), 1)
    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        if pd.api.types.is_bool_dtype(s):
            logical_type = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            logical_type = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            logical_type = "numeric"
        else:
            logical_type = "categorical"

        n_unique = int(s.nunique(dropna=True))
        missing_pct = float(s.isna().mean() * 100.0)
        cardinality_ratio = float(n_unique / n_rows)

        if logical_type == "numeric":
            encoding = "scale_or_transform"
        elif logical_type == "boolean":
            encoding = "binary_ready"
        elif logical_type == "datetime":
            encoding = "extract_time_parts_or_cyclical_encode"
        elif col == "user_id" or cardinality_ratio > 0.20 or n_unique > 50:
            encoding = "hash_or_frequency_encode"
        elif n_unique <= 2:
            encoding = "binary_encode"
        elif n_unique <= 12:
            encoding = "one_hot_encode"
        else:
            encoding = "count_or_target_encode"

        if logical_type in {"categorical", "boolean"} or n_unique <= 20:
            top_values = s.astype("string").fillna("<NA>").value_counts().head(3).index.tolist()
        else:
            top_values = []
        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "logical_type": logical_type,
                "n_unique": n_unique,
                "missing_pct": missing_pct,
                "cardinality_ratio": cardinality_ratio,
                "encoding_readiness": encoding,
                "top_values": " | ".join(top_values),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["logical_type", "n_unique"], ascending=[True, False]).reset_index(drop=True)


def plot_distribution_diagnostics(
    num_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    plots_dir: Path,
    logger: logging.Logger,
) -> list[str]:
    if not HAS_PLOT or outlier_df.empty:
        return []

    features = outlier_df["feature"].head(MAX_DISTRIBUTION_FEATURES).tolist()
    if not features:
        return []

    plot_df = sample_dataframe(num_df[features], max_rows=MAX_DIST_ROWS)
    n = len(features)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig_hist, axes_hist = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
    axes_hist = np.atleast_1d(axes_hist).reshape(-1)
    for ax, feature in zip(axes_hist, features):
        series = plot_df[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            ax.set_visible(False)
            continue
        bins = min(40, max(10, int(np.sqrt(len(series)))))
        if HAS_SNS:
            sns.histplot(series, bins=bins, kde=series.nunique() > 20, color="#3a6ea5", ax=ax)
        else:
            ax.hist(series, bins=bins, color="#3a6ea5", alpha=0.85)
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("count")
    for ax in axes_hist[n:]:
        ax.set_visible(False)
    fig_hist.suptitle("Distributions numeriques les plus atypiques", fontsize=16, fontweight="bold")
    fig_hist.tight_layout()
    fig_hist.savefig(plots_dir / "distribution_histograms_top_features.png", dpi=220, bbox_inches="tight")
    plt.close(fig_hist)

    fig_box, axes_box = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.6 * nrows))
    axes_box = np.atleast_1d(axes_box).reshape(-1)
    for ax, feature in zip(axes_box, features):
        series = plot_df[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            ax.set_visible(False)
            continue
        if HAS_SNS:
            sns.boxplot(x=series, ax=ax, color="#e09f3e", orient="h")
        else:
            ax.boxplot(series, vert=False)
        ax.set_title(feature)
        ax.set_xlabel(feature)
    for ax in axes_box[n:]:
        ax.set_visible(False)
    fig_box.suptitle("Boxplots numeriques les plus atypiques", fontsize=16, fontweight="bold")
    fig_box.tight_layout()
    fig_box.savefig(plots_dir / "distribution_boxplots_top_features.png", dpi=220, bbox_inches="tight")
    plt.close(fig_box)

    logger.info("Distribution diagnostics saved for features: %s", ", ".join(features))
    return features


def temporal_analysis_outputs(
    df: pd.DataFrame,
    num_df: pd.DataFrame,
    target_df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "window" not in df.columns or df["window"].notna().sum() == 0:
        return pd.DataFrame(), pd.DataFrame()

    features = merge_ranked_features(num_df, [target_df], limit=MAX_TIME_FEATURES)
    if not features:
        return pd.DataFrame(), pd.DataFrame()

    work_cols = ["window"] + features
    if "is_insider" in df.columns:
        work_cols.append("is_insider")
    work = df[work_cols].copy()
    work = work.dropna(subset=["window"]).sort_values("window")
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    time_index = work.set_index("window")
    value_cols = features + (["is_insider"] if "is_insider" in work.columns else [])
    if time_index.index.nunique() > 500:
        temporal = time_index[value_cols].resample("D").mean()
        counts = time_index.resample("D").size().rename("row_count")
        grain = "day"
    else:
        temporal = time_index[value_cols].groupby(level=0).mean()
        counts = time_index.groupby(level=0).size().rename("row_count")
        grain = "window"

    temporal = temporal.join(counts, how="left")
    if "is_insider" in temporal.columns:
        temporal = temporal.rename(columns={"is_insider": "is_insider_rate"})
    temporal = temporal.reset_index()
    temporal.to_csv(out_dir / "temporal_feature_means.csv", index=False)
    logger.info("Temporal means saved using aggregation grain=%s", grain)

    drift_source = temporal.set_index("window")[features]
    n_steps = len(drift_source)
    if n_steps >= 4:
        span = max(1, n_steps // 4)
        early = drift_source.iloc[:span]
        late = drift_source.iloc[-span:]
        base_std = drift_source.std(ddof=0).replace(0, np.nan)
        rows = []
        for feature in features:
            early_mean = float(early[feature].mean())
            late_mean = float(late[feature].mean())
            delta = late_mean - early_mean
            drift_std = float(delta / base_std.get(feature)) if pd.notna(base_std.get(feature)) else np.nan
            pct_change = float(delta / (abs(early_mean) + 1e-9) * 100.0)
            rows.append(
                {
                    "feature": feature,
                    "early_mean": early_mean,
                    "late_mean": late_mean,
                    "mean_delta": delta,
                    "pct_change_vs_early": pct_change,
                    "drift_std_score": drift_std,
                }
            )
        drift_df = pd.DataFrame(rows)
        drift_df["abs_drift_std_score"] = drift_df["drift_std_score"].abs()
        drift_df = drift_df.sort_values("abs_drift_std_score", ascending=False).reset_index(drop=True)
        drift_df.to_csv(out_dir / "temporal_drift_summary.csv", index=False)
    else:
        drift_df = pd.DataFrame()

    seasonality_df = work[["window"]].copy()
    seasonality_df["hour_of_day"] = seasonality_df["window"].dt.hour
    seasonality_df["day_of_week"] = pd.Categorical(
        seasonality_df["window"].dt.day_name().str[:3],
        categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ordered=True,
    )
    if "is_insider" in work.columns:
        seasonality_df["heat_value"] = work["is_insider"].to_numpy()
    else:
        seasonality_df["heat_value"] = work[features[0]].to_numpy()
    pivot = seasonality_df.pivot_table(
        index="day_of_week",
        columns="hour_of_day",
        values="heat_value",
        aggfunc="mean",
    )
    if not pivot.empty:
        pivot.to_csv(out_dir / "temporal_seasonality_heatmap_values.csv")

    if HAS_PLOT:
        plot_temporal_lines(temporal, features, plots_dir, logger)
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(12, 4.8))
            if HAS_SNS:
                sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
            else:
                ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index.astype(str))
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns.astype(str))
            ax.set_title("Saisonnalite temporelle", fontweight="bold")
            ax.set_xlabel("hour_of_day")
            ax.set_ylabel("day_of_week")
            fig.tight_layout()
            fig.savefig(plots_dir / "temporal_seasonality_heatmap.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    return temporal, drift_df


def plot_temporal_lines(
    temporal_df: pd.DataFrame,
    features: Sequence[str],
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    if not HAS_PLOT or temporal_df.empty:
        return

    x = temporal_df["window"]
    plot_cols = list(features)
    if "is_insider_rate" in temporal_df.columns:
        plot_cols = ["is_insider_rate"] + plot_cols

    n = len(plot_cols)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14, 3.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    palette = ["#d1495b", "#3a6ea5", "#4c956c", "#ff9f1c", "#6c757d", "#577590", "#1d3557"]
    for idx, col in enumerate(plot_cols):
        ax = axes[idx]
        ax.plot(x, temporal_df[col], color=palette[idx % len(palette)], linewidth=1.6)
        ax.set_title(col)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("window")
    fig.suptitle("Evolution temporelle des features UEBA", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "temporal_feature_trends.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info("Temporal trend plot saved")


def class_comparison_outputs(
    df: pd.DataFrame,
    num_df: pd.DataFrame,
    target_df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    target_col = "is_insider"
    if target_col not in df.columns or target_col not in num_df.columns:
        return pd.DataFrame()
    if num_df[target_col].nunique(dropna=True) < 2:
        return pd.DataFrame()

    cols = feature_columns(num_df)
    if not cols:
        return pd.DataFrame()

    mask_normal = num_df[target_col] == 0
    mask_insider = num_df[target_col] == 1
    if int(mask_normal.sum()) == 0 or int(mask_insider.sum()) == 0:
        return pd.DataFrame()

    normal = num_df.loc[mask_normal, cols]
    insider = num_df.loc[mask_insider, cols]
    mean_normal = normal.mean()
    mean_insider = insider.mean()
    median_normal = normal.median()
    median_insider = insider.median()
    pooled_std = np.sqrt((normal.var(ddof=1) + insider.var(ddof=1)) / 2.0).replace(0, np.nan)
    cohen_d = (mean_insider - mean_normal).div(pooled_std)

    out = pd.DataFrame(
        {
            "feature": cols,
            "normal_mean": mean_normal.reindex(cols).values,
            "insider_mean": mean_insider.reindex(cols).values,
            "mean_gap": (mean_insider - mean_normal).reindex(cols).values,
            "normal_median": median_normal.reindex(cols).values,
            "insider_median": median_insider.reindex(cols).values,
            "median_gap": (median_insider - median_normal).reindex(cols).values,
            "cohen_d": cohen_d.reindex(cols).values,
        }
    )
    out["abs_cohen_d"] = out["cohen_d"].abs()
    out = out.sort_values("abs_cohen_d", ascending=False).reset_index(drop=True)
    out.to_csv(out_dir / "class_comparison_summary.csv", index=False)

    if HAS_PLOT:
        ranked_for_plots = merge_ranked_features(
            num_df,
            [out, target_df],
            limit=MAX_CLASS_FEATURES,
            continuous_only=True,
        )
        plot_classwise_distributions(df, ranked_for_plots, plots_dir, logger)

    return out


def plot_classwise_distributions(
    df: pd.DataFrame,
    features: Sequence[str],
    plots_dir: Path,
    logger: logging.Logger,
    target_col: str = "is_insider",
) -> None:
    if not HAS_PLOT or target_col not in df.columns or not features:
        return

    cols = [target_col] + list(features)
    plot_df = df[cols].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col])
    plot_df = sample_per_class(plot_df, target_col=target_col, max_rows_per_class=MAX_CLASS_ROWS_PER_CLASS)
    plot_df["_class_label"] = target_label_series(plot_df[target_col])

    n = len(features)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig_box, axes_box = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
    axes_box = np.atleast_1d(axes_box).reshape(-1)
    for ax, feature in zip(axes_box, features):
        cur = plot_df[[feature, "_class_label"]].dropna()
        if cur.empty:
            ax.set_visible(False)
            continue
        if HAS_SNS:
            sns.boxplot(data=cur, x="_class_label", y=feature, ax=ax, palette=["#3a6ea5", "#d1495b"])
        else:
            groups = [cur.loc[cur["_class_label"] == label, feature] for label in ["normal", "insider"]]
            ax.boxplot(groups, labels=["normal", "insider"])
        ax.set_title(feature)
        ax.set_xlabel("class")
        ax.grid(alpha=0.2)
    for ax in axes_box[n:]:
        ax.set_visible(False)
    fig_box.suptitle("Comparaison par classe - boxplots", fontsize=16, fontweight="bold")
    fig_box.tight_layout()
    fig_box.savefig(plots_dir / "class_boxplots_top_features.png", dpi=220, bbox_inches="tight")
    plt.close(fig_box)

    fig_hist, axes_hist = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
    axes_hist = np.atleast_1d(axes_hist).reshape(-1)
    for idx, (ax, feature) in enumerate(zip(axes_hist, features)):
        cur = plot_df[[feature, "_class_label"]].dropna()
        if cur.empty:
            ax.set_visible(False)
            continue
        if HAS_SNS:
            sns.histplot(
                data=cur,
                x=feature,
                hue="_class_label",
                common_norm=False,
                stat="density",
                bins=30,
                element="step",
                fill=False,
                ax=ax,
            )
            if idx > 0 and ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            for label, color in [("normal", "#3a6ea5"), ("insider", "#d1495b")]:
                series = cur.loc[cur["_class_label"] == label, feature]
                ax.hist(series, bins=30, alpha=0.45, density=True, label=label, color=color)
            if idx == 0:
                ax.legend()
        ax.set_title(feature)
        ax.grid(alpha=0.2)
    for ax in axes_hist[n:]:
        ax.set_visible(False)
    fig_hist.suptitle("Comparaison par classe - distributions", fontsize=16, fontweight="bold")
    fig_hist.tight_layout()
    fig_hist.savefig(plots_dir / "class_distributions_top_features.png", dpi=220, bbox_inches="tight")
    plt.close(fig_hist)
    logger.info("Class-wise distribution plots saved")


def compute_mutual_information(
    num_df: pd.DataFrame,
    target_col: str = "is_insider",
    max_features: int = MAX_PROXY_IMPORTANCE_FEATURES,
    max_rows: int = MAX_PROXY_IMPORTANCE_ROWS,
) -> pd.DataFrame:
    if not HAS_SKLEARN or target_col not in num_df.columns:
        return pd.DataFrame()
    if num_df[target_col].nunique(dropna=True) < 2:
        return pd.DataFrame()

    ranked = variance_ranked_features(num_df, cols=feature_columns(num_df))[:max_features]
    if not ranked:
        return pd.DataFrame()

    sample = sample_dataframe(num_df[ranked + [target_col]], max_rows=max_rows, stratify_col=target_col)
    x = prepare_numeric_matrix(sample[ranked])
    y = sample[target_col].fillna(0).astype(int)
    discrete_mask = [sample[c].nunique(dropna=True) <= 10 for c in ranked]
    mi = mutual_info_classif(x, y, discrete_features=discrete_mask, random_state=RANDOM_STATE)
    out = pd.DataFrame({"feature": ranked, "mutual_information": mi})
    return out.sort_values("mutual_information", ascending=False).reset_index(drop=True)


def compute_random_forest_importance(
    num_df: pd.DataFrame,
    target_col: str = "is_insider",
    max_features: int = MAX_PROXY_IMPORTANCE_FEATURES,
    max_rows: int = MAX_PROXY_IMPORTANCE_ROWS,
) -> pd.DataFrame:
    if not HAS_SKLEARN or target_col not in num_df.columns:
        return pd.DataFrame()
    if num_df[target_col].nunique(dropna=True) < 2:
        return pd.DataFrame()

    ranked = variance_ranked_features(num_df, cols=feature_columns(num_df))[:max_features]
    if not ranked:
        return pd.DataFrame()

    sample = sample_dataframe(num_df[ranked + [target_col]], max_rows=max_rows, stratify_col=target_col)
    x = prepare_numeric_matrix(sample[ranked])
    y = sample[target_col].fillna(0).astype(int)

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x, y)
    out = pd.DataFrame({"feature": ranked, "rf_importance": model.feature_importances_})
    return out.sort_values("rf_importance", ascending=False).reset_index(drop=True)


def multivariate_outputs(
    df: pd.DataFrame,
    num_df: pd.DataFrame,
    target_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not HAS_PLOT:
        return pd.DataFrame()

    ranked = merge_ranked_features(
        num_df,
        [rf_df, mi_df, target_df],
        limit=max(MAX_PCA_FEATURES, MAX_PAIRPLOT_FEATURES),
        continuous_only=True,
    )
    if len(ranked) < 2:
        return pd.DataFrame()

    plot_pairplot(df, ranked[:MAX_PAIRPLOT_FEATURES], plots_dir, logger)

    if not HAS_SKLEARN:
        return pd.DataFrame()

    pca_features = ranked[:MAX_PCA_FEATURES]
    cols = pca_features + (["is_insider"] if "is_insider" in num_df.columns else [])
    sample = sample_dataframe(num_df[cols], max_rows=MAX_PCA_ROWS, stratify_col="is_insider" if "is_insider" in cols else None)
    if len(sample) < 2:
        return pd.DataFrame()
    x = prepare_numeric_matrix(sample[pca_features])
    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame({"pc1": components[:, 0], "pc2": components[:, 1]}, index=sample.index)
    pca_df["explained_variance_ratio_pc1"] = float(pca.explained_variance_ratio_[0])
    pca_df["explained_variance_ratio_pc2"] = float(pca.explained_variance_ratio_[1])

    n_clusters = min(KMEANS_CLUSTERS, max(2, len(pca_df) // 50))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    pca_df["cluster"] = kmeans.fit_predict(components)

    if "is_insider" in sample.columns:
        pca_df["is_insider"] = sample["is_insider"].astype(int).to_numpy()
        pca_df["class_label"] = target_label_series(sample["is_insider"]).to_numpy()
        cluster_target = pd.crosstab(pca_df["cluster"], pca_df["class_label"], normalize="index")
        cluster_target.to_csv(out_dir / "cluster_vs_target_summary.csv")
    pca_df.to_csv(out_dir / "pca_projection.csv", index=False)

    plot_pca_projection(pca_df, plots_dir, logger)
    return pca_df


def plot_pairplot(
    df: pd.DataFrame,
    features: Sequence[str],
    plots_dir: Path,
    logger: logging.Logger,
) -> None:
    if not HAS_SNS or len(features) < 2:
        return

    cols = list(features)
    hue = None
    plot_df = df[cols + (["is_insider"] if "is_insider" in df.columns else [])].copy()
    if "is_insider" in plot_df.columns:
        hue = "class_label"
        plot_df["class_label"] = target_label_series(plot_df["is_insider"])
        plot_df = plot_df.drop(columns=["is_insider"])
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        return
    plot_df = sample_dataframe(plot_df, max_rows=MAX_PAIRPLOT_ROWS, stratify_col=hue)
    pair = sns.pairplot(
        plot_df,
        vars=cols,
        hue=hue,
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.65, "s": 18, "edgecolor": "none"},
    )
    pair.fig.suptitle("Pairplot echantillonne des features discriminantes", y=1.02, fontweight="bold")
    pair.savefig(plots_dir / "pairplot_top_features.png", dpi=180, bbox_inches="tight")
    plt.close(pair.fig)
    logger.info("Pairplot saved")


def plot_pca_projection(pca_df: pd.DataFrame, plots_dir: Path, logger: logging.Logger) -> None:
    if not HAS_PLOT or pca_df.empty:
        return

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5))
    if "class_label" in pca_df.columns and HAS_SNS:
        sns.scatterplot(
            data=pca_df,
            x="pc1",
            y="pc2",
            hue="class_label",
            palette={"normal": "#3a6ea5", "insider": "#d1495b", "unknown": "#6c757d"},
            alpha=0.7,
            s=28,
            ax=axes[0],
        )
        axes[0].set_title("Projection PCA coloree par classe")
    else:
        axes[0].scatter(pca_df["pc1"], pca_df["pc2"], alpha=0.65, s=18, color="#3a6ea5")
        axes[0].set_title("Projection PCA")

    if HAS_SNS:
        sns.scatterplot(
            data=pca_df,
            x="pc1",
            y="pc2",
            hue="cluster",
            palette="tab10",
            alpha=0.7,
            s=28,
            ax=axes[1],
        )
    else:
        axes[1].scatter(pca_df["pc1"], pca_df["pc2"], c=pca_df["cluster"], cmap="tab10", alpha=0.65, s=18)
    axes[1].set_title("Projection PCA coloree par cluster")

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlabel("pc1")
        ax.set_ylabel("pc2")
    fig.suptitle("Visualisations multivariees echantillonnees", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "pca_and_clusters.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info("PCA and clustering visual saved")


def write_summary(
    out_file: Path,
    files: list[Path],
    df: pd.DataFrame,
    dup_count: int,
    miss_df: pd.DataFrame,
    target_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    vif_df: pd.DataFrame,
    class_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    col_profile_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("FEATURES_V5_EDA_SUMMARY")
    lines.append(f"files_count: {len(files)}")
    lines.append(f"rows_total: {len(df)}")
    lines.append(f"cols_total: {len(df.columns)}")
    if "user_id" in df.columns:
        lines.append(f"users: {df['user_id'].nunique()}")
    if "window" in df.columns and df["window"].notna().any():
        lines.append(f"window_min: {df['window'].min()} window_max: {df['window'].max()}")
    if "is_insider" in df.columns:
        insider_count = int(pd.to_numeric(df["is_insider"], errors="coerce").fillna(0).sum())
        lines.append(f"is_insider_count: {insider_count}")
        lines.append(f"is_insider_rate_pct: {float(pd.to_numeric(df['is_insider'], errors='coerce').fillna(0).mean() * 100.0):.6f}")
    lines.append(f"duplicate_user_window: {dup_count}")
    lines.append(f"plotting_enabled: {HAS_PLOT}")
    lines.append(f"seaborn_enabled: {HAS_SNS}")
    lines.append(f"vif_enabled: {HAS_VIF}")
    lines.append(f"sklearn_enabled: {HAS_SKLEARN}")
    lines.append("")

    lines.append("MISSING_TOP_15")
    for _, row in miss_df.head(15).iterrows():
        lines.append(f"{row['feature']}: {row['missing_pct']:.4f}%")
    lines.append("")

    if not target_df.empty:
        lines.append("TOP_CORR_WITH_IS_INSIDER_10")
        for _, row in target_df.head(10).iterrows():
            lines.append(f"{row['feature']}: corr={row['corr_with_is_insider']:.6f}")
        lines.append("")

    if not outlier_df.empty:
        lines.append("TOP_OUTLIERS_10")
        for _, row in outlier_df.head(10).iterrows():
            lines.append(
                f"{row['feature']}: iqr_pct={row['iqr_outlier_pct']:.4f} zscore_pct={row['zscore_outlier_pct']:.4f} abs_skew={row['abs_skew']:.4f}"
            )
        lines.append("")

    if not vif_df.empty:
        lines.append("TOP_VIF_10")
        for _, row in vif_df.head(10).iterrows():
            lines.append(f"{row['feature']}: vif={row['vif']:.4f}")
        lines.append("")

    if not class_df.empty:
        lines.append("TOP_CLASS_GAPS_10")
        for _, row in class_df.head(10).iterrows():
            lines.append(f"{row['feature']}: cohen_d={row['cohen_d']:.6f} mean_gap={row['mean_gap']:.6f}")
        lines.append("")

    if not drift_df.empty:
        lines.append("TOP_TEMPORAL_DRIFT_10")
        for _, row in drift_df.head(10).iterrows():
            lines.append(
                f"{row['feature']}: drift_std={row['drift_std_score']:.6f} pct_change={row['pct_change_vs_early']:.4f}"
            )
        lines.append("")

    if not mi_df.empty:
        lines.append("TOP_MUTUAL_INFORMATION_10")
        for _, row in mi_df.head(10).iterrows():
            lines.append(f"{row['feature']}: mi={row['mutual_information']:.6f}")
        lines.append("")

    if not rf_df.empty:
        lines.append("TOP_RANDOM_FOREST_IMPORTANCE_10")
        for _, row in rf_df.head(10).iterrows():
            lines.append(f"{row['feature']}: importance={row['rf_importance']:.6f}")
        lines.append("")

    cat_high = col_profile_df[
        (col_profile_df["logical_type"] == "categorical")
        & ((col_profile_df["n_unique"] > 50) | (col_profile_df["cardinality_ratio"] > 0.20))
    ]
    if not cat_high.empty:
        lines.append("HIGH_CARDINALITY_CATEGORICALS_10")
        for _, row in cat_high.head(10).iterrows():
            lines.append(
                f"{row['column']}: n_unique={int(row['n_unique'])} ratio={row['cardinality_ratio']:.6f} encode={row['encoding_readiness']}"
            )
        lines.append("")

    lines.append("NOTES")
    lines.append("- Inspect features with heavy tails or strong skew before assuming Gaussian behavior.")
    lines.append("- Investigate features with high IQR/Z-score outlier rates; they often carry UEBA anomaly signal.")
    lines.append("- Review VIF > 10 and correlation pairs with |corr| > 0.95.")
    lines.append("- Compare early vs late windows to catch temporal drift before training.")
    lines.append("- Prefer sampled plots and top-ranked features to keep EDA resource usage stable.")

    out_file.write_text("\n".join(lines), encoding="utf-8")


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    features_dir = Path(cfg["output"]["features_dir"])
    reports_dir = Path(cfg["output"]["reports_dir"])
    out_dir, plots_dir = ensure_dirs(reports_dir)
    logger = get_logger(out_dir / "logs")
    
    logger.info("=" * 72)
    logger.info("EDA FEATURES V5 - START")
    logger.info("=" * 72)
    logger.info("features_dir=%s", features_dir)
    logger.info("reports_dir=%s", reports_dir)
    logger.info(
        "plotting=%s | seaborn=%s | vif=%s | sklearn=%s",
        HAS_PLOT,
        HAS_SNS,
        HAS_VIF,
        HAS_SKLEARN,
    )

    files = list_feature_files(features_dir)
    logger.info("feature_files_detected=%d", len(files))
    
    if not files:
        logger.warning("No feature files found in %s/slice_*/features.parquet", features_dir)
        return

    # Load data
    logger.info("Loading features...")
    df = load_features_df(files)
    logger.info("loaded_rows=%d | loaded_cols=%d", len(df), len(df.columns))
    
    num_df = safe_numeric_df(df)
    dup_count = duplicate_key_count(df)
    logger.info("numeric_cols=%d | duplicate_user_window=%d", len(num_df.columns), dup_count)
    
    miss_df = missing_summary(df)
    desc_df = descriptive_stats(num_df)
    col_profile_df = column_profile_summary(df)
    miss_df.to_csv(out_dir / "missing_summary.csv", index=False)
    if not desc_df.empty:
        desc_df.to_csv(out_dir / "descriptive_stats.csv", index=False)
    col_profile_df.to_csv(out_dir / "column_type_cardinality_summary.csv", index=False)
    
    # ✅ CORRELATION MATRICES (OPTIMIZED)
    top_corr = correlation_outputs_optimized(num_df, out_dir, plots_dir, logger)
    _ = top_corr

    target_df = target_correlation(num_df, target_col="is_insider")
    if not target_df.empty:
        target_df.to_csv(out_dir / "target_correlation_is_insider.csv", index=False)

        logger.info("Top 10 features correlated with is_insider:")
        for _, row in target_df.head(10).iterrows():
            logger.info("  %-40s %+0.4f", row["feature"], row["corr_with_is_insider"])

    outlier_df = outlier_summary(num_df)
    if not outlier_df.empty:
        outlier_df.to_csv(out_dir / "outlier_summary_iqr_zscore.csv", index=False)
        plot_distribution_diagnostics(num_df, outlier_df, plots_dir, logger)

    vif_df = compute_vif(num_df, max_features=MAX_VIF_FEATURES)
    if not vif_df.empty:
        vif_df.to_csv(out_dir / "vif_report.csv", index=False)
        logger.info("VIF report saved with %d features", len(vif_df))

    temporal_df, drift_df = temporal_analysis_outputs(df, num_df, target_df, out_dir, plots_dir, logger)
    _ = temporal_df

    class_df = class_comparison_outputs(df, num_df, target_df, out_dir, plots_dir, logger)

    mi_df = compute_mutual_information(num_df, target_col="is_insider")
    if not mi_df.empty:
        mi_df.to_csv(out_dir / "feature_importance_mutual_information.csv", index=False)

    rf_df = compute_random_forest_importance(num_df, target_col="is_insider")
    if not rf_df.empty:
        rf_df.to_csv(out_dir / "feature_importance_random_forest.csv", index=False)

    _ = multivariate_outputs(df, num_df, target_df, mi_df, rf_df, out_dir, plots_dir, logger)

    summary_file = out_dir / "features_v5_summary.txt"
    write_summary(
        summary_file,
        files,
        df,
        dup_count,
        miss_df,
        target_df,
        outlier_df,
        vif_df,
        class_df,
        drift_df,
        mi_df,
        rf_df,
        col_profile_df,
    )
    logger.info("Summary written: %s", summary_file)
    
    elapsed = time.perf_counter() - t0
    logger.info("Outputs: %s", out_dir)
    logger.info("Plots:   %s", plots_dir)
    logger.info("EDA FEATURES V5 - DONE | total_elapsed=%.2fs", elapsed)
    logger.info("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("EDAFeaturesV5").exception("EDA FEATURES V5 - FAILED")
        raise
df = pd.read_csv("datasets/cert4-2/logon.csv")
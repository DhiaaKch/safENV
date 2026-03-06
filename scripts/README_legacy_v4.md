# Legacy V4 Bundle - Documentation Complete

Ce README decrit le bundle **pipeline v4 (baseline F1 ~0.48)** sur la branche
`legacy-pipelinev4-f1-048`.

## Scripts supprimes sur cette branche

Les scripts suivants ont ete supprimes comme demande:

- `scripts/splicer.py`
- `scripts/splicer_ldap.py`
- `scripts/diagnostic.py`
- `scripts/cleaner.py`
- `scripts/cleaner_ldap.py`
- `scripts/features_engineering.py`

## Remplacements / equivalents utilises

- `splicer.py` -> `slicer.py`
- `cleaner.py` -> `cleaner_v4.py`
- `features_engineering.py` -> `feature_engineering_v4.py`
- `splicer_ldap.py` + `cleaner_ldap.py` -> `ldap_loader.py`
- `diagnostic.py` -> `eda_featuresv4.py` (analyse exploratoire des features)

## Description complete des scripts actifs

### 1) `scripts/slicer.py`
- Role: decouper les CSV bruts CERT en slices temporelles (`slice_001`, `slice_002`, ...).
- Points clefs: support gros fichiers (ex. `http.csv`) via traitement par chunks, conversion date robuste.
- Input: `data/00_raw/r4.2/*.csv`
- Output: `data/01_slices/slice_NNN/{source}.parquet` + `meta.parquet`
- Execution: `python scripts/slicer.py`

### 2) `scripts/cleaner_v4.py`
- Role: nettoyer les parquets de chaque slice (normalisation et qualite de donnees par source).
- Points clefs: SQL DuckDB robuste, filtrage temporel via `meta.parquet`, enrichissements de colonnes utiles.
- Input: `data/01_slices/slice_NNN/{source}.parquet`
- Output: memes fichiers ecrases en version nettoyee
- Execution: `python scripts/cleaner_v4.py`

### 3) `scripts/label_joiner.py`
- Role: joindre les labels insiders (is_insider/scenario) aux users presents dans chaque slice.
- Logique: chevauchement fenetre malveillante (`start/end`) avec fenetre temporelle de la slice.
- Input:
  - `cfg.data.answers_file` (ex: `answers.csv` / insiders)
  - `data/01_slices/slice_NNN/*.parquet`
- Output: `data/03_features/labels/slice_NNN/labels.parquet`
- Execution: `python scripts/label_joiner.py`

### 4) `scripts/ldap_loader.py`
- Role: construire les snapshots LDAP par slice et enrichir les metadonnees RH.
- Points clefs: gestion des snapshots mensuels, detection changements org (`department/role/supervisor`), `tenure_days`.
- Input: `cfg.data.ldap_folder` (CSV LDAP mensuels)
- Output: `data/03_features/ldap/slice_NNN/ldap.parquet`
- Execution: `python scripts/ldap_loader.py`

### 5) `scripts/feature_engineering_v4.py`
- Role: generer les features UEBA v4 (40+ features) a partir des slices nettoyees.
- Integre:
  - features comportementales multi-sources (logon/device/file/http/email)
  - labels (`label_joiner.py`)
  - enrichment LDAP (`ldap_loader.py`)
- Input:
  - `data/01_slices/slice_NNN/*.parquet`
  - `data/03_features/labels/slice_NNN/labels.parquet`
  - `data/03_features/ldap/slice_NNN/ldap.parquet`
- Output: `data/03_features/slice_NNN/features.parquet`
- Execution:
  - `python scripts/feature_engineering_v4.py`
  - `python scripts/feature_engineering_v4.py --slice 1`

### 6) `scripts/eda_featuresv4.py`
- Role: EDA complet sur les features v4 avant/modelisation.
- Produit:
  - checks schema/qualite
  - stats descriptives
  - correlations Pearson/Spearman
  - correlation cible `is_insider`
  - VIF (si `statsmodels` present)
  - rapport texte + CSV + plots
- Input: `data/03_features/slice_*/features.parquet`
- Output: `reports/features_v4_eda/` (tables, logs, plots, summary)
- Execution: `python scripts/eda_featuresv4.py`

### 7) `scripts/pipelinev4.py`
- Role: pipeline ML baseline v4 (LightGBM) avec selection top features.
- Etapes:
  - filtre correlation
  - reequilibrage (SMOTE ou `--no-smote`)
  - CV stratifiee
  - selection top N
  - evaluation finale + plots
- Input: `data/03_features/slice_*/features.parquet`
- Output:
  - `data/04_model/model.pkl`
  - `data/04_model/selected_features.json`
  - `data/04_model/evaluation_report.json`
  - `data/04_model/plots/*`
- Execution:
  - `python scripts/pipelinev4.py`
  - `python scripts/pipelinev4.py --no-smote`

## Ordre d'execution recommande

1. `python scripts/slicer.py`
2. `python scripts/cleaner_v4.py`
3. `python scripts/label_joiner.py`
4. `python scripts/ldap_loader.py`
5. `python scripts/feature_engineering_v4.py`
6. `python scripts/eda_featuresv4.py`
7. `python scripts/pipelinev4.py`

## Notes

- Les chemins reels sont pilotes par `configs/config.yaml`.
- Le pipeline v4 reste une baseline historique; les versions plus recentes peuvent etre plus performantes selon le dataset.

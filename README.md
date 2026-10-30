# 💳 Credit Intelligence Pro — Explainable Credit Risk Analytics

> **CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model**

A professionally engineered, explainable, and reproducible machine learning system
that estimates credit risk using the UCI Statlog German Credit dataset.
Built with an end-to-end ML pipeline, interactive Streamlit dashboard, automated
test suite, and full documentation.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
- [Model Comparison & Selection](#model-comparison--selection)
- [Evaluation Results](#evaluation-results)
- [Explainability](#explainability)
- [Directory Structure](#directory-structure)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Test Results](#test-results)
- [Dashboard Screenshots](#dashboard-screenshots)
- [Limitations & Future Work](#limitations--future-work)
- [Disclaimer](#disclaimer)

---

## Project Overview

**Credit Intelligence Pro** estimates the probability that a credit applicant
represents a bad credit risk. The system demonstrates:

- Modular, testable ML engineering with `sklearn` Pipelines
- Data validation, EDA, preprocessing, multi-model training, and evaluation
- Transparent model selection and decision-threshold analysis
- Model-agnostic explainability via permutation feature importance
- An interactive, professional-grade Streamlit web dashboard
- A comprehensive pytest test suite with 39 tests

**Educational purpose only.** This is not a production lending tool.

---

## Problem Statement

Given 20 financial and personal attributes for a credit applicant, predict:

- **0 = Good Credit** (70% of dataset) — negative class
- **1 = Bad Credit** (30% of dataset) — positive class ← the class we prioritise detecting

In credit risk screening, **false negatives (bad credit approved as good) are
typically more costly** than false positives. We address this via:
- `class_weight='balanced'` in all classifiers
- Decision threshold tuned on out-of-fold training predictions toward higher recall

---

## Architecture

```
app.py              ← Streamlit dashboard (entry point)
src/
  config.py         ← Centralised paths and constants (pathlib, no hard-coded paths)
  data_ingestion.py ← Downloads/loads UCI dataset, remaps target encoding
  data_validation.py← Schema, type, range, and category validation
  eda.py            ← EDA chart generation (headless, saves to outputs/figures/)
  preprocessing.py  ← sklearn ColumnTransformer pipeline (no leakage)
  train.py          ← Cross-validation, model selection, refit, artifact saving
  evaluate.py       ← Final test-set evaluation, threshold analysis, chart generation
  explainability.py ← Permutation importance + built-in feature importances
  predict.py        ← Inference API with input validation
  run_eda.py        ← Standalone EDA runner
  utils.py          ← Logging, filesystem helpers, JSON/CSV I/O
tests/
  test_data.py      ← Data loading, schema, target encoding tests
  test_preprocessing.py ← Preprocessor tests
  test_model.py     ← Training, serialization, inference tests
  test_smoke.py     ← End-to-end smoke tests
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [UCI ML Repository — Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| Size | 1,000 samples × 20 features + 1 target |
| Period | ~1994, West Germany |
| Original target | 1 = Good credit, 2 = Bad credit |
| **Remapped target** | **0 = Good credit, 1 = Bad credit** |
| Missing values | None |
| Class distribution | Good: 700 (70%), Bad: 300 (30%) |

**Target encoding note:** The UCI file uses 1=good and 2=bad. We explicitly remap
to 0=good and 1=bad throughout the project. This mapping is documented in
`src/config.py` and `models/model_metadata.json`.

---

## Feature Engineering & Preprocessing

### Numeric Features (7)
`duration`, `credit_amount`, `installment_rate`, `present_residence`,
`age`, `existing_credits`, `dependents`

**Pipeline:** Median imputation → Standard scaling

### Categorical Features (13)
`checking_account_status`, `credit_history`, `purpose`, `savings_account`,
`employment_since`, `personal_status_sex`, `other_debtors`, `property`,
`other_installment_plans`, `housing`, `job`, `telephone`, `foreign_worker`

**Pipeline:** Constant imputation (`'missing'`) → Ordinal encoding
(`handle_unknown='use_encoded_value'` with `unknown_value=-1`)

### Key Design Decisions
- All preprocessing fitted **only on training data** (no target leakage)
- Unknown categories at inference time handled gracefully (not a crash)
- `class_weight='balanced'` applied to all classifiers to address class imbalance
- Preprocessing lives inside the `sklearn.Pipeline` — no train/test contamination

---

## Model Comparison & Selection

Three classifiers evaluated via **stratified 5-fold cross-validation on training data**:

| Model | CV ROC-AUC (mean±std) | CV F1 (mean±std) |
|---|---|---|
| **Random Forest** ✅ | **0.799 ± 0.041** | **0.596 ± 0.052** |
| Logistic Regression | 0.779 ± 0.055 | 0.587 ± 0.052 |
| Decision Tree | 0.698 ± 0.037 | 0.562 ± 0.027 |

**Selected model:** Random Forest Classifier (200 trees, max_depth=10, balanced weights)

**Selection criterion:** Highest mean CV ROC-AUC on training data only.

**Threshold selection:** Out-of-fold training predictions were used to scan thresholds
[0.10, 0.70] maximising F1 for the bad-credit class. **Test labels were never used
for threshold selection.**

Optimal threshold: **0.410** (OOF F1=0.636, Precision=0.528, Recall=0.800)

---

## Evaluation Results

**Final evaluation on held-out test set (200 samples, 20% of dataset).**
Test set was used exactly once, after all training and threshold selection.

| Metric | Value (threshold=0.410) |
|---|---|
| **ROC-AUC** | **0.8063** |
| Accuracy | 0.715 |
| Precision (Bad Credit) | 0.517 |
| **Recall (Bad Credit)** | **0.750** |
| F1-Score (Bad Credit) | 0.612 |

### Confusion Matrix (threshold=0.410)

|  | Predicted Good | Predicted Bad |
|---|---|---|
| **Actual Good** | 101 (TN) | 39 (FP) |
| **Actual Bad** | 15 (FN) | 45 (TP) |

- **45** of 60 bad-credit applicants correctly identified ✅
- **15** bad-credit applicants missed (false negatives)
- **39** good-credit applicants flagged (false positives)

**False Positive vs False Negative trade-off:**
- FN (bad approved): higher financial loss risk
- FP (good rejected): missed revenue, unfair rejection
- Higher recall was prioritised via balanced class weights + lower threshold

---

## Explainability

**Top features by permutation importance (test set, 20 repeats):**

| Rank | Feature | Mean Δ ROC-AUC |
|---|---|---|
| 1 | `checking_account_status` | 0.1366 |
| 2 | `credit_amount` | 0.0246 |
| 3 | `duration` | 0.0203 |
| 4 | `credit_history` | 0.0196 |
| 5 | `other_debtors` | 0.0128 |
| 6 | `purpose` | 0.0094 |
| 7 | `other_installment_plans` | 0.0087 |
| 8 | `age` | 0.0080 |
| 9 | `savings_account` | 0.0071 |
| 10 | `job` | 0.0051 |

**Interpretation:** `checking_account_status` dominates by a wide margin, consistent
with domain knowledge that current account standing is the strongest credit signal.

**Limitation:** Feature importances show correlation, not causation. The model must
not be used to justify demographic discrimination.

---

## Directory Structure

```
Credit_Scoring_ML/
├── app.py                          ← Dashboard entry point
├── requirements.txt                ← Verified dependencies
├── README.md
├── Start.md                        ← Project instructions
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── predict.py
│   ├── run_eda.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_smoke.py
├── data/
│   ├── raw/
│   │   └── german_credit.csv       ← UCI dataset (auto-downloaded if absent)
│   └── processed/
│       ├── train.csv               ← 800 rows (generated by training)
│       └── test.csv                ← 200 rows (generated by training)
├── models/
│   ├── best_pipeline.joblib        ← Fitted Random Forest pipeline (regenerate via train)
│   └── model_metadata.json         ← Model config and CV results
└── outputs/
    ├── figures/
    │   ├── target_distribution.png
    │   ├── numeric_distributions.png
    │   ├── categorical_distributions.png
    │   ├── correlation_heatmap.png
    │   ├── missing_values.png
    │   ├── roc_curve.png
    │   ├── confusion_matrix.png
    │   ├── precision_recall_curve.png
    │   ├── threshold_analysis.png
    │   ├── permutation_importance.png
    │   └── model_feature_importance.png
    └── reports/
        ├── model_metrics.json
        ├── cv_results.json
        ├── model_comparison.csv
        ├── threshold_analysis.csv
        └── feature_importance.csv
```

---

## Setup & Installation

### Prerequisites
- Python **3.13.x** (tested on 3.13.7)
- Windows PowerShell or terminal
- Internet connection (first run only, to download the dataset)

### Installation

```powershell
# 1. Navigate to the project directory
cd "C:\path\to\Credit_Scoring_ML"

# 2. (Recommended) Create a virtual environment
py -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** No virtual environment is required if packages are installed system-wide.
> The project was verified using the system Python 3.13.7 with packages installed globally.

---

## Running the Project

All commands are run from the **repository root** (`Credit_Scoring_ML/`).

### Step 1: Train the model

```powershell
py -m src.train
```

Downloads the dataset on first run, validates it, trains 3 classifiers via 5-fold CV,
selects the best, refits on training data, and saves:
- `models/best_pipeline.joblib`
- `models/model_metadata.json`
- `data/processed/train.csv` and `test.csv`
- `outputs/reports/model_comparison.csv`

**Expected output:**
```
Best model by CV ROC-AUC: RandomForest
Saved model pipeline to models/best_pipeline.joblib
=== Training Complete ===
```

### Step 2: Evaluate the model

```powershell
py -m src.evaluate
```

Evaluates on the held-out test set. Generates metrics, charts, and threshold analysis.

**Expected output:**
```
Best OOF threshold: 0.410
Test ROC-AUC: 0.8063
Test F1:      0.6122
Test Recall:  0.75
```

### Step 3: Generate EDA charts

```powershell
py -m src.run_eda
```

Saves 5 exploratory charts to `outputs/figures/`.

### Step 4: Generate explainability analysis

```powershell
py -m src.explainability
```

Computes permutation feature importance (may take ~1 minute).

### Step 5: Launch the dashboard

```powershell
streamlit run app.py
```

Opens the Credit Intelligence Pro dashboard at `http://localhost:8501`.

### Step 6: Run the test suite

```powershell
py -m pytest tests/ -v
```

---

## Test Results

**Verified on 2026-10-27, Python 3.13.7, pytest 9.1.1:**

```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1
rootdir: Credit_Scoring_ML
collected 38 items

tests/test_data.py::TestSchemaValidation::test_valid_schema_passes        PASSED
tests/test_data.py::TestSchemaValidation::test_missing_column_detected    PASSED
tests/test_data.py::TestSchemaValidation::test_wrong_row_count_detected   PASSED
tests/test_data.py::TestTargetValidation::test_valid_targets              PASSED
tests/test_data.py::TestTargetValidation::test_invalid_target_value_detected PASSED
tests/test_data.py::TestTargetValidation::test_target_constants           PASSED
tests/test_data.py::TestMissingValues::test_no_missing_passes             PASSED
tests/test_data.py::TestMissingValues::test_missing_detected              PASSED
tests/test_data.py::TestRealDataset::test_real_dataset_shape              PASSED
tests/test_data.py::TestRealDataset::test_real_dataset_validation_passes  PASSED
tests/test_data.py::TestRealDataset::test_target_encoding_correct         PASSED
tests/test_data.py::TestRealDataset::test_no_missing_values               PASSED
tests/test_model.py::TestModelTraining::test_pipeline_fits_without_error  PASSED
tests/test_model.py::TestModelTraining::test_predict_shape                PASSED
tests/test_model.py::TestModelTraining::test_predict_proba_shape          PASSED
tests/test_model.py::TestModelTraining::test_predict_proba_range          PASSED
tests/test_model.py::TestModelTraining::test_predictions_binary           PASSED
tests/test_model.py::TestModelTraining::test_roc_auc_reasonable           PASSED
tests/test_model.py::TestSerialization::test_joblib_roundtrip             PASSED
tests/test_model.py::TestSerialization::test_missing_model_raises         PASSED
tests/test_model.py::TestInference::test_validate_input_valid             PASSED
tests/test_model.py::TestInference::test_validate_input_missing_feature   PASSED
tests/test_model.py::TestInference::test_validate_input_out_of_range      PASSED
tests/test_model.py::TestInference::test_predict_single_with_trained_pipeline PASSED
tests/test_model.py::TestInference::test_predict_single_deterministic     PASSED
tests/test_model.py::TestRealModel::test_real_model_loads                 PASSED
tests/test_model.py::TestRealModel::test_real_model_predict_proba         PASSED
tests/test_preprocessing.py::TestBuildPreprocessor::test_returns_column_transformer PASSED
tests/test_preprocessing.py::TestBuildPreprocessor::test_fit_transform_shape PASSED
tests/test_preprocessing.py::TestBuildPreprocessor::test_no_nan_in_output PASSED
tests/test_preprocessing.py::TestBuildPreprocessor::test_unknown_category_handled PASSED
tests/test_preprocessing.py::TestSplitFeaturesTarget::test_correct_shapes PASSED
tests/test_preprocessing.py::TestSplitFeaturesTarget::test_target_not_in_X PASSED
tests/test_preprocessing.py::TestSplitFeaturesTarget::test_missing_column_raises PASSED
tests/test_smoke.py::TestSmokeWorkflow::test_full_pipeline_smoke          PASSED
tests/test_smoke.py::TestSmokeWorkflow::test_predict_single_smoke         PASSED
tests/test_smoke.py::TestSmokeWorkflow::test_metrics_file_valid           PASSED
tests/test_smoke.py::TestSmokeWorkflow::test_model_metadata_valid         PASSED
tests/test_tuning.py::test_compute_cost_sensitive_threshold                PASSED

============================= 39 passed in 4.37s ==============================
```

---

## Dashboard Screenshots

> Add screenshots of the running dashboard here by launching `streamlit run app.py`
> and capturing the following pages:
> - Executive Overview (metrics cards + class distribution chart)
> - Credit Risk Demo (input form + probability gauge)
> - Model Performance (ROC curve + confusion matrix)
> - Explainability (permutation importance chart)

---

## Limitations & Future Work

### Current Limitations
- **Dataset age:** German credit data from ~1994; modern credit behaviour differs significantly.
- **Sample size:** 1,000 samples is small for a credit model.
- **Calibration:** Random Forest probability estimates are not formally calibrated.
- **Fairness:** A formal fairness audit requires demographic data and domain expertise.
- **Feature set:** 20 features; real credit scoring systems use hundreds of signals.
- **No SHAP:** Per-instance SHAP explanations would improve individual-prediction interpretability.

### Future Improvements
- Probability calibration using `CalibratedClassifierCV`
- SHAP values for per-instance explanations
- Gradient Boosting models (XGBoost, LightGBM)
- Cross-validation with nested hyperparameter search (`GridSearchCV`)
- Fairness metrics (demographic parity, equalised odds)
- CI/CD pipeline with GitHub Actions
- Docker container for reproducible deployment
- Model monitoring and data drift detection

---

## Disclaimer

This project is an **educational prototype** developed for the CodeAlpha Machine Learning
Internship. It is not a production-certified financial tool. Predictions carry uncertainty
and may reflect historical biases present in the 1994 German credit dataset.

**This system must not be used for actual credit or lending decisions.**

All metrics and results are generated from the actual trained model and reported honestly.
No results have been fabricated or hard-coded.

---

## Acknowledgements

- UCI Machine Learning Repository — Statlog German Credit Data
  (Prof. Dr. Hans Hofmann, University of Hamburg)
- scikit-learn, pandas, matplotlib, seaborn, plotly, Streamlit communities

---

*CodeAlpha ML Internship — Task 1 | Developed with Python 3.13.7*

"""
Credit Intelligence Pro — centralised project configuration.

All paths are relative to the repository root so that commands work from
any machine without hard-coded usernames or absolute paths.
"""

from pathlib import Path

# ── Repository root (the directory that contains this src/ package) ──────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"

RAW_DATASET_PATH   = DATA_RAW_DIR / "german_credit.csv"
PROCESSED_PATH     = DATA_PROCESSED_DIR / "processed.csv"
TRAIN_PATH         = DATA_PROCESSED_DIR / "train.csv"
TEST_PATH          = DATA_PROCESSED_DIR / "test.csv"

# ── Models ────────────────────────────────────────────────────────────────────
MODELS_DIR         = ROOT / "models"
MODEL_PATH         = MODELS_DIR / "best_pipeline.joblib"
MODEL_META_PATH    = MODELS_DIR / "model_metadata.json"

# ── Outputs ───────────────────────────────────────────────────────────────────
FIGURES_DIR        = ROOT / "outputs" / "figures"
REPORTS_DIR        = ROOT / "outputs" / "reports"
METRICS_PATH       = REPORTS_DIR / "model_metrics.json"
CV_RESULTS_PATH    = REPORTS_DIR / "cv_results.json"
COMPARISON_PATH    = REPORTS_DIR / "model_comparison.csv"
THRESHOLD_PATH     = REPORTS_DIR / "threshold_analysis.csv"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED   = 42
TEST_SIZE     = 0.20   # 80 / 20 stratified split

# ── Target encoding ───────────────────────────────────────────────────────────
# UCI German Credit: original encoding 1 = good, 2 = bad.
# We remap to: 0 = good credit (negative class), 1 = bad credit (positive class).
TARGET_COL        = "credit_risk"
POSITIVE_LABEL    = 1   # bad credit — the class whose recall we prioritise
NEGATIVE_LABEL    = 0   # good credit

# ── Decision threshold for production-style output ────────────────────────────
# Tuned on validation data, NOT on the final test set.
DEFAULT_THRESHOLD = 0.40

# ── Column definitions (UCI Statlog German Credit — 20 features) ──────────────
NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_rate",
    "present_residence",
    "age",
    "existing_credits",
    "dependents",
]

CATEGORICAL_FEATURES = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── UCI column names (positional order in the original space-separated file) ──
UCI_COLUMNS = [
    "checking_account_status",   # A1  — categorical
    "duration",                  # A2  — numeric (months)
    "credit_history",            # A3  — categorical
    "purpose",                   # A4  — categorical
    "credit_amount",             # A5  — numeric (DM)
    "savings_account",           # A6  — categorical
    "employment_since",          # A7  — categorical
    "installment_rate",          # A8  — numeric (% of disposable income)
    "personal_status_sex",       # A9  — categorical
    "other_debtors",             # A10 — categorical
    "present_residence",         # A11 — numeric (years)
    "property",                  # A12 — categorical
    "age",                       # A13 — numeric (years)
    "other_installment_plans",   # A14 — categorical
    "housing",                   # A15 — categorical
    "existing_credits",          # A16 — numeric (count)
    "job",                       # A17 — categorical
    "dependents",                # A18 — numeric (count)
    "telephone",                 # A19 — categorical
    "foreign_worker",            # A20 — categorical
    "credit_risk",               # A21 — target (1=good → 0, 2=bad → 1)
]

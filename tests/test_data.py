"""
tests/test_data.py — unit tests for data loading, schema validation, and target encoding.

Tests use synthetic mini-DataFrames or the actual downloaded dataset (if present).
They do NOT require a network connection when the dataset is cached.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_DATASET_PATH,
    TARGET_COL,
    POSITIVE_LABEL,
    NEGATIVE_LABEL,
)
from src.data_validation import (
    run_full_validation,
    validate_missing,
    validate_schema,
    validate_target,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_row() -> dict:
    """Return a minimal valid row matching the German Credit schema."""
    return {
        "checking_account_status": "A11",
        "duration": 12,
        "credit_history": "A32",
        "purpose": "A43",
        "credit_amount": 3000,
        "savings_account": "A61",
        "employment_since": "A73",
        "installment_rate": 2,
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "present_residence": 2,
        "property": "A121",
        "age": 35,
        "other_installment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1,
        "job": "A173",
        "dependents": 1,
        "telephone": "A192",
        "foreign_worker": "A201",
        "credit_risk": 0,
    }


def _make_df(n: int = 1000) -> pd.DataFrame:
    """Return a synthetic DataFrame of n rows using the valid schema."""
    row = _make_synthetic_row()
    return pd.DataFrame([row] * n)


# ── Schema validation ─────────────────────────────────────────────────────────

class TestSchemaValidation:
    def test_valid_schema_passes(self):
        df = _make_df()
        errors = validate_schema(df)
        assert errors == [], f"Unexpected schema errors: {errors}"

    def test_missing_column_detected(self):
        df = _make_df().drop(columns=["duration"])
        errors = validate_schema(df)
        assert any("duration" in e for e in errors), "Missing column not detected."

    def test_wrong_row_count_detected(self):
        df = _make_df(500)
        errors = validate_schema(df)
        assert any("500" in e for e in errors), "Wrong row count not detected."


# ── Target validation ─────────────────────────────────────────────────────────

class TestTargetValidation:
    def test_valid_targets(self):
        df = _make_df()
        df[TARGET_COL] = [0, 1] * 500
        errors = validate_target(df)
        assert errors == []

    def test_invalid_target_value_detected(self):
        df = _make_df()
        df.at[0, TARGET_COL] = 2   # original UCI encoding — must be rejected
        errors = validate_target(df)
        assert any("2" in e for e in errors), "Invalid target value not caught."

    def test_target_constants(self):
        assert POSITIVE_LABEL == 1, "Bad credit must be encoded as 1."
        assert NEGATIVE_LABEL == 0, "Good credit must be encoded as 0."


# ── Missing value detection ───────────────────────────────────────────────────

class TestMissingValues:
    def test_no_missing_passes(self):
        df = _make_df()
        warnings = validate_missing(df)
        assert warnings == []

    def test_missing_detected(self):
        df = _make_df()
        df.at[0, "age"] = None
        warnings = validate_missing(df)
        assert any("age" in w for w in warnings), "Missing value in 'age' not reported."


# ── Full validation on real dataset (skipped if dataset absent) ───────────────

@pytest.mark.skipif(
    not RAW_DATASET_PATH.exists(),
    reason="Raw dataset not downloaded yet — run 'py -m src.train' first.",
)
class TestRealDataset:
    def test_real_dataset_shape(self):
        df = pd.read_csv(RAW_DATASET_PATH)
        assert df.shape == (1000, 21), f"Unexpected shape: {df.shape}"

    def test_real_dataset_validation_passes(self):
        df = pd.read_csv(RAW_DATASET_PATH)
        result = run_full_validation(df, strict=False)
        assert result["passed"], f"Validation errors: {result['errors']}"

    def test_target_encoding_correct(self):
        df = pd.read_csv(RAW_DATASET_PATH)
        unique = set(df[TARGET_COL].unique())
        assert unique.issubset({0, 1}), f"Unexpected target values: {unique}"

    def test_no_missing_values(self):
        df = pd.read_csv(RAW_DATASET_PATH)
        assert df.isnull().sum().sum() == 0, "Dataset has missing values."

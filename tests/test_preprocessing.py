"""
tests/test_preprocessing.py — unit tests for preprocessing pipeline.

Uses synthetic mini-DataFrames to test transformation correctness without
requiring a network connection or full model training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL
from src.preprocessing import build_preprocessor, split_features_target


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n: int = 50) -> pd.DataFrame:
    """Synthetic valid DataFrame for preprocessing tests."""
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n):
        rows.append({
            "checking_account_status": rng.choice(["A11", "A12", "A13", "A14"]),
            "duration": int(rng.integers(6, 60)),
            "credit_history": rng.choice(["A30", "A31", "A32", "A33", "A34"]),
            "purpose": rng.choice(["A40", "A41", "A42", "A43"]),
            "credit_amount": int(rng.integers(500, 20000)),
            "savings_account": rng.choice(["A61", "A62", "A63", "A64", "A65"]),
            "employment_since": rng.choice(["A71", "A72", "A73", "A74", "A75"]),
            "installment_rate": int(rng.integers(1, 5)),
            "personal_status_sex": rng.choice(["A91", "A92", "A93", "A94"]),
            "other_debtors": rng.choice(["A101", "A102", "A103"]),
            "present_residence": int(rng.integers(1, 5)),
            "property": rng.choice(["A121", "A122", "A123", "A124"]),
            "age": int(rng.integers(20, 70)),
            "other_installment_plans": rng.choice(["A141", "A142", "A143"]),
            "housing": rng.choice(["A151", "A152", "A153"]),
            "existing_credits": int(rng.integers(1, 5)),
            "job": rng.choice(["A171", "A172", "A173", "A174"]),
            "dependents": int(rng.integers(1, 3)),
            "telephone": rng.choice(["A191", "A192"]),
            "foreign_worker": rng.choice(["A201", "A202"]),
            "credit_risk": int(rng.choice([0, 1])),
        })
    return pd.DataFrame(rows)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        pre = build_preprocessor()
        assert isinstance(pre, ColumnTransformer)

    def test_fit_transform_shape(self):
        df = _make_df(100)
        X, y = split_features_target(df)
        pre = build_preprocessor()
        X_transformed = pre.fit_transform(X)
        # Shape: same number of rows, numeric + categorical columns
        assert X_transformed.shape[0] == 100
        assert X_transformed.shape[1] == len(ALL_FEATURES)

    def test_no_nan_in_output(self):
        df = _make_df(100)
        X, y = split_features_target(df)
        pre = build_preprocessor()
        X_transformed = pre.fit_transform(X)
        assert not np.isnan(X_transformed).any(), "NaN values in preprocessor output."

    def test_unknown_category_handled(self):
        """Preprocessor fitted on train data must not crash on unknown categories at test time."""
        train = _make_df(80)
        test  = _make_df(20)
        # Inject an unknown category value in test data
        test.at[0, "checking_account_status"] = "UNKNOWN_VAL"

        X_train, _ = split_features_target(train)
        X_test, _  = split_features_target(test)

        pre = build_preprocessor()
        pre.fit(X_train)
        # Should NOT raise; unknown_value=-1 handles this
        X_out = pre.transform(X_test)
        assert X_out.shape[0] == 20


class TestSplitFeaturesTarget:
    def test_correct_shapes(self):
        df = _make_df(50)
        X, y = split_features_target(df)
        assert X.shape == (50, len(ALL_FEATURES))
        assert y.shape == (50,)
        assert y.name == TARGET_COL

    def test_target_not_in_X(self):
        df = _make_df(50)
        X, y = split_features_target(df)
        assert TARGET_COL not in X.columns

    def test_missing_column_raises(self):
        df = _make_df(50).drop(columns=["duration"])
        with pytest.raises(ValueError, match="missing expected columns"):
            split_features_target(df)

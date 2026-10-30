"""
tests/test_smoke.py — smoke tests for the overall workflow.

These tests verify that the complete pipeline can run end-to-end on
a small synthetic dataset without requiring downloaded data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURES,
    METRICS_PATH,
    MODEL_META_PATH,
    MODEL_PATH,
    TARGET_COL,
)
from src.preprocessing import build_preprocessor, split_features_target


def _make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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


class TestSmokeWorkflow:
    """End-to-end smoke test using a tiny synthetic dataset."""

    def test_full_pipeline_smoke(self, tmp_path):
        """Train a tiny pipeline and make a prediction — no network required."""
        df = _make_df(100)
        X, y = split_features_target(df)

        pipe = Pipeline([
            ("pre", build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ])
        pipe.fit(X, y)

        # Prediction
        sample = X.iloc[:1]
        proba = pipe.predict_proba(sample)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0

    def test_predict_single_smoke(self, tmp_path):
        """predict_single should work with a temporarily saved pipeline."""
        df = _make_df(100)
        X, y = split_features_target(df)
        pipe = Pipeline([
            ("pre", build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ])
        pipe.fit(X, y)

        model_path = tmp_path / "test_model.joblib"
        joblib.dump(pipe, model_path)
        loaded = joblib.load(model_path)

        from src.predict import predict_single
        data = {
            "checking_account_status": "A11",
            "duration": 24,
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
        }
        result = predict_single(data, pipeline=loaded, threshold=0.5)
        assert 0.0 <= result["bad_credit_probability"] <= 1.0
        assert result["predicted_class"] in {0, 1}

    def test_metrics_file_valid(self):
        """If metrics file exists, it must have required keys."""
        if not METRICS_PATH.exists():
            pytest.skip("Metrics file not generated yet. Run 'py -m src.evaluate' first.")
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        assert "tuned_threshold" in metrics
        assert "test_rows" in metrics
        assert "metrics_at_tuned_threshold" in metrics
        m = metrics["metrics_at_tuned_threshold"]
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0

    def test_model_metadata_valid(self):
        """If model metadata file exists, it must have required keys."""
        if not MODEL_META_PATH.exists():
            pytest.skip("Model metadata not found. Run 'py -m src.train' first.")
        with open(MODEL_META_PATH) as f:
            meta = json.load(f)
        assert "model_name" in meta
        assert "random_seed" in meta
        assert "feature_columns" in meta
        assert meta["feature_columns"] == ALL_FEATURES
        assert meta["target_encoding"] == "0=good_credit, 1=bad_credit"

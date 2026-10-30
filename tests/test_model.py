"""
tests/test_model.py — unit tests for model training, serialization, prediction, and metrics.

These tests use a small synthetic dataset and do NOT require downloading data
or running full cross-validation, keeping the test suite fast.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import ALL_FEATURES, MODEL_PATH, TARGET_COL
from src.preprocessing import build_preprocessor, split_features_target


# ── Synthetic dataset fixture ─────────────────────────────────────────────────

def _synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
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


@pytest.fixture(scope="module")
def synthetic_data():
    return _synthetic_df(200)


@pytest.fixture(scope="module")
def trained_pipeline(synthetic_data):
    """Train a small pipeline on synthetic data."""
    X, y = split_features_target(synthetic_data)
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


# ── Training tests ────────────────────────────────────────────────────────────

class TestModelTraining:
    def test_pipeline_fits_without_error(self, synthetic_data):
        X, y = split_features_target(synthetic_data)
        pipe = Pipeline([
            ("pre", build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ])
        pipe.fit(X, y)  # must not raise

    def test_predict_shape(self, trained_pipeline, synthetic_data):
        X, y = split_features_target(synthetic_data)
        preds = trained_pipeline.predict(X)
        assert preds.shape == (len(synthetic_data),)

    def test_predict_proba_shape(self, trained_pipeline, synthetic_data):
        X, _ = split_features_target(synthetic_data)
        proba = trained_pipeline.predict_proba(X)
        assert proba.shape == (len(synthetic_data), 2)

    def test_predict_proba_range(self, trained_pipeline, synthetic_data):
        X, _ = split_features_target(synthetic_data)
        proba = trained_pipeline.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predictions_binary(self, trained_pipeline, synthetic_data):
        X, _ = split_features_target(synthetic_data)
        preds = trained_pipeline.predict(X)
        assert set(preds).issubset({0, 1})

    def test_roc_auc_reasonable(self, trained_pipeline, synthetic_data):
        """AUC on training data should be at least 0.6 (basic sanity)."""
        X, y = split_features_target(synthetic_data)
        proba = trained_pipeline.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        assert auc >= 0.6, f"ROC-AUC too low: {auc:.4f}"


# ── Serialization tests ───────────────────────────────────────────────────────

class TestSerialization:
    def test_joblib_roundtrip(self, trained_pipeline, synthetic_data, tmp_path):
        model_file = tmp_path / "test_model.joblib"
        joblib.dump(trained_pipeline, model_file)
        loaded = joblib.load(model_file)
        X, _ = split_features_target(synthetic_data)
        preds_original = trained_pipeline.predict(X)
        preds_loaded   = loaded.predict(X)
        np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_missing_model_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.joblib"
        with pytest.raises(FileNotFoundError):
            from src.predict import load_pipeline
            # Temporarily monkey-patch MODEL_PATH
            import src.predict as pred_module
            original = pred_module.MODEL_PATH
            pred_module.MODEL_PATH = missing
            try:
                pred_module.load_pipeline()
            finally:
                pred_module.MODEL_PATH = original


# ── Inference tests ───────────────────────────────────────────────────────────

class TestInference:
    def _valid_input(self) -> dict:
        return {
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

    def test_validate_input_valid(self):
        from src.predict import validate_input
        errors = validate_input(self._valid_input())
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_validate_input_missing_feature(self):
        from src.predict import validate_input
        data = self._valid_input()
        del data["age"]
        errors = validate_input(data)
        assert any("age" in e or "Missing" in e for e in errors)

    def test_validate_input_out_of_range(self):
        from src.predict import validate_input
        data = self._valid_input()
        data["age"] = 5  # below minimum 18
        errors = validate_input(data)
        assert any("age" in e for e in errors)

    def test_predict_single_with_trained_pipeline(self, trained_pipeline):
        from src.predict import predict_single
        data = self._valid_input()
        result = predict_single(data, pipeline=trained_pipeline, threshold=0.5)
        assert "bad_credit_probability" in result
        assert 0.0 <= result["bad_credit_probability"] <= 1.0
        assert result["predicted_class"] in {0, 1}
        assert "disclaimer" in result

    def test_predict_single_deterministic(self, trained_pipeline):
        from src.predict import predict_single
        data = self._valid_input()
        r1 = predict_single(data, pipeline=trained_pipeline, threshold=0.5)
        r2 = predict_single(data, pipeline=trained_pipeline, threshold=0.5)
        assert r1["bad_credit_probability"] == r2["bad_credit_probability"]


# ── Real model tests (skipped if model not trained) ───────────────────────────

@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not found — run 'py -m src.train' first.",
)
class TestRealModel:
    def test_real_model_loads(self):
        pipeline = joblib.load(MODEL_PATH)
        assert hasattr(pipeline, "predict_proba")

    def test_real_model_predict_proba(self):
        pipeline = joblib.load(MODEL_PATH)
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
        df_in = pd.DataFrame([data])[ALL_FEATURES]
        proba = pipeline.predict_proba(df_in)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0

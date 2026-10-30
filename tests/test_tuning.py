"""
tests/test_tuning.py — Automated tests for hyperparameter tuning & cost-sensitive optimization.
"""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import TRAIN_PATH
from src.model_tuning import compute_cost_sensitive_threshold, tune_random_forest
from src.preprocessing import split_features_target
from src.utils import load_csv


@pytest.fixture(scope="module")
def sample_train_data():
    df = load_csv(TRAIN_PATH)
    # Use small subset for rapid testing
    subset = df.head(100)
    X, y = split_features_target(subset)
    return X, y


def test_compute_cost_sensitive_threshold(sample_train_data):
    X, y = sample_train_data
    from sklearn.ensemble import RandomForestClassifier
    from src.preprocessing import build_preprocessor

    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
    ])
    opt_thresh, cost_df = compute_cost_sensitive_threshold(
        pipe, X, y, cost_fn=5.0, cost_fp=1.0, cv_folds=2
    )

    assert 0.05 <= opt_thresh <= 0.95
    assert not cost_df.empty
    assert "total_cost" in cost_df.columns
    assert "threshold" in cost_df.columns

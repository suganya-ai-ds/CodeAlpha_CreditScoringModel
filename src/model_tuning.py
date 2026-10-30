"""
src/model_tuning.py — Hyperparameter tuning & cost-sensitive optimization.

Provides utilities for:
1. Systematic GridSearchCV / RandomizedSearchCV tuning on training data only.
2. Cost-sensitive threshold tuning based on asymmetric financial risk matrices
   (e.g., Cost(FN) = 5x Cost(FP) reflecting missed default risk vs customer friction).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED, REPORTS_DIR, TARGET_COL, TRAIN_PATH
from src.preprocessing import build_preprocessor, split_features_target
from src.utils import ensure_dirs, get_logger, load_csv, save_json

logger = get_logger(__name__)


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    GridSearch tuning for RandomForest on training data using ROC-AUC.
    """
    preprocessor = build_preprocessor()
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1)),
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    logger.info("Running GridSearchCV for RandomForest on %d folds...", cv_folds)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)
    logger.info("Best RF Params: %s | Best CV ROC-AUC: %.4f", best_params, best_score)

    tuning_summary = {
        "best_score_roc_auc": best_score,
        "best_params": best_params,
        "n_splits": cv_folds,
    }
    return grid_search.best_estimator_, tuning_summary


def compute_cost_sensitive_threshold(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    cv_folds: int = 5,
) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal decision threshold minimizing total cost using out-of-fold training predictions.
    Total Cost = (FN * cost_fn) + (FP * cost_fp)
    Defaults to 5:1 penalty for False Negatives (loan defaults) vs False Positives (rejected solvent applicants).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_probs = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    thresholds = np.linspace(0.05, 0.95, 91)
    results = []

    y_true = np.asarray(y_train)

    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        tp = int(np.sum((y_true == 1) & (preds == 1)))
        tn = int(np.sum((y_true == 0) & (preds == 0)))
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        fn = int(np.sum((y_true == 1) & (preds == 0)))

        total_cost = (fn * cost_fn) + (fp * cost_fp)
        results.append({
            "threshold": round(float(t), 3),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
            "total_cost": float(total_cost),
        })

    df_cost = pd.DataFrame(results)
    best_row = df_cost.loc[df_cost["total_cost"].idxmin()]
    optimal_threshold = float(best_row["threshold"])

    logger.info(
        "Optimal cost-sensitive threshold: %.3f (Min Total Cost: %.1f, FN=%d, FP=%d)",
        optimal_threshold,
        best_row["total_cost"],
        best_row["fn"],
        best_row["fp"],
    )
    return optimal_threshold, df_cost


if __name__ == "__main__":
    train_df = load_csv(TRAIN_PATH)
    X_train, y_train = split_features_target(train_df)
    best_pipe, summary = tune_random_forest(X_train, y_train)
    opt_thresh, cost_df = compute_cost_sensitive_threshold(best_pipe, X_train, y_train)
    summary["cost_sensitive_threshold"] = opt_thresh
    ensure_dirs(REPORTS_DIR)
    save_json(summary, REPORTS_DIR / "tuning_summary.json")
    print(json.dumps(summary, indent=2))
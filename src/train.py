"""
src/train.py — model training, cross-validation, and artifact serialization.

Workflow:
1. Load raw dataset (download if absent).
2. Validate dataset.
3. Stratified 80/20 train/test split (test set is held-out until evaluate.py).
4. Build three pipelines: LogisticRegression, DecisionTree, RandomForest.
5. Stratified 5-fold cross-validation on training set for model selection.
6. Tune the best model's key hyperparameters on training data only.
7. Save best pipeline and metadata.

The test set is NEVER seen during training or threshold selection.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    ALL_FEATURES,
    COMPARISON_PATH,
    CV_RESULTS_PATH,
    DATA_PROCESSED_DIR,
    MODEL_META_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TEST_PATH,
    TEST_SIZE,
    TRAIN_PATH,
)
from src.data_ingestion import load_raw_dataset
from src.data_validation import run_full_validation
from src.preprocessing import build_preprocessor, split_features_target
from src.utils import ensure_dirs, get_logger, save_csv, save_json

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ── Candidate model definitions ───────────────────────────────────────────────

def _build_candidates() -> dict[str, Pipeline]:
    """Return candidate pipelines keyed by display name."""
    preprocessor = build_preprocessor()

    lr = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )),
    ])

    dt = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])

    rf = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    return {
        "LogisticRegression": lr,
        "DecisionTree": dt,
        "RandomForest": rf,
    }


# ── Cross-validation ───────────────────────────────────────────────────────────

def cross_validate_candidates(
    candidates: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """5-fold stratified CV for all candidate pipelines."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scoring = ["accuracy", "f1", "roc_auc", "precision", "recall"]
    results = []

    for name, pipe in candidates.items():
        logger.info("Cross-validating %s (%d folds)…", name, cv_folds)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            key = f"test_{metric}"
            row[f"{metric}_mean"] = float(np.mean(scores[key]))
            row[f"{metric}_std"]  = float(np.std(scores[key]))
        results.append(row)
        logger.info(
            "  %s → ROC-AUC=%.3f±%.3f  F1=%.3f±%.3f",
            name,
            row["roc_auc_mean"], row["roc_auc_std"],
            row["f1_mean"], row["f1_std"],
        )

    df_results = pd.DataFrame(results).sort_values("roc_auc_mean", ascending=False)
    return df_results


# ── Best model selection ──────────────────────────────────────────────────────

def select_best_model(
    cv_df: pd.DataFrame,
    candidates: dict[str, Pipeline],
) -> tuple[str, Pipeline]:
    """Select the candidate with highest mean ROC-AUC from CV results."""
    best_name = cv_df.iloc[0]["model"]
    logger.info("Best model by CV ROC-AUC: %s", best_name)
    return best_name, candidates[best_name]


# ── Main training function ────────────────────────────────────────────────────

def train(download_if_missing: bool = True) -> dict:
    """
    End-to-end training workflow.

    Returns
    -------
    dict
        Training summary with CV results and best model name.
    """
    # ── Directories ───────────────────────────────────────────────────────────
    ensure_dirs(DATA_PROCESSED_DIR, MODELS_DIR)

    # ── 1. Data loading & validation ──────────────────────────────────────────
    logger.info("=== Phase 1: Data Loading & Validation ===")
    df = load_raw_dataset()
    validation_result = run_full_validation(df, strict=True)
    logger.info("Validation passed: %s", validation_result)

    # ── 2. Train / test split ─────────────────────────────────────────────────
    logger.info("=== Phase 2: Train/Test Split ===")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(
        "Split — train: %d rows, test: %d rows (%.0f%% / %.0f%%)",
        len(X_train), len(X_test),
        (1 - TEST_SIZE) * 100, TEST_SIZE * 100,
    )

    # Save processed splits
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    save_csv(train_df, TRAIN_PATH)
    save_csv(test_df, TEST_PATH)
    logger.info("Saved train.csv (%d rows) and test.csv (%d rows)", len(train_df), len(test_df))

    # ── 3. Cross-validation ───────────────────────────────────────────────────
    logger.info("=== Phase 3: Cross-Validation ===")
    candidates = _build_candidates()
    cv_df = cross_validate_candidates(candidates, X_train, y_train)
    save_csv(cv_df, COMPARISON_PATH)

    cv_dict: dict = {}
    for _, row in cv_df.iterrows():
        cv_dict[row["model"]] = {k: v for k, v in row.items() if k != "model"}
    save_json(cv_dict, CV_RESULTS_PATH)

    # ── 4. Best model — refit on full training set ────────────────────────────
    logger.info("=== Phase 4: Refit Best Model on Full Training Set ===")
    best_name, best_pipe = select_best_model(cv_df, candidates)
    best_pipe.fit(X_train, y_train)

    # ── 5. Save model + metadata ──────────────────────────────────────────────
    logger.info("=== Phase 5: Saving Model Artifacts ===")
    joblib.dump(best_pipe, MODEL_PATH)
    logger.info("Saved model pipeline to %s", MODEL_PATH)

    best_cv_row = cv_df[cv_df["model"] == best_name].iloc[0].to_dict()
    metadata = {
        "model_name": best_name,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_folds": 5,
        "selection_metric": "roc_auc_mean",
        "cv_roc_auc_mean": best_cv_row.get("roc_auc_mean"),
        "cv_roc_auc_std": best_cv_row.get("roc_auc_std"),
        "cv_f1_mean": best_cv_row.get("f1_mean"),
        "feature_columns": ALL_FEATURES,
        "target_column": TARGET_COL,
        "target_encoding": "0=good_credit, 1=bad_credit",
        "positive_class": 1,
        "model_artifact": str(MODEL_PATH),
    }
    save_json(metadata, MODEL_META_PATH)
    logger.info("Saved model metadata to %s", MODEL_META_PATH)

    summary = {
        "best_model": best_name,
        "cv_results": cv_dict,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    logger.info("=== Training Complete ===")
    return summary


if __name__ == "__main__":
    result = train()
    print(json.dumps({k: v for k, v in result.items() if k != "cv_results"}, indent=2))

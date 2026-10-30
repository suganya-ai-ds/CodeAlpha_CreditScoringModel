"""
src/evaluate.py — final model evaluation on the held-out test set.

IMPORTANT: The test set is used ONCE, after all training and model selection is complete.
Threshold selection uses validation-fold predictions from cross-validation, NOT test labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

from src.config import (
    CV_RESULTS_PATH,
    DEFAULT_THRESHOLD,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_META_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    TARGET_COL,
    TEST_PATH,
    THRESHOLD_PATH,
    TRAIN_PATH,
)
from src.preprocessing import split_features_target
from src.utils import ensure_dirs, get_logger, load_csv, load_json, save_csv, save_json

logger = get_logger(__name__)


def _save_fig(fig: plt.Figure, name: str) -> Path:
    ensure_dirs(FIGURES_DIR)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


# ── Threshold analysis (on training data via cross-val scores) ────────────────

def threshold_analysis_from_train(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[float, pd.DataFrame]:
    """
    Select a decision threshold using out-of-fold predictions on the training set.
    Scans thresholds [0.10, 0.70] and picks the one maximising F1 for the bad-credit class.

    The test set is NEVER consulted here.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from src.config import RANDOM_SEED

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_probs = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    thresholds = np.linspace(0.10, 0.70, 61)
    rows = []
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        rows.append({
            "threshold":  round(float(t), 3),
            "precision":  round(float(precision_score(y_train, preds, zero_division=0)), 4),
            "recall":     round(float(recall_score(y_train, preds, zero_division=0)), 4),
            "f1":         round(float(f1_score(y_train, preds, zero_division=0)), 4),
        })

    df_thresh = pd.DataFrame(rows)
    best_row  = df_thresh.loc[df_thresh["f1"].idxmax()]
    best_threshold = float(best_row["threshold"])
    logger.info(
        "Best OOF threshold: %.3f  (F1=%.4f, Precision=%.4f, Recall=%.4f)",
        best_threshold, best_row["f1"], best_row["precision"], best_row["recall"],
    )
    return best_threshold, df_thresh


# ── Evaluation charts ─────────────────────────────────────────────────────────

def plot_roc_curve(y_test: pd.Series, y_prob: np.ndarray) -> Path:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#3498DB", label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, "roc_curve.png")


def plot_precision_recall_curve(y_test: pd.Series, y_prob: np.ndarray) -> Path:
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, color="#E67E22")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, "precision_recall_curve.png")


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> Path:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good", "Bad"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, "confusion_matrix.png")


def plot_threshold_analysis(df_thresh: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df_thresh["threshold"], df_thresh["f1"],        label="F1",        lw=2, color="#2ECC71")
    ax.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision",  lw=2, color="#3498DB")
    ax.plot(df_thresh["threshold"], df_thresh["recall"],    label="Recall",     lw=2, color="#E74C3C")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis (Out-of-Fold — Training Data Only)", fontsize=11, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, "threshold_analysis.png")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate() -> dict:
    """
    Load best pipeline and test set, evaluate, save metrics and charts.

    Returns
    -------
    dict
        Full metrics dictionary.
    """
    ensure_dirs(REPORTS_DIR, FIGURES_DIR)

    # ── Load artifacts ────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'py -m src.train' first."
        )
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Loaded model pipeline from %s", MODEL_PATH)

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test set not found at {TEST_PATH}. Run 'py -m src.train' first."
        )
    test_df  = load_csv(TEST_PATH)
    X_test, y_test = split_features_target(test_df)

    train_df = load_csv(TRAIN_PATH)
    X_train, y_train = split_features_target(train_df)

    # ── Threshold selection from training OOF predictions ─────────────────────
    logger.info("Selecting decision threshold using out-of-fold training predictions…")
    best_threshold, df_thresh = threshold_analysis_from_train(pipeline, X_train, y_train)
    save_csv(df_thresh, THRESHOLD_PATH)
    plot_threshold_analysis(df_thresh)

    # ── Test set predictions ──────────────────────────────────────────────────
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default   = (y_prob >= DEFAULT_THRESHOLD).astype(int)
    y_pred_tuned     = (y_prob >= best_threshold).astype(int)

    # ── Metrics (using tuned threshold for primary report) ────────────────────
    def _metrics(y_true, y_pred, y_prob, label=""):
        return {
            "label": label,
            "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred,
                target_names=["Good (0)", "Bad (1)"],
                output_dict=True,
            ),
        }

    metrics_default = _metrics(y_test, y_pred_default, y_prob, f"threshold={DEFAULT_THRESHOLD}")
    metrics_tuned   = _metrics(y_test, y_pred_tuned,   y_prob, f"threshold={best_threshold:.3f}")

    full_metrics = {
        "tuned_threshold": best_threshold,
        "default_threshold": DEFAULT_THRESHOLD,
        "test_rows": int(len(y_test)),
        "metrics_at_tuned_threshold": metrics_tuned,
        "metrics_at_default_threshold": metrics_default,
    }
    save_json(full_metrics, METRICS_PATH)
    logger.info("Saved evaluation metrics to %s", METRICS_PATH)

    # ── Charts ────────────────────────────────────────────────────────────────
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    plot_confusion_matrix(y_test, y_pred_tuned)

    # ── Summary to console ────────────────────────────────────────────────────
    m = metrics_tuned
    logger.info(
        "=== TEST SET RESULTS (threshold=%.3f) ===\n"
        "  Accuracy:  %.4f\n  Precision: %.4f\n  Recall:    %.4f\n"
        "  F1:        %.4f\n  ROC-AUC:   %.4f",
        best_threshold, m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"],
    )
    return full_metrics


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(
        {k: v for k, v in result.items() if k != "metrics_at_tuned_threshold"},
        indent=2
    ))
    m = result["metrics_at_tuned_threshold"]
    print(f"\nTest ROC-AUC: {m['roc_auc']}")
    print(f"Test F1:      {m['f1']}")
    print(f"Test Recall:  {m['recall']}")

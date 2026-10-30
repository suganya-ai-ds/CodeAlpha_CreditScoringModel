"""
src/explainability.py — feature importance and model interpretation.

Methods used:
- Permutation feature importance (model-agnostic, valid for any estimator).
- Logistic regression coefficients (when the best model is LR).
- Decision tree feature importances (when applicable).

All charts saved to outputs/figures/.
Importance values saved to outputs/reports/.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

from src.config import (
    ALL_FEATURES,
    FIGURES_DIR,
    MODEL_PATH,
    REPORTS_DIR,
    TEST_PATH,
    TRAIN_PATH,
    RANDOM_SEED,
)
from src.preprocessing import split_features_target
from src.utils import ensure_dirs, get_logger, load_csv, save_csv, save_json

logger = get_logger(__name__)
_REPORTS_IMPORTANCE = REPORTS_DIR / "feature_importance.csv"
_FIGURE_PERM        = FIGURES_DIR / "permutation_importance.png"
_FIGURE_MODEL       = FIGURES_DIR / "model_feature_importance.png"


def _save_fig(fig: plt.Figure, path) -> None:
    ensure_dirs(FIGURES_DIR)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def compute_permutation_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 20,
) -> pd.DataFrame:
    """Compute permutation feature importance on test set (model-agnostic)."""
    logger.info("Computing permutation importance (%d repeats)…", n_repeats)
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring="roc_auc",
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature":        ALL_FEATURES,
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def plot_permutation_importance(df_imp: pd.DataFrame, top_n: int = 15) -> None:
    top = df_imp.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        top["feature"][::-1],
        top["importance_mean"][::-1],
        xerr=top["importance_std"][::-1],
        color="#3498DB",
        edgecolor="white",
        capsize=4,
    )
    ax.set_xlabel("Mean Decrease in ROC-AUC")
    ax.set_title(f"Permutation Feature Importance (Top {top_n})", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    fig.tight_layout()
    _save_fig(fig, _FIGURE_PERM)


def compute_model_feature_importance(pipeline) -> pd.DataFrame | None:
    """
    Extract built-in feature importances from the classifier step if available.
    Returns None for models without .feature_importances_ or .coef_.
    """
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return None

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        method = "gini_impurity"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        method = "abs_coefficient"
    else:
        return None

    df = pd.DataFrame({
        "feature":    ALL_FEATURES,
        "importance": importances,
        "method":     method,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def plot_model_feature_importance(df_imp: pd.DataFrame, top_n: int = 15) -> None:
    method = df_imp["method"].iloc[0] if "method" in df_imp.columns else "built-in"
    top = df_imp.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top["feature"][::-1],
        top["importance"][::-1],
        color="#E67E22",
        edgecolor="white",
    )
    ax.set_xlabel(f"Importance ({method})")
    ax.set_title(
        f"Model Feature Importance — {method.replace('_', ' ').title()} (Top {top_n})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, _FIGURE_MODEL)


def run_explainability() -> dict:
    """Full explainability analysis: permutation + built-in importance."""
    ensure_dirs(FIGURES_DIR, REPORTS_DIR)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Loaded pipeline from %s", MODEL_PATH)

    test_df = load_csv(TEST_PATH)
    X_test, y_test = split_features_target(test_df)

    # Permutation importance
    df_perm = compute_permutation_importance(pipeline, X_test, y_test)
    save_csv(df_perm, _REPORTS_IMPORTANCE)
    plot_permutation_importance(df_perm)

    # Built-in importance
    df_model = compute_model_feature_importance(pipeline)
    if df_model is not None:
        plot_model_feature_importance(df_model)
        df_model_path = REPORTS_DIR / "model_feature_importance.csv"
        save_csv(df_model, df_model_path)

    result = {
        "top_features_permutation": df_perm.head(10)["feature"].tolist(),
        "top_importance_values":    df_perm.head(10)["importance_mean"].round(4).tolist(),
    }
    if df_model is not None:
        result["top_features_model"] = df_model.head(10)["feature"].tolist()

    logger.info("Explainability analysis complete.")
    return result


if __name__ == "__main__":
    import json
    result = run_explainability()
    print(json.dumps(result, indent=2))

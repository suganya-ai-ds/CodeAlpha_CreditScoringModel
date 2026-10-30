"""
src/eda.py — exploratory data analysis and chart generation.

All charts are saved to outputs/figures/ as PNG files.
No display calls (plt.show()) are used — all figures are saved to disk
to support headless / CI environments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    TARGET_COL,
)
from src.utils import ensure_dirs, get_logger

logger = get_logger(__name__)

# Colour palette — dark navy / teal accent
_PALETTE = ["#2ECC71", "#E74C3C"]  # green=good, red=bad
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.0)


def _save(fig: plt.Figure, name: str) -> Path:
    ensure_dirs(FIGURES_DIR)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


# ── Individual chart functions ────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> Path:
    """Bar chart of class distribution."""
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = {0: "Good Credit", 1: "Bad Credit"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [labels.get(k, str(k)) for k in counts.index],
        counts.values,
        color=_PALETTE,
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val} ({val/len(df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Credit Risk")
    fig.tight_layout()
    return _save(fig, "target_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame) -> Path:
    """Grid of histograms for all numeric features, split by target class."""
    n = len(NUMERIC_FEATURES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes_flat = axes.flatten()

    for i, col in enumerate(NUMERIC_FEATURES):
        ax = axes_flat[i]
        for label, grp in df.groupby(TARGET_COL):
            ax.hist(
                grp[col].dropna(),
                bins=20,
                alpha=0.6,
                label=f"{'Bad' if label == 1 else 'Good'}",
                color=_PALETTE[label],
                edgecolor="white",
            )
        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Credit Risk", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "numeric_distributions.png")


def plot_categorical_counts(df: pd.DataFrame) -> Path:
    """Count plots for categorical features."""
    n = len(CATEGORICAL_FEATURES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes_flat = axes.flatten()

    for i, col in enumerate(CATEGORICAL_FEATURES):
        ax = axes_flat[i]
        order = df[col].value_counts().index
        sns.countplot(
            data=df,
            x=col,
            hue=TARGET_COL,
            order=order,
            ax=ax,
            palette=_PALETTE,
        )
        ax.set_title(col.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.legend(
            title="Risk",
            labels=["Good", "Bad"],
            fontsize=7,
            title_fontsize=7,
        )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Categorical Feature Distributions by Credit Risk", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "categorical_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Pearson correlation heatmap of numeric features."""
    corr = df[NUMERIC_FEATURES + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = __import__("numpy").triu(__import__("numpy").ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        mask=mask,
    )
    ax.set_title("Pearson Correlation — Numeric Features + Target", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "correlation_heatmap.png")


def plot_missing_values(df: pd.DataFrame) -> Path:
    """Bar chart of missing value counts per column."""
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values found!", ha="center", va="center", fontsize=13)
        ax.axis("off")
    else:
        ax.barh(missing.index, missing.values, color="#E74C3C")
        ax.set_xlabel("Missing Count")
        ax.set_title("Missing Values per Column", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "missing_values.png")


def run_full_eda(df: pd.DataFrame) -> dict[str, Path]:
    """Generate all EDA charts and return a dict of {name: path}."""
    logger.info("Running full EDA — saving charts to %s", FIGURES_DIR)
    paths: dict[str, Path] = {}
    paths["target_distribution"]    = plot_target_distribution(df)
    paths["numeric_distributions"]  = plot_numeric_distributions(df)
    paths["categorical_counts"]     = plot_categorical_counts(df)
    paths["correlation_heatmap"]    = plot_correlation_heatmap(df)
    paths["missing_values"]         = plot_missing_values(df)
    logger.info("EDA complete — %d charts generated.", len(paths))
    return paths

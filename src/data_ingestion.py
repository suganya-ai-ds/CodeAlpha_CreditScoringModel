"""
src/data_ingestion.py — downloads or loads the UCI German Credit dataset.

The UCI Statlog German Credit dataset has 1 000 rows and 20 predictor attributes
plus 1 target attribute.  The original file is space-separated with no header.
Target encoding in the original: 1 = good credit, 2 = bad credit.
We remap to: 0 = good, 1 = bad (POSITIVE_LABEL = bad credit).

Reference: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    RAW_DATASET_PATH,
    UCI_COLUMNS,
    TARGET_COL,
)
from src.utils import get_logger, ensure_dirs, save_csv

logger = get_logger(__name__)

# Primary URL (UCI ML Repository direct file link)
_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)
# Mirror / fallback
_MIRROR_URL = (
    "https://raw.githubusercontent.com/gagolews/teaching-data/master/"
    "marek/german_credit_data.csv"
)


def _remap_target(series: pd.Series) -> pd.Series:
    """Map original UCI encoding (1=good, 2=bad) → (0=good, 1=bad)."""
    mapping = {1: 0, 2: 1}
    remapped = series.map(mapping)
    if remapped.isna().any():
        raise ValueError(
            f"Unexpected values in target column: {series.unique().tolist()}"
        )
    return remapped.astype(int)


def download_dataset(url: str = _UCI_URL) -> pd.DataFrame:
    """Download the raw German Credit dataset from UCI and save to data/raw/."""
    ensure_dirs(DATA_RAW_DIR)
    logger.info("Downloading German Credit dataset from %s", url)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(
            f"Dataset download failed from {url}: {exc}\n"
            "Check your internet connection or place 'german.data' in data/raw/ manually."
        ) from exc

    rows = [line.split() for line in content.strip().splitlines() if line.strip()]
    df = pd.DataFrame(rows, columns=UCI_COLUMNS)
    # Numeric columns are stored as strings after split — cast them
    numeric_cols = [c for c in UCI_COLUMNS if c not in [TARGET_COL] and _is_numeric_col(df, c)]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df[TARGET_COL] = _remap_target(pd.to_numeric(df[TARGET_COL], errors="raise"))
    save_csv(df, RAW_DATASET_PATH)
    logger.info("Saved %d rows × %d cols to %s", *df.shape, RAW_DATASET_PATH)
    return df


def _is_numeric_col(df: pd.DataFrame, col: str) -> bool:
    """Heuristic: try converting first non-null value; True if it works."""
    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
    try:
        float(sample)
        return True
    except (ValueError, TypeError):
        return False


def load_raw_dataset() -> pd.DataFrame:
    """
    Load raw dataset from disk.  Downloads automatically if not present.

    Returns
    -------
    pd.DataFrame
        1 000-row German Credit DataFrame with remapped target.
    """
    if RAW_DATASET_PATH.exists():
        logger.info("Loading existing raw dataset from %s", RAW_DATASET_PATH)
        df = pd.read_csv(RAW_DATASET_PATH)
        # Validate columns are present
        missing = set(UCI_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Existing raw dataset is missing columns: {missing}. "
                "Delete data/raw/german_credit.csv and re-run to re-download."
            )
        return df
    logger.info("Raw dataset not found — attempting download.")
    return download_dataset()

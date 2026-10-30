"""
src/run_eda.py — standalone script to generate all EDA charts.

Run from the repository root:
    py -m src.run_eda
"""

from __future__ import annotations

from src.data_ingestion import load_raw_dataset
from src.eda import run_full_eda
from src.utils import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("Loading dataset…")
    df = load_raw_dataset()
    logger.info("Generating EDA charts…")
    paths = run_full_eda(df)
    logger.info("EDA complete. Generated %d figures:", len(paths))
    for name, path in paths.items():
        logger.info("  %s → %s", name, path)

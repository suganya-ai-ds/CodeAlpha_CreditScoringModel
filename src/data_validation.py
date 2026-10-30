"""
src/data_validation.py — schema and content validation for the German Credit dataset.

Raises ValueError or TypeError with actionable messages on failure.
All checks are deterministic and do not depend on model training.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    POSITIVE_LABEL,
    NEGATIVE_LABEL,
)
from src.utils import get_logger

logger = get_logger(__name__)

EXPECTED_ROWS = 1_000
EXPECTED_COLS = 21  # 20 features + 1 target

# Categorical value sets as defined by UCI documentation
VALID_CATEGORICAL_VALUES: dict[str, set[str]] = {
    "checking_account_status": {"A11", "A12", "A13", "A14"},
    "credit_history":          {"A30", "A31", "A32", "A33", "A34"},
    "purpose":                 {"A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A410"},
    "savings_account":         {"A61", "A62", "A63", "A64", "A65"},
    "employment_since":        {"A71", "A72", "A73", "A74", "A75"},
    "personal_status_sex":     {"A91", "A92", "A93", "A94", "A95"},
    "other_debtors":           {"A101", "A102", "A103"},
    "property":                {"A121", "A122", "A123", "A124"},
    "other_installment_plans": {"A141", "A142", "A143"},
    "housing":                 {"A151", "A152", "A153"},
    "job":                     {"A171", "A172", "A173", "A174"},
    "telephone":               {"A191", "A192"},
    "foreign_worker":          {"A201", "A202"},
}

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "duration":            (1, 120),    # months
    "credit_amount":       (1, 200_000),
    "installment_rate":    (1, 4),
    "present_residence":   (1, 4),
    "age":                 (18, 120),
    "existing_credits":    (1, 4),
    "dependents":          (1, 2),
}


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Check columns, dtypes, and shape.  Returns list of error strings."""
    errors: list[str] = []
    missing_cols = set(ALL_FEATURES + [TARGET_COL]) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {sorted(missing_cols)}")
    if df.shape[0] != EXPECTED_ROWS:
        errors.append(
            f"Expected {EXPECTED_ROWS} rows, found {df.shape[0]}"
        )
    return errors


def validate_target(df: pd.DataFrame) -> list[str]:
    """Check target column contains only {0, 1}."""
    errors: list[str] = []
    if TARGET_COL not in df.columns:
        return [f"Target column '{TARGET_COL}' is absent."]
    unique = set(df[TARGET_COL].unique())
    valid = {POSITIVE_LABEL, NEGATIVE_LABEL}
    if not unique.issubset(valid):
        errors.append(
            f"Target column has unexpected values: {unique - valid}. "
            f"Expected {valid}."
        )
    return errors


def validate_numerics(df: pd.DataFrame) -> list[str]:
    """Check numeric feature ranges."""
    errors: list[str] = []
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if not out_of_range.empty:
            errors.append(
                f"Column '{col}': {len(out_of_range)} values outside [{lo}, {hi}]."
            )
    return errors


def validate_categoricals(df: pd.DataFrame) -> list[str]:
    """Check that categorical columns contain only documented levels."""
    errors: list[str] = []
    for col, valid_set in VALID_CATEGORICAL_VALUES.items():
        if col not in df.columns:
            continue
        found = set(df[col].astype(str).unique())
        unknown = found - valid_set
        if unknown:
            errors.append(
                f"Column '{col}': unknown categories {unknown}."
            )
    return errors


def validate_missing(df: pd.DataFrame) -> list[str]:
    """Report columns with missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return []
    return [
        f"Missing values detected: {missing.to_dict()}"
    ]


def validate_duplicates(df: pd.DataFrame) -> list[str]:
    """Report duplicate rows."""
    n_dup = df.duplicated().sum()
    if n_dup:
        return [f"{n_dup} duplicate rows found."]
    return []


def run_full_validation(df: pd.DataFrame, strict: bool = True) -> dict[str, Any]:
    """
    Run all validation checks and return a summary dict.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    strict : bool
        If True, raise ValueError when any errors are found.

    Returns
    -------
    dict with keys: errors, warnings, passed (bool).
    """
    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(validate_schema(df))
    errors.extend(validate_target(df))
    errors.extend(validate_numerics(df))
    errors.extend(validate_categoricals(df))
    warnings.extend(validate_missing(df))
    warnings.extend(validate_duplicates(df))

    passed = len(errors) == 0
    result: dict[str, Any] = {
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "target_distribution": df[TARGET_COL].value_counts().to_dict() if TARGET_COL in df.columns else {},
    }

    for w in warnings:
        logger.warning("VALIDATION WARNING: %s", w)
    for e in errors:
        logger.error("VALIDATION ERROR: %s", e)

    if errors and strict:
        raise ValueError(
            f"Dataset validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    logger.info(
        "Validation %s — rows=%d, cols=%d, errors=%d, warnings=%d",
        "PASSED" if passed else "FAILED",
        df.shape[0], df.shape[1], len(errors), len(warnings),
    )
    return result

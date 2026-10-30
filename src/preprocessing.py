"""
src/preprocessing.py — build the sklearn preprocessing pipeline.

Design decisions:
- All learned transformations (imputation, scaling, encoding) are fitted
  ONLY on training data to prevent target leakage and train-test contamination.
- Categorical features use OrdinalEncoder with handle_unknown='use_encoded_value'
  to gracefully handle unknown categories at inference time.
- Numeric features are imputed (median) then standard-scaled.
- The pipeline is compatible with RandomForest, LogisticRegression, and DecisionTree.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL


def build_preprocessor() -> ColumnTransformer:
    """
    Return an unfitted ColumnTransformer that preprocesses the German Credit features.

    Numeric pipeline:  median imputation → standard scaling
    Categorical pipeline: constant imputation ('missing') → ordinal encoding
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",   # drop any extra columns silently
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return feature names after fitting (numeric + categorical order)."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate X (features) from y (target) using config column lists."""
    from src.config import ALL_FEATURES

    missing = set(ALL_FEATURES + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing expected columns: {missing}")

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y

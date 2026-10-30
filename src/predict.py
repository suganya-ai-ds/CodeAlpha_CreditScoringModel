"""
src/predict.py — inference on new data using the saved pipeline.

Usage:
    py -m src.predict --input '{"duration": 24, "credit_amount": 5000, ...}'
    py -m src.predict --csv path/to/new_data.csv

The loaded pipeline includes the fitted preprocessor, so raw feature values
(same schema as the training data) can be passed directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_THRESHOLD,
    METRICS_PATH,
    MODEL_META_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
)
from src.utils import get_logger, load_json

logger = get_logger(__name__)


# ── Pipeline loading ──────────────────────────────────────────────────────────

def load_pipeline():
    """Load the saved sklearn pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{MODEL_PATH}'.\n"
            "Run 'py -m src.train' to train and save the model."
        )
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Pipeline loaded from %s", MODEL_PATH)
    return pipeline


def get_decision_threshold() -> float:
    """Return the tuned threshold from saved metrics, falling back to default."""
    if METRICS_PATH.exists():
        try:
            metrics = load_json(METRICS_PATH)
            return float(metrics.get("tuned_threshold", DEFAULT_THRESHOLD))
        except Exception:
            pass
    return DEFAULT_THRESHOLD


# ── Input validation ──────────────────────────────────────────────────────────

NUMERIC_BOUNDS = {
    "duration":          (1, 120),
    "credit_amount":     (1, 200_000),
    "installment_rate":  (1, 4),
    "present_residence": (1, 4),
    "age":               (18, 120),
    "existing_credits":  (1, 4),
    "dependents":        (1, 2),
}


def validate_input(data: dict) -> list[str]:
    """
    Validate a single prediction input dictionary.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []
    missing = [f for f in ALL_FEATURES if f not in data]
    if missing:
        errors.append(f"Missing features: {missing}")
        return errors  # can't validate further without all features

    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        val = data.get(col)
        if val is None:
            errors.append(f"'{col}' is required.")
        elif not isinstance(val, (int, float)):
            errors.append(f"'{col}' must be numeric, got {type(val).__name__}.")
        elif not (lo <= val <= hi):
            errors.append(f"'{col}' value {val} is outside expected range [{lo}, {hi}].")

    return errors


# ── Core prediction function ──────────────────────────────────────────────────

def predict_single(
    input_data: dict,
    pipeline=None,
    threshold: float | None = None,
) -> dict:
    """
    Predict credit risk for a single applicant dictionary.

    Parameters
    ----------
    input_data : dict
        Feature values keyed by column name (same as training schema).
    pipeline : fitted sklearn Pipeline, optional
        Loaded automatically if not supplied.
    threshold : float, optional
        Decision threshold.  Loaded from saved metrics if not supplied.

    Returns
    -------
    dict with keys: bad_credit_probability, predicted_class, risk_category,
                    threshold_used, disclaimer.
    """
    errors = validate_input(input_data)
    if errors:
        raise ValueError(f"Input validation failed: {errors}")

    if pipeline is None:
        pipeline = load_pipeline()
    if threshold is None:
        threshold = get_decision_threshold()

    df_input = pd.DataFrame([input_data])[ALL_FEATURES]
    prob = float(pipeline.predict_proba(df_input)[0, 1])
    pred = int(prob >= threshold)

    if prob < 0.30:
        risk_category = "Low Risk"
    elif prob < threshold:
        risk_category = "Moderate Risk"
    elif prob < 0.70:
        risk_category = "High Risk"
    else:
        risk_category = "Very High Risk"

    return {
        "bad_credit_probability": round(prob, 4),
        "predicted_class":        pred,
        "risk_category":          risk_category,
        "threshold_used":         round(threshold, 4),
        "disclaimer": (
            "⚠ This is an EDUCATIONAL PROTOTYPE only. "
            "It must not be used for actual lending decisions. "
            "Predictions carry uncertainty and may reflect historical biases in the training data."
        ),
    }


def predict_dataframe(
    df: pd.DataFrame,
    pipeline=None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Run batch predictions on a DataFrame with the same feature schema."""
    if pipeline is None:
        pipeline = load_pipeline()
    if threshold is None:
        threshold = get_decision_threshold()

    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing features: {missing}")

    X = df[ALL_FEATURES]
    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    result = df.copy()
    result["bad_credit_probability"] = probs.round(4)
    result["predicted_class"]        = preds
    return result


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Credit Intelligence Pro — Inference")
    parser.add_argument("--input",  type=str, help="JSON string of feature values")
    parser.add_argument("--csv",    type=str, help="Path to a CSV file for batch prediction")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override")
    parser.add_argument("--sample", action="store_true", help="Run prediction on a representative test applicant")
    args = parser.parse_args()

    if args.sample:
        sample_applicant = {
            "duration": 24,
            "credit_amount": 4870,
            "installment_rate": 3,
            "present_residence": 2,
            "age": 35,
            "existing_credits": 2,
            "dependents": 1,
            "checking_account_status": "<0_DM",
            "credit_history": "existing_paid",
            "purpose": "car_new",
            "savings_account": "<100_DM",
            "employment_since": "1_to_4_years",
            "personal_status_sex": "male_single",
            "other_debtors": "none",
            "property": "car_or_other",
            "other_installment_plans": "none",
            "housing": "own",
            "job": "skilled",
            "telephone": "yes",
            "foreign_worker": "yes",
        }
        result = predict_single(sample_applicant, threshold=args.threshold)
        print(json.dumps(result, indent=2))
    elif args.input:
        data = json.loads(args.input)
        result = predict_single(data, threshold=args.threshold)
        print(json.dumps(result, indent=2))
    elif args.csv:
        df = pd.read_csv(args.csv)
        pipeline = load_pipeline()
        result_df = predict_dataframe(df, pipeline, threshold=args.threshold)
        print(result_df[["bad_credit_probability", "predicted_class"]].to_string())
    else:
        parser.print_help()
        sys.exit(1)

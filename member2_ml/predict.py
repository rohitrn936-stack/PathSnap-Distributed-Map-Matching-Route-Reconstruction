"""Score candidate roads with a trained Member 2 model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    BASE_DIR / "outputs" / "real_baseline" / "HistGradientBoosting.joblib"
)
FEATURE_ORDER_PATH = BASE_DIR / "outputs" / "real_baseline" / "feature_order.json"
IDENTIFIER_COLUMNS = ["split", "trip_id", "observation_idx", "arc_id"]


def _load_feature_order():
    with FEATURE_ORDER_PATH.open(encoding="utf-8") as file:
        features = json.load(file)

    if (
        not isinstance(features, list)
        or len(features) != 6
        or not all(isinstance(feature, str) for feature in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError(f"Expected six unique feature names in {FEATURE_ORDER_PATH}")
    return features


def _allows_missing_values(model):
    """Use sklearn estimator tags to determine whether NaN input is supported."""
    try:
        from sklearn.utils import get_tags

        if get_tags(model).input_tags.allow_nan:
            return True
    except (ImportError, AttributeError, TypeError):
        try:
            if model._get_tags().get("allow_nan", False):
                return True
        except (AttributeError, TypeError):
            pass

    # A pipeline may accept NaNs through preprocessing even when its combined
    # estimator tag does not advertise that capability.
    try:
        from sklearn.impute import SimpleImputer

        estimators = model.get_params(deep=True).values()
        return any(
            isinstance(estimator, SimpleImputer)
            and bool(pd.isna(estimator.missing_values))
            for estimator in estimators
        )
    except (AttributeError, TypeError):
        return False


def predict_candidates(candidates, model_path=DEFAULT_MODEL_PATH):
    """Return candidate identifiers and their model probability for class 1."""
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")

    features = _load_feature_order()
    required_columns = IDENTIFIER_COLUMNS + features
    missing_columns = [
        column for column in required_columns if column not in candidates.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if candidates[IDENTIFIER_COLUMNS].isna().any().any():
        raise ValueError("Candidate identifiers must not be missing")

    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("The loaded model must provide predict_proba() and classes_")

    positive_columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive_columns) != 1:
        raise ValueError("Model classes_ must contain class 1 exactly once")

    model_features = pd.DataFrame(index=candidates.index)
    has_missing = False
    for feature in features:
        original = candidates[feature]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = (
            original.notna()
            & original.astype("string").str.strip().ne("")
            & numeric.isna()
        )
        if invalid.any():
            raise ValueError(
                f"Non-numeric values in {feature}: {original[invalid].head().tolist()}"
            )
        if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values in feature {feature}")

        has_missing |= numeric.isna().any()
        model_features[feature] = numeric

    if has_missing and not _allows_missing_values(model):
        raise ValueError("Missing feature values are not supported by this model")

    result = candidates[IDENTIFIER_COLUMNS].copy()
    if candidates.empty:
        result["ml_score"] = pd.Series(index=result.index, dtype=float)
        return result

    probabilities = model.predict_proba(model_features)[:, positive_columns[0]]
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Model returned probabilities outside the finite [0, 1] range")

    result["ml_score"] = probabilities
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Model joblib path (defaults to the baseline model).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Score five rows from data/real_candidates.csv.",
    )
    args = parser.parse_args()

    if not args.smoke_test:
        parser.error("--smoke-test is required to run the included smoke test")

    sample = pd.read_csv(BASE_DIR / "data" / "real_candidates.csv", nrows=5)
    scored = predict_candidates(sample, model_path=args.model_path)
    expected_columns = IDENTIFIER_COLUMNS + ["ml_score"]
    if list(scored.columns) != expected_columns or len(scored) != len(sample):
        raise AssertionError("Smoke test returned an unexpected output schema or row count")
    if not np.isfinite(scored["ml_score"]).all() or not scored["ml_score"].between(0, 1).all():
        raise AssertionError("Smoke test returned invalid scores")

    print(f"Smoke test passed: {len(scored)} candidate rows scored using {args.model_path}")


if __name__ == "__main__":
    main()

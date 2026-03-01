# src/analysis/clinical/abcd_analysis.py

import pandas as pd
from pathlib import Path
from typing import Dict, List


ABCD_FEATURES = ["asymmetry", "border", "color", "diameter"]
REQUIRED_PRED_COLUMNS = {"image_id", "y_true", "y_pred"}


def run_abcd_error_analysis(
    exp_dir: Path,
    metadata_path: Path,
    min_samples: int = 5
) -> Dict[str, List[Dict]]:
    """
    Analyze prediction performance by ABCD clinical features.

    Parameters
    ----------
    exp_dir : Path
    metadata_path : Path
    min_samples : int
        Minimum number of samples per subgroup to include.

    Returns
    -------
    dict:
        {
            feature_name: [
                {
                    "value": ...,
                    "n_samples": ...,
                    "accuracy": ...,
                    "error_rate": ...
                }
            ]
        }
    """

    pred_path = exp_dir / "predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {exp_dir}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    preds = pd.read_csv(pred_path)
    meta = pd.read_csv(metadata_path)

    if not REQUIRED_PRED_COLUMNS.issubset(preds.columns):
        missing = REQUIRED_PRED_COLUMNS - set(preds.columns)
        raise ValueError(f"Missing prediction columns: {missing}")

    if "image_id" not in meta.columns:
        raise ValueError("Metadata must contain 'image_id' column.")

    missing_features = [f for f in ABCD_FEATURES if f not in meta.columns]
    if missing_features:
        raise ValueError(f"Missing ABCD feature columns: {missing_features}")

    df = preds.merge(meta, on="image_id", how="inner")

    if len(df) != len(preds):
        raise ValueError(
            "Mismatch after merge: some predictions have no metadata."
        )

    df["correct"] = df["y_true"] == df["y_pred"]

    results = {}

    for feature in ABCD_FEATURES:

        # Drop missing feature values
        feature_df = df.dropna(subset=[feature])

        grouped = (
            feature_df
            .groupby(feature)
            .agg(
                n_samples=("correct", "count"),
                accuracy=("correct", "mean")
            )
            .reset_index()
        )

        # Filter small groups
        grouped = grouped[grouped["n_samples"] >= min_samples]

        grouped["error_rate"] = 1.0 - grouped["accuracy"]

        grouped = grouped.sort_values("accuracy")

        results[feature] = grouped.to_dict(orient="records")

    return results
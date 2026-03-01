# src/analysis/statistics/mcnemar_test.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from statsmodels.stats.contingency_tables import mcnemar


REQUIRED_COLUMNS = {"image_id", "y_true", "y_pred"}


def load_predictions(exp_dir: Path) -> pd.DataFrame:
    """
    Load predictions.csv from experiment directory.
    Must contain:
        - image_id
        - y_true
        - y_pred
    """

    pred_path = exp_dir / "predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {exp_dir}")

    df = pd.read_csv(pred_path)

    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing = REQUIRED_COLUMNS - set(df.columns)
        raise ValueError(
            f"Missing required columns {missing} in {pred_path}"
        )

    return df[list(REQUIRED_COLUMNS)].copy()


def run_mcnemar_test(
    exp1_dir: Path,
    exp2_dir: Path,
    exact: bool = False,
    correction: bool = True
) -> Dict:
    """
    McNemar test between two experiments.

    Parameters
    ----------
    exp1_dir : Path
    exp2_dir : Path
    exact : bool
        Use exact binomial test (recommended if b+c < 25).
    correction : bool
        Apply continuity correction (for chi-square approximation).

    Returns
    -------
    dict
        {
            "n_samples": int,
            "a": int,
            "b": int,
            "c": int,
            "d": int,
            "statistic": float,
            "p_value": float
        }
    """

    df1 = load_predictions(exp1_dir)
    df2 = load_predictions(exp2_dir)

    merged = df1.merge(
        df2,
        on=["image_id", "y_true"],
        suffixes=("_1", "_2")
    )

    if len(merged) == 0:
        raise ValueError("No overlapping samples between experiments.")

    correct_1 = merged["y_pred_1"] == merged["y_true"]
    correct_2 = merged["y_pred_2"] == merged["y_true"]

    a = np.sum((correct_1 == True) & (correct_2 == True))
    b = np.sum((correct_1 == True) & (correct_2 == False))
    c = np.sum((correct_1 == False) & (correct_2 == True))
    d = np.sum((correct_1 == False) & (correct_2 == False))

    if (b + c) == 0:
        return {
            "n_samples": int(len(merged)),
            "a": int(a),
            "b": int(b),
            "c": int(c),
            "d": int(d),
            "statistic": 0.0,
            "p_value": 1.0,
            "note": "Models have identical predictions."
        }

    table = [[a, b],
             [c, d]]

    result = mcnemar(
        table,
        exact=exact,
        correction=correction
    )

    return {
        "n_samples": int(len(merged)),
        "a": int(a),
        "b": int(b),
        "c": int(c),
        "d": int(d),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue)
    }
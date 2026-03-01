# src/analysis/performance/confusion_analysis.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from src.analysis.core.load_results import load_all_experiments
from src.analysis.core.metrics_utils import validate_class_consistency


def _validate_confusion_matrix(exp: dict) -> None:
    """
    Internal validation for confusion matrix structure.
    """

    exp_name = exp.get("experiment", exp.get("__exp_name__"))

    if "confusion_matrix" not in exp:
        raise ValueError(f"'confusion_matrix' missing in experiment {exp_name}")

    cm_data = exp["confusion_matrix"]

    if "matrix" not in cm_data or "class_names" not in cm_data:
        raise ValueError(
            f"Invalid confusion_matrix structure in experiment {exp_name}"
        )

    matrix = np.array(cm_data["matrix"])
    class_names = cm_data["class_names"]

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Confusion matrix not square in experiment {exp_name}"
        )

    if matrix.shape[0] != len(class_names):
        raise ValueError(
            f"Matrix size does not match class_names in experiment {exp_name}"
        )


def find_top_confusions(
    results_dir: Path,
    top_k: int = 5,
    aggregate: bool = False
) -> pd.DataFrame:
    """
    Find most frequent misclassification pairs.

    Parameters
    ----------
    results_dir : Path
        Path to results directory.
    top_k : int
        Number of top confusion pairs to return.
    aggregate : bool
        If True, aggregate confusion counts across experiments.
        If False, return per-experiment confusion rows.

    Returns
    -------
    pd.DataFrame
        Columns:
            - experiment (if aggregate=False)
            - true_class
            - predicted_class
            - count
    """

    experiments = load_all_experiments(results_dir)

    if len(experiments) == 0:
        raise ValueError(f"No experiments found in {results_dir}")

    validate_class_consistency(experiments)

    all_rows = []

    for exp in experiments:
        _validate_confusion_matrix(exp)

        exp_name = exp.get("experiment", exp.get("__exp_name__"))

        cm = np.array(exp["confusion_matrix"]["matrix"])
        class_names = exp["confusion_matrix"]["class_names"]

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i == j:
                    continue

                count = cm[i, j]

                if count > 0:
                    row = {
                        "true_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(count),
                    }

                    if not aggregate:
                        row["experiment"] = exp_name

                    all_rows.append(row)

    if len(all_rows) == 0:
        return pd.DataFrame(
            columns=["experiment", "true_class", "predicted_class", "count"]
            if not aggregate
            else ["true_class", "predicted_class", "count"]
        )

    df = pd.DataFrame(all_rows)

    if aggregate:
        df = (
            df
            .groupby(["true_class", "predicted_class"], as_index=False)
            .sum()
        )

    df = (
        df
        .sort_values(by="count", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    return df
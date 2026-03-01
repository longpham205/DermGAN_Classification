# src/analysis/performance/per_class_analysis.py

from pathlib import Path
from typing import List
import pandas as pd

from src.analysis.core.load_results import load_all_experiments
from src.analysis.core.metrics_utils import validate_class_consistency


def build_per_class_table(
    results_dir: Path,
    metric: str = "f1",
    sort_by_mean: bool = False
) -> pd.DataFrame:
    """
    Build per-class comparison table across experiments.

    Parameters
    ----------
    results_dir : Path
        Path to results directory.
    metric : str
        Per-class metric name (e.g., 'f1', 'precision', 'recall').
    sort_by_mean : bool
        If True, sort experiments by mean class score (descending).

    Returns
    -------
    pd.DataFrame
        Table with shape (num_experiments, num_classes + 1)
    """

    experiments = load_all_experiments(results_dir)

    if len(experiments) == 0:
        raise ValueError(f"No experiments found in {results_dir}")

    validate_class_consistency(experiments)

    class_names: List[str] = experiments[0]["dataset"]["class_names"]

    rows = []

    for exp in experiments:
        exp_name = exp.get("experiment", exp.get("__exp_name__"))

        if "per_class" not in exp:
            raise ValueError(f"'per_class' not found in experiment {exp_name}")

        row = {"experiment": exp_name}

        for cls in class_names:
            if cls not in exp["per_class"]:
                raise ValueError(
                    f"Class '{cls}' missing in per_class of experiment {exp_name}"
                )

            if metric not in exp["per_class"][cls]:
                raise ValueError(
                    f"Metric '{metric}' missing for class '{cls}' in experiment {exp_name}"
                )

            row[cls] = exp["per_class"][cls][metric]

        rows.append(row)

    df = pd.DataFrame(rows)

    if sort_by_mean:
        df["__mean__"] = df.drop(columns=["experiment"]).mean(axis=1)
        df = df.sort_values("__mean__", ascending=False).drop(columns="__mean__")

    return df


def find_worst_classes(
    results_dir: Path,
    metric: str = "f1"
) -> pd.DataFrame:
    """
    Identify weakest class per experiment.

    Parameters
    ----------
    results_dir : Path
        Path to results directory.
    metric : str
        Metric used to determine worst class.

    Returns
    -------
    pd.DataFrame
        Columns:
            - experiment
            - worst_class
            - metric
    """

    df = build_per_class_table(results_dir, metric)

    worst_rows = []

    for _, row in df.iterrows():
        experiment = row["experiment"]
        class_scores = row.drop(labels=["experiment"])

        worst_class = class_scores.idxmin()
        worst_score = class_scores.min()

        worst_rows.append({
            "experiment": experiment,
            "worst_class": worst_class,
            metric: worst_score
        })

    result_df = pd.DataFrame(worst_rows)
    result_df = result_df.sort_values(metric, ascending=True).reset_index(drop=True)

    return result_df
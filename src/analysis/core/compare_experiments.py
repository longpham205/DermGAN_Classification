# src/analysis/core/compare_experiments.py

from typing import List
import pandas as pd

from .metrics_utils import extract_global_metric, validate_class_consistency
from .load_results import ExperimentResult


# =========================================
# Compare by single metric
# =========================================

def compare_by_metric(
    experiments: List[ExperimentResult],
    metric_name: str = "accuracy",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Compare experiments by a specific metric.

    Parameters
    ----------
    experiments : List[ExperimentResult]
    metric_name : str
        Metric to compare (e.g., 'accuracy', 'f1-score', 'macro_auc')
    ascending : bool
        Sort direction
    """

    validate_class_consistency(experiments)

    rows = []

    for exp in experiments:
        try:
            metric_value = extract_global_metric(exp, metric_name)
        except ValueError:
            # Skip experiments missing this metric
            continue

        row = {
            "experiment": exp.exp_name,
            metric_name: metric_value,
        }

        # Optional: best_epoch if exists in metrics
        if "best_epoch" in exp.metrics:
            row["best_epoch"] = exp.metrics["best_epoch"]

        rows.append(row)

    if len(rows) == 0:
        raise ValueError(f"No experiments contain metric '{metric_name}'")

    df = pd.DataFrame(rows)
    df = df.sort_values(by=metric_name, ascending=ascending).reset_index(drop=True)

    return df


# =========================================
# Compare all scalar metrics
# =========================================

def compare_all_global_metrics(
    experiments: List[ExperimentResult],
) -> pd.DataFrame:
    """
    Build comparison table including all scalar metrics
    found at top-level of metrics.json.
    """

    validate_class_consistency(experiments)

    rows = []

    for exp in experiments:
        row = {
            "experiment": exp.exp_name,
        }

        for key, value in exp.metrics.items():
            if isinstance(value, (int, float)):
                row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No scalar metrics found across experiments.")

    return df
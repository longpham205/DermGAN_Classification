# src/analysis/core/metrics_utils.py

from typing import List
from pathlib import Path

from .load_results import ExperimentResult


# =====================================
# Metric Extraction
# =====================================

def extract_global_metric(exp: ExperimentResult, metric_name: str) -> float:
    """
    Extract a global metric from metrics.json.

    Supports:
        - Top-level metrics (e.g., "accuracy")
        - Nested metrics (e.g., macro avg -> f1-score)
    """

    metrics = exp.metrics

    # Case 1: direct key
    if metric_name in metrics:
        return metrics[metric_name]

    # Case 2: macro avg or weighted avg
    for key in ["macro avg", "weighted avg"]:
        if key in metrics and metric_name in metrics[key]:
            return metrics[key][metric_name]

    raise ValueError(
        f"Metric '{metric_name}' not found in experiment '{exp.exp_name}'."
    )


# =====================================
# Class Consistency Validation
# =====================================

def validate_class_consistency(experiments: List[ExperimentResult]) -> None:
    """
    Ensure all experiments share identical class ordering.
    This is critical for:
        - Confusion matrix comparison
        - McNemar test
        - Bootstrap CI
    """

    if len(experiments) == 0:
        return

    reference_exp = experiments[0]

    if reference_exp.confusion_matrix is None:
        raise ValueError(
            f"Experiment '{reference_exp.exp_name}' missing confusion_matrix.json"
        )

    reference_classes = reference_exp.confusion_matrix.get("class_names")

    for exp in experiments[1:]:
        if exp.confusion_matrix is None:
            raise ValueError(
                f"Experiment '{exp.exp_name}' missing confusion_matrix.json"
            )

        current_classes = exp.confusion_matrix.get("class_names")

        if current_classes != reference_classes:
            raise ValueError(
                "Class name mismatch between experiments.\n"
                f"Expected: {reference_classes}\n"
                f"Got: {current_classes}"
            )
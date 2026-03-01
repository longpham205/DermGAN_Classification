# src/analysis/statistics/significance_report.py

"""
Statistical Significance Report
================================

Includes:
- McNemar test (paired prediction disagreement)
- Bootstrap confidence intervals
- Effect size (accuracy difference)

Outputs:
- significance_report.json
- significance_report.txt
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..core.load_results import ExperimentResult
from .mcnemar_test import run_mcnemar_test
from .bootstrap_ci import bootstrap_confidence_interval


# ==========================================================
# Helpers
# ==========================================================

REQUIRED_COLUMNS = {"image_id", "y_true", "y_pred"}


def _load_predictions(exp: ExperimentResult) -> pd.DataFrame:
    """
    Load predictions.csv safely from ExperimentResult.
    """
    if exp.predictions_path is None:
        raise ValueError(
            f"Experiment '{exp.exp_name}' does not contain predictions.csv"
        )

    df = pd.read_csv(exp.predictions_path)

    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing = REQUIRED_COLUMNS - set(df.columns)
        raise ValueError(
            f"{exp.exp_name} predictions.csv missing columns: {missing}"
        )

    return df[list(REQUIRED_COLUMNS)].copy()


def compute_basic_metrics(y_true, y_pred) -> Dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro")
    }


def compute_effect_size(acc1, acc2):
    """
    Absolute accuracy difference (Model2 - Model1).
    """
    return acc2 - acc1


# ==========================================================
# Main function
# ==========================================================

def run_significance_tests(
    exp1: ExperimentResult,
    exp2: ExperimentResult,
    output_dir: Path
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning significance tests: {exp1.exp_name} vs {exp2.exp_name}")

    # ======================================================
    # Load predictions
    # ======================================================

    df1 = _load_predictions(exp1)
    df2 = _load_predictions(exp2)

    # Align by image_id
    merged = df1.merge(
        df2,
        on="image_id",
        suffixes=("_1", "_2")
    )

    if len(merged) == 0:
        raise ValueError("No overlapping samples between experiments.")

    # Ensure ground truth consistency
    if not (merged["y_true_1"] == merged["y_true_2"]).all():
        raise ValueError("Ground truth mismatch between experiments.")

    y_true = merged["y_true_1"].values
    y_pred1 = merged["y_pred_1"].values
    y_pred2 = merged["y_pred_2"].values

    # ======================================================
    # 1️⃣ Basic metrics
    # ======================================================

    metrics1 = compute_basic_metrics(y_true, y_pred1)
    metrics2 = compute_basic_metrics(y_true, y_pred2)

    # ======================================================
    # 2️⃣ McNemar Test (paired)
    # ======================================================

    mcnemar_result = run_mcnemar_test(
        exp1.exp_dir,
        exp2.exp_dir
    )

    # ======================================================
    # 3️⃣ Bootstrap CI
    # ======================================================

    acc_ci_1 = bootstrap_confidence_interval(
        y_true=y_true,
        y_pred=y_pred1,
        metric="accuracy",
        stratified=True
    )

    acc_ci_2 = bootstrap_confidence_interval(
        y_true=y_true,
        y_pred=y_pred2,
        metric="accuracy",
        stratified=True
    )

    f1_ci_1 = bootstrap_confidence_interval(
        y_true=y_true,
        y_pred=y_pred1,
        metric="f1_macro",
        stratified=True
    )

    f1_ci_2 =  bootstrap_confidence_interval(
        y_true=y_true,
        y_pred=y_pred2,
        metric="f1_macro",
        stratified=True
    )

    # ======================================================
    # 4️⃣ Effect Size
    # ======================================================

    effect_size = compute_effect_size(
        metrics1["accuracy"],
        metrics2["accuracy"]
    )

    # ======================================================
    # 5️⃣ Interpretation
    # ======================================================

    if mcnemar_result["p_value"] < 0.05:
        significance_statement = "Statistically significant difference (p < 0.05)"
    else:
        significance_statement = "No statistically significant difference"

    # ======================================================
    # 6️⃣ Final Report
    # ======================================================

    report = {
        "experiment_1": exp1.exp_name,
        "experiment_2": exp2.exp_name,
        "model_1": {
            "accuracy": metrics1["accuracy"],
            "accuracy_ci_95": acc_ci_1,
            "f1_macro": metrics1["f1_macro"],
            "f1_macro_ci_95": f1_ci_1
        },
        "model_2": {
            "accuracy": metrics2["accuracy"],
            "accuracy_ci_95": acc_ci_2,
            "f1_macro": metrics2["f1_macro"],
            "f1_macro_ci_95": f1_ci_2
        },
        "effect_size_accuracy": effect_size,
        "mcnemar": mcnemar_result,
        "interpretation": significance_statement
    }

    # Save JSON
    with open(output_dir / "significance_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # Save TXT summary
    txt_report = generate_text_report(report)

    with open(output_dir / "significance_report.txt", "w", encoding="utf-8") as f:
        f.write(txt_report)

    print("Significance report saved.\n")

    return report


# ==========================================================
# Text Summary (Paper-ready)
# ==========================================================

def generate_text_report(report):

    txt = []
    txt.append("=" * 60)
    txt.append("Statistical Significance Report")
    txt.append("=" * 60)
    txt.append("")

    txt.append(f"Comparison: {report['experiment_1']} vs {report['experiment_2']}")
    txt.append("")

    txt.append("Model 1:")
    txt.append(f"  Accuracy: {report['model_1']['accuracy']:.4f}")
    txt.append(f"  95% CI: {report['model_1']['accuracy_ci_95']}")
    txt.append(f"  F1-macro: {report['model_1']['f1_macro']:.4f}")
    txt.append("")

    txt.append("Model 2:")
    txt.append(f"  Accuracy: {report['model_2']['accuracy']:.4f}")
    txt.append(f"  95% CI: {report['model_2']['accuracy_ci_95']}")
    txt.append(f"  F1-macro: {report['model_2']['f1_macro']:.4f}")
    txt.append("")

    txt.append(
        f"Effect size (Accuracy difference): {report['effect_size_accuracy']:.4f}"
    )
    txt.append("")

    txt.append("McNemar Test:")
    txt.append(f"  Statistic: {report['mcnemar']['statistic']:.4f}")
    txt.append(f"  p-value: {report['mcnemar']['p_value']:.6f}")
    txt.append("")

    txt.append("Conclusion:")
    txt.append(report["interpretation"])
    txt.append("")

    return "\n".join(txt)
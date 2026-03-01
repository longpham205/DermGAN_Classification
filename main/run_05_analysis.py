# main/run_05_analysis.py

"""
DermGAN Research Analysis Pipeline
==================================

Research-grade evaluation pipeline for:

1. Overall performance (from metrics.json)
2. Per-class analysis (from classification_report.json)
3. Confusion matrix
4. Statistical significance testing (if predictions.csv exists)
5. Representation analysis (optional)
6. Clinical ABCD evaluation (optional)

Author: DermGAN Classification
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict

# ==================================================
# Project root & imports
# ==================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# ==============================
# Core
# ==============================
from src.analysis.core.load_results import (
    load_experiment,
    ExperimentResult
)
from src.analysis.core.compare_experiments import compare_all_global_metrics

# ==============================
# Statistics
# ==============================
from src.analysis.statistics.significance_report import run_significance_tests

# ==============================
# Representation
# ==============================
from src.analysis.representation.embedding_analysis import run_embedding_analysis

# ==============================
# Clinical
# ==============================
from src.analysis.clinical.abcd_analysis import run_abcd_error_analysis


# ==========================================================
# Utilities
# ==========================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def validate_experiment_dir(exp_dir: Path):
    if not (exp_dir / "metrics.json").exists():
        raise FileNotFoundError(
            f"Missing metrics.json in {exp_dir}"
        )


# ==========================================================
# Performance Summary Extraction
# ==========================================================

def extract_performance_summary(exp: ExperimentResult) -> Dict:
    """
    Extract key metrics from metrics.json
    """

    metrics = exp.metrics

    return {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "auc_macro": metrics.get("auc_macro"),
    }


# ==========================================================
# Main
# ==========================================================

def main(args):

    print("=" * 60)
    print("DermGAN Research Analysis Pipeline")
    print("=" * 60)

    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    experiments: Dict[str, ExperimentResult] = {}
    performance_results = {}

    # ======================================================
    # 1️⃣ Load Experiments
    # ======================================================

    for exp_path in args.experiments:

        exp_dir = Path(exp_path)
        exp_name = exp_dir.name

        validate_experiment_dir(exp_dir)

        exp = load_experiment(exp_dir.parent, exp_name)
        experiments[exp_name] = exp

        print(f"Loaded experiment: {exp_name}")

    # ======================================================
    # 2️⃣ Save & Summarize Performance
    # ======================================================

    for name, exp in experiments.items():

        exp_output = output_root / name
        ensure_dir(exp_output)

        # Save raw files again for reproducibility
        with open(exp_output / "metrics.json", "w") as f:
            json.dump(exp.metrics, f, indent=4)

        if exp.confusion_matrix:
            with open(exp_output / "confusion_matrix.json", "w") as f:
                json.dump(exp.confusion_matrix, f, indent=4)

        if exp.classification_report:
            with open(exp_output / "classification_report.json", "w") as f:
                json.dump(exp.classification_report, f, indent=4)

        performance_results[name] = extract_performance_summary(exp)

    # ======================================================
    # 3️⃣ Compare Experiments
    # ======================================================

    if len(experiments) > 1:

        print("\nComparing experiments...")

        exp_list = list(experiments.values())

        comparison_df = compare_all_global_metrics(exp_list)

        comparison_df.to_csv(
            output_root / "experiment_comparison.csv",
            index=False
        )

    # ======================================================
    # 4️⃣ Statistical Significance (if predictions exist)
    # ======================================================

    if len(experiments) == 2:

        exp_list = list(experiments.values())
        exp1, exp2 = exp_list[0], exp_list[1]

        if exp1.predictions_path and exp2.predictions_path:

            print("\nRunning statistical significance tests...")

            run_significance_tests(
                exp1,
                exp2,
                output_root
            )
        else:
            print("Skipping statistical tests (predictions.csv missing).")

    # ======================================================
    # 5️⃣ Representation (Optional)
    # ======================================================

    if args.run_representation:

        for name, exp in experiments.items():

            if not (exp.exp_dir / "embeddings.npy").exists():
                print(f"No embeddings found for {name}. Skipping.")
                continue

            print(f"\nRunning embedding analysis: {name}")

            stats = run_embedding_analysis(exp.exp_dir)

            exp_output = output_root / name
            with open(exp_output / "embedding_analysis.json", "w") as f:
                json.dump(stats, f, indent=4)

    # ======================================================
    # 6️⃣ Clinical ABCD (Optional)
    # ======================================================

    if args.run_clinical:

        if args.metadata_path is None:
            raise ValueError(
                "--metadata_path is required when --run_clinical is used."
            )

        metadata_path = Path(os.path.join(ROOT_DIR, args.metadata_path))

        for name, exp in experiments.items():

            print(f"\nRunning ABCD analysis: {name}")

            stats = run_abcd_error_analysis(
                exp.exp_dir,
                metadata_path
            )

            exp_output = output_root / name
            with open(exp_output / "abcd_analysis.json", "w") as f:
                json.dump(stats, f, indent=4)

    print("\nAnalysis complete.")
    print(f"Results saved to: {output_root}")


# ==========================================================
# CLI
# ==========================================================

def run_analysis():

    parser = argparse.ArgumentParser(
        description="DermGAN Research Analysis Pipeline"
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Paths to experiment folders (each must contain metrics.json)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Directory to save results"
    )

    parser.add_argument(
        "--run_representation",
        action="store_true",
        help="Run embedding analysis"
    )

    parser.add_argument(
        "--run_clinical",
        action="store_true",
        help="Run ABCD clinical analysis"
    )

    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Metadata CSV path (required if --run_clinical)"
    )

    if len(sys.argv) == 1:
        # ===== DEV MODE (no CLI args provided) =====
        args = parser.parse_args([
            "--experiments",
            "results/exp_01_baseline",
            "results/exp_02_base_aug",
            #"results/exp_03_gan",
            # "--run_clinical",
            "--metadata_path",
            "data/HAM10000/metadata.csv",
            "--output_dir",
            "results/analysis_results"
        ])
    else:
        # ===== NORMAL CLI MODE =====
        args = parser.parse_args()

    main(args)
    
if __name__ == "__main__":
    run_analysis()
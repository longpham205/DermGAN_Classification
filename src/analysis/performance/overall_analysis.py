# src/analysis/performance/overall_analysis.py

from pathlib import Path
from typing import Optional
import pandas as pd

from src.analysis.core.load_results import load_all_experiments, ExperimentResult
from src.analysis.core.compare_experiments import compare_by_metric
from src.analysis.core.metrics_utils import extract_global_metric


def run_overall_analysis(
    results_dir: Path,
    metric_name: str = "accuracy",
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare experiments by selected global metric
    and rank them in descending order.
    """

    print(f"\n[INFO] Loading experiments from: {results_dir}")

    experiments = load_all_experiments(results_dir)

    print(f"[INFO] Loaded {len(experiments)} experiments.")

    # Validate metric exists in at least one experiment
    valid_count = 0
    for exp in experiments:
        try:
            extract_global_metric(exp, metric_name)
            valid_count += 1
        except ValueError:
            continue

    if valid_count == 0:
        raise ValueError(
            f"Metric '{metric_name}' not found in any experiment."
        )

    if valid_count < len(experiments):
        print(
            f"[WARNING] Metric '{metric_name}' missing in "
            f"{len(experiments) - valid_count} experiment(s). "
            "They will be skipped."
        )

    df = compare_by_metric(
        experiments,
        metric_name=metric_name,
        ascending=False,
    )

    print("\n=== Overall Experiment Ranking ===")
    print(df)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[INFO] Saved ranking to: {save_path}")

    return df
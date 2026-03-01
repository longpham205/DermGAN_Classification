import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


# ================================
# Data Structure
# ================================

@dataclass
class ExperimentResult:
    exp_name: str
    exp_dir: Path
    metrics: Dict[str, Any]                  # global metrics only
    raw_metrics: Dict[str, Any]              # full metrics.json
    confusion_matrix: Optional[Dict] = None
    classification_report: Optional[Dict] = None
    predictions_path: Optional[Path] = None


# ================================
# Internal Helpers
# ================================

def _safe_load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load {path}: {e}")
        return None


def _extract_global_metrics(raw_metrics: Dict, exp_name: str) -> Dict:
    """
    Extract global metrics in a schema-robust way.

    Supported formats:
    1) Flat:
        { "accuracy": 0.8, ... }

    2) Structured:
        {
            "global_metrics": {...},
            ...
        }
    """

    # Structured format (recommended)
    if "global_metrics" in raw_metrics:
        global_metrics = raw_metrics["global_metrics"]

        if not isinstance(global_metrics, dict):
            raise ValueError(
                f"'global_metrics' in experiment '{exp_name}' must be a dict."
            )

        return global_metrics

    # Backward compatibility (flat format)
    warnings.warn(
        f"Experiment '{exp_name}' uses flat metrics format. "
        f"Consider migrating to 'global_metrics' schema."
    )
    return raw_metrics


def _validate_metrics(metrics: Dict, exp_name: str):
    """
    Minimal validation.
    Only enforce essential metric(s).
    """

    required_keys = ["accuracy"]

    for key in required_keys:
        if key not in metrics:
            raise ValueError(
                f"Experiment '{exp_name}' missing required metric: {key}"
            )

    if not isinstance(metrics["accuracy"], (int, float)):
        raise ValueError(
            f"Experiment '{exp_name}' has invalid accuracy value."
        )


# ================================
# Public API
# ================================

def load_experiment(results_dir: Path, exp_name: str) -> ExperimentResult:
    """
    Load a single experiment folder.

    Expected files:
        - metrics.json (required)
        - confusion_matrix.json (optional)
        - classification_report.json (optional)
        - predictions.csv (optional)
    """

    exp_dir = results_dir / exp_name

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # ----------------------------
    # Load metrics.json
    # ----------------------------
    metrics_path = exp_dir / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {exp_dir}")

    raw_metrics = _safe_load_json(metrics_path)

    if raw_metrics is None:
        raise ValueError(f"metrics.json corrupted in '{exp_name}'")

    global_metrics = _extract_global_metrics(raw_metrics, exp_name)

    _validate_metrics(global_metrics, exp_name)

    # ----------------------------
    # Optional files
    # ----------------------------
    confusion_matrix = _safe_load_json(exp_dir / "confusion_matrix.json")
    classification_report = _safe_load_json(exp_dir / "classification_report.json")

    # If confusion_matrix embedded inside metrics.json
    if confusion_matrix is None and "confusion_matrix" in raw_metrics:
        confusion_matrix = raw_metrics["confusion_matrix"]

    # predictions.csv
    predictions_path = exp_dir / "predictions.csv"
    if not predictions_path.exists():
        predictions_path = None

    return ExperimentResult(
        exp_name=exp_name,
        exp_dir=exp_dir,
        metrics=global_metrics,
        raw_metrics=raw_metrics,
        confusion_matrix=confusion_matrix,
        classification_report=classification_report,
        predictions_path=predictions_path,
    )


def load_all_experiments(results_dir: Path) -> List[ExperimentResult]:
    """
    Load all experiments in results_dir.
    Only folders starting with 'exp_' are considered.
    """

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    experiments: List[ExperimentResult] = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        if not exp_dir.name.startswith("exp_"):
            continue

        try:
            exp = load_experiment(results_dir, exp_dir.name)
            experiments.append(exp)
        except Exception as e:
            warnings.warn(f"Skipping experiment '{exp_dir.name}': {e}")

    if len(experiments) == 0:
        raise ValueError(f"No valid experiments found in {results_dir}")

    return experiments
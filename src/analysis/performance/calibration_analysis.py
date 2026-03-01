# src/analysis/performance/calibration_analysis.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    return_bins: bool = False
) -> Tuple[float, Optional[pd.DataFrame]]:
    """
    Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    confidences : np.ndarray
        Maximum predicted probabilities.
    predictions : np.ndarray
        Predicted class indices.
    labels : np.ndarray
        Ground truth class indices.
    n_bins : int
        Number of confidence bins.
    return_bins : bool
        If True, also return per-bin statistics.

    Returns
    -------
    ece : float
    bin_df : pd.DataFrame (optional)
    """

    if len(confidences) == 0:
        raise ValueError("Empty confidence array.")

    if not (len(confidences) == len(predictions) == len(labels)):
        raise ValueError("Input arrays must have same length.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []

    for i in range(n_bins):
        if i == 0:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])

        bin_size = np.sum(mask)

        if bin_size == 0:
            continue

        acc = np.mean(predictions[mask] == labels[mask])
        conf = np.mean(confidences[mask])

        gap = np.abs(acc - conf)
        weight = bin_size / len(labels)

        ece += gap * weight

        if return_bins:
            bin_rows.append({
                "bin_lower": bins[i],
                "bin_upper": bins[i + 1],
                "count": int(bin_size),
                "accuracy": float(acc),
                "confidence": float(conf),
                "gap": float(gap),
            })

    if return_bins:
        return float(ece), pd.DataFrame(bin_rows)

    return float(ece), None


def run_calibration_analysis(
    exp_dir: Path,
    n_bins: int = 15,
    return_bins: bool = False
) -> Tuple[float, Optional[pd.DataFrame]]:
    """
    Compute ECE for one experiment.

    Requires predictions.csv with:
        - true_label
        - prob_* columns

    Returns
    -------
    ece : float
    bin_df : pd.DataFrame (optional)
    """

    pred_path = exp_dir / "predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {exp_dir}")

    df = pd.read_csv(pred_path)

    if "true_label" not in df.columns:
        raise ValueError("Column 'true_label' not found in predictions.csv")

    prob_cols = [c for c in df.columns if c.startswith("prob_")]

    if len(prob_cols) == 0:
        raise ValueError("No probability columns starting with 'prob_' found.")

    probs = df[prob_cols].values
    labels = df["true_label"].values

    if probs.ndim != 2:
        raise ValueError("Probability array must be 2D.")

    if len(probs) != len(labels):
        raise ValueError("Mismatch between probabilities and labels length.")

    # Optional sanity check (can be removed if too strict)
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Probabilities must be in range [0, 1].")

    row_sums = np.sum(probs, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError("Probabilities do not sum to 1 across classes.")

    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    ece, bin_df = compute_ece(
        confidences,
        predictions,
        labels,
        n_bins=n_bins,
        return_bins=return_bins
    )

    return ece, bin_df
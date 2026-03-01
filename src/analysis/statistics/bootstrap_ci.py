# src/analysis/statistics/bootstrap_ci.py

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import f1_score, roc_auc_score


SUPPORTED_METRICS = {"f1_macro", "accuracy", "macro_auc"}


# ==========================================================
# Metric computation
# ==========================================================

def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    metric: str = "f1_macro"
) -> float:

    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'")

    if metric == "f1_macro":
        return f1_score(y_true, y_pred, average="macro")

    elif metric == "accuracy":
        return float(np.mean(y_true == y_pred))

    elif metric == "macro_auc":

        if y_prob is None:
            raise ValueError("macro_auc requires probability estimates.")

        if len(np.unique(y_true)) < 2:
            raise ValueError("AUC undefined for single-class sample.")

        return roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        )


# ==========================================================
# Bootstrap CI (clean version – no filesystem dependency)
# ==========================================================

def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    metric: str = "f1_macro",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = 42,
    stratified: bool = False,
) -> Dict:
    """
    Compute bootstrap confidence interval for selected metric.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    y_prob : np.ndarray or None
    metric : str
    n_bootstrap : int
    alpha : float
    random_state : int or None
    stratified : bool

    Returns
    -------
    dict with:
        metric, mean, ci_lower, ci_upper, ci_width
    """

    if len(y_true) == 0:
        raise ValueError("Empty dataset.")

    rng = np.random.default_rng(random_state)
    scores = []
    n = len(y_true)

    for _ in range(n_bootstrap):

        if stratified:
            indices = []
            for cls in np.unique(y_true):
                cls_idx = np.where(y_true == cls)[0]
                sampled = rng.choice(cls_idx, size=len(cls_idx), replace=True)
                indices.extend(sampled)
            idx = np.array(indices)
        else:
            idx = rng.choice(n, n, replace=True)

        try:
            score = compute_metric(
                y_true=y_true[idx],
                y_pred=y_pred[idx],
                y_prob=y_prob[idx] if y_prob is not None else None,
                metric=metric
            )
            scores.append(score)

        except ValueError:
            # Skip invalid bootstrap sample (e.g., single-class AUC)
            continue

    if len(scores) == 0:
        raise RuntimeError("All bootstrap samples failed.")

    scores = np.array(scores)

    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return {
        "metric": metric,
        "mean": float(np.mean(scores)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_width": float(upper - lower),
        "n_successful_bootstrap": int(len(scores))
    }
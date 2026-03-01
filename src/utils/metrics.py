# src/utils/metrics.py
"""
Evaluation metrics for skin lesion classification.

Designed for imbalanced medical image datasets.
Fully compatible with train_with_gan.py
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    return_confusion: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray]]:

    metrics: Dict[str, float] = {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = len(np.unique(y_true))

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # ==================================================
    # Global metrics
    # ==================================================
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    metrics["precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision_weighted"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics["recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall_weighted"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics["f1_macro"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["f1_weighted"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # ==================================================
    # Per-class metrics  ✅ FIX hoàn toàn KeyError
    # ==================================================
    per_class: Dict[str, Dict[str, float]] = {}

    precisions = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    recalls = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    f1s = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )

    for i, cls_name in enumerate(class_names):
        per_class[cls_name] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
        }

    metrics["per_class"] = per_class

    # ==================================================
    # AUC
    # ==================================================
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                metrics["auc_macro"] = roc_auc_score(
                    y_true, y_prob[:, 1]
                )
            else:
                metrics["auc_macro"] = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                )
        except ValueError:
            metrics["auc_macro"] = float("nan")

    # ==================================================
    # Binary-only metrics
    # ==================================================
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["specificity"] = tn / (tn + fp + 1e-8)
        metrics["sensitivity"] = tp / (tp + fn + 1e-8)

    # ==================================================
    # Confusion matrix
    # ==================================================
    if return_confusion:
        cm = confusion_matrix(y_true, y_pred)
        return metrics, cm

    return metrics

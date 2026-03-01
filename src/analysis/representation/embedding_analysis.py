# src/analysis/representation/embedding_analysis.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Literal
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


def load_embeddings(exp_dir: Path):
    """
    Load embeddings.npy and corresponding labels from predictions.csv
    """

    emb_path = exp_dir / "embeddings.npy"
    pred_path = exp_dir / "predictions.csv"

    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.npy not found in {exp_dir}")

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {exp_dir}")

    embeddings = np.load(emb_path)
    df = pd.read_csv(pred_path)

    if "y_true" not in df.columns:
        raise ValueError("Column 'y_true' missing in predictions.csv")

    labels = df["y_true"].values

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D array.")

    if embeddings.shape[0] != len(labels):
        raise ValueError(
            "Mismatch between number of embeddings and labels."
        )

    return embeddings, labels


def compute_intra_inter_distance(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: Literal["euclidean", "cosine"] = "euclidean"
) -> Dict[str, float]:
    """
    Compute average intra-class and inter-class distances.
    """

    if metric == "euclidean":
        dist_matrix = euclidean_distances(embeddings)
    elif metric == "cosine":
        dist_matrix = cosine_distances(embeddings)
    else:
        raise ValueError("Unsupported distance metric.")

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError("At least 2 classes required.")

    intra_distances = []
    inter_distances = []

    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        other = np.where(labels != lbl)[0]

        if len(idx) > 1:
            intra = dist_matrix[np.ix_(idx, idx)]

            # remove diagonal
            intra_no_diag = intra[~np.eye(len(idx), dtype=bool)]
            intra_distances.append(np.mean(intra_no_diag))

        if len(other) > 0:
            inter = dist_matrix[np.ix_(idx, other)]
            inter_distances.append(np.mean(inter))

    intra_mean = float(np.mean(intra_distances))
    inter_mean = float(np.mean(inter_distances))

    return {
        "intra_class_distance": intra_mean,
        "inter_class_distance": inter_mean,
        "separation_ratio": inter_mean / intra_mean
    }


def compute_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: Literal["euclidean", "cosine"] = "euclidean"
) -> float:
    """
    Compute silhouette score safely.
    """

    if len(np.unique(labels)) < 2:
        return 0.0

    return float(silhouette_score(embeddings, labels, metric=metric))


def run_embedding_analysis(
    exp_dir: Path,
    metric: Literal["euclidean", "cosine"] = "euclidean"
) -> Dict[str, float]:
    """
    Full embedding quality analysis.

    Returns:
        - intra_class_distance
        - inter_class_distance
        - separation_ratio
        - silhouette_score
    """

    embeddings, labels = load_embeddings(exp_dir)

    stats = compute_intra_inter_distance(
        embeddings,
        labels,
        metric=metric
    )

    sil = compute_silhouette(
        embeddings,
        labels,
        metric=metric
    )

    stats["silhouette_score"] = sil

    return stats
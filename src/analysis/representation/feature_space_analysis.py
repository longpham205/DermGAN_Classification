# src/analysis/representation/feature_space_analysis.py

import numpy as np
from pathlib import Path
from typing import Dict, Literal, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_embeddings(exp_dir: Path) -> np.ndarray:
    emb_path = exp_dir / "embeddings.npy"

    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.npy not found in {exp_dir}")

    embeddings = np.load(emb_path)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if embeddings.shape[0] < 2:
        raise ValueError("At least 2 samples required.")

    return embeddings


def compute_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
    standardize: bool = True
) -> Dict:
    """
    Compute PCA projection.
    """

    if standardize:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)

    return {
        "reduced": reduced,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_explained_variance": float(
            np.sum(pca.explained_variance_ratio_)
        )
    }


def compute_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    subsample: Optional[int] = None,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute t-SNE projection.

    Parameters
    ----------
    subsample : int or None
        If provided, randomly subsample embeddings before t-SNE.
    """

    n_samples = embeddings.shape[0]

    if subsample is not None and subsample < n_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, subsample, replace=False)
        embeddings = embeddings[idx]
        n_samples = subsample

    if perplexity >= n_samples:
        raise ValueError(
            "Perplexity must be less than number of samples."
        )

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca"
    )

    reduced = tsne.fit_transform(embeddings)

    return reduced


def run_feature_space_analysis(
    exp_dir: Path,
    use_tsne: bool = True,
    tsne_subsample: Optional[int] = None
) -> Dict:
    """
    Run PCA and optionally t-SNE analysis.

    Returns:
        {
            "pca": {...},
            "tsne": array (optional)
        }
    """

    embeddings = load_embeddings(exp_dir)

    results = {}

    pca_result = compute_pca(embeddings)
    results["pca"] = pca_result

    if use_tsne:
        tsne_result = compute_tsne(
            embeddings,
            subsample=tsne_subsample
        )
        results["tsne"] = tsne_result

    return results
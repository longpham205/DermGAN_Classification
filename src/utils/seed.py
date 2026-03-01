# utils/seed.py
"""
Utility functions for reproducibility.

This module enforces deterministic behavior across Python, NumPy, and PyTorch
to support reproducible experiments, which is critical for scientific research
and fair comparison between models.
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seed for full reproducibility.

    Args:
        seed (int): Random seed value (must be provided explicitly).
        deterministic (bool): If True, enforce deterministic behavior.
                              This may reduce training speed.
    """
    if seed is None:
        raise ValueError("Seed must be explicitly provided for reproducibility.")

    # =========================
    # Python & NumPy
    # =========================
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # =========================
    # PyTorch (CPU & CUDA)
    # =========================
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # =========================
    # Deterministic settings
    # =========================
    if deterministic:
        # cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enforce deterministic algorithms (PyTorch >= 1.8)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        # Recommended by PyTorch for full CUDA determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # =========================
    # Logging (important for papers)
    # =========================
    logging.info(
        f"Reproducibility set | seed={seed}, deterministic={deterministic}"
    )


def seed_worker(worker_id: int):
    """
    Initialize random seed for PyTorch DataLoader workers.

    Each worker is seeded based on the initial seed set by PyTorch to ensure:
    - Deterministic behavior across runs
    - Different randomness across workers

    This function should be passed to:
        DataLoader(worker_init_fn=seed_worker)
    """
    # torch.initial_seed() is set by the DataLoader using the global seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

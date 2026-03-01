# preprocessing/color_constancy.py
"""
Color constancy using Shade-of-Gray algorithm.

WARNING:
- This method may alter diagnostically relevant color cues.
- It should be evaluated via ablation study.
"""

import cv2
import numpy as np


def shade_of_gray(
    img: np.ndarray,
    power: int = 6,
    to_rgb: bool = True,
    eps: float = 1e-6,
    output_dtype: str = "float32",
):
    """
    Apply Shade-of-Gray color constancy.

    Args:
        img (np.ndarray): Input image (H, W, C).
        power (int): Power parameter (typically 4-10).
        to_rgb (bool): Convert BGR -> RGB before processing.
        eps (float): Numerical stability term.
        output_dtype (str): 'float32' or 'uint8'.

    Returns:
        np.ndarray: Color-normalized image.
    """

    if img is None:
        raise ValueError("Input image is None")

    # Handle grayscale
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Handle alpha channel
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # Convert color space
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)

    # Shade-of-Gray computation
    img_power = np.power(img, power)
    mean_rgb = np.mean(img_power, axis=(0, 1))
    mean_rgb = np.power(mean_rgb, 1.0 / power)

    # Prevent division by zero
    mean_rgb = np.maximum(mean_rgb, eps)

    img = img / mean_rgb

    # Rescale to original dynamic range
    img = img * np.mean(mean_rgb)
    img = np.clip(img, 0, 255)

    if output_dtype == "uint8":
        img = img.astype(np.uint8)
    elif output_dtype == "float32":
        pass
    else:
        raise ValueError("output_dtype must be 'float32' or 'uint8'")

    return img

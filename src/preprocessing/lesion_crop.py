# preprocessing/lesion_crop.py
"""
Simple center-based lesion cropping.

NOTE:
- This is NOT a true lesion segmentation.
- It assumes lesions are approximately centered.
- Used as a lightweight preprocessing baseline.
"""

import cv2
import numpy as np


def center_crop(
    img: np.ndarray,
    ratio: float = 0.9,
    min_size: int = 32,
):
    """
    Center crop an image by a given ratio.

    Args:
        img (np.ndarray): Input image (H, W, C).
        ratio (float): Fraction of image to keep (0 < ratio <= 1).
        min_size (int): Minimum allowed crop size.

    Returns:
        np.ndarray: Cropped image.
    """

    if img is None:
        raise ValueError("Input image is None")

    # Handle grayscale
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Handle alpha channel
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    if not (0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    h, w, _ = img.shape

    new_h = max(int(h * ratio), min_size)
    new_w = max(int(w * ratio), min_size)

    # Safety clamp
    new_h = min(new_h, h)
    new_w = min(new_w, w)

    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2

    cropped = img[
        start_y : start_y + new_h,
        start_x : start_x + new_w
    ]

    return cropped

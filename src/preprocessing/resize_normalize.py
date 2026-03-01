# preprocessing/resize_normalize.py
"""
Resize and normalize image for classifier / GAN pipeline.

Design choices:
- Explicit color space handling (BGR -> RGB)
- Explicit interpolation method
- Optional ImageNet normalization
"""

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_and_normalize(
    img: np.ndarray,
    size: tuple = (224, 224),
    normalize: bool = True,
    imagenet_norm: bool = False,
    to_rgb: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
):
    """
    Resize and normalize an image.

    Args:
        img (np.ndarray): Input image (H, W, C) in BGR or RGB format.
        size (tuple): Target size (width, height).
        normalize (bool): Scale pixel values to [0, 1].
        imagenet_norm (bool): Apply ImageNet mean/std normalization.
        to_rgb (bool): Convert BGR -> RGB.
        interpolation (int): OpenCV interpolation flag.

    Returns:
        np.ndarray: Processed image (H, W, C), dtype float32.
    """

    if img is None:
        raise ValueError("Input image is None")

    # Handle grayscale images
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Handle alpha channel
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # Convert BGR to RGB if needed
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, size, interpolation=interpolation)

    img = img.astype(np.float32)

    if normalize:
        img /= 255.0

    if imagenet_norm:
        if not normalize:
            raise ValueError(
                "ImageNet normalization requires normalize=True"
            )
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

    return img

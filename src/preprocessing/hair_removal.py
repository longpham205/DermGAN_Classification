# preprocessing/hair_removal.py
import cv2
import numpy as np


def remove_hair(
    img,
    kernel_size: int = 9,
    threshold: int = 10,
    inpaint_radius: int = 1
):
    """
    Remove hair artifacts using black-hat + inpainting.

    Args:
        img (np.ndarray): BGR image (uint8, 3-channel)
        kernel_size (int): size of structuring element (odd number)
        threshold (int): threshold for hair mask
        inpaint_radius (int): radius for inpainting

    Returns:
        np.ndarray: inpainted BGR image (uint8)
    """

    # -------------------------
    # Safety checks
    # -------------------------
    if img is None:
        raise ValueError("remove_hair(): input image is None")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] != 3:
        raise ValueError(
            f"remove_hair(): expected 3-channel image, got shape {img.shape}"
        )

    # -------------------------
    # Hair detection
    # -------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, mask = cv2.threshold(
        blackhat,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # -------------------------
    # Inpainting
    # -------------------------
    img_inpaint = cv2.inpaint(
        img,
        mask,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA
    )

    return img_inpaint

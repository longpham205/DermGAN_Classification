# src/gan/datasets.py
"""
GAN Dataset for Skin Lesion Image Generation
============================================

Design principles:
- One GAN per class (single-class GAN)
- GAN sees REAL images from TRAIN split only
- No conditional labels (avoid label leakage)
- Preprocessing is GAN-consistent ([-1, 1] normalization)
"""

from pathlib import Path
from typing import List, Dict

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from configs.config import DATASET_ROOT, CLASSES


# ======================================================
# Image Transforms (GAN-CONSISTENT)
# ======================================================
def build_gan_transforms(image_size: int = 128) -> transforms.Compose:
    """
    IMPORTANT:
    - GAN is trained in [-1, 1] space (tanh output)
    - Classifier normalization is applied AFTER generation
    - No data augmentation here
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])


# ======================================================
# GAN Dataset (Single-Class)
# ======================================================
class SkinLesionGANDataset(Dataset):
    """
    Dataset for training a GAN on a SINGLE skin lesion class.

    Data source:
    - dataset/processed/train
    - dataset/splits/train.csv
    """

    def __init__(
        self,
        class_name: str,
        image_size: int = 128,
        split: str = "train",
        return_image_id: bool = False,
    ):
        # -----------------------------
        # Safety checks
        # -----------------------------
        if split != "train":
            raise ValueError("GAN is allowed to use TRAIN split only")

        if class_name not in CLASSES:
            raise ValueError(f"Unknown class: {class_name}")

        self.class_name = class_name
        self.return_image_id = return_image_id

        # -----------------------------
        # Paths
        # -----------------------------
        self.images_dir = DATASET_ROOT / "processed" / split
        self.split_csv = DATASET_ROOT / "splits" / f"{split}.csv"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")

        # -----------------------------
        # Load split metadata
        # -----------------------------
        df = pd.read_csv(self.split_csv)

        required_cols = {"image_id", "dx"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Split CSV must contain columns {required_cols}")

        df = df[df["dx"] == class_name].reset_index(drop=True)

        if df.empty:
            raise RuntimeError(f"No training samples found for class: {class_name}")

        # -----------------------------
        # Resolve image paths
        # -----------------------------
        self.image_paths: List[Path] = []

        for image_id in df["image_id"].tolist():
            for ext in ("", ".jpg", ".png", ".jpeg"):
                candidate = self.images_dir / f"{image_id}{ext}"
                if candidate.exists():
                    self.image_paths.append(candidate)
                    break
            else:
                raise FileNotFoundError(f"Missing image file for ID: {image_id}")

        # -----------------------------
        # Transforms
        # -----------------------------
        self.transform = build_gan_transforms(image_size)

        # -----------------------------
        # Logging
        # -----------------------------
        print(
            f"[GAN DATASET] Class='{class_name}' | "
            f"Images={len(self.image_paths)} | Split=train"
        )

    # --------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    # --------------------------------------------------
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Dummy label for interface compatibility only
        dummy_label = torch.zeros(1, dtype=torch.long)

        if self.return_image_id:
            return image, dummy_label, img_path.name

        return image, dummy_label


# ======================================================
# Helper: Dataset sanity check
# ======================================================
def summarize_gan_dataset(dataset: SkinLesionGANDataset) -> Dict[str, int]:
    """
    Print dataset summary for sanity checking.
    """
    dist = {dataset.class_name: len(dataset)}

    print("\n[GAN DATASET SUMMARY]")
    for cls, count in dist.items():
        print(f"  {cls:<10}: {count}")

    return dist

# src/classifier/datasets_gan.py

"""
Dataset for skin lesion classification with GAN augmentation.

Design:
- Train:
    + Real images (processed/train)
    + Synthetic images (GAN-generated)
- Val / Test:
    + Real images ONLY (no GAN, no leakage)

Paper-safe & comparison-ready.
"""

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from configs.config import CLASSES


# ======================================================
# IMAGE TRANSFORMS
# ======================================================
def build_transforms(mode: str, img_size: int = 224):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


# ======================================================
# DATASET
# ======================================================
class SkinLesionGANDataset(Dataset):
    """
    Train-time dataset with optional GAN augmentation.

    Modes:
    - train : real + synthetic (GAN)
    - val   : real only
    - test  : real only
    """

    def __init__(
        self,
        csv_file: str,
        images_dir: str,
        mode: str = "train",                  # train | val | test
        synthetic_root: Optional[str] = None, # dataset/synthetic
        synthetic_csv: Optional[str] = None,  # synthetic_metadata.csv
        img_size: int = 224,
        return_image_id: bool = False,
    ):
        assert mode in {"train", "val", "test"}

        self.mode = mode
        self.return_image_id = return_image_id
        self.transform = build_transforms(mode, img_size)

        class_to_idx = {c: i for i, c in enumerate(CLASSES)}

        # --------------------------------------------------
        # REAL DATA
        # --------------------------------------------------
        real_root = Path(images_dir) / mode
        real_df = pd.read_csv(csv_file)

        if not {"image_id", "label"}.issubset(real_df.columns):
            raise ValueError(f"{csv_file} must contain image_id, label")

        real_samples = []
        for _, row in real_df.iterrows():
            img_path = real_root / f"{row['image_id']}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(f"Missing real image: {img_path}")

            real_samples.append({
                "path": img_path,
                "label": int(row["label"]),
                "source": "real",
                "image_id": row["image_id"],
            })

        # --------------------------------------------------
        # SYNTHETIC DATA (TRAIN ONLY)
        # --------------------------------------------------
        synthetic_samples = []

        if mode == "train" and synthetic_csv is not None:
            syn_df = pd.read_csv(synthetic_csv)

            if not {"image_path", "label"}.issubset(syn_df.columns):
                raise ValueError(
                    f"{synthetic_csv} must contain image_path, label"
                )

            synthetic_root = Path(synthetic_root) if synthetic_root else None

            for _, row in syn_df.iterrows():
                # ---- path handling (FIX GAN=0 bug) ----
                img_path = Path(str(row["image_path"]).replace("\\", "/"))

                if not img_path.exists() and synthetic_root is not None:
                    img_path = synthetic_root / img_path

                if not img_path.exists():
                    continue  # skip broken synthetic sample

                # ---- label mapping ----
                lbl = row["label"]
                if isinstance(lbl, str):
                    if lbl not in class_to_idx:
                        continue
                    lbl = class_to_idx[lbl]

                synthetic_samples.append({
                    "path": img_path,
                    "label": int(lbl),
                    "source": "gan",
                    "image_id": img_path.stem,
                })

        # --------------------------------------------------
        # MERGE
        # --------------------------------------------------
        self.samples = real_samples + synthetic_samples

        if len(self.samples) == 0:
            raise RuntimeError("Dataset is empty!")

        self.num_real = len(real_samples)
        self.num_gan = len(synthetic_samples)

        print(
            f"[GAN DATASET] Mode={mode} | "
            f"Real={self.num_real} | GAN={self.num_gan} | "
            f"Total={len(self.samples)}"
        )

    # --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------
    def __getitem__(self, idx) -> Tuple:
        sample = self.samples[idx]

        image = Image.open(sample["path"]).convert("RGB")
        image = self.transform(image)
        label = sample["label"]

        if self.return_image_id:
            return image, label, sample["image_id"], sample["source"]

        return image, label

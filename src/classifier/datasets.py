# classifier/datasets.py
"""
Dataset definition for skin lesion classification.

FINAL DESIGN:
- train / val / test split by CSV
- Optional class-aware augmentation (TRAIN ONLY)
- Baseline & Aug use SAME dataset class
- use_class_aug = PIPELINE SWITCH (baseline vs base_aug)
- All policies come from experiment.yaml & classifier.yaml
- Explicit reporting of real vs augmented samples
- Paper-ready
"""

from pathlib import Path
from typing import Tuple, Dict

import yaml
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from configs.config import (
    EXPERIMENT_CFG_PATH,
    CLASSIFIER_CFG_PATH,
)

# ======================================================
# CONFIG LOADER
# ======================================================
def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


EXP_CFG = load_yaml(EXPERIMENT_CFG_PATH)
CLS_CFG = load_yaml(CLASSIFIER_CFG_PATH)


# ======================================================
# IMAGE TRANSFORMS (FROM CONFIG)
# ======================================================
def build_train_transform(img_size: int, tf_cfg: Dict):
    tf_list = [transforms.Resize((img_size, img_size))]

    if tf_cfg.get("horizontal_flip", False):
        tf_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if tf_cfg.get("vertical_flip", False):
        tf_list.append(transforms.RandomVerticalFlip(p=0.3))

    if tf_cfg.get("rotation", 0) > 0:
        tf_list.append(transforms.RandomRotation(tf_cfg["rotation"]))

    if tf_cfg.get("color_jitter", False):
        tf_list.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            )
        )

    tf_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(tf_list)


def build_eval_transform(img_size: int):
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
class SkinLesionDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        images_dir: str,
        mode: str = "train",              # train | val | test
        use_class_aug: bool = False,      # 🔑 PIPELINE SWITCH
        return_image_id: bool = False,
        verbose: bool = True,
    ):
        assert mode in {"train", "val", "test"}
        self.mode = mode
        self.return_image_id = return_image_id

        # -------------------------
        # CONFIG RESOLUTION
        # -------------------------
        img_size = CLS_CFG["training"]["input_size"]

        exp_aug_cfg = EXP_CFG["augmentation"]
        cls_aug_cfg = CLS_CFG["augmentation"]

        # 🔑 FINAL DECISION: whether basic augmentation is applied
        self.use_basic_aug = (
            use_class_aug                 # pipeline decision
            and mode == "train"
            and exp_aug_cfg["enabled"]
            and exp_aug_cfg["type"] == "basic"
            and cls_aug_cfg["enabled"]
        )

        self.class_aug_factor = cls_aug_cfg.get("class_aug_factor", {})

        strength_policy = exp_aug_cfg["basic"]["strength_policy"]
        self.majority_strength = strength_policy["majority"]
        self.minority_strength = strength_policy["minority"]

        # -------------------------
        # PATHS
        # -------------------------
        self.images_dir = Path(images_dir) / mode

        # -------------------------
        # LOAD CSV
        # -------------------------
        self.df = pd.read_csv(csv_file)

        required_cols = {"image_id", "label"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"{csv_file} must contain {required_cols}")

        if len(self.df) == 0:
            raise RuntimeError(f"Empty CSV: {csv_file}")

        # -------------------------
        # REAL DISTRIBUTION
        # -------------------------
        real_counts = self.df["label"].value_counts().sort_index()

        # -------------------------
        # EXPAND DATASET (TRAIN + BASIC AUG)
        # -------------------------
        if self.use_basic_aug:
            expanded_rows = []
            for _, row in self.df.iterrows():
                label_name = EXP_CFG["dataset"]["classes"][int(row["label"])]["name"]
                factor = self.class_aug_factor.get(label_name, 1)
                for _ in range(factor):
                    expanded_rows.append(row)

            self.df = pd.DataFrame(expanded_rows).reset_index(drop=True)

        expanded_counts = self.df["label"].value_counts().sort_index()

        # -------------------------
        # REPORT (TRAIN ONLY)
        # -------------------------
        if verbose and mode == "train":
            print("\n[Dataset Summary]")
            print(f"  Mode            : {mode}")
            print(f"  Pipeline aug    : {use_class_aug}")
            print(f"  Basic aug used  : {self.use_basic_aug}")
            print("  ------------------------------------")
            print("  label | real | after_aug | factor")

            for label in sorted(real_counts.index):
                real = real_counts[label]
                after = expanded_counts.get(label, 0)
                factor = after // real if real > 0 else 0
                print(f"   {label:>3}  | {real:>4} | {after:>9} | {factor}")

            print("  ------------------------------------")
            print(f"  Total images: {len(self.df)}\n")

        # -------------------------
        # TRANSFORMS
        # -------------------------
        if mode == "train":
            self.weak_tf = build_train_transform(
                img_size,
                CLS_CFG["augmentation"]["train"]["weak"],
            )
            self.strong_tf = build_train_transform(
                img_size,
                CLS_CFG["augmentation"]["train"]["strong"],
            )
        else:
            self.eval_tf = build_eval_transform(img_size)

        # -------------------------
        # SANITY CHECK
        # -------------------------
        for img_id in self.df["image_id"].head(5):
            if not (self.images_dir / f"{img_id}.jpg").exists():
                raise FileNotFoundError(
                    f"Missing image: {img_id}.jpg in {self.images_dir}"
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple:
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])

        image_path = self.images_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        if self.mode == "train":
            if self.use_basic_aug:
                label_name = EXP_CFG["dataset"]["classes"][label]["name"]
                factor = self.class_aug_factor.get(label_name, 1)

                # minority → strong, majority → weak
                if factor >= 3 and self.minority_strength == "strong":
                    image = self.strong_tf(image)
                else:
                    image = self.weak_tf(image)
            else:
                image = self.weak_tf(image)
        else:
            image = self.eval_tf(image)

        if self.return_image_id:
            return image, label, image_id

        return image, label

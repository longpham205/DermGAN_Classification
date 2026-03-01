# main/run_01_preprocess.py
"""
Unified preprocessing pipeline.

Principles:
- NO data leakage: heavy preprocessing applied to TRAIN only
- VAL / TEST: minimal preprocessing (resize only)
- Fully controlled by config
- Reproducible & auditable
"""

import os
import sys
import cv2
import yaml
import pandas as pd
from tqdm import tqdm

# ==================================================
# Project root & imports
# ==================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing.generate_splits import generate_splits

from src.preprocessing.resize_normalize import resize_and_normalize
from src.preprocessing.color_constancy import shade_of_gray
from src.preprocessing.hair_removal import remove_hair
from src.preprocessing.lesion_crop import center_crop


# ==================================================
# Load config
# ==================================================
def load_config(path="configs/experiment.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==================================================
# Preprocess one split
# ==================================================
def preprocess_split(split_name: str, config: dict):

    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocess"]

    # ---------- RAW DATA ----------
    raw_root = dataset_cfg["raw"]["root"]
    raw_images_dir = os.path.join(
        raw_root,
        dataset_cfg["raw"]["images_dir"]
    )

    # ---------- SPLIT FILE ----------
    splits_dir = dataset_cfg["splits"]["splits_dir"]
    split_csv = os.path.join(splits_dir, f"{split_name}.csv")

    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"Missing split file: {split_csv}")

    df_split = pd.read_csv(split_csv)

    if "image_id" not in df_split.columns:
        raise ValueError("Split CSV must contain column: image_id")

    image_ids = df_split["image_id"].tolist()

    # ---------- OUTPUT DIR ----------
    output_root = dataset_cfg["processed"]["processed_images_dir"]
    split_out_dir = os.path.join(output_root, split_name)
    os.makedirs(split_out_dir, exist_ok=True)

    # ---------- PROCESS ----------
    for img_id in tqdm(image_ids, desc=f"Preprocessing [{split_name}]"):

        img_path = os.path.join(raw_images_dir, f"{img_id}.jpg")
        save_path = os.path.join(split_out_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        # =========================
        # TRAIN-ONLY preprocessing
        # =========================
        if split_name == "train":

            if preprocess_cfg.get("color_constancy", False):
                img = shade_of_gray(
                    img,
                    power=preprocess_cfg.get("cc_power", 6)
                )

            if preprocess_cfg.get("hair_removal", False):
                img = remove_hair(
                    img,
                    kernel_size=preprocess_cfg.get("hair_kernel", 9),
                    threshold=preprocess_cfg.get("hair_threshold", 10),
                    inpaint_radius=preprocess_cfg.get("inpaint_radius", 1)
                )

            if preprocess_cfg.get("center_crop", False):
                img = center_crop(
                    img,
                    ratio=preprocess_cfg.get("crop_ratio", 0.9)
                )

        # =========================
        # Resize (ALL splits)
        # =========================
        img = resize_and_normalize(
            img,
            size=tuple(preprocess_cfg["image_size"]),
            normalize=False
        )

        # Ensure uint8 for saving
        if img.dtype != "uint8":
            img = img.astype("uint8")

        cv2.imwrite(save_path, img)


# ==================================================
# Entry point
# ==================================================
def run_preprocess():
    
    #==================
    #Split Data
    #==================
    generate_splits()

    #==================
    #Preprocess
    #==================
    config = load_config()

    for split in ["train", "val", "test"]:
        preprocess_split(split, config)

    print("Preprocessing completed successfully.")
    
    
if __name__ == "__main__":
    run_preprocess()

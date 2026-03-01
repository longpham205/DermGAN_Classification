# src/preprocessing/generate_splits.py
"""
Generate stratified train/val/test splits for HAM10000

- Lesion-level split (NO data leakage)
- Stratified by diagnosis
- Label mapping loaded STRICTLY from configs/experiment.yaml
- Save CSV only: image_id, dx, label
"""

import os, sys 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
import yaml
import random
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_checksum(ids):
    m = hashlib.md5()
    for _id in sorted(ids):
        m.update(str(_id).encode("utf-8"))
    return m.hexdigest()


def validate_ratios(split_cfg):
    total = (
        split_cfg["train_ratio"]
        + split_cfg["val_ratio"]
        + split_cfg["test_ratio"]
    )
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.4f}"
        )


# =========================
# Main
# =========================

def generate_splits(config_path: str = "configs/experiment.yaml"):

    # -------------------------
    # Load config
    # -------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["reproducibility"]["seed"]
    dataset_cfg = config["dataset"]
    split_cfg = config["split"]

    set_seed(seed)
    validate_ratios(split_cfg)

    # -------------------------
    # Paths
    # -------------------------
    raw_cfg = dataset_cfg["raw"]
    splits_cfg = dataset_cfg["splits"]

    metadata_path = os.path.join(
        raw_cfg["root"],
        raw_cfg["metadata_csv"]
    )

    splits_dir = splits_cfg["splits_dir"]
    ensure_dir(splits_dir)

    # -------------------------
    # Load metadata
    # -------------------------
    df = pd.read_csv(metadata_path)

    required_cols = ["lesion_id", "image_id", "dx"]
    if not all(col in df.columns for col in required_cols):
        raise KeyError(
            f"Metadata must contain columns: {required_cols}"
        )

    # -------------------------
    # Load label mapping (STRICT)
    # -------------------------
    class_cfg = dataset_cfg["classes"]
    num_classes_cfg = dataset_cfg["num_classes"]

    if len(class_cfg) != num_classes_cfg:
        raise ValueError(
            "num_classes does not match number of class definitions"
        )

    dx_to_label = {c["name"]: c["id"] for c in class_cfg}
    label_to_dx = {c["id"]: c["name"] for c in class_cfg}

    # Validate dx values
    unknown_dx = set(df["dx"].unique()) - set(dx_to_label.keys())
    if unknown_dx:
        raise ValueError(
            f"Metadata contains unknown dx not in config: {unknown_dx}"
        )

    df["label"] = df["dx"].map(dx_to_label)

    # -------------------------
    # Lesion-level table
    # -------------------------
    lesion_df = (
        df.groupby("lesion_id")
        .agg({"dx": "first"})
        .reset_index()
    )

    # -------------------------
    # Validate stratification
    # -------------------------
    lesion_class_counts = lesion_df["dx"].value_counts()
    if (lesion_class_counts < 2).any():
        rare = lesion_class_counts[lesion_class_counts < 2].to_dict()
        raise ValueError(
            f"Some classes have <2 lesions, cannot stratify: {rare}"
        )

    # -------------------------
    # Stratified split (LESION LEVEL)
    # -------------------------
    train_lesions, temp_lesions = train_test_split(
        lesion_df,
        test_size=(1.0 - split_cfg["train_ratio"]),
        stratify=lesion_df["dx"],
        random_state=seed,
    )

    val_ratio_adj = split_cfg["val_ratio"] / (
        split_cfg["val_ratio"] + split_cfg["test_ratio"]
    )

    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=(1.0 - val_ratio_adj),
        stratify=temp_lesions["dx"],
        random_state=seed,
    )

    split_map = {
        "train": set(train_lesions["lesion_id"]),
        "val": set(val_lesions["lesion_id"]),
        "test": set(test_lesions["lesion_id"]),
    }

    # -------------------------
    # Save CSVs
    # -------------------------
    split_stats = {}

    for split, lesion_ids in split_map.items():
        split_df = df[df["lesion_id"].isin(lesion_ids)].copy()

        out_csv = os.path.join(splits_dir, f"{split}.csv")
        split_df[["image_id", "dx", "label"]].to_csv(
            out_csv, index=False
        )

        split_stats[split] = {
            "num_images": len(split_df),
            "num_lesions": split_df["lesion_id"].nunique(),
            "class_distribution": split_df["dx"]
            .value_counts()
            .to_dict(),
            "lesion_checksum": compute_checksum(
                split_df["lesion_id"].unique()
            ),
            "csv_path": out_csv,
        }

        print(f"[{split.upper()}]")
        print(f"  Images : {split_stats[split]['num_images']}")
        print(f"  Lesions: {split_stats[split]['num_lesions']}")
        print(f"  Saved  : {out_csv}\n")

    # -------------------------
    # Save split protocol
    # -------------------------
    protocol = {
        "project": config["project_name"],
        "experiment": config["experiment_name"],
        "dataset": dataset_cfg["name"],
        "seed": seed,
        "label_mapping": label_to_dx,
        "ratios": split_cfg,
        "splits": split_stats,
        "note": (
            "Lesion-level stratified split. "
            "Labels strictly follow experiment.yaml."
        ),
    }

    protocol_path = os.path.join(
        splits_dir,
        splits_cfg["split_version"]
    )

    with open(protocol_path, "w", encoding="utf-8") as f:
        yaml.dump(protocol, f, sort_keys=False)

    print("Split generation completed successfully.")
    print(f"Protocol saved to: {protocol_path}")


# =========================
# Entry
# =========================

if __name__ == "__main__":
    generate_splits()

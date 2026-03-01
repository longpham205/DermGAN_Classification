#preprocessing/split_data.py
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(root_dir)
sys.path.insert(0, root_dir)
import shutil
import yaml
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# Load config
# =========================
def load_config(config_path="configs/experiment.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# =========================
# Main split function
# =========================
def split_ham10000(config):
    random.seed(config["seed"])

    dataset_cfg = config["dataset"]
    split_cfg = config["split"]
    output_cfg = config["output"]

    data_root = dataset_cfg["data_root"]
    images_dir = os.path.join(data_root, dataset_cfg["images_dir"])
    metadata_path = os.path.join(data_root, dataset_cfg["metadata_file"])

    output_root = output_cfg["processed_dir"]

    # Create output dirs
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_root, split), exist_ok=True)

    # =========================
    # Load metadata
    # =========================
    df = pd.read_csv(metadata_path)

    # Columns required
    required_cols = ["lesion_id", "image_id", "dx"]
    assert all(col in df.columns for col in required_cols), "Missing required columns"

    # =========================
    # Group by lesion_id
    # =========================
    lesion_df = (
        df.groupby("lesion_id")
          .first()
          .reset_index()[["lesion_id", "dx"]]
    )

    # =========================
    # Split lesion_id
    # =========================
    train_lesions, temp_lesions = train_test_split(
        lesion_df,
        test_size=(1 - split_cfg["train_ratio"]),
        stratify=lesion_df["dx"],
        random_state=config["seed"]
    )

    val_ratio_adjusted = split_cfg["val_ratio"] / (split_cfg["val_ratio"] + split_cfg["test_ratio"])

    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_lesions["dx"],
        random_state=config["seed"]
    )

    split_map = {
        "train": set(train_lesions["lesion_id"]),
        "val": set(val_lesions["lesion_id"]),
        "test": set(test_lesions["lesion_id"]),
    }

    # =========================
    # Assign images
    # =========================
    for split, lesion_ids in split_map.items():
        split_df = df[df["lesion_id"].isin(lesion_ids)]

        for _, row in split_df.iterrows():
            image_name = row["image_id"] + ".jpg"
            src_path = os.path.join(images_dir, image_name)
            dst_path = os.path.join(output_root, split, image_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

    # =========================
    # Save split metadata
    # =========================
    for split in ["train", "val", "test"]:
        split_df = df[df["lesion_id"].isin(split_map[split])]

        save_path = os.path.join(output_root, f"{split}_metadata.csv")
        split_df.to_csv(save_path, index=False)

        print(f"Saved {split.upper()} metadata:")
        print(f"Path  : {save_path}")
        print(f"Samples: {len(split_df)}\n")

    print(f"Split completed successfully!")
    print(f"Train lesions: {len(train_lesions)}")
    print(f"Val lesions:   {len(val_lesions)}")
    print(f"Test lesions:  {len(test_lesions)}")

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    config = load_config()
    split_ham10000(config)

# src/gan/generate.py
"""
Auto-generate synthetic images for ALL trained GAN classes
==========================================================

Workflow:
- Read train split CSV
- Compute optimal generation plan (same logic as training)
- For each trained class:
    - Load Generator checkpoint
    - Generate required number of images
    - Save to output directory

Design:
- One-class GAN
- Fully automatic (NO CLI args)
"""

import yaml
from pathlib import Path
from collections import Counter
import statistics
import os, sys 
# ======================================================
# Project root
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import torch
import pandas as pd
from torchvision.utils import save_image
from tqdm import tqdm

from src.gan.models.generator import Generator


# ======================================================
# Utils
# ======================================================
def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_gan_targets(label_counts: dict, gan_cfg: dict):
    """
    Decide how many images to generate per class
    """
    max_count = max(label_counts.values())
    median_count = int(torch.tensor(list(label_counts.values())).median().item())

    tau = gan_cfg["generation"]["threshold_ratio"]
    max_growth = gan_cfg["generation"]["max_growth_factor"]

    plan = {}

    for cls, n_real in label_counts.items():
        if n_real >= tau * max_count:
            continue

        target = min(median_count, int(n_real * max_growth))
        to_generate = max(0, target - n_real)

        if to_generate > 0:
            plan[cls] = {
                "real": n_real,
                "target": target,
                "to_generate": to_generate,
            }

    return plan


# ======================================================
# Main
# ======================================================
def generate():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    gan_cfg = load_yaml(PROJECT_ROOT / "configs" / "gan.yaml")["gan"]
    exp_cfg = load_yaml(PROJECT_ROOT / "configs" / "experiment.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # --------------------------------------------------
    # Load training labels
    # --------------------------------------------------
    train_csv = (
        PROJECT_ROOT
        / exp_cfg["dataset"]["splits"]["splits_dir"]
        / exp_cfg["dataset"]["splits"]["train"]
    )

    df = pd.read_csv(train_csv)
    label_counts = Counter(df["dx"])

    plan = select_gan_targets(label_counts, gan_cfg)

    print("\n[GAN GENERATION PLAN]")
    for cls, info in plan.items():
        print(
            f"  {cls:<10} | real={info['real']} → "
            f"target={info['target']} (+{info['to_generate']})"
        )

    if len(plan) == 0:
        print("[INFO] No class needs GAN augmentation.")
        return

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    ckpt_root = (
        PROJECT_ROOT
        / gan_cfg["checkpoints"]["root"]
        / gan_cfg["experiment_name"]
    )

    output_root = (
        PROJECT_ROOT
        / gan_cfg["generation"]["output_dir"]
    )
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_path = (
        PROJECT_ROOT
        / gan_cfg["generation"]["output_dir"]
        / "synthetic_metadata.csv"
    )

    latent_dim = gan_cfg["latent"]["dim"]
    image_size = gan_cfg["image"]["size"]

    gan_exp_name = gan_cfg["experiment_name"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metadata_rows = []

    # --------------------------------------------------
    # Generate images
    # --------------------------------------------------
    for cls, info in plan.items():
        ckpt_path = ckpt_root / cls / gan_cfg["checkpoints"]["ckpt_path"]

        if not ckpt_path.exists():
            print(f"[SKIP] No checkpoint for class '{cls}'")
            continue

        print(f"\n[GENERATE] Class = {cls} | N = {info['to_generate']}")

        G = Generator(
            latent_dim=latent_dim,
            image_size=image_size,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ckpt["generator"])
        G.eval()

        out_dir = output_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for i in tqdm(range(info["to_generate"]), desc=cls):
                z = torch.randn(1, latent_dim, device=device)
                img = G(z)

                filename = f"{cls}_gan_{i:05d}.jpg"
                save_path = out_dir / filename

                save_image(
                    img,
                    save_path,
                    normalize=True,
                    value_range=(-1, 1),
                )

                metadata_rows.append({
                    "image_path": str(save_path.relative_to(PROJECT_ROOT)),
                    "label": cls,
                    "source": "gan",
                    "gan_experiment": gan_exp_name,
                    "gan_checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                    "real_count": info["real"],
                    "target_count": info["target"],
                    "generated_index": i,
                    "created_at": timestamp,
                })

        print(f"[✓] Saved {info['to_generate']} images → {out_dir}")

    # --------------------------------------------------
    # Save metadata
    # --------------------------------------------------
    if metadata_rows:
        meta_df = pd.DataFrame(metadata_rows)

        if metadata_path.exists() and metadata_path.stat().st_size > 0:
            try:
                old_df = pd.read_csv(metadata_path)
                meta_df = pd.concat([old_df, meta_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                pass  # treat as new file


        meta_df.to_csv(metadata_path, index=False)
        print(f"\n[✓] synthetic_metadata.csv updated → {metadata_path}")

    print("\n[SUCCESS] GAN data augmentation completed.")


if __name__ == "__main__":
    generate()

# src/gan/train_gan.py

"""
Auto-Selective Single-Class WGAN-GP Trainer
===========================================

Purpose:
- Train ONE WGAN-GP per minority class
- Learn data distribution ONLY
- Save trained Generator checkpoints (.pt)
- Log training metrics per epoch
"""

import sys
import yaml
import json
import random
from pathlib import Path
from collections import Counter

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

# ======================================================
# Project root
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================
# Imports
# ======================================================
from src.gan.datasets import SkinLesionGANDataset
from src.gan.models.generator import Generator
from src.gan.models.discriminator import Discriminator
from src.gan.models.losses import generator_loss, critic_loss_with_gp


# ======================================================
# Utils
# ======================================================
def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_instance_noise(x: torch.Tensor, std: float = 0.03):
    if std > 0:
        return x + torch.randn_like(x) * std
    return x


# ======================================================
# Auto-selective GAN target selection
# ======================================================
def select_gan_targets(label_counts: dict, gan_cfg: dict):
    max_count = max(label_counts.values())
    median_count = int(np.median(list(label_counts.values())))

    tau = gan_cfg["generation"]["threshold_ratio"]
    max_growth = gan_cfg["generation"]["max_growth_factor"]

    plan = {}

    for cls, n_real in label_counts.items():
        if n_real >= tau * max_count:
            continue

        target = min(median_count, int(n_real * max_growth))
        n_generate = max(0, target - n_real)

        if n_generate > 0:
            plan[cls] = {
                "real": n_real,
                "target": target,
                "to_generate": n_generate,
            }

    return plan


# ======================================================
# Train GAN for ONE class
# ======================================================
def train_gan_for_class(class_name: str, gan_cfg: dict, device: torch.device):
    print(f"\n[GAN TRAIN] Class = {class_name}")

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = SkinLesionGANDataset(
        class_name=class_name,
        image_size=gan_cfg["image"]["size"],
        split="train",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=gan_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    # -----------------------------
    # Models
    # -----------------------------
    latent_dim = gan_cfg["latent"]["dim"]

    G = Generator(
        latent_dim=latent_dim,
        image_size=gan_cfg["image"]["size"],
    ).to(device)

    D = Discriminator(
        image_size=gan_cfg["image"]["size"],
    ).to(device)

    # -----------------------------
    # Optimizers (IMPORTANT CHANGE)
    # -----------------------------
    opt_G = optim.Adam(
        G.parameters(),
        lr=gan_cfg["training"]["lr"],
        betas=(0.0, 0.9),
    )

    opt_D = optim.Adam(
        D.parameters(),
        lr=gan_cfg["training"]["lr"] * 2,
        betas=(0.0, 0.9),
    )

    # -----------------------------
    # Training params
    # -----------------------------
    epochs = gan_cfg["training"]["epochs"]
    n_critic = gan_cfg["training"]["n_critic"]
    lambda_gp = gan_cfg["training"]["lambda_gp"]

    epoch_logs = []

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, epochs + 1):
        g_losses, d_losses, w_losses, gp_losses = [], [], [], []

        pbar = tqdm(
            dataloader,
            desc=f"[{class_name}] Epoch {epoch}/{epochs}",
            leave=False,
        )

        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # =====================
            # Train Critic
            # =====================
            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim, device=device)

                with torch.no_grad():
                    fake_imgs = G(z)

                real_noisy = add_instance_noise(real_imgs, std=0.03)
                fake_noisy = add_instance_noise(fake_imgs, std=0.03)

                d_loss, w_loss, gp = critic_loss_with_gp(
                    D,
                    real_noisy,
                    fake_noisy,
                    labels=None,
                    lambda_gp=lambda_gp,
                )

                opt_D.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_D.step()

            # =====================
            # Train Generator
            # =====================
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = G(z)
            fake_scores = D(fake_imgs)

            g_loss = generator_loss(fake_scores)

            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_G.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            w_losses.append(w_loss.item())
            gp_losses.append(gp.item())

            pbar.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")

        # -----------------------------
        # Epoch summary
        # -----------------------------
        epoch_summary = {
            "epoch": epoch,
            "g_loss": float(np.mean(g_losses)),
            "d_loss": float(np.mean(d_losses)),
            "w_loss": float(np.mean(w_losses)),
            "gp": float(np.mean(gp_losses)),
        }

        epoch_logs.append(epoch_summary)

        print(
            f"[Epoch {epoch:03d}] "
            f"G={epoch_summary['g_loss']:.4f} | "
            f"D={epoch_summary['d_loss']:.4f} | "
            f"W={epoch_summary['w_loss']:.4f} | "
            f"GP={epoch_summary['gp']:.4f}"
        )

    # ======================================================
    # Save checkpoint
    # ======================================================
    ckpt_dir = (
        PROJECT_ROOT
        / gan_cfg["checkpoints"]["root"]
        / "gan"
        / class_name
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / gan_cfg["checkpoints"]["ckpt_path"]

    torch.save(
        {
            "generator": G.state_dict(),
            "config": gan_cfg,
            "class_name": class_name,
            "metrics": epoch_logs,
        },
        ckpt_path,
    )

    print(f"[✓] Saved Generator → {ckpt_path}")

    # ======================================================
    # Save results
    # ======================================================
    results_dir = (
        PROJECT_ROOT
        / gan_cfg["results"]["root"]
        / "gan"
        / "train"
        / class_name
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / gan_cfg["results"]["matrix_json"], "w") as f:
        json.dump(epoch_logs, f, indent=2)

    pd.DataFrame(epoch_logs).to_csv(
        results_dir / gan_cfg["results"]["matrix_csv"], index=False
    )

    print(f"[✓] Saved training results → {results_dir}")


# ======================================================
# Main
# ======================================================
def train_gan():
    gan_cfg = load_yaml(PROJECT_ROOT / "configs" / "gan.yaml")["gan"]
    exp_cfg = load_yaml(PROJECT_ROOT / "configs" / "experiment.yaml")

    set_seed(gan_cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    split_csv = (
        PROJECT_ROOT
        / exp_cfg["dataset"]["splits"]["splits_dir"]
        / exp_cfg["dataset"]["splits"]["train"]
    )

    df = pd.read_csv(split_csv)
    label_counts = Counter(df["dx"])

    plan = select_gan_targets(label_counts, gan_cfg)

    print("\n[GAN TRAIN PLAN]")
    for cls, info in plan.items():
        print(f"  {cls:<10} | real={info['real']} → target={info['target']}")

    for class_name in plan.keys():
        train_gan_for_class(class_name, gan_cfg, device)

    print("\n[SUCCESS] All GANs trained and results saved.")


if __name__ == "__main__":
    train_gan()

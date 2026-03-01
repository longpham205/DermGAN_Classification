# # src/classifier/train_with_gan.py

import os, sys, time
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_dir)

import numpy as np
import csv
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src.utils.seed import set_seed, seed_worker
from src.utils.metrics import compute_metrics
from src.classifier.datasets_gan import SkinLesionGANDataset
from src.classifier.models import create_model
from configs.config import (
    CLASSES,
    CLASSIFIER_CHECKPOINT_DIR,
    EXPERIMENT_CFG_PATH,
    CLASSIFIER_CFG_PATH,
    LOGGING_CFG_PATH,
)

# ====================
# Utils
# ====================
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_table_header(f):
    f.write(
        f"{'Epoch':>5} | {'Loss':>8} | {'LR':>9} | "
        f"{'Acc':>7} | {'BalAcc':>7} | "
        f"{'F1_macro':>9} | {'F1_weight':>9} | {'Time(s)':>7}\n"
    )
    f.write("-" * 90 + "\n")


# ====================
# Main
# ====================
def train_with_gan():
    # ====================
    # Configs
    # ====================
    exp_cfg = load_yaml(EXPERIMENT_CFG_PATH)
    cls_cfg = load_yaml(CLASSIFIER_CFG_PATH)
    log_cfg = load_yaml(LOGGING_CFG_PATH)

    # ====================
    # Reproducibility
    # ====================
    set_seed(
        seed=exp_cfg["reproducibility"]["seed"],
        deterministic=exp_cfg["reproducibility"]["deterministic"],
    )
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ====================
    # Output dirs
    # ====================
    exp_name = log_cfg["logging"]["experiment_gan"]
    result_dir = os.path.join(log_cfg["logging"]["root_dir"], exp_name)
    os.makedirs(result_dir, exist_ok=True)

    ckpt_dir = os.path.join(CLASSIFIER_CHECKPOINT_DIR, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, exp_cfg["checkpoints"]["model"])

    log_txt_path = os.path.join(result_dir, exp_cfg["results"]["log_txt"])
    log_csv_path = os.path.join(result_dir, exp_cfg["results"]["log_csv"])
    metrics_path = os.path.join(result_dir, exp_cfg["results"]["metrics"])

    # ====================
    # Dataset
    # ====================
    real_images_dir = exp_cfg["dataset"]["processed"]["root"]

    train_csv = os.path.join(
        exp_cfg["dataset"]["splits"]["splits_dir"],
        exp_cfg["dataset"]["splits"]["train"],
    )
    val_csv = os.path.join(
        exp_cfg["dataset"]["splits"]["splits_dir"],
        exp_cfg["dataset"]["splits"]["val"],
    )

    train_set = SkinLesionGANDataset(
        csv_file=train_csv,
        images_dir=real_images_dir,
        mode="train",
        synthetic_root=exp_cfg["dataset"]["synthetic"]["root"],
        synthetic_csv=os.path.join(
            exp_cfg["dataset"]["synthetic"]["root"],
            exp_cfg["dataset"]["synthetic"]["metadata"],
        ),
    )

    val_set = SkinLesionGANDataset(
        csv_file=val_csv,
        images_dir=real_images_dir,
        mode="val",
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cls_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cls_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ====================
    # Logging header
    # ====================
    gan_ratio = train_set.num_gan / (train_set.num_real + train_set.num_gan)

    with open(log_txt_path, "w") as f:
        f.write(f"Experiment   : {exp_name}\n")
        f.write(f"Start time   : {datetime.now()}\n")
        f.write("DATASET STATS\n")
        f.write(f"Train REAL   : {train_set.num_real}\n")
        f.write(f"Train GAN    : {train_set.num_gan}\n")
        f.write(f"GAN ratio    : {gan_ratio:.2%}\n")
        f.write(f"Val REAL     : {len(val_set)}\n")
        f.write("-" * 90 + "\n")
        f.write("EPOCH METRICS (VALIDATION)\n")
        write_table_header(f)

    # ====================
    # Model
    # ====================
    model = create_model(cls_cfg["model"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cls_cfg["training"]["lr"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cls_cfg["training"]["epochs"]
    )

    scaler = torch.amp.GradScaler("cuda")

    best_f1 = 0.0
    best_epoch = -1
    best_metrics = None
    csv_header_written = False

    # reset CSV
    with open(log_csv_path, "w", newline="") as f:
        pass

    # ====================
    # Training loop
    # ====================
    for epoch in range(cls_cfg["training"]["epochs"]):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train+GAN]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ====================
        # Validation
        # ====================
        model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        metrics, cm = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=np.array(y_prob),
            class_names=CLASSES,
            return_confusion=True,
        )

        metrics["confusion_matrix"] = cm.tolist()

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        # ====================
        # TXT LOG
        # ====================
        with open(log_txt_path, "a") as f:
            f.write(
                f"{epoch+1:5d} | "
                f"{train_loss:8.4f} | "
                f"{lr:9.2e} | "
                f"{metrics['accuracy']:7.4f} | "
                f"{metrics['balanced_accuracy']:7.4f} | "
                f"{metrics['f1_macro']:9.4f} | "
                f"{metrics['f1_weighted']:9.4f} | "
                f"{epoch_time:7.1f}\n"
            )

            f.write("  Per-class:\n")
            for cls, vals in metrics["per_class"].items():
                f.write(
                    f"    {cls:10s} | "
                    f"P={vals['precision']:.3f} "
                    f"R={vals['recall']:.3f} "
                    f"F1={vals['f1']:.3f}\n"
                )
            f.write("-" * 90 + "\n")

        # ====================
        # CSV LOG
        # ====================
        metrics_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "f1_macro": metrics.get("f1_macro"),
            "f1_weighted": metrics.get("f1_weighted"),
            "auc_macro": metrics.get("auc_macro"),
        }

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_row.keys())
            if not csv_header_written:
                writer.writeheader()
                csv_header_written = True
            writer.writerow(metrics_row)

        # ====================
        # Checkpoint
        # ====================
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_epoch = epoch + 1
            best_metrics = metrics.copy()
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"[Epoch {epoch+1:03d}] "
            f"loss={train_loss:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"lr={lr:.2e} | "
            f"time={epoch_time:.1f}s"
        )

    # ====================
    # Save final metrics.json
    # ====================
    metrics_json = {
        "experiment": exp_name,
        "timestamp": datetime.now().isoformat(),
        "best_epoch": best_epoch,
        "gan_stats": {
            "num_real": train_set.num_real,
            "num_gan": train_set.num_gan,
            "gan_ratio": gan_ratio,
        },
        "global_metrics": {
            "accuracy": best_metrics.get("accuracy"),
            "balanced_accuracy": best_metrics.get("balanced_accuracy"),
            "precision_macro": best_metrics.get("precision_macro"),
            "recall_macro": best_metrics.get("recall_macro"),
            "f1_macro": best_metrics.get("f1_macro"),
            "f1_weighted": best_metrics.get("f1_weighted"),
            "auc_macro": best_metrics.get("auc_macro"),
            "specificity": best_metrics.get("specificity"),
            "sensitivity": best_metrics.get("sensitivity"),
        },
        "per_class": best_metrics.get("per_class", {}),
        "confusion_matrix": best_metrics.get("confusion_matrix"),
        "dataset": {
            "num_classes": len(CLASSES),
            "class_names": CLASSES,
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=4)

    print(
        f"\nTraining finished. Best macro-F1 = {best_f1:.4f} "
        f"(epoch {best_epoch})"
    )


if __name__ == "__main__":
    train_with_gan()
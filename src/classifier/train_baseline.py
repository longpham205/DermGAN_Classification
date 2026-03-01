#src/classifier/train_baseline.py

import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_dir)

import numpy as np
import csv
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from datetime import datetime
from collections import Counter

from src.utils.seed import set_seed, seed_worker
from src.utils.metrics import compute_metrics
from src.classifier.datasets import SkinLesionDataset
from src.classifier.models import create_model
from configs.config import (
    CLASSES, 
    NUM_CLASSES, 
    CLASSIFIER_CHECKPOINT_DIR,
    EXPERIMENT_CFG_PATH,
    CLASSIFIER_CFG_PATH,
    LOGGING_CFG_PATH
)


# ====================
# Utils
# ====================
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_class_weights_from_csv(csv_path, num_classes):
    labels = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))

    counter = Counter(labels)
    counts = np.array([counter[i] for i in range(num_classes)], dtype=np.float32)

    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float)


def compute_sample_weights(csv_path):
    labels = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))

    counter = Counter(labels)
    sample_weights = [1.0 / counter[label] for label in labels]
    return torch.DoubleTensor(sample_weights)


# ====================
# Main
# ====================
def train_baseline():
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

    class_names = CLASSES

    # ====================
    # Output dirs
    # ====================
    exp_name = log_cfg["logging"]["experiment_name"]
    result_dir = os.path.join(log_cfg["logging"]["root_dir"], exp_name)
    os.makedirs(result_dir, exist_ok=True)

    ckpt_dir = os.path.join(CLASSIFIER_CHECKPOINT_DIR, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, exp_cfg["checkpoints"]["model"])

    log_txt_path = os.path.join(result_dir, exp_cfg["results"]["log_txt"])
    log_csv_path = os.path.join(result_dir, exp_cfg["results"]["log_csv"])
    metrics_path = os.path.join(result_dir, exp_cfg["results"]["metrics"])

    with open(log_txt_path, "w") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Start time: {datetime.now()}\n")
        f.write("-" * 60 + "\n")

    # ====================
    # Dataset & Loader
    # ====================
    images_dir = os.path.join(
        exp_cfg["dataset"]["processed"]["root"],
    )

    train_csv = os.path.join(
        exp_cfg["dataset"]["splits"]["splits_dir"],
        exp_cfg["dataset"]["splits"]["train"],
    )
    val_csv = os.path.join(
        exp_cfg["dataset"]["splits"]["splits_dir"],
        exp_cfg["dataset"]["splits"]["val"],
    )

    train_set = SkinLesionDataset(
        images_dir=images_dir,
        csv_file=train_csv,
        mode="train",
    )

    val_set = SkinLesionDataset(
        images_dir=images_dir,
        csv_file=val_csv,
        mode="val",
    )

    # -------- Imbalance handling --------
    class_weights = compute_class_weights_from_csv(
        train_csv, NUM_CLASSES
    ).to(device)

    sample_weights = compute_sample_weights(train_csv)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    g = torch.Generator().manual_seed(exp_cfg["reproducibility"]["seed"])

    train_loader = DataLoader(
        train_set,
        batch_size=cls_cfg["training"]["batch_size"],
        sampler=sampler,            # 🔥 imbalance-aware
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cls_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ====================
    # Model
    # ====================
    model = create_model(cls_cfg["model"]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cls_cfg["training"]["lr"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cls_cfg["training"]["epochs"]
    )

    scaler = torch.amp.GradScaler("cuda")

    best_f1 = 0.0
    best_epoch = -1

    # ====================
    # Training loop
    # ====================
    csv_header_written = False
    best_metrics = None

    # Reset CSV file at start
    with open(log_csv_path, "w", newline="") as f:
        pass


    for epoch in range(cls_cfg["training"]["epochs"]):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
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
                x = x.to(device, non_blocking=True)
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
            class_names=class_names,
            return_confusion=True,
        )

        # lưu confusion matrix dạng list để dump JSON
        metrics["confusion_matrix"] = cm.tolist()

        # ====================
        # Logging TXT
        # ====================
        with open(log_txt_path, "a") as f:
            f.write(
                f"Epoch {epoch+1:03d} | "
                f"loss={train_loss:.4f} | "
                f"f1_macro={metrics['f1_macro']:.4f} | "
                f"balanced_acc={metrics['balanced_accuracy']:.4f} | "
                f"auc_macro={metrics.get('auc_macro', float('nan')):.4f}\n"
            )

        # ====================
        # Logging CSV (chỉ scalar)
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
        # Checkpoint (best macro-F1)
        # ====================
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_epoch = epoch + 1
            best_metrics = metrics.copy()
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"[Epoch {epoch+1}] "
            f"loss={train_loss:.4f} | "
            f"val_macro_f1={metrics['f1_macro']:.4f}"
        )

    # ====================
    # Summary
    # ====================
    with open(log_txt_path, "a") as f:
        f.write("-" * 60 + "\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best macro-F1: {best_f1:.4f}\n")

    # ====================
    # Save final metrics.json (cấu trúc mới chuẩn)
    # ====================
    metrics_json = {
        "experiment": exp_name,
        "timestamp": datetime.now().isoformat(),
        "best_epoch": best_epoch,

        "dataset": {
            "num_classes": NUM_CLASSES,
            "class_names": class_names,
        },

        "global_metrics": {
            "accuracy": best_metrics.get("accuracy"),
            "balanced_accuracy": best_metrics.get("balanced_accuracy"),
            "precision_macro": best_metrics.get("precision_macro"),
            "recall_macro": best_metrics.get("recall_macro"),
            "f1_macro": best_metrics.get("f1_macro"),
            "f1_weighted": best_metrics.get("f1_weighted"),
            "macro_auc": best_metrics.get("auc_macro"),
            "specificity": best_metrics.get("specificity"),
            "sensitivity": best_metrics.get("sensitivity"),
        },

        "per_class": best_metrics.get("per_class", {}),

        "confusion_matrix": {
            "matrix": best_metrics.get("confusion_matrix"),
            "class_names": class_names,
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=4)

    print(f"Saved metrics to {metrics_path}")
    print(
        f"Training finished. Best macro-F1 = {best_f1:.4f} "
        f"(epoch {best_epoch})"
    )

if __name__ == "__main__":
    train_baseline()

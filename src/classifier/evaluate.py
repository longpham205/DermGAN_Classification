# src/classifier/evaluate.py

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    balanced_accuracy_score,
)

import pandas as pd


# ======================================================
# Project root
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================
# Imports (SYNC WITH TRAIN)
# ======================================================
from src.classifier.datasets import SkinLesionDataset
from src.classifier.models import create_model
from configs.config import (
    CLASSES,
    NUM_CLASSES,
    CLASSIFIER_CHECKPOINT_DIR,
    EXPERIMENT_CFG_PATH,
    CLASSIFIER_CFG_PATH,
    LOGGING_CFG_PATH,
)


# ======================================================
# Utils
# ======================================================
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ======================================================
# Main
# ======================================================
def evaluate(exp_name = "exp_01_baseline" ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    print(f"\nEvaluate: {exp_name}\n" )

    # ------------------
    # Load configs
    # ------------------
    exp_cfg = load_yaml(EXPERIMENT_CFG_PATH)
    cls_cfg = load_yaml(CLASSIFIER_CFG_PATH)
    log_cfg = load_yaml(LOGGING_CFG_PATH)

    # exp_name = log_cfg["logging"]["experiment_name"]

    result_dir = Path(log_cfg["logging"]["root_dir"]) / exp_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # ------------------
    # Dataset (TEST)
    # ------------------
    images_dir = (
        Path(exp_cfg["dataset"]["processed"]["root"])
    )

    test_csv = (
        Path(exp_cfg["dataset"]["splits"]["splits_dir"])
        / exp_cfg["dataset"]["splits"]["test"]
    )

    test_set = SkinLesionDataset(
        images_dir=str(images_dir),
        csv_file=str(test_csv),
        mode="test",
        return_image_id=True,  # ⭐ dùng cho analysis
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cls_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------
    # Model
    # ------------------
    model = create_model(cls_cfg["model"]).to(device)
    
    ckpt_path = Path(CLASSIFIER_CHECKPOINT_DIR) / exp_name / "model_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ------------------
    # Inference
    # ------------------
    y_true, y_pred, y_prob, image_ids = [], [], [], []

    with torch.no_grad():
        for x, y, img_ids in tqdm(test_loader, desc="Evaluating"):
            x = x.to(device, non_blocking=True)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            image_ids.extend(img_ids)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # ------------------
    # Metrics (GLOBAL)
    # ------------------
    global_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }

    metrics_json = {
        "experiment": exp_name,
        "timestamp": datetime.now().isoformat(),
        "split": "test",
        "global_metrics": global_metrics,
        "dataset": {
            "num_classes": NUM_CLASSES,
            "class_names": CLASSES,
        },
        "num_samples": len(y_true),
    }

    with open(result_dir / exp_cfg["results"]["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4)

    # ------------------
    # Classification report (PER CLASS)
    # ------------------
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    with open(result_dir / exp_cfg["results"]["classification_report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # ------------------
    # Confusion matrix (RAW, NO PLOT)
    # ------------------
    cm = confusion_matrix(y_true, y_pred)

    cm_json = {
        "class_names": CLASSES,
        "matrix": cm.tolist(),
    }

    with open(result_dir / exp_cfg["results"]["confusion_matrix"], "w", encoding="utf-8") as f:
        json.dump(cm_json, f, indent=4)

    # ------------------
    # Save predictions.csv (FOR ANALYSIS)
    # ------------------
    pred_df = pd.DataFrame({
        "image_id": image_ids,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    for i, cls in enumerate(CLASSES):
        pred_df[f"prob_{cls}"] = y_prob[:, i]

    pred_df.to_csv(result_dir / exp_cfg["results"]["predictions"], index=False)

    # ------------------
    # Done
    # ------------------
    print("[INFO] Evaluation complete.")
    print(json.dumps(global_metrics, indent=2))
    print(f"Saved: {result_dir}")


if __name__ == "__main__":
    evaluate("exp_02_base_aug")

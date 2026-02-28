"""
Phase 7 - Global Model Evaluation (6-class version)
=====================================================
Loads models/global_model.pth, evaluates on data/partitions/global_test,
prints per-class Precision / Recall / F1-Score and saves a
Confusion Matrix image to models/confusion_matrix.png.

Usage (run from project root):
    python src/evaluate_global.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
    precision_score, recall_score,
)

from dataset import CLASS_NAMES, NUM_CLASSES, get_test_loader, MedicalImageDataset
from model import MedicalCNN

PROJ_ROOT   = Path(__file__).parent.parent
MODEL_PATH  = PROJ_ROOT / "models" / "global_model.pth"
OUTPUT_PATH = PROJ_ROOT / "models" / "confusion_matrix.png"
DEVICE      = "cpu"

SHORT_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
    "Normal (CXR)",
    "Pneumonia",
]


def load_model(path: Path) -> MedicalCNN:
    model = MedicalCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def run_inference(model, loader):
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            _, preds = outputs.max(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(
        "Global Federated Model - Confusion Matrix\n"
        "(Brain Tumor MRI 4-class + Pneumonia X-Ray 2-class)",
        fontsize=13, pad=14,
    )
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved -> {save_path}")


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Run federated training first (run.bat).")
        sys.exit(1)

    print(f"Loading global model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    print("Loading global test dataset ...")
    loader = get_test_loader(batch_size=64)
    ds = loader.dataset
    print(f"  {len(ds)} test samples  |  {len(loader)} batches")

    if hasattr(ds, "class_distribution"):
        dist = ds.class_distribution()
        print("  Per-class counts in test set:")
        for cls, cnt in dist.items():
            print(f"    {cls:12s}: {cnt}")

    print("\nRunning inference ...")
    y_true, y_pred = run_inference(model, loader)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  Accuracy  (overall)    : {acc  * 100:6.2f} %")
    print(f"  Precision (weighted)   : {prec * 100:6.2f} %")
    print(f"  Recall    (weighted)   : {rec  * 100:6.2f} %")
    print(f"  F1-Score  (weighted)   : {f1   * 100:6.2f} %")
    print(sep)
    print("\nPer-class report:")
    print(classification_report(
        y_true, y_pred,
        target_names=SHORT_NAMES,
        zero_division=0,
    ))
    plot_confusion_matrix(y_true, y_pred, OUTPUT_PATH)


if __name__ == "__main__":
    main()


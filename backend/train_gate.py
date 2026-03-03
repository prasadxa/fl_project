"""
train_gate.py — Train the 4-class Scan-Type Gatekeeper (ScanGate)
==================================================================
Trains an EfficientNet-B0 classifier to distinguish:

    0  chest_xray
    1  brain_mri
    2  ct_scan       (sourced from existing partitions or downloaded)
    3  non_medical   (CIFAR-10 subset: cats, dogs, cars, ships, horses, deer)

Data sources (auto-detected):
    Chest X-Ray  → data/partitions/global_test/normal  +  data/partitions/global_test/pneumonia
    Brain MRI    → data/partitions/global_test/glioma  +  data/partitions/global_test/meningioma
                   + data/partitions/global_test/notumor + data/partitions/global_test/pituitary
    CT Scan      → data/gate_data/ct_scan  (if present, else skipped / synthetic)
    Non-Medical  → data/gate_data/non_medical  (if present)
                   PLUS CIFAR-10 downloaded automatically via torchvision

Output:
    models/scan_gate.pth   — state_dict only (loaded by scan_classifier.py)

Usage:
    cd fl_project
    python backend/train_gate.py
    python backend/train_gate.py --epochs 20 --batch-size 64 --lr 3e-4
    python backend/train_gate.py --no-cifar   # skip CIFAR-10 download
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ── project paths ──────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent
_PROJ_ROOT = _BACKEND_DIR.parent
sys.path.insert(0, str(_BACKEND_DIR))

from scan_classifier import (
    GATE_CLASS_IDX,
    GATE_CLASSES,
    NUM_GATE_CLASSES,
    build_gate_model,
)

# ── directories ────────────────────────────────────────────────────────────────
PARTITIONS_DIR = _PROJ_ROOT / "data" / "partitions"
GATE_DATA_DIR = _PROJ_ROOT / "data" / "gate_data"
CIFAR_DIR = _PROJ_ROOT / "data" / "cifar10_cache"
MODELS_DIR = _PROJ_ROOT / "models"
GATE_MODEL_OUT = MODELS_DIR / "scan_gate.pth"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
GATE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── CIFAR-10 classes used as "non_medical" ─────────────────────────────────────
# Deliberately exclude "airplane" and "truck" which could look slightly like
# radiology screenshots; keep natural / everyday objects.
CIFAR_NON_MEDICAL_CLASSES = [
    "cat",
    "dog",
    "horse",
    "deer",
    "ship",
    "automobile",
    "bird",
    "frog",
]

# ── transforms ─────────────────────────────────────────────────────────────────
_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ── dataset helpers ────────────────────────────────────────────────────────────


def _collect_image_paths(directory: Path) -> List[Path]:
    """Recursively collect all image paths under a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    if not directory.exists():
        return paths
    for p in directory.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            paths.append(p)
    return paths


class GateDataset(Dataset):
    """
    Dataset for scan-gate training.  Each sample is (image_path, label_index).

    Medical images are loaded as RGB (even though they are grayscale — the
    EfficientNet needs 3-channel input and colour channels carry the saturation
    signal needed to reject coloured photos).
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform=None,
    ):
        self.samples = samples
        self.transform = transform or _VAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return a black image on load failure rather than crashing
            img = Image.new("RGB", (224, 224), 0)
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in GATE_CLASSES}
        for _, lbl in self.samples:
            counts[GATE_CLASSES[lbl]] += 1
        return counts


class CIFARRGBDataset(Dataset):
    """
    Wraps torchvision CIFAR-10 and filters to the non_medical classes.
    Returns (RGB tensor, NON_MEDICAL_LABEL).
    """

    def __init__(
        self, root: Path, train: bool = True, transform=None, max_per_class: int = 800
    ):
        from torchvision.datasets import CIFAR10

        raw = CIFAR10(root=str(root), train=train, download=True)
        # CIFAR-10 class names
        cifar_classes = raw.classes  # list of 10 names
        allowed_idx = {
            i for i, c in enumerate(cifar_classes) if c in CIFAR_NON_MEDICAL_CLASSES
        }

        # Filter and balance
        class_buckets: Dict[int, List[int]] = {i: [] for i in allowed_idx}
        for sample_idx, (_, target) in enumerate(raw):
            if target in allowed_idx:
                class_buckets[target].append(sample_idx)

        self.indices: List[int] = []
        for bucket in class_buckets.values():
            random.shuffle(bucket)
            self.indices.extend(bucket[:max_per_class])

        self.raw = raw
        self.transform = transform or _VAL_TRANSFORM
        self.label = GATE_CLASS_IDX["non_medical"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        pil_img, _ = self.raw[real_idx]
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, self.label


# ── data assembly ──────────────────────────────────────────────────────────────


def _build_medical_samples(
    max_per_class: int = 1200,
) -> Dict[str, List[Tuple[Path, int]]]:
    """
    Collect image paths from existing partitions and return a dict
    {gate_class_label: [(path, label_idx), ...]}.
    """
    samples: Dict[str, List[Tuple[Path, int]]] = {c: [] for c in GATE_CLASSES}

    # ── Chest X-Ray ────────────────────────────────────────────────────────────
    xray_dirs = [
        PARTITIONS_DIR / "global_test" / "normal",
        PARTITIONS_DIR / "global_test" / "pneumonia",
        GATE_DATA_DIR / "chest_xray",
    ]
    lbl = GATE_CLASS_IDX["chest_xray"]
    for d in xray_dirs:
        for p in _collect_image_paths(d):
            samples["chest_xray"].append((p, lbl))

    # ── Brain MRI ──────────────────────────────────────────────────────────────
    mri_dirs = [
        PARTITIONS_DIR / "global_test" / "glioma",
        PARTITIONS_DIR / "global_test" / "meningioma",
        PARTITIONS_DIR / "global_test" / "notumor",
        PARTITIONS_DIR / "global_test" / "pituitary",
        GATE_DATA_DIR / "brain_mri",
    ]
    lbl = GATE_CLASS_IDX["brain_mri"]
    for d in mri_dirs:
        for p in _collect_image_paths(d):
            samples["brain_mri"].append((p, lbl))

    # ── CT Scan ────────────────────────────────────────────────────────────────
    ct_dirs = [GATE_DATA_DIR / "ct_scan"]
    lbl = GATE_CLASS_IDX["ct_scan"]
    for d in ct_dirs:
        for p in _collect_image_paths(d):
            samples["ct_scan"].append((p, lbl))

    # ── Non-Medical (file-based, if present) ───────────────────────────────────
    non_med_dirs = [GATE_DATA_DIR / "non_medical"]
    lbl = GATE_CLASS_IDX["non_medical"]
    for d in non_med_dirs:
        for p in _collect_image_paths(d):
            samples["non_medical"].append((p, lbl))

    # ── Cap per class to avoid severe imbalance ────────────────────────────────
    for cls in GATE_CLASSES:
        lst = samples[cls]
        random.shuffle(lst)
        samples[cls] = lst[:max_per_class]

    return samples


def _split_samples(
    samples_per_class: Dict[str, List[Tuple[Path, int]]],
    val_frac: float = 0.15,
) -> Tuple[List, List]:
    """Split each class into train/val, preserving class ratio."""
    train_all: List = []
    val_all: List = []
    for cls, lst in samples_per_class.items():
        random.shuffle(lst)
        n_val = max(1, int(len(lst) * val_frac))
        val_all.extend(lst[:n_val])
        train_all.extend(lst[n_val:])
    return train_all, val_all


def _print_class_dist(label: str, counts: Dict[str, int]) -> None:
    total = sum(counts.values())
    print(f"\n  {label} class distribution (total={total}):")
    for cls, cnt in counts.items():
        bar = "█" * (cnt * 30 // (total + 1))
        print(f"    {cls:<14}  {cnt:>5}  {bar}")


# ── training loop ──────────────────────────────────────────────────────────────


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_weighted_sampler(samples: List[Tuple]) -> WeightedRandomSampler:
    """Over-sample minority classes so each class appears equally per epoch."""
    class_counts = [0] * NUM_GATE_CLASSES
    for _, lbl in samples:
        class_counts[lbl] += 1
    class_weights = [1.0 / (c + 1e-8) for c in class_counts]
    sample_weights = [class_weights[lbl] for _, lbl in samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    """Returns (loss, accuracy) on a data loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)
    return total_loss / (total + 1e-8), correct / (total + 1e-8)


def _per_class_acc(
    model: nn.Module, loader: DataLoader, device: str
) -> Dict[str, float]:
    model.eval()
    correct = [0] * NUM_GATE_CLASSES
    totals = [0] * NUM_GATE_CLASSES
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs).argmax(dim=1)
            for p, g in zip(preds.cpu(), labels.cpu()):
                totals[g.item()] += 1
                if p.item() == g.item():
                    correct[g.item()] += 1
    return {
        GATE_CLASSES[i]: (correct[i] / totals[i] if totals[i] > 0 else 0.0)
        for i in range(NUM_GATE_CLASSES)
    }


def train(
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    use_cifar: bool = True,
    max_per_class: int = 1200,
    cifar_per_class: int = 800,
    val_frac: float = 0.15,
    seed: int = 42,
    patience: int = 6,
) -> Path:
    """
    Full training pipeline.  Returns path to the saved model.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _pick_device()
    print(f"\n{'=' * 60}")
    print(f"  ScanGate Training")
    print(f"  device={device}  epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"{'=' * 60}")

    # ── assemble file-based samples ────────────────────────────────────────────
    print("\n[1/4] Collecting image paths from partitions …")
    med_samples = _build_medical_samples(max_per_class=max_per_class)

    for cls in GATE_CLASSES:
        n = len(med_samples.get(cls, []))
        print(f"      {cls:<14}  {n} images (file-based)")

    if use_cifar and len(med_samples.get("non_medical", [])) < 200:
        print("\n[2/4] Downloading / loading CIFAR-10 for non_medical class …")
        CIFAR_DIR.mkdir(parents=True, exist_ok=True)
        try:
            cifar_train = CIFARRGBDataset(
                CIFAR_DIR, train=True, max_per_class=cifar_per_class
            )
            cifar_val = CIFARRGBDataset(
                CIFAR_DIR, train=False, max_per_class=int(cifar_per_class * val_frac)
            )
            print(
                f"      CIFAR-10 non_medical: {len(cifar_train)} train  {len(cifar_val)} val"
            )
        except Exception as e:
            print(f"      WARNING: CIFAR-10 failed ({e}). Continuing without it.")
            cifar_train = cifar_val = None
    else:
        print("\n[2/4] Skipping CIFAR-10 (non_medical data present or --no-cifar).")
        cifar_train = cifar_val = None

    # ── split medical samples ──────────────────────────────────────────────────
    all_samples_map = {cls: samples for cls, samples in med_samples.items()}
    train_samples, val_samples = _split_samples(all_samples_map, val_frac=val_frac)

    # ── check we have enough data ──────────────────────────────────────────────
    available_classes = [
        cls
        for cls in GATE_CLASSES
        if any(GATE_CLASSES[lbl] == cls for _, lbl in train_samples)
    ]

    if "ct_scan" not in available_classes:
        print("\n  NOTE: No CT scan data found. The model will only distinguish")
        print("        chest_xray / brain_mri / non_medical. Place CT images in")
        print(f"        {GATE_DATA_DIR / 'ct_scan'} and retrain for best results.")

    # ── build datasets ─────────────────────────────────────────────────────────
    print("\n[3/4] Building datasets …")
    train_ds = GateDataset(train_samples, transform=_TRAIN_TRANSFORM)
    val_ds = GateDataset(val_samples, transform=_VAL_TRANSFORM)
    _print_class_dist("Train", train_ds.class_counts())

    # Attach CIFAR non_medical if available
    if cifar_train is not None:
        from torch.utils.data import ConcatDataset

        train_ds = ConcatDataset([train_ds, cifar_train])
        val_ds = ConcatDataset([val_ds, cifar_val])
        print(f"\n      After CIFAR concat: {len(train_ds)} train  {len(val_ds)} val")

    # Weighted sampler on the file-based train dataset only (before concat)
    # After concat we can't easily do per-sample weighting, so use simple shuffle
    if cifar_train is None:
        sampler = _make_weighted_sampler(train_samples)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # ── build model ────────────────────────────────────────────────────────────
    print("\n[4/4] Building EfficientNet-B0 (ImageNet pretrained) …")
    model = build_gate_model(pretrained=True)
    model.to(device)

    # Freeze feature extractor for first few epochs, then unfreeze all
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # ── training loop ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(
        f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}"
    )
    print(f"{'─' * 60}")

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    UNFREEZE_EPOCH = max(3, epochs // 5)  # unfreeze after ~20% of training

    for epoch in range(1, epochs + 1):
        # Unfreeze all layers after warm-up
        if epoch == UNFREEZE_EPOCH:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr * 0.1,  # lower LR for fine-tuning backbone
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch
            )
            print(f"  [Epoch {epoch}] Unfreezing backbone with lr={lr * 0.1:.2e}")

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_correct += out.argmax(1).eq(labels).sum().item()
            running_total += imgs.size(0)

        scheduler.step()

        train_loss = running_loss / (running_total + 1e-8)
        train_acc = running_correct / (running_total + 1e-8)
        val_loss, val_acc = _evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc * 100:>8.1f}%  "
            f"{val_loss:>9.4f}  {val_acc * 100:>7.1f}%  {current_lr:>8.2e}  "
            f"({elapsed:.1f}s)"
        )

        # ── save best ─────────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), GATE_MODEL_OUT)
            print(f"  ✓ Saved best model (val_acc={val_acc * 100:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)."
                )
                break

    print(f"\n{'─' * 60}")
    print(
        f"  Training complete. Best val_acc={best_val_acc * 100:.1f}% at epoch {best_epoch}"
    )
    print(f"  Model saved → {GATE_MODEL_OUT}")

    # ── final per-class evaluation ─────────────────────────────────────────────
    # Reload best checkpoint
    model.load_state_dict(
        torch.load(GATE_MODEL_OUT, map_location=device, weights_only=True)
    )
    per_cls = _per_class_acc(model, val_loader, device)
    print(f"\n  Per-class accuracy on validation set:")
    for cls, acc in per_cls.items():
        bar = "█" * int(acc * 20)
        print(f"    {cls:<14}  {acc * 100:>6.1f}%  {bar}")

    print(f"\n{'=' * 60}\n")
    return GATE_MODEL_OUT


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the ScanGate 4-class scan-type classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    p.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay"
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=1200,
        help="Max images per class from file system (before CIFAR)",
    )
    p.add_argument(
        "--cifar-per-class",
        type=int,
        default=800,
        help="Max CIFAR-10 images per class for non_medical",
    )
    p.add_argument(
        "--val-frac", type=float, default=0.15, help="Validation split fraction"
    )
    p.add_argument(
        "--patience", type=int, default=6, help="Early stopping patience (epochs)"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--no-cifar",
        action="store_true",
        default=False,
        help="Skip CIFAR-10 download / non_medical augmentation",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Confidence threshold baked into log (informational only)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_cifar=not args.no_cifar,
        max_per_class=args.max_per_class,
        cifar_per_class=args.cifar_per_class,
        val_frac=args.val_frac,
        seed=args.seed,
        patience=args.patience,
    )
    print(f"Done. Model at: {out}")

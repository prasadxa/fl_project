"""
Phase 3 - Dataset Utilities (6-class Brain Tumor + Pneumonia version)
=====================================================================
Classes:
    0  glioma       -- Brain Tumor MRI
    1  meningioma   -- Brain Tumor MRI
    2  notumor      -- Brain Tumor MRI (no tumor)
    3  pituitary    -- Brain Tumor MRI
    4  normal       -- Chest X-Ray (healthy)
    5  pneumonia    -- Chest X-Ray (pneumonia)

MedicalImageDataset : loads preprocessed images from a partition folder.
get_client_loader   : DataLoader for a specific client partition.
get_test_loader     : DataLoader for the global test partition.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ── Project layout ────────────────────────────────────────────────────────────
_SRC_DIR       = Path(__file__).parent
PROJ_ROOT      = _SRC_DIR.parent
PARTITIONS_DIR = PROJ_ROOT / "data" / "partitions"

# ── Class registry (6 classes: brain tumor + pneumonia) ──────────────────────────
CLASS_NAMES: List[str] = [
    "glioma",       # 0
    "meningioma",   # 1
    "notumor",      # 2
    "pituitary",    # 3
    "normal",       # 4
    "pneumonia",    # 5
]
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)  # 6

# ── CLAHE custom transform ────────────────────────────────────────────────────
class CLAHETransform:
    """
    Contrast Limited Adaptive Histogram Equalisation (CLAHE).
    Applied on PIL greyscale images before converting to tensor.
    Significantly improves tumour edge visibility in MRI and X-ray scans.
    clipLimit   : controls contrast amplification (2.0 is standard clinical setting)
    tileGridSize: neighbourhood size for local histogram (8x8 is standard)
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                       tileGridSize=tile_grid_size)

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img, dtype=np.uint8)
        enhanced = self.clahe.apply(arr)
        return Image.fromarray(enhanced)


# ── Default transform (inference / evaluation — NO augmentation) ──────────────
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),  # enhance contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ── Training transform (WITH augmentation to improve generalisation) ──────────
# Augmentations chosen specifically for medical images (MRI + chest X-ray):
#   RandomHorizontalFlip  — tumours can appear on either side
#   RandomVerticalFlip    — valid for axial MRI slices
#   RandomRotation(15°)   — scanner/patient positioning variation
#   RandomAffine          — small translation + scale simulate patient movement
#   RandomAutocontrast    — simulate different scanner brightness levels
#   RandomAdjustSharpness — simulate varying MRI sharpness settings
#   CLAHE                 — always applied: enhances tumour boundary visibility
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
    ),
    transforms.RandomAutocontrast(p=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])



# ─────────────────────────────────────────────────────────────────────────────
class MedicalImageDataset(Dataset):
    """
    Multi-class medical image dataset.

    root_dir should contain one sub-folder per class named after CLASS_NAMES.
    Unrecognised folders are silently skipped.
    """

    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM
        self.samples: List[Tuple[Path, int]] = []

        for cls_name, label in CLASS_TO_IDX.items():
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in '{self.root_dir}'. "
                "Run src/preprocess.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_distribution(self) -> dict:
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
def get_client_loader(
    client_id: int,
    batch_size: int  = 32,
    num_workers: int = 0,
    augment: bool    = True,
) -> DataLoader:
    """
    Return a DataLoader for client_id's partition.

    Uses WeightedRandomSampler so that each training batch sees a balanced
    class distribution regardless of the raw class counts in the partition.
    WeightedRandomSampler is used instead of (or together with) class-weighted
    loss to give the model equal exposure to every class at the batch level.
    """
    transform = TRAIN_TRANSFORM if augment else DEFAULT_TRANSFORM
    dataset   = MedicalImageDataset(
        PARTITIONS_DIR / f"client_{client_id}",
        transform=transform,
    )

    # -- WeightedRandomSampler: assign higher sampling probability to rare classes
    dist = dataset.class_distribution()    # {class_name: count}
    class_counts = [dist.get(cls, 0) for cls in CLASS_NAMES]
    # weight per class = 1 / count  (0-count classes get weight 0)
    class_weight_map = [
        1.0 / c if c > 0 else 0.0 for c in class_counts
    ]
    # assign each sample the weight of its class
    sample_weights = [
        class_weight_map[label] for _, label in dataset.samples
    ]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )

    # shuffle=False because WeightedRandomSampler already randomises order
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = num_workers,
    )


def get_test_loader(
    batch_size: int  = 32,
    num_workers: int = 0,
) -> DataLoader:
    dataset = MedicalImageDataset(PARTITIONS_DIR / "global_test")
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)


def compute_class_weights(loader: DataLoader) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a DataLoader's dataset.

    Formula (sklearn convention):
        weight_i = total_samples / (num_classes * count_i)

    A class with fewer samples gets a HIGHER weight, so the loss penalises
    misclassifying rare tumour types (e.g. meningioma) proportionally more
    than over-represented classes (e.g. notumor).  This directly addresses
    the problem of the model defaulting to 'No Tumor' predictions.

    Returns a float32 tensor of shape (num_classes,) on CPU.
    """
    dist   = loader.dataset.class_distribution()       # {class_name: count}
    total  = sum(dist.values())
    weights: List[float] = []
    for cls in CLASS_NAMES:
        count = dist.get(cls, 0)
        if count == 0:
            weights.append(1.0)   # fallback: neutral weight for absent class
        else:
            weights.append(total / (NUM_CLASSES * count))
    return torch.tensor(weights, dtype=torch.float32)

"""
Phase 3 - Dataset Utilities (6-class multi-dataset version)
=============================================================
Classes:
    0  glioma       -- Brain Tumor MRI
    1  meningioma   -- Brain Tumor MRI
    2  notumor      -- Brain Tumor MRI (no tumor)
    3  pituitary    -- Brain Tumor MRI
    4  normal       -- Pneumonia X-Ray (healthy lung)
    5  pneumonia    -- Pneumonia X-Ray

MedicalImageDataset : loads preprocessed images from a partition folder.
get_client_loader   : DataLoader for a specific client partition.
get_test_loader     : DataLoader for the global test partition.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Project layout ────────────────────────────────────────────────────────────
_SRC_DIR       = Path(__file__).parent
PROJ_ROOT      = _SRC_DIR.parent
PARTITIONS_DIR = PROJ_ROOT / "data" / "partitions"

# ── Class registry ────────────────────────────────────────────────────────────
CLASS_NAMES: List[str] = [
    "glioma",       # 0
    "meningioma",   # 1
    "notumor",      # 2
    "pituitary",    # 3
    "normal",       # 4
    "pneumonia",    # 5
]
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)

# ── Default transform (inference / evaluation) ───────────────────────────────
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
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
    shuffle: bool    = True,
) -> DataLoader:
    dataset = MedicalImageDataset(
        PARTITIONS_DIR / f"client_{client_id}",
        transform=DEFAULT_TRANSFORM,
    )
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)


def get_test_loader(
    batch_size: int  = 32,
    num_workers: int = 0,
) -> DataLoader:
    dataset = MedicalImageDataset(PARTITIONS_DIR / "global_test")
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)

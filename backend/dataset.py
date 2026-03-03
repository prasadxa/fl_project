from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

PARTITIONS_DIR = Path(__file__).parent.parent / "data" / "partitions"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary", "normal", "pneumonia"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = None  # created lazily per-process so cv2.CLAHE is picklable

    def _get_clahe(self):
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
            )
        return self._clahe

    def __call__(self, pil_img):
        return Image.fromarray(
            self._get_clahe().apply(np.array(pil_img, dtype=np.uint8))
        )

    # Make picklable for multiprocessing DataLoader workers:
    # drop the unpicklable cv2.CLAHE handle; it will be recreated in each worker.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_clahe"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        CLAHETransform(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or DEFAULT_TRANSFORM
        self.samples = []
        for cls_name, label in CLASS_TO_IDX.items():
            cls_dir = self.root_dir / cls_name
            if cls_dir.exists():
                self.samples.extend(
                    [
                        (p, label)
                        for p in cls_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    ]
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        return self.transform(img) if self.transform else img, label

    def class_distribution(self):
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts


def get_client_loader(
    client_id,
    batch_size=32,
    num_workers=0,
    augment=True,
    pin_memory=False,
    prefetch_factor=None,
    persistent_workers=False,
):
    dataset = MedicalImageDataset(
        PARTITIONS_DIR / f"client_{client_id}",
        transform=TRAIN_TRANSFORM if augment else DEFAULT_TRANSFORM,
    )
    dist = dataset.class_distribution()
    class_counts = [dist.get(cls, 0) for cls in CLASS_NAMES]
    weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [weights[label] for _, label in dataset.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    loader_kwargs = dict(
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def get_test_loader(
    batch_size=32,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=None,
    persistent_workers=False,
):
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        MedicalImageDataset(PARTITIONS_DIR / "global_test"),
        **loader_kwargs,
    )


def compute_class_weights(loader):
    dist = loader.dataset.class_distribution()
    total = sum(dist.values())
    weights = [
        total / (NUM_CLASSES * dist.get(cls, 0)) if dist.get(cls, 0) > 0 else 1.0
        for cls in CLASS_NAMES
    ]
    return torch.tensor(weights, dtype=torch.float32)

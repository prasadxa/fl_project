"""
Phase 4 - Flower Federated-Learning Client (6-class) with Continual Learning
=============================================================================
Before each training round the client:
  1. Scans data/new_collected_data/ for doctor-verified images.
  2. Merges them into the existing partition dataset.
  3. Trains on the combined (base + new) data.
  4. Archives processed images to data/archived_data/ so they are
     not treated as 'new' in future rounds.

Usage:
    python src/client.py --client_id 1
    python src/client.py --client_id 2
    python src/client.py --client_id 3
"""

from __future__ import annotations
import argparse, shutil, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Suppress Flower's verbose "DEPRECATED FEATURE" log records.
from utils import suppress_flwr_deprecation_warnings
suppress_flwr_deprecation_warnings()

import torch
from torch.utils.data import ConcatDataset, DataLoader
import flwr as fl
from flwr.client import start_client  # non-deprecated replacement for start_numpy_client
from dataset import CLASS_NAMES, NUM_CLASSES, MedicalImageDataset, get_client_loader
from model import MedicalCNN, evaluate, get_parameters, set_parameters, train_one_round

warnings.filterwarnings("ignore")
DEVICE = "cpu"

PROJ_ROOT    = Path(__file__).parent.parent
NEW_DATA_DIR = PROJ_ROOT / "data" / "new_collected_data"
ARCHIVE_DIR  = PROJ_ROOT / "data" / "archived_data"


# ── continual-learning helpers ────────────────────────────────────────────────

def _count_new_images() -> int:
    """Return total number of images waiting in new_collected_data/."""
    if not NEW_DATA_DIR.exists():
        return 0
    return sum(
        1 for p in NEW_DATA_DIR.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )


def _build_combined_loader(
    base_loader: DataLoader,
    batch_size: int,
) -> DataLoader:
    """
    If new_collected_data contains doctor-verified images, merge them with the
    base partition dataset and return a combined DataLoader.
    Uses torchvision.datasets.ImageFolder so it correctly reads class subfolders.
    """
    n_new = _count_new_images()
    if n_new == 0:
        return base_loader

    print(f"[Continual] Found {n_new} new doctor-verified image(s) — merging into training set")

    from torchvision import datasets as tvdatasets
    from dataset import DEFAULT_TRANSFORM

    try:
        new_ds = tvdatasets.ImageFolder(str(NEW_DATA_DIR), transform=DEFAULT_TRANSFORM)
        combined = ConcatDataset([base_loader.dataset, new_ds])
        print(f"[Continual] Combined dataset: {len(base_loader.dataset)} base "
              f"+ {len(new_ds)} new = {len(combined)} total")
        return DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception as exc:
        print(f"[Continual] WARNING: could not load new images ({exc}) — using base loader only")
        return base_loader


def _archive_new_images() -> int:
    """Move all files from new_collected_data/ to archived_data/ after a round."""
    if not NEW_DATA_DIR.exists():
        return 0
    moved = 0
    for src in list(NEW_DATA_DIR.rglob("*")):
        if src.is_file() and src.suffix.lower() in (".jpg", ".jpeg", ".png"):
            relative = src.relative_to(NEW_DATA_DIR)
            dst      = ARCHIVE_DIR / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved += 1
    if moved:
        print(f"[Continual] Archived {moved} image(s) to data/archived_data/")
    return moved


# ── Flower client ─────────────────────────────────────────────────────────────

class MedicalFLClient(fl.client.NumPyClient):

    def __init__(self, client_id: int, batch_size: int = 16, local_epochs: int = 2):
        self.client_id    = client_id
        self.local_epochs = local_epochs
        self.batch_size   = batch_size
        self.model        = MedicalCNN(num_classes=NUM_CLASSES)
        self.base_loader  = get_client_loader(client_id, batch_size=batch_size)
        n    = len(self.base_loader.dataset)
        dist = self.base_loader.dataset.class_distribution()
        print(f"[Client {client_id}] Ready - {n} samples, {local_epochs} local epochs")
        print(f"[Client {client_id}] " + "  ".join(f"{k}={v}" for k, v in dist.items()))

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", self.local_epochs))

        # --- continual learning: merge any new doctor-verified images ---
        loader = _build_combined_loader(self.base_loader, self.batch_size)

        loss, acc = train_one_round(self.model, loader, epochs=local_epochs, device=DEVICE)
        print(f"[Client {self.client_id}] fit  -> loss={loss:.4f}  acc={acc:.4f}  "
              f"(samples={len(loader.dataset)})")

        # --- archive new images so they are not re-used next round ---
        _archive_new_images()

        return get_parameters(self.model), len(loader.dataset), {"loss": float(loss), "accuracy": float(acc)}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.base_loader, device=DEVICE)
        print(f"[Client {self.client_id}] eval -> loss={loss:.4f}  acc={acc:.4f}")
        return float(loss), len(self.base_loader.dataset), {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id",  type=int, required=True)
    parser.add_argument("--server",     type=str, default="localhost:8080")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=2)
    args = parser.parse_args()
    client = MedicalFLClient(
        client_id=args.client_id,
        batch_size=args.batch_size,
        local_epochs=args.epochs,
    )
    # to_client() wraps NumPyClient → Client; insecure=True matches the plain-gRPC server
    start_client(server_address=args.server, client=client.to_client(), insecure=True)


if __name__ == "__main__":
    main()

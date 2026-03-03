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

import argparse
import shutil
import sys
import warnings
from pathlib import Path

# Fix module structure
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets as tvdatasets
import flwr as fl
from flwr.client import start_client

from utils import suppress_flwr_deprecation_warnings
from dataset import CLASS_NAMES, NUM_CLASSES, MedicalImageDataset, compute_class_weights, get_client_loader, DEFAULT_TRANSFORM
from model import MedicalCNN, evaluate, get_parameters, set_parameters, train_one_round

suppress_flwr_deprecation_warnings()
warnings.filterwarnings("ignore")

DEVICE = "cpu"
PROJ_ROOT = Path(__file__).parent.parent
NEW_DATA_DIR = PROJ_ROOT / "data" / "new_collected_data"
ARCHIVE_DIR = PROJ_ROOT / "data" / "archived_data"


def _count_new_images() -> int:
    return sum(1 for p in NEW_DATA_DIR.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()) if NEW_DATA_DIR.exists() else 0

def _build_combined_loader(base_loader: DataLoader, batch_size: int) -> DataLoader:
    if _count_new_images() == 0:
        return base_loader
    try:
        new_ds = tvdatasets.ImageFolder(str(NEW_DATA_DIR), transform=DEFAULT_TRANSFORM)
        return DataLoader(ConcatDataset([base_loader.dataset, new_ds]), batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception:
        return base_loader

def _archive_new_images():
    if not NEW_DATA_DIR.exists(): return
    for src in NEW_DATA_DIR.rglob("*"):
        if src.is_file() and src.suffix.lower() in (".jpg", ".jpeg", ".png"):
            dst = ARCHIVE_DIR / src.relative_to(NEW_DATA_DIR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

class MedicalFLClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, batch_size: int = 16, local_epochs: int = 10):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.model = MedicalCNN(num_classes=NUM_CLASSES)
        self.base_loader = get_client_loader(client_id, batch_size=batch_size)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        loader = _build_combined_loader(self.base_loader, self.batch_size)
        loss, acc = train_one_round(
            self.model, loader,
            epochs=int(config.get("local_epochs", self.local_epochs)),
            device=DEVICE,
            class_weights=compute_class_weights(loader)
        )
        _archive_new_images()
        return get_parameters(self.model), len(loader.dataset), {"loss": float(loss), "accuracy": float(acc)}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.base_loader, device=DEVICE)
        return float(loss), len(self.base_loader.dataset), {"accuracy": float(acc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--server", type=str, default="localhost:8080")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    
    start_client(server_address=args.server, client=MedicalFLClient(args.client_id, args.batch_size, args.epochs).to_client(), insecure=True)


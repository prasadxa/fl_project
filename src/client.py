"""
Phase 4 - Flower Federated-Learning Client (6-class)
=====================================================
Usage:
    python src/client.py --client_id 1
    python src/client.py --client_id 2
    python src/client.py --client_id 3
"""

from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import flwr as fl
from dataset import NUM_CLASSES, get_client_loader
from model import MedicalCNN, evaluate, get_parameters, set_parameters, train_one_round

warnings.filterwarnings("ignore")
DEVICE = "cpu"


class MedicalFLClient(fl.client.NumPyClient):

    def __init__(self, client_id: int, batch_size: int = 16, local_epochs: int = 2):
        self.client_id    = client_id
        self.local_epochs = local_epochs
        self.model        = MedicalCNN(num_classes=NUM_CLASSES)
        self.loader       = get_client_loader(client_id, batch_size=batch_size)
        n    = len(self.loader.dataset)
        dist = self.loader.dataset.class_distribution()
        print(f"[Client {client_id}] Ready - {n} samples, {local_epochs} local epochs")
        print(f"[Client {client_id}] " + "  ".join(f"{k}={v}" for k, v in dist.items()))

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        loss, acc = train_one_round(self.model, self.loader, epochs=local_epochs, device=DEVICE)
        print(f"[Client {self.client_id}] fit  -> loss={loss:.4f}  acc={acc:.4f}")
        return get_parameters(self.model), len(self.loader.dataset), {"loss": float(loss), "accuracy": float(acc)}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.loader, device=DEVICE)
        print(f"[Client {self.client_id}] eval -> loss={loss:.4f}  acc={acc:.4f}")
        return float(loss), len(self.loader.dataset), {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id",  type=int, required=True)
    parser.add_argument("--server",     type=str, default="localhost:8080")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=2)
    args = parser.parse_args()
    client = MedicalFLClient(client_id=args.client_id, batch_size=args.batch_size, local_epochs=args.epochs)
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()

"""
Phase 5 - Flower Federated-Learning Server (6-class)
=====================================================
FedAvg, 5 rounds, 3 clients, saves global model checkpoints.

Usage (run from project root):
    python src/server.py
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import flwr as fl
import torch
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from dataset import NUM_CLASSES
from model import MedicalCNN, get_parameters, set_parameters

PROJ_ROOT  = Path(__file__).parent.parent
MODELS_DIR = PROJ_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

NUM_ROUNDS  = 5
MIN_CLIENTS = 3


class FedAvgSaveModel(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            model    = MedicalCNN(num_classes=NUM_CLASSES)
            set_parameters(model, ndarrays)

            round_path  = MODELS_DIR / f"global_model_round_{server_round}.pth"
            latest_path = MODELS_DIR / "global_model.pth"
            torch.save(model.state_dict(), round_path)
            torch.save(model.state_dict(), latest_path)
            print(f"[Server] Round {server_round}/{NUM_ROUNDS} - model saved -> {round_path.name}")

        return aggregated_parameters, aggregated_metrics


def main():
    init_model      = MedicalCNN(num_classes=NUM_CLASSES)
    init_parameters = ndarrays_to_parameters(get_parameters(init_model))

    strategy = FedAvgSaveModel(
        initial_parameters=init_parameters,
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        on_fit_config_fn=lambda rnd: {"round": rnd, "local_epochs": 2},
    )

    print(f"[Server] Starting - 6-class medical image FL (Brain Tumor + Pneumonia)")
    print(f"[Server] {NUM_ROUNDS} rounds | waiting for {MIN_CLIENTS} clients")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

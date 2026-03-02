"""
Phase 5 - Flower Federated-Learning Server (Continuous / Always-On Mode)
=========================================================================
FedAvg with no fixed round limit — the server runs continuously and
aggregates whenever at least MIN_FIT_CLIENTS hospital clients connect.

Key changes from the fixed-round version:
  - NUM_ROUNDS replaced with a very large ceiling (effectively infinite).
  - MIN_FIT_CLIENTS = 2 (aggregation triggers with any 2 hospitals).
  - Timestamps on every log line so operators know exactly when each
    global model version was produced.
  - global_model.pth is always overwritten with the latest aggregated weights.

Usage (run from project root):
    python src/server.py
    python src/server.py --rounds 9999   # explicit ceiling (default = 9999)
    python src/server.py --min_clients 3 # override minimum clients
"""

from __future__ import annotations
import argparse, sys, warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# Suppress Flower's verbose "DEPRECATED FEATURE" log records while keeping
# all INFO messages (round progress, model saves, etc.) and other warnings.
from utils import suppress_flwr_deprecation_warnings
suppress_flwr_deprecation_warnings()

import flwr as fl
import torch
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
# Use the compat layer directly to avoid the top-level deprecation wrapper
from flwr.compat.server.app import start_server as _start_server
from flwr.server import ServerConfig

from dataset import NUM_CLASSES
from model import MedicalCNN, get_parameters, set_parameters

PROJ_ROOT  = Path(__file__).parent.parent
MODELS_DIR = PROJ_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Continuous mode defaults
DEFAULT_ROUNDS      = 9999   # effectively infinite; stop manually with Ctrl+C
DEFAULT_MIN_CLIENTS = 2      # aggregate as soon as 2 hospitals send updates


def _ts() -> str:
    """Return a compact UTC timestamp string for log lines."""
    return datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


class FedAvgSaveModel(fl.server.strategy.FedAvg):
    """FedAvg that saves a timestamped checkpoint after every aggregation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._round_counter = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self._round_counter += 1
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            model    = MedicalCNN(num_classes=NUM_CLASSES)
            set_parameters(model, ndarrays)

            # Timestamped checkpoint so no previous round is ever overwritten
            ts_tag      = datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
            round_path  = MODELS_DIR / f"global_model_round_{server_round}_{ts_tag}.pth"
            latest_path = MODELS_DIR / "global_model.pth"

            torch.save(model.state_dict(), round_path)
            torch.save(model.state_dict(), latest_path)

            n_clients = len(results)
            print(
                f"[{_ts()}] [Server] Round {server_round} complete | "
                f"{n_clients} client(s) aggregated | "
                f"New global model -> {round_path.name}"
            )

        return aggregated_parameters, aggregated_metrics


def main():
    parser = argparse.ArgumentParser(description="Tecnomate FL Server (continuous mode)")
    parser.add_argument("--rounds",      type=int, default=DEFAULT_ROUNDS,
                        help="Max rounds ceiling (default=9999, effectively infinite)")
    parser.add_argument("--min_clients", type=int, default=DEFAULT_MIN_CLIENTS,
                        help="Min clients required to trigger aggregation (default=2)")
    parser.add_argument("--address",     type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    init_model  = MedicalCNN(num_classes=NUM_CLASSES)
    latest_path = MODELS_DIR / "global_model.pth"
    if latest_path.exists():
        init_model.load_state_dict(torch.load(latest_path, map_location="cpu"))
        print(f"[{_ts()}] [Server] Resuming from checkpoint : {latest_path.name}")
    else:
        print(f"[{_ts()}] [Server] No checkpoint found — starting from random weights")
    init_parameters = ndarrays_to_parameters(get_parameters(init_model))

    strategy = FedAvgSaveModel(
        initial_parameters=init_parameters,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        on_fit_config_fn=lambda rnd: {"round": rnd, "local_epochs": 2},
    )

    print(f"[{_ts()}] [Server] Tecnomate FL Server starting (continuous mode)")
    print(f"[{_ts()}] [Server] Min clients to aggregate : {args.min_clients}")
    print(f"[{_ts()}] [Server] Round ceiling            : {args.rounds}")
    print(f"[{_ts()}] [Server] Listening on             : {args.address}")
    print(f"[{_ts()}] [Server] Press Ctrl+C to stop gracefully")

    _start_server(
        server_address=args.address,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

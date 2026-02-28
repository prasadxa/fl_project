"""
Standalone FL Simulation — train_sim.py
========================================
Runs Federated Averaging (FedAvg) in a single Python process.

No network.  No gRPC.  No Ray.  No server/client processes.
The FedAvg math is identical to the real server.py + client.py stack:
  each client trains locally, then the server performs a sample-weighted
  average of all client weight tensors.

Checkpoint behaviour (mirrors server.py):
  - On startup, loads models/global_model.pth if it exists (resume by default).
  - After every round saves:
      models/global_model.pth                          ← always overwritten
      models/global_model_round_N.pth                  ← clean per-round file
      models/global_model_round_N_YYYYMMDD_HHMMSS.pth  ← timestamped archive

Usage (run from project root):
    python src/train_sim.py                        # 5 rounds, resume from checkpoint
    python src/train_sim.py --rounds 10            # 10 rounds
    python src/train_sim.py --rounds 5 --fresh     # ignore checkpoint, random init
    python src/train_sim.py --rounds 5 --eval      # evaluate on test set every round
    python src/train_sim.py --rounds 3 --epochs 1  # 1 local epoch per client (faster)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dataset import CLASS_NAMES, NUM_CLASSES, get_client_loader, get_test_loader
from model import (
    MedicalCNN,
    evaluate,
    get_parameters,
    set_parameters,
    train_one_round,
)

# ── constants ─────────────────────────────────────────────────────────────────
PROJ_ROOT   = Path(__file__).parent.parent
MODELS_DIR  = PROJ_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

NUM_CLIENTS    = 3
DEVICE         = "cpu"
DEFAULT_ROUNDS = 5
DEFAULT_EPOCHS = 2
DEFAULT_BATCH  = 32


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    """UTC timestamp string for log lines."""
    return datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _banner(text: str, width: int = 65, char: str = "=") -> None:
    print(char * width)
    print(f"  {text}")
    print(char * width)


def _hline(width: int = 65, char: str = "─") -> None:
    print(char * width)


# ─────────────────────────────────────────────────────────────────────────────
# FedAvg aggregation
# ─────────────────────────────────────────────────────────────────────────────

def fedavg(
    client_weights: List[List[np.ndarray]],
    client_sizes:   List[int],
) -> List[np.ndarray]:
    """
    Compute the sample-weighted average of model parameters across all clients.

    FedAvg formula:
        W_global = sum_i( n_i / N  *  W_i )
    where n_i is the number of training samples used by client i
    and N = sum(n_i).

    Parameters
    ----------
    client_weights : list of parameter lists, one per client
                     (output of get_parameters() for each trained local model)
    client_sizes   : number of training samples each client trained on

    Returns
    -------
    Averaged parameter list with the same structure as one client's list.
    """
    total = sum(client_sizes)
    if total == 0:
        raise ValueError("fedavg: total sample count is zero — no clients trained?")

    averaged: List[np.ndarray] = []
    num_layers = len(client_weights[0])

    for layer_idx in range(num_layers):
        weighted_sum = np.zeros_like(
            client_weights[0][layer_idx], dtype=np.float64
        )
        for i, weights in enumerate(client_weights):
            weighted_sum += weights[layer_idx].astype(np.float64) * (
                client_sizes[i] / total
            )
        averaged.append(weighted_sum.astype(client_weights[0][layer_idx].dtype))

    return averaged


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(model: MedicalCNN) -> bool:
    """
    Load models/global_model.pth into *model* in-place.

    Returns True if a checkpoint was loaded, False if none was found.
    """
    latest = MODELS_DIR / "global_model.pth"
    if not latest.exists():
        return False
    model.load_state_dict(torch.load(latest, map_location=DEVICE))
    return True


def save_checkpoint(model: MedicalCNN, round_num: int) -> None:
    """
    Save three files after each round (mirrors server.py behaviour):
      1. global_model.pth                          — latest, always overwritten
      2. global_model_round_N.pth                  — clean per-round copy
      3. global_model_round_N_YYYYMMDD_HHMMSS.pth  — timestamped archive
    """
    state    = model.state_dict()
    ts_tag   = datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

    latest      = MODELS_DIR / "global_model.pth"
    round_plain = MODELS_DIR / f"global_model_round_{round_num}.pth"
    round_ts    = MODELS_DIR / f"global_model_round_{round_num}_{ts_tag}.pth"

    torch.save(state, latest)
    torch.save(state, round_plain)
    torch.save(state, round_ts)

    print(f"[{_ts()}] [Save] global_model.pth  (latest)")
    print(f"[{_ts()}] [Save] {round_plain.name}")
    print(f"[{_ts()}] [Save] {round_ts.name}")


# ─────────────────────────────────────────────────────────────────────────────
# One FL round
# ─────────────────────────────────────────────────────────────────────────────

def run_round(
    global_model:   MedicalCNN,
    client_loaders: List[DataLoader],
    local_epochs:   int,
    round_num:      int,
) -> Tuple[MedicalCNN, float, float]:
    """
    Execute one complete FedAvg round:
      1. Distribute current global weights to all clients.
      2. Each client trains independently for *local_epochs* epochs.
      3. Aggregate updated weights using sample-weighted FedAvg.
      4. Write aggregated weights back into *global_model*.

    Returns
    -------
    global_model   : updated in-place and also returned for chaining
    round_loss     : sample-weighted mean training loss across all clients
    round_acc      : sample-weighted mean training accuracy across all clients
    """
    global_params = get_parameters(global_model)

    all_weights: List[List[np.ndarray]] = []
    all_sizes:   List[int]              = []
    all_losses:  List[float]            = []
    all_accs:    List[float]            = []

    for cid, loader in enumerate(client_loaders, start=1):
        t_start = time.perf_counter()

        # --- give client a fresh copy of the global weights ---
        local_model = MedicalCNN(num_classes=NUM_CLASSES)
        set_parameters(local_model, global_params)

        # --- local training ---
        loss, acc = train_one_round(
            local_model, loader,
            epochs=local_epochs,
            device=DEVICE,
        )
        elapsed = time.perf_counter() - t_start

        n = len(loader.dataset)
        print(
            f"[{_ts()}] [Client {cid}] "
            f"loss={loss:.4f}  acc={acc*100:.2f}%  "
            f"samples={n}  time={elapsed:.1f}s"
        )

        all_weights.append(get_parameters(local_model))
        all_sizes.append(n)
        all_losses.append(loss)
        all_accs.append(acc)

    # --- FedAvg aggregation ---
    avg_weights = fedavg(all_weights, all_sizes)
    set_parameters(global_model, avg_weights)

    total         = sum(all_sizes)
    round_loss    = sum(l * s for l, s in zip(all_losses, all_sizes)) / total
    round_acc     = sum(a * s for a, s in zip(all_accs,  all_sizes)) / total

    print(
        f"[{_ts()}] [Aggregated] "
        f"weighted loss={round_loss:.4f}  weighted acc={round_acc*100:.2f}%"
    )

    return global_model, round_loss, round_acc


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model: MedicalCNN, round_num: int) -> Tuple[float, float]:
    """
    Evaluate *model* on the global test partition.

    Returns (loss, accuracy) as floats.
    """
    loader     = get_test_loader(batch_size=64)
    loss, acc  = evaluate(model, loader, device=DEVICE)
    n          = len(loader.dataset)
    print(
        f"[{_ts()}] [Eval R{round_num:02d}] "
        f"test loss={loss:.4f}  test acc={acc*100:.2f}%  "
        f"({n} images)"
    )
    return loss, acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tecnomate — standalone FedAvg simulation (no network required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rounds",  type=int, default=DEFAULT_ROUNDS,
        help="Number of federated rounds to run",
    )
    parser.add_argument(
        "--epochs",  type=int, default=DEFAULT_EPOCHS,
        help="Local training epochs per client per round",
    )
    parser.add_argument(
        "--batch",   type=int, default=DEFAULT_BATCH,
        help="Mini-batch size for client data loaders",
    )
    parser.add_argument(
        "--fresh",   action="store_true",
        help="Start from random weights — ignore any existing global_model.pth",
    )
    parser.add_argument(
        "--eval",    action="store_true",
        help="Evaluate on global test set after every round (adds ~10s per round)",
    )
    args = parser.parse_args()

    # ── startup banner ────────────────────────────────────────────────────────
    _banner("Tecnomate  —  FedAvg Simulation  (single-process)")
    print(f"  Rounds        : {args.rounds}")
    print(f"  Local epochs  : {args.epochs}")
    print(f"  Batch size    : {args.batch}")
    print(f"  Clients       : {NUM_CLIENTS}")
    print(f"  Device        : {DEVICE}")
    print(f"  Eval per round: {args.eval}")
    print(f"  Fresh start   : {args.fresh}")
    print("=" * 65)

    # ── initialise global model ───────────────────────────────────────────────
    global_model = MedicalCNN(num_classes=NUM_CLASSES)

    if args.fresh:
        print(f"\n[{_ts()}] [Init] Starting from random weights (--fresh)")
    else:
        if load_checkpoint(global_model):
            print(f"[{_ts()}] [Init] Resumed from models/global_model.pth")
        else:
            print(f"[{_ts()}] [Init] No checkpoint found — starting from random weights")

    # ── load client data loaders ──────────────────────────────────────────────
    print(f"\n[{_ts()}] [Data] Loading client partitions...")
    client_loaders: List[DataLoader] = []
    for cid in range(1, NUM_CLIENTS + 1):
        loader = get_client_loader(cid, batch_size=args.batch)
        dist   = loader.dataset.class_distribution()
        print(
            f"  Client {cid}: {len(loader.dataset):,} samples | "
            + "  ".join(f"{k}={v}" for k, v in dist.items())
        )
        client_loaders.append(loader)

    # ── baseline evaluation ───────────────────────────────────────────────────
    print(f"\n[{_ts()}] [Eval R00] Baseline (before any training):")
    init_loss, init_acc = run_evaluation(global_model, round_num=0)

    # ── training loop ─────────────────────────────────────────────────────────
    history: List[dict] = []
    wall_start = time.perf_counter()

    for rnd in range(1, args.rounds + 1):
        _hline()
        print(f"[{_ts()}]  ROUND  {rnd} / {args.rounds}")
        _hline()

        t0 = time.perf_counter()
        global_model, r_loss, r_acc = run_round(
            global_model, client_loaders, args.epochs, rnd
        )
        round_time = time.perf_counter() - t0

        print(f"[{_ts()}] [Time ] Round {rnd} completed in {round_time:.1f}s")

        # save checkpoint after every round — partial progress is never lost
        save_checkpoint(global_model, rnd)

        # optional per-round test evaluation
        if args.eval:
            eval_loss, eval_acc = run_evaluation(global_model, round_num=rnd)
        else:
            eval_loss, eval_acc = None, None

        history.append({
            "round":      rnd,
            "train_loss": r_loss,
            "train_acc":  r_acc,
            "eval_loss":  eval_loss,
            "eval_acc":   eval_acc,
            "time_s":     round_time,
        })

    total_time = time.perf_counter() - wall_start

    # ── final evaluation ──────────────────────────────────────────────────────
    _banner("Training Complete  —  Final Evaluation")
    final_loss, final_acc = run_evaluation(global_model, round_num=args.rounds)

    # ── summary table ─────────────────────────────────────────────────────────
    print()
    _banner("Training Summary")

    col_w  = 8
    header = (
        f"  {'Round':>6}  {'Train Loss':>10}  {'Train Acc':>10}"
        + (f"  {'Test Loss':>10}  {'Test Acc':>10}" if args.eval else "")
        + f"  {'Time(s)':>8}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for h in history:
        row = (
            f"  {h['round']:>6}  {h['train_loss']:>10.4f}  {h['train_acc']*100:>9.2f}%"
        )
        if args.eval and h["eval_acc"] is not None:
            row += f"  {h['eval_loss']:>10.4f}  {h['eval_acc']*100:>9.2f}%"
        row += f"  {h['time_s']:>8.1f}"
        print(row)

    print()
    print(f"  Baseline test accuracy  : {init_acc*100:.2f}%")
    print(f"  Final   test accuracy   : {final_acc*100:.2f}%")
    delta = (final_acc - init_acc) * 100
    arrow = "+" if delta >= 0 else ""
    print(f"  Delta                   : {arrow}{delta:.2f}%")
    print(f"  Total wall-clock time   : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print()
    print(f"  Best model saved to     : models/global_model.pth")
    print(f"  Run for full report     : python src/evaluate_global.py")
    print("=" * 65)


if __name__ == "__main__":
    main()

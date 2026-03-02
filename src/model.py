"""
Phase 3 - PyTorch Model (ResNet-18, 6-class, FedProx version)
=============================================================
MedicalCNN      : ResNet-18 adapted for 1-channel 192x192 grayscale input,
                  6-class output. NO Softmax — raw logits only.
train_fedprox   : FedProx local training with proximal regularisation term.
                  L = CrossEntropyLoss + (mu/2) * ||W_local − W_global||^2
train_one_round : alias kept for backward compatibility.
evaluate        : inference + per-class accuracy, returns (loss, acc, per_class_acc).
get_parameters  : model -> list[np.ndarray]
set_parameters  : list[np.ndarray] -> model
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MedicalCNN(nn.Module):
    """
    ResNet-18 adapted for medical image classification.

    Changes from the standard ImageNet ResNet-18:
      - conv1 accepts 1 input channel (grayscale) instead of 3.
      - fc outputs num_classes logits instead of 1000.
      - NO Softmax — CrossEntropyLoss works on raw logits.
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        # Load standard ResNet-18 (no pretrained weights — we train from scratch
        # on medical data; ImageNet weights are harmful for grayscale MRI/X-ray)
        backbone = models.resnet18(weights=None)

        # Replace first conv: 3 channels -> 1 channel (grayscale)
        # Keep all other hyperparameters (kernel 7, stride 2, padding 3)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final fc: 512 -> num_classes (raw logits, no Softmax)
        backbone.fc = nn.Linear(512, num_classes)

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # returns raw logits


# ---------------------------------------------------------------------------
# Parameter helpers (FedProx / FedAvg aggregation)
# ---------------------------------------------------------------------------

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Flatten all model parameters to a list of numpy arrays."""
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load a list of numpy arrays back into the model."""
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=model.state_dict()[k].dtype)
         for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# FedProx local training
# ---------------------------------------------------------------------------

def train_fedprox(
    model: nn.Module,
    global_params: List[np.ndarray],
    loader: DataLoader,
    epochs: int = 2,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    mu: float = 0.01,
    device: str = "cpu",
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """
    FedProx local training for one federated round.

    Loss = CrossEntropyLoss(logits, labels)
         + (mu / 2) * ||W_local - W_global||^2

    The proximal term (mu/2)*||W_local - W_global||^2 prevents local models
    from diverging too far from the global model, which is the key advantage
    of FedProx over plain FedAvg on heterogeneous (non-IID) data.

    Parameters
    ----------
    model         : local model (initialised with global weights before calling)
    global_params : snapshot of global weights at the start of this round
    loader        : client's local DataLoader (WeightedRandomSampler inside)
    epochs        : number of local epochs (default 2)
    lr            : SGD learning rate
    momentum      : SGD momentum
    weight_decay  : L2 regularisation for SGD
    mu            : FedProx proximal coefficient (0.01 is the paper default)
    device        : 'cpu' or 'cuda'
    class_weights : optional tensor of shape (num_classes,) for imbalanced data

    Returns
    -------
    (last_epoch_loss, last_epoch_accuracy)
    """
    model.to(device).train()

    # Weighted CrossEntropyLoss — handles class imbalance across splits
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    # SGD is preferred in FL: it does not maintain per-parameter momentum state
    # across rounds (which would be stale after global aggregation), and its
    # update direction is more predictable for convergence proofs.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    # CosineAnnealingLR smoothly decays LR each epoch, avoiding abrupt drops
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # Build per-parameter global tensors aligned with model.parameters().
    # global_params comes from state_dict() which includes BN buffers; we must
    # select only the entries whose keys appear in named_parameters().
    sd_keys    = list(model.state_dict().keys())
    param_names = {name for name, _ in model.named_parameters()}
    global_param_map = {
        key: torch.tensor(global_params[i], dtype=torch.float32, device=device)
        for i, key in enumerate(sd_keys)
        if key in param_names
    }
    # ordered list matching model.parameters() iteration order
    global_tensors = [
        global_param_map[name]
        for name, _ in model.named_parameters()
    ]

    last_loss = last_acc = 0.0

    for _epoch in range(epochs):
        running_loss = correct = total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)                    # raw logits
            ce_loss = criterion(outputs, labels)

            # --- FedProx proximal term ---
            # Penalise deviation of local weights from the frozen global weights
            prox = sum(
                torch.sum((lw - gw) ** 2)
                for lw, gw in zip(model.parameters(), global_tensors)
            )
            loss = ce_loss + (mu / 2.0) * prox

            loss.backward()
            # Gradient clipping prevents exploding gradients on deep ResNet
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += ce_loss.item() * images.size(0)  # log CE loss only
            _, preds = outputs.max(dim=1)
            correct  += preds.eq(labels).sum().item()
            total    += images.size(0)

        last_loss = running_loss / total
        last_acc  = correct / total
        scheduler.step()

    return last_loss, last_acc


# Keep the old name so api.py / other callers don't break
def train_one_round(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 2,
    lr: float = 0.01,
    device: str = "cpu",
    class_weights: Optional[torch.Tensor] = None,
    global_params: Optional[List[np.ndarray]] = None,
    mu: float = 0.01,
) -> Tuple[float, float]:
    """Thin wrapper — calls train_fedprox when global_params is provided,
    otherwise falls back to plain SGD (no proximal term)."""
    if global_params is None:
        global_params = get_parameters(model)
    return train_fedprox(
        model, global_params, loader,
        epochs=epochs, lr=lr, mu=mu,
        device=device, class_weights=class_weights,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    num_classes: int = 6,
) -> Tuple[float, float, Dict[int, float]]:
    """
    Evaluate model on a DataLoader.

    Returns
    -------
    (loss, overall_accuracy, per_class_accuracy_dict)
    per_class_accuracy_dict: {class_idx: accuracy_float}
    """
    model.to(device).eval()
    criterion    = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct      = 0
    total        = 0

    # Track per-class correct / total
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(dim=1)
            correct  += preds.eq(labels).sum().item()
            total    += images.size(0)

            for label, pred in zip(labels.cpu(), preds.cpu()):
                class_total[label.item()]   += 1
                class_correct[label.item()] += int(pred.item() == label.item())

    overall_acc = correct / total if total > 0 else 0.0
    per_class   = {
        i: (class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0)
        for i in range(num_classes)
    }
    return running_loss / total, overall_acc, per_class

"""
Phase 3 - PyTorch Model (6-class version)
==========================================
MedicalCNN      : 3-block CNN (1-ch 128x128 input -> 6 classes).
train_one_round : local training, returns (loss, accuracy).
evaluate        : inference only, returns (loss, accuracy).
get_parameters  : model -> list[np.ndarray]   (Flower helper)
set_parameters  : list[np.ndarray] -> model   (Flower helper)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MedicalCNN(nn.Module):
    """
    Lightweight 3-block CNN for 1-channel 128x128 medical images.

    Architecture:
        Block 1: Conv(1->32)   + BN + ReLU + MaxPool  -> 64x64
        Block 2: Conv(32->64)  + BN + ReLU + MaxPool  -> 32x32
        Block 3: Conv(64->128) + BN + ReLU + MaxPool  -> 16x16
        AdaptiveAvgPool                               ->  4x4
        Dropout(0.5) -> FC(2048->256) -> ReLU -> Dropout(0.3) -> FC(256->num_classes)
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.features = nn.Sequential(
            conv_block(1,   32),
            conv_block(32,  64),
            conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Alias for backward compatibility
BrainTumorCNN = MedicalCNN


def train_one_round(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 2,
    lr: float   = 1e-3,
    device: str = "cpu",
) -> Tuple[float, float]:
    model.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    last_loss = last_acc = 0.0
    for _ in range(epochs):
        running_loss = correct = total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(dim=1)
            correct  += preds.eq(labels).sum().item()
            total    += images.size(0)
        last_loss = running_loss / total
        last_acc  = correct / total
        scheduler.step()

    return last_loss, last_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, float]:
    model.to(device).eval()
    criterion    = nn.CrossEntropyLoss()
    running_loss = correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(dim=1)
            correct  += preds.eq(labels).sum().item()
            total    += images.size(0)
    return running_loss / total, correct / total


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=model.state_dict()[k].dtype)
         for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


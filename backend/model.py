import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models

try:
    from tqdm import tqdm as _tqdm_cls

    _TQDM = True
except ImportError:
    _tqdm_cls = None  # type: ignore[assignment]
    _TQDM = False


def _progress_bar(iterable, desc="", total=None, leave=False, colour=None):
    """Wrap iterable with tqdm if available, otherwise return plain iterable."""
    if _TQDM and _tqdm_cls is not None:
        return _tqdm_cls(
            iterable,
            desc=desc,
            total=total,
            leave=leave,
            colour=colour,
            dynamic_ncols=True,
            file=sys.stdout,
        )
    return iterable


class MedicalCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


def get_parameters(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model, parameters):
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


def train_fedprox(
    model,
    global_params,
    loader,
    epochs=2,
    lr=0.01,
    mu=0.01,
    device="cpu",
    class_weights=None,
    client_id: int = 0,
    round_num: int = 0,
):
    base = getattr(model, "_orig_mod", model)
    base.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    # Align global tensors with *trainable* parameters by name so batch-norm
    # running stats (which live in state_dict but not in model.parameters())
    # never cause a shape mismatch in the proximal term.
    param_keys = [
        k
        for k, v in model.state_dict().items()
        if v.requires_grad or k in dict(model.named_parameters())
    ]
    named_params = dict(model.named_parameters())  # only trainable params
    global_state = dict(zip(model.state_dict().keys(), global_params))
    valid_globals = [
        torch.tensor(global_state[k]).to(device)
        for k in named_params
        if k in global_state and global_state[k].shape == named_params[k].shape
    ]

    last_loss, last_acc = 0.0, 0.0
    num_batches = len(loader)

    for epoch in range(1, epochs + 1):
        running_loss, correct, total = 0.0, 0, 0

        bar_desc = f"  R{round_num:02d} C{client_id} E{epoch}/{epochs}"
        bar = _progress_bar(
            loader,
            desc=bar_desc,
            total=num_batches,
            leave=(epoch == epochs),  # keep final epoch bar; clear intermediate
            colour="cyan",
        )

        for images, labels in bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(
                device
            )  # blocking — non_blocking on labels causes MPS async corruption
            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            prox = (
                sum(
                    torch.sum((lw - gw) ** 2)
                    for lw, gw in zip(model.parameters(), valid_globals)
                )
                if valid_globals
                else torch.tensor(0.0, device=device)
            )
            loss = ce_loss + (mu / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += ce_loss.item() * images.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += images.size(0)

            # Live postfix: rolling loss + acc on every batch
            if _TQDM:
                bar.set_postfix(
                    loss=f"{running_loss / total:.4f}",
                    acc=f"{correct / total * 100:.1f}%",
                    refresh=False,
                )

        last_loss = running_loss / total
        last_acc = correct / total

    return last_loss, last_acc


def train_one_round(
    model,
    loader,
    epochs=2,
    lr=0.01,
    device="cpu",
    class_weights=None,
    global_params=None,
    mu=0.01,
    client_id: int = 0,
    round_num: int = 0,
):
    if global_params is None:
        global_params = get_parameters(model)
    return train_fedprox(
        model,
        global_params,
        loader,
        epochs,
        lr,
        mu,
        device,
        class_weights,
        client_id=client_id,
        round_num=round_num,
    )


def evaluate(model, loader, device="cpu", num_classes=6):
    # Move the underlying model to device. If model is a compiled wrapper
    # (torch.compile), we need to access ._orig_mod to call .to() safely.
    base = getattr(model, "_orig_mod", model)
    base.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    class_correct, class_total = [0] * num_classes, [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(
                device
            )  # blocking — non_blocking on labels causes MPS async corruption
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.max(1)[1]
            correct += preds.eq(labels).sum().item()
            total += images.size(0)
            for label, pred in zip(labels.cpu(), preds.cpu()):
                class_total[label.item()] += 1
                class_correct[label.item()] += int(pred.item() == label.item())

    overall_acc = correct / total if total > 0 else 0.0
    per_class = {
        i: (class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0)
        for i in range(num_classes)
    }
    return running_loss / total, overall_acc, per_class

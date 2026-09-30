"""Conditional Optuna search for binary-classification losses and optimizers.

All losses consume model logits (never probabilities). The same resolved
configuration is used by the inner folds and the outer-train refit.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class BinaryLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)


def suggest_training_params(
    trial,
    *,
    optimizer_choices=("AdamW", "Adam", "SGD"),
    loss_choices=("bce", "bce_label_smoothing", "focal"),
    lr_range=(1e-5, 5e-4),
    sgd_lr_range=(1e-4, 1e-2),
    weight_decay_range=(1e-6, 1e-3),
    epoch_range=(5, 30),
):
    """Suggest one complete recipe; branch-specific keys avoid distribution conflicts."""
    choices_o = tuple(optimizer_choices)
    choices_l = tuple(loss_choices)
    if not choices_o or not set(choices_o).issubset({"AdamW", "Adam", "SGD"}):
        raise ValueError("optimizer_choices must be a nonempty subset of AdamW, Adam, SGD")
    if not choices_l or not set(choices_l).issubset({"bce", "bce_label_smoothing", "focal"}):
        raise ValueError("loss_choices must be a nonempty subset of bce, bce_label_smoothing, focal")
    name = trial.suggest_categorical("optimizer", list(choices_o))
    if name == "SGD":
        lr = trial.suggest_float("lr_sgd", *sgd_lr_range, log=True)
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
    else:
        lr = trial.suggest_float("lr_adaptive", *lr_range, log=True)
        momentum = None
    wd = trial.suggest_float("weight_decay", *weight_decay_range, log=True)
    epochs = trial.suggest_int("epochs", *epoch_range)
    loss = trial.suggest_categorical("loss", list(choices_l))
    resolved = {"optimizer": name, "loss": loss, "lr": lr,
                "weight_decay": wd, "epochs": epochs}
    if momentum is not None:
        resolved["momentum"] = momentum
    if loss == "focal":
        resolved["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)
    elif loss == "bce_label_smoothing":
        resolved["label_smoothing"] = trial.suggest_float("label_smoothing", 0.01, 0.15)
    return resolved


def resolve_best_params(params: dict) -> dict:
    """Map Optuna's branch-specific LR keys to one stable configuration."""
    config = dict(params)
    name = config["optimizer"]
    config["lr"] = float(config["lr_sgd"] if name == "SGD" else config["lr_adaptive"])
    return config


def make_loss(params: dict, nout: int):
    """Support 1-logit BCE and 2-logit CE checkpoints with matching loss families."""
    name = params["loss"]
    if nout == 1:
        if name == "bce":
            return nn.BCEWithLogitsLoss()
        if name == "bce_label_smoothing":
            return BinaryLabelSmoothingLoss(params["label_smoothing"])
        if name == "focal":
            return BinaryFocalLoss(params["focal_gamma"])
    elif nout == 2:
        if name == "bce":
            return nn.CrossEntropyLoss()
        if name == "bce_label_smoothing":
            return nn.CrossEntropyLoss(label_smoothing=float(params["label_smoothing"]))
        if name == "focal":
            gamma = float(params["focal_gamma"])
            class MulticlassFocal(nn.Module):
                def forward(self, logits, targets):
                    ce = F.cross_entropy(logits, targets.long(), reduction="none")
                    pt = torch.exp(-ce)
                    return (((1.0 - pt) ** gamma) * ce).mean()
            return MulticlassFocal()
    raise ValueError(f"Unsupported loss={name!r} or output dimension={nout}")


def make_optimizer(model: nn.Module, params: dict):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters")
    options = dict(lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    name = params["optimizer"]
    if name == "AdamW":
        return torch.optim.AdamW(trainable, **options)
    if name == "Adam":
        return torch.optim.Adam(trainable, **options)
    if name == "SGD":
        return torch.optim.SGD(trainable, momentum=float(params["momentum"]), **options)
    raise ValueError(f"Unknown optimizer: {name}")

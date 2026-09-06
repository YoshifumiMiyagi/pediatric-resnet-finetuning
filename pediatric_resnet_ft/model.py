from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torchvision import models

VALID_MODES = {"head", "layer4", "layer3_4", "full"}


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def build_resnet50(
    pretrained: str = "imagenet",
    pretrained_path: Optional[str] = None,
    mode: str = "layer4",
    num_outputs: int = 1,
    reset_head: bool = False,
) -> nn.Module:
    """Build a ResNet50 for pediatric fine-tuning.

    Parameters
    ----------
    pretrained:
        "imagenet", "adult", or "none".
    pretrained_path:
        Required when pretrained="adult". The adult checkpoint should be a
        ResNet50 with a compatible architecture.
    mode:
        "head", "layer4", "layer3_4", or "full".
    num_outputs:
        Number of output logits. Use 1 for binary classification.
    reset_head:
        If True, reinitialize the final fc layer after loading adult weights.
        Useful for comparing transferred representations with a fresh classifier.
    """
    pretrained = pretrained.lower()
    mode = mode.lower()

    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")

    if pretrained == "imagenet":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)

    elif pretrained == "adult":
        if pretrained_path is None:
            raise ValueError("pretrained_path is required when pretrained='adult'.")

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)

        checkpoint = torch.load(Path(pretrained_path), map_location="cpu")
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Adult checkpoint could not be loaded strictly. Ensure the adult "
                "model is torchvision ResNet50 and has the same output dimension."
            ) from exc

        if reset_head:
            model.fc = nn.Linear(model.fc.in_features, num_outputs)

    elif pretrained == "none":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)

    else:
        raise ValueError("pretrained must be 'imagenet', 'adult', or 'none'.")

    set_finetune_mode(model, mode)
    return model


def set_finetune_mode(model: nn.Module, mode: str) -> nn.Module:
    """Freeze/unfreeze ResNet50 parameters according to a fine-tuning mode."""
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    if mode == "head":
        modules: Iterable[nn.Module] = (model.fc,)
    elif mode == "layer4":
        modules = (model.layer4, model.fc)
    elif mode == "layer3_4":
        modules = (model.layer3, model.layer4, model.fc)
    else:
        modules = (model,)

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    return model


def trainable_parameter_names(model: nn.Module):
    return [name for name, p in model.named_parameters() if p.requires_grad]

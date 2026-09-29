from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torchvision import models

VALID_MODES = {"head", "layer4", "layer3_4", "full"}


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def build_resnet50(
    pretrained: str = "imagenet",
    pretrained_path: Optional[str] = None,
    mode: str = "layer4",
    num_outputs: Optional[int] = None,
    reset_head: bool = False,
) -> nn.Module:
    """Build a ResNet50 for pediatric fine-tuning.

    For adult checkpoints, the classifier output dimension is auto-detected from
    fc.weight when num_outputs is None. This allows transfer from 2-logit
    CrossEntropy classifiers as well as 1-logit BCE classifiers.
    """
    pretrained = pretrained.lower()
    mode = mode.lower()

    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")

    if pretrained == "imagenet":
        out_dim = 1 if num_outputs is None else int(num_outputs)
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, out_dim)

    elif pretrained == "adult":
        if pretrained_path is None:
            raise ValueError("pretrained_path is required when pretrained='adult'.")

        checkpoint = torch.load(Path(pretrained_path), map_location="cpu")
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))

        if "fc.weight" not in state_dict:
            raise RuntimeError("Adult checkpoint does not contain fc.weight.")

        checkpoint_out = int(state_dict["fc.weight"].shape[0])
        out_dim = checkpoint_out if num_outputs is None else int(num_outputs)

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, checkpoint_out)
        model.load_state_dict(state_dict, strict=True)

        if reset_head:
            model.fc = nn.Linear(model.fc.in_features, out_dim)
        elif out_dim != checkpoint_out:
            raise ValueError(
                f"Adult checkpoint has {checkpoint_out} outputs but num_outputs={out_dim}. "
                "Use num_outputs=None to preserve the adult head, or reset_head=True "
                "to replace the classifier."
            )

    elif pretrained == "none":
        out_dim = 1 if num_outputs is None else int(num_outputs)
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_dim)

    else:
        raise ValueError("pretrained must be 'imagenet', 'adult', or 'none'.")

    model._pediatric_resnet_num_outputs = int(model.fc.out_features)
    set_finetune_mode(model, mode)
    return model


def set_finetune_mode(model: nn.Module, mode: str) -> nn.Module:
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

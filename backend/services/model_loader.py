"""Model Architecture Builders and Safe Checkpoint Loader."""

from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models

from backend.config import DEVICE, MODELS_DIR, MONOLITHIC_CLASSES
from backend.logging_config import get_logger

logger = get_logger("ophthalmoai.model_loader")


def build_monolithic_model(num_classes: int = len(MONOLITHIC_CLASSES)) -> nn.Module:
    """Constructs EfficientNet-B4 backbone configured for retinal classification."""
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_convnext_small(num_classes: int = len(MONOLITHIC_CLASSES)) -> nn.Module:
    """Constructs ConvNeXt-Small backbone configured for retinal classification."""
    m = models.convnext_small(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    return m


def build_densenet201(num_classes: int = len(MONOLITHIC_CLASSES)) -> nn.Module:
    """Constructs DenseNet-201 backbone configured for retinal classification."""
    m = models.densenet201(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m


def build_efficientnet_v2_m(num_classes: int = len(MONOLITHIC_CLASSES)) -> nn.Module:
    """Constructs EfficientNet-V2-M backbone configured for retinal classification."""
    m = models.efficientnet_v2_m(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def load_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    device: torch.device = DEVICE,
) -> bool:
    """Safely loads checkpoint weights with weights_only=True and places on device."""
    if not os.path.exists(ckpt_path):
        logger.warning("model_loader.checkpoint_missing", path=ckpt_path)
        return False

    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        return True
    except Exception as exc:
        logger.error("model_loader.checkpoint_failed", path=ckpt_path, error=str(exc))
        return False


def load_all_models(
    models_dir: str = MODELS_DIR,
    device: torch.device = DEVICE,
) -> Tuple[nn.Module | None, Dict[str, nn.Module]]:
    """Initializes and loads all production screening models:"""
    num_classes = len(MONOLITHIC_CLASSES)

    monolith = build_monolithic_model(num_classes)
    monolith_path = os.path.join(models_dir, "efficientnet_b4.pth")
    monolithic_model: nn.Module | None = None

    if load_checkpoint(monolith, monolith_path, device=device):
        monolithic_model = monolith
        logger.info("model_loader.monolithic_model_ready", path=monolith_path)
    else:
        logger.warning("model_loader.monolithic_model_unavailable", path=monolith_path)

    ensemble_specs: Dict[str, Tuple[Callable[[int], nn.Module], str]] = {
        "densenet201": (build_densenet201, "densenet201.pth"),
        "convnext_small": (build_convnext_small, "convnext_small.pth"),
        "efficientnet_v2_m": (build_efficientnet_v2_m, "efficientnet_v2_m.pth"),
    }
    loaded_ensemble: Dict[str, nn.Module] = {}

    for arch_name, (builder_fn, ckpt_file) in ensemble_specs.items():
        ckpt_path = os.path.join(models_dir, ckpt_file)
        m = builder_fn(num_classes)
        if load_checkpoint(m, ckpt_path, device=device):
            loaded_ensemble[arch_name] = m
            logger.info(f"model_loader.{arch_name}_ready")
        else:
            logger.warning(f"model_loader.{arch_name}_unavailable", path=ckpt_path)

    if len(loaded_ensemble) == 3:
        logger.info("model_loader.tri_backbone_ensemble_ready", models=list(loaded_ensemble.keys()))
    else:
        logger.warning("model_loader.ensemble_partial_or_fallback", loaded=list(loaded_ensemble.keys()))

    return monolithic_model, loaded_ensemble

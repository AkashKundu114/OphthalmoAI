"""
Post-Hoc Probability Calibration via Temperature Scaling (Guo et al.).

Scales logits by a learned or configured temperature parameter T > 0 to align
predicted model softmax confidences with true empirical accuracy (ECE optimization).
Includes safeguards against numerical overflow, zero division, and non-positive temperatures.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TEMPERATURE: float = 1.0
MIN_TEMPERATURE: float = 1e-3
MAX_TEMPERATURE: float = 50.0
EPSILON: float = 1e-7


def sanitize_temperature(temperature: Optional[Union[float, int]]) -> float:
    """
    Sanitizes temperature scalar, ensuring strictly positive bounded float T in (0, MAX_TEMPERATURE].
    Falls back to DEFAULT_TEMPERATURE if input is null, non-numeric, or non-finite.
    """
    if temperature is None:
        return DEFAULT_TEMPERATURE
    try:
        val = float(temperature)
        if not math.isfinite(val) or val <= 0.0:
            return DEFAULT_TEMPERATURE
        return max(MIN_TEMPERATURE, min(val, MAX_TEMPERATURE))
    except (TypeError, ValueError):
        return DEFAULT_TEMPERATURE


class TemperatureScaler(nn.Module):
    """
    Learns a scalar temperature parameter on validation holdout logits
    using Negative Log-Likelihood (NLL) minimization via L-BFGS.
    """

    def __init__(self, model: Optional[nn.Module] = None):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def get_clamped_temperature(self) -> float:
        """Returns scalar temperature clamped to physiological bounds."""
        with torch.no_grad():
            t = float(self.temperature.item())
            return sanitize_temperature(t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling and division-by-zero safeguard."""
        t_safe = self.get_clamped_temperature()
        if self.model is not None:
            logits = self.model(x)
        else:
            logits = x
        return logits / max(t_safe, EPSILON)

    def fit(
        self,
        val_loader_or_logits: Union[torch.Tensor, Any],
        device_or_labels: Optional[Any] = None,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """
        Fits temperature parameter to validation holdout set.
        Guarantees final temperature remains strictly within [MIN_TEMPERATURE, MAX_TEMPERATURE].
        """
        if isinstance(val_loader_or_logits, torch.Tensor):
            logits = val_loader_or_logits
            labels = device_or_labels
            if labels is None:
                raise ValueError("labels must be provided when logits tensor is passed to fit().")
        else:
            val_loader = val_loader_or_logits
            device = device_or_labels if device_or_labels is not None else torch.device("cpu")
            if isinstance(device, str):
                device = torch.device(device)
            if self.model is not None:
                self.model.eval()

            logits_list, labels_list = [], []
            with torch.no_grad():
                for inputs, labels_batch in val_loader:
                    inputs = inputs.to(device)
                    if self.model is not None:
                        logits_list.append(self.model(inputs))
                    else:
                        logits_list.append(inputs)
                    labels_list.append(labels_batch.to(device))

            if not logits_list:
                raise ValueError("Validation loader yielded no batches for calibration.")

            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        if logits.numel() == 0 or labels.numel() == 0:
            raise ValueError("Logits or labels tensor cannot be empty.")

        if not torch.isfinite(logits).all():
            raise ValueError("Logits tensor contains non-finite values (NaN or Inf).")

        logits = logits.to(self.temperature.device)
        labels = labels.to(self.temperature.device)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            clamped_t = torch.clamp(self.temperature, min=MIN_TEMPERATURE, max=MAX_TEMPERATURE)
            scaled_logits = logits / clamped_t
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            self.temperature.clamp_(min=MIN_TEMPERATURE, max=MAX_TEMPERATURE)

        return float(self.temperature.item())


def apply_temperature(logits: torch.Tensor, temperature: Optional[float]) -> torch.Tensor:
    """
    Applies temperature scaling to raw logits with strictly enforced T > 0 bounds
    and numerical safeguards against division by zero.
    """
    t_safe = sanitize_temperature(temperature)
    return logits / max(t_safe, EPSILON)


def calibrated_softmax(logits: torch.Tensor, temperature: Optional[float], dim: int = -1) -> torch.Tensor:
    """
    Computes numerically stable softmax on temperature-scaled logits.
    Subtracts max before exponentiation to prevent overflow.
    """
    scaled = apply_temperature(logits, temperature)
    return F.softmax(scaled, dim=dim)


class CalibrationRegistry:
    """
    In-memory registry for per-disease / per-model calibration temperatures
    with persistent JSON backing.
    """

    def __init__(self, path: str):
        self.path = path
        self._temperatures: Dict[str, float] = {}
        self.reload()

    def reload(self) -> None:
        """Loads and validates temperature parameters from disk."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._temperatures = {
                    k: sanitize_temperature(v)
                    for k, v in data.items()
                    if isinstance(k, str)
                }
            except Exception:
                self._temperatures = {}
        else:
            self._temperatures = {}

    def get(self, group_key: str) -> float:
        """Returns calibrated temperature for given group, or DEFAULT_TEMPERATURE."""
        return self._temperatures.get(group_key, DEFAULT_TEMPERATURE)

    def is_calibrated(self, group_key: str) -> bool:
        """Checks if calibrated temperature is registered for group."""
        return group_key in self._temperatures

    def all(self) -> Dict[str, float]:
        """Returns a copy of all registered temperature mappings."""
        return dict(self._temperatures)

    @staticmethod
    def save(path: str, temperatures: Dict[str, float]) -> None:
        """Atomically persists validated temperatures to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sanitized = {k: sanitize_temperature(v) for k, v in temperatures.items()}
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=2)
        os.replace(temp_path, path)

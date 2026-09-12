from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn


DEFAULT_TEMPERATURE = 1.0


class TemperatureScaler(nn.Module):

    def __init__(self, model: Optional[nn.Module] = None):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        if self.model is not None:
            return self.model(x) / self.temperature
        return x / self.temperature

    def fit(
        self,
        val_loader_or_logits,
        device_or_labels=None,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        if isinstance(val_loader_or_logits, torch.Tensor):
            logits = val_loader_or_logits
            labels = device_or_labels
            if labels is None:
                raise ValueError("labels must be provided when logits tensor is passed to fit()")
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
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        logits = logits.to(self.temperature.device)
        labels = labels.to(self.temperature.device)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        temperature = DEFAULT_TEMPERATURE
    return logits / temperature


class CalibrationRegistry:

    def __init__(self, path: str):
        self.path = path
        self._temperatures: Dict[str, float] = {}
        self.loaded_at = None
        self.reload()

    def reload(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self._temperatures = {k: float(v) for k, v in data.items()}
            except Exception as e:
                print(f"Failed to load calibration file at {self.path}: {e}")
                self._temperatures = {}
        else:
            self._temperatures = {}

    def get(self, group_key: str) -> float:
        return self._temperatures.get(group_key, DEFAULT_TEMPERATURE)

    def is_calibrated(self, group_key: str) -> bool:
        return group_key in self._temperatures

    def all(self) -> Dict[str, float]:
        return dict(self._temperatures)

    @staticmethod
    def save(path: str, temperatures: Dict[str, float]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(temperatures, f, indent=2)

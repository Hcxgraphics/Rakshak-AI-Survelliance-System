from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

LOGGER = logging.getLogger(__name__)


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.model = models.resnet50(weights=None)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_accident_model(
    model_path: Path,
    *,
    num_classes: int = 6,
    device: torch.device | str = "cpu",
) -> nn.Module:
    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Accident model weights not found: {resolved_path}")

    checkpoint = torch.load(resolved_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    # Dynamically resolve number of classes from weight matrix shape to avoid size mismatch
    fc_weight_keys = ["model.fc.1.weight", "fc.1.weight", "model.fc.weight", "fc.weight"]
    for key in fc_weight_keys:
        if key in state_dict:
            num_classes = state_dict[key].shape[0]
            break

    model = CarClassifierResNet(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    LOGGER.info("Loaded accident model from %s", resolved_path)
    return model

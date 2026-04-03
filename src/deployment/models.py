from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError

import tensorflow as tf
import torch
from PIL import Image
from tensorflow.keras.models import load_model as keras_load_model
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from ultralytics import YOLO

from accidentModel import load_accident_model

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TORCH_CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

POLICE_CLASS_NAMES = ["NonViolence", "Violence", "guns", "knife", "police"]
ACCIDENT_CLASS_NAMES = ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5"]


def _resolve_model_path(env_name: str, default_name: str) -> Path:
    configured = os.getenv(env_name)
    candidate = Path(configured).expanduser() if configured else MODELS_DIR / default_name
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


def _resolve_optional_path(env_name: str, default_path: Path | None = None) -> Path | None:
    configured = os.getenv(env_name)
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate
    return default_path


@dataclass(frozen=True)
class ModelPaths:
    weapon: Path
    violence: Path
    police: Path
    accident: Path
    police_backbone_weights: Path | None


@dataclass(frozen=True)
class DeploymentModels:
    device: torch.device
    paths: ModelPaths
    weapon_model: YOLO
    violence_model: tf.keras.Model
    police_head: tf.keras.Model
    police_backbone: torch.nn.Module
    accident_model: torch.nn.Module
    police_transform: transforms.Compose
    transform_accident: transforms.Compose
    police_class_names: list[str]
    accident_class_names: list[str]


@lru_cache(maxsize=1)
def load_models() -> DeploymentModels:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = ModelPaths(
        weapon=_resolve_model_path("WEAPON_MODEL_PATH", "weapon/best.pt"),
        violence=_resolve_model_path("VIOLENCE_MODEL_PATH", "violence/final_model.h5"),
        police=_resolve_model_path("POLICE_MODEL_PATH", "police/police_or_danger.h5"),
        accident=_resolve_model_path("ACCIDENT_MODEL_PATH", "accident/accident_model.pth"),
        police_backbone_weights=_resolve_optional_path(
            "POLICE_BACKBONE_WEIGHTS",
            TORCH_CACHE_DIR / "mobilenet_v2-b0353104.pth",
        ),
    )

    for field_name, path in paths.__dict__.items():
        if path is None and field_name == "police_backbone_weights":
            continue
        if not Path(path).exists():
            raise FileNotFoundError(f"Required model file not found: {path}")

    LOGGER.info("Loading deployment models from %s", MODELS_DIR)

    weapon_model = YOLO(str(paths.weapon))
    violence_model = keras_load_model(str(paths.violence), compile=False)
    police_head = keras_load_model(str(paths.police), compile=False)
    police_backbone = _load_police_backbone(device, paths.police_backbone_weights)
    accident_model = load_accident_model(paths.accident, device=device)

    police_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_accident = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return DeploymentModels(
        device=device,
        paths=paths,
        weapon_model=weapon_model,
        violence_model=violence_model,
        police_head=police_head,
        police_backbone=police_backbone,
        accident_model=accident_model,
        police_transform=police_transform,
        transform_accident=transform_accident,
        police_class_names=POLICE_CLASS_NAMES,
        accident_class_names=ACCIDENT_CLASS_NAMES,
    )


def _load_police_backbone(device: torch.device, local_weights_path: Path | None) -> torch.nn.Module:
    if local_weights_path is not None and local_weights_path.exists():
        backbone = mobilenet_v2(weights=None)
        state_dict = torch.load(local_weights_path, map_location="cpu")
        backbone.load_state_dict(state_dict)
        LOGGER.info("Loaded MobileNetV2 backbone from local cached weights at %s", local_weights_path)
        return backbone.features.eval().to(device)

    try:
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        LOGGER.info("Loaded MobileNetV2 backbone with pretrained ImageNet weights")
    except (URLError, OSError, PermissionError, RuntimeError) as exc:
        raise RuntimeError(
            "Unable to load pretrained MobileNetV2 backbone weights. "
            "Provide POLICE_BACKBONE_WEIGHTS or populate the local torch cache."
        ) from exc
    return backbone.features.eval().to(device)


def pil_image_from_bgr(frame_bgr) -> Image.Image:
    import cv2

    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

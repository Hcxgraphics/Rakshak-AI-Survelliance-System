from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError

import tensorflow as tf
import torch
import h5py
from PIL import Image
from tensorflow.keras.models import load_model as keras_load_model
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from accidentModel import load_accident_model

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TORCH_CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

from ultralytics import YOLO  # noqa: E402

POLICE_CLASS_NAMES = ["NonViolence", "Violence", "guns", "knife", "police"]
ACCIDENT_CLASS_NAMES = ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5"]
POLICE_INPUT_SIZE = 224
ACCIDENT_INPUT_SIZE = 224


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


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(64).startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def _validate_model_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required model file not found: {path}")
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"Model file is a Git LFS pointer, not real weights: {path}. "
            "Install Git LFS and run `git lfs pull`, or replace this file with the actual model binary."
        )


def _patch_legacy_keras_config(node: object) -> bool:
    patched = False
    if isinstance(node, dict):
        if node.get("class_name") == "InputLayer":
            config = node.get("config")
            if isinstance(config, dict) and "batch_shape" in config:
                batch_shape = config.pop("batch_shape")
                config.setdefault("batch_input_shape", batch_shape)
                patched = True
        for key, value in list(node.items()):
            if isinstance(value, dict) and value.get("class_name") == "DTypePolicy":
                config = value.get("config")
                if isinstance(config, dict) and isinstance(config.get("name"), str):
                    node[key] = config["name"]
                    patched = True
                    continue
            patched = _patch_legacy_keras_config(value) or patched
    elif isinstance(node, list):
        for value in node:
            patched = _patch_legacy_keras_config(value) or patched
    return patched


def _load_keras_model(path: Path) -> tf.keras.Model:
    import sys
    try:
        return keras_load_model(str(path), compile=False)
    except Exception as exc:
        if "DTypePolicy" in str(exc) or "as_list" in str(exc):
            raise RuntimeError(
                f"Keras checkpoint is incompatible with the installed TensorFlow/Keras runtime: {path}.\n"
                f"Current Python executable: {sys.executable}\n"
                "It looks like you are running the server using the wrong virtual environment!\n"
                "Please activate the correct virtual environment at the project root:\n"
                "  .venv\\Scripts\\activate\n"
                "And run the backend using:\n"
                "  uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"
            ) from exc
        if "batch_shape" not in str(exc) and "DTypePolicy" not in str(exc):
            raise

        LOGGER.warning("Patching legacy Keras config for %s", path)
        with tempfile.NamedTemporaryFile(suffix=path.suffix, delete=False) as handle:
            patched_path = Path(handle.name)
        shutil.copy2(path, patched_path)
        try:
            with h5py.File(patched_path, "r+") as h5_file:
                raw_config = h5_file.attrs.get("model_config")
                if raw_config is None:
                    raise RuntimeError(f"Keras model has no model_config: {path}") from exc
                if isinstance(raw_config, bytes):
                    raw_config = raw_config.decode("utf-8")
                model_config = json.loads(raw_config)
                if not _patch_legacy_keras_config(model_config):
                    raise RuntimeError(f"No legacy Keras config entries found in: {path}") from exc
                h5_file.attrs.modify("model_config", json.dumps(model_config).encode("utf-8"))
            try:
                return keras_load_model(str(patched_path), compile=False)
            except Exception as patched_exc:
                raise RuntimeError(
                    f"Keras checkpoint is incompatible with the installed TensorFlow/Keras runtime: {path}.\n"
                    f"Current Python executable: {sys.executable}\n"
                    "Please activate the correct virtual environment at the project root:\n"
                    "  .venv\\Scripts\\activate\n"
                    "And run the backend using:\n"
                    "  uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"
                ) from patched_exc
        finally:
            patched_path.unlink(missing_ok=True)


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
        if field_name == "police_backbone_weights":
            if os.getenv("POLICE_BACKBONE_WEIGHTS") is not None:
                _validate_model_file(Path(path))
            continue
        _validate_model_file(Path(path))

    LOGGER.info("Loading deployment models from %s", MODELS_DIR)

    weapon_model = YOLO(str(paths.weapon))
    violence_model = _load_keras_model(paths.violence)
    police_head = _load_keras_model(paths.police)
    police_backbone = _load_police_backbone(device, paths.police_backbone_weights)
    accident_model = load_accident_model(paths.accident, device=device)

    police_transform = transforms.Compose(
        [
            transforms.Resize((POLICE_INPUT_SIZE, POLICE_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_accident = transforms.Compose(
        [
            transforms.Resize((ACCIDENT_INPUT_SIZE, ACCIDENT_INPUT_SIZE)),
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

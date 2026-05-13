from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import load_models, pil_image_from_bgr

LOGGER = logging.getLogger(__name__)

DEFAULT_POLICE_THRESHOLD = float(os.getenv("POLICE_THRESHOLD", "0.35"))
DEFAULT_WEAPON_THRESHOLD = float(os.getenv("WEAPON_THRESHOLD", "0.5"))
DEFAULT_VIOLENCE_THRESHOLD = float(os.getenv("VIOLENCE_THRESHOLD", "0.6"))
DEFAULT_ACCIDENT_THRESHOLD = float(os.getenv("ACCIDENT_THRESHOLD", "0.7"))
DEFAULT_FRAME_SKIP = int(os.getenv("LIVE_FRAME_SKIP", "2"))
WEAPON_IMAGE_SIZE = int(os.getenv("WEAPON_IMAGE_SIZE", "640"))
WEAPON_DETECT_CONF = float(os.getenv("WEAPON_DETECT_CONF", "0.05"))
VIOLENCE_FRAME_SIZE = int(os.getenv("VIOLENCE_FRAME_SIZE", "64"))
WEAPON_CLASS_NAMES = {
    0: "weapon",
    1: "weapon",
    2: "weapon",
    3: "weapon",
}

INPUT_IMAGE = "image"
INPUT_VIDEO = "video"
INPUT_STREAM = "stream"

PRIORITY_ORDER = ["weapon", "violence", "accident", "police", "normal"]
FUSION_WEIGHTS = {
    "weapon": 1.0,
    "violence": 0.86,
    "accident": 0.78,
    "police": 0.48,
    "normal": 0.20,
}

DETECTION_LOGS: deque[dict[str, Any]] = deque(maxlen=300)


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceConfig:
    threshold_police: float = DEFAULT_POLICE_THRESHOLD
    threshold_weapon: float = DEFAULT_WEAPON_THRESHOLD
    threshold_violence: float = DEFAULT_VIOLENCE_THRESHOLD
    threshold_accident: float = DEFAULT_ACCIDENT_THRESHOLD
    detection_enabled: bool = True
    frame_index: int = 0
    frame_skip: int = DEFAULT_FRAME_SKIP
    source_type: str = INPUT_IMAGE


@dataclass
class ModelSignal:
    label: str
    confidence: float
    source: str
    passed: bool
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        return round(float(self.confidence) * float(self.weight), 4)


def get_detection_logs(limit: int = 100) -> list[dict[str, Any]]:
    return list(DETECTION_LOGS)[-limit:]


def _append_log(entry: dict[str, Any]) -> None:
    DETECTION_LOGS.append({"timestamp": time.strftime("%H:%M:%S"), **entry})


def _clamp_threshold(value: float, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(max(parsed, 0.0), 1.0)


def validate_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise InferenceError("Input frame could not be decoded.")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
        raise InferenceError(f"Expected an image frame with 3 or 4 channels, received shape {frame.shape}.")
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    height, width = frame.shape[:2]
    if height < 32 or width < 32:
        raise InferenceError("Input frame is too small for reliable detection.")
    return np.ascontiguousarray(frame)


def should_run_model(model_name: str, config: InferenceConfig, frame_count: int) -> bool:
    if not config.detection_enabled:
        return False
    if model_name == "violence" and config.source_type == INPUT_IMAGE:
        return False
    if config.source_type == INPUT_IMAGE:
        return True
    if model_name in {"weapon", "police"}:
        return True
    if model_name == "violence":
        return config.source_type in {INPUT_VIDEO, INPUT_STREAM} and frame_count >= 1
    if model_name == "accident":
        return config.frame_index % max(config.frame_skip, 1) == 0
    return True


def build_violence_clip(frames_bgr: list[np.ndarray], sequence_length: int = 16) -> np.ndarray:
    if not frames_bgr:
        raise InferenceError("Violence model needs at least one frame.")
    if len(frames_bgr) >= sequence_length:
        indices = np.linspace(0, len(frames_bgr) - 1, num=sequence_length, dtype=int)
        selected = [frames_bgr[index] for index in indices]
    else:
        selected = [*frames_bgr]
        while len(selected) < sequence_length:
            selected.append(selected[-1].copy())
    clip = np.stack(
        [
            cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (VIOLENCE_FRAME_SIZE, VIOLENCE_FRAME_SIZE))
            for frame in selected
        ],
        axis=0,
    ).astype(np.float32) / 255.0
    return np.expand_dims(clip, axis=0)


def _as_float_dict(class_names: list[str], predictions: np.ndarray) -> dict[str, float]:
    return {name: float(score) for name, score in zip(class_names, predictions.tolist())}


def _safe_component(name: str, runner: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], str | None, float]:
    started = time.perf_counter()
    try:
        return runner(), None, (time.perf_counter() - started) * 1000
    except (ValueError, RuntimeError, OSError, FileNotFoundError, cv2.error) as exc:
        LOGGER.exception("%s inference failed", name)
        return {}, str(exc), (time.perf_counter() - started) * 1000


def police_detect(frame_bgr: np.ndarray, threshold: float = DEFAULT_POLICE_THRESHOLD) -> dict[str, Any]:
    models = load_models()
    image = pil_image_from_bgr(frame_bgr)
    tensor = models.police_transform(image).unsqueeze(0).to(models.device)

    with torch.no_grad():
        features = models.police_backbone(tensor)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(1, -1).cpu().numpy()

    predictions = np.asarray(models.police_head.predict(pooled, verbose=0)[0], dtype=np.float32)
    scores = _as_float_dict(models.police_class_names, predictions)
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]
    return {
        "has_police": scores["police"] >= threshold,
        "police_score": scores["police"],
        "violence_score_police": scores["Violence"],
        "gun_score_police": scores["guns"],
        "knife_score_police": scores["knife"],
        "police_scores": scores,
        "police_top_label": top_label,
        "police_top_score": top_score,
    }


def _predict_violence_score(frame_rgb: np.ndarray, violence_input: np.ndarray | None) -> float:
    models = load_models()
    if violence_input is None:
        frame_64 = cv2.resize(frame_rgb, (64, 64)).astype(np.float32) / 255.0
        violence_input = np.expand_dims(np.repeat(frame_64[None, ...], 16, axis=0), axis=0)

    predictions = np.asarray(models.violence_model.predict(violence_input, verbose=0)[0], dtype=np.float32)
    if predictions.ndim == 0:
        return float(predictions)
    if predictions.shape[0] == 1:
        return float(predictions[0])
    return float(predictions[1])


def _predict_accident(frame_bgr: np.ndarray) -> tuple[int, float]:
    models = load_models()
    tensor = models.transform_accident(pil_image_from_bgr(frame_bgr)).unsqueeze(0).to(models.device)
    with torch.no_grad():
        logits = models.accident_model(tensor)[0]
        probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
    accident_class = int(np.argmax(probabilities))
    accident_conf = float(probabilities[accident_class])
    return accident_class, accident_conf


def _detect_weapon(frame_rgb: np.ndarray, annotated_frame: np.ndarray, threshold: float) -> dict[str, Any]:
    models = load_models()
    prediction = models.weapon_model.predict(
        source=frame_rgb,
        imgsz=WEAPON_IMAGE_SIZE,
        conf=WEAPON_DETECT_CONF,
        iou=0.45,
        stream=False,
        verbose=False,
    )[0]
    boxes = prediction.boxes.data.detach().cpu().numpy() if prediction.boxes is not None else np.empty((0, 6))
    valid_boxes = [box for box in boxes if float(box[4]) >= WEAPON_DETECT_CONF]
    detections = []
    for x1, y1, x2, y2, conf, cls in valid_boxes:
        label_name = WEAPON_CLASS_NAMES.get(int(cls), prediction.names.get(int(cls), "weapon") if hasattr(prediction, "names") else "weapon")
        detection = {
            "label": label_name,
            "class_id": int(cls),
            "confidence": round(float(conf), 4),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        }
        detections.append(detection)
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (30, 210, 255), 2)
        cv2.putText(
            annotated_frame,
            f"{label_name} {float(conf):.2f}",
            (int(x1), max(int(y1) - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 210, 255),
            2,
        )
    max_conf = max([item["confidence"] for item in detections], default=0.0)
    return {
        "weapon_detections": detections,
        "weapons_detected": len(detections),
        "has_weapon_yolo": bool(detections),
        "weapon_score_yolo": max_conf,
    }


def _build_signals(results: dict[str, Any], config: InferenceConfig) -> list[ModelSignal]:
    raw_weapon_score = max(results["weapon_score_yolo"], results["gun_score_police"], results["knife_score_police"])
    weapon_score = max(raw_weapon_score, 0.75 if results["has_weapon_yolo"] else 0.0)
    violence_score = max(results["violence_score_lstm"], results["violence_score_police"])
    signals = [
        ModelSignal("weapon", weapon_score, "YOLOv8 + police classifier", results["has_weapon_yolo"] or raw_weapon_score >= config.threshold_weapon, FUSION_WEIGHTS["weapon"]),
        ModelSignal("violence", violence_score, "MobileNet-LSTM + police classifier", violence_score >= config.threshold_violence, FUSION_WEIGHTS["violence"]),
        ModelSignal("accident", results["accident_conf"], "ResNet50", results["accident_conf"] >= config.threshold_accident, FUSION_WEIGHTS["accident"]),
        ModelSignal("police", results["police_score"], "MobileNetV2", results["police_score"] >= config.threshold_police, FUSION_WEIGHTS["police"]),
    ]
    return signals


def _fuse_scene(results: dict[str, Any], config: InferenceConfig) -> dict[str, Any]:
    signals = _build_signals(results, config)
    active = [signal for signal in signals if signal.passed]
    if not active:
        return {
            "scene": "Normal",
            "summary": "Normal scene",
            "risk_level": "low",
            "confidence": 1.0 - max([signal.confidence for signal in signals], default=0.0),
            "signals": [signal.__dict__ | {"weighted_score": signal.weighted_score} for signal in signals],
        }

    active.sort(key=lambda signal: (PRIORITY_ORDER.index(signal.label), -signal.weighted_score))
    winner = active[0]
    summary_map = {
        "weapon": "Weapon detected",
        "violence": "Violence detected",
        "accident": "Accident detected",
        "police": "Police present",
    }
    risk_map = {"weapon": "critical", "violence": "high", "accident": "high", "police": "watch"}
    if winner.label == "violence" and results.get("has_police"):
        summary = "Police present with violence detected"
    elif winner.label == "accident" and results.get("accident_class") is not None:
        summary = f"Accident detected (class {results['accident_class']})"
    else:
        summary = summary_map[winner.label]

    return {
        "scene": winner.label.title(),
        "summary": summary,
        "risk_level": risk_map[winner.label],
        "confidence": winner.weighted_score,
        "signals": [signal.__dict__ | {"weighted_score": signal.weighted_score} for signal in signals],
    }


def _draw_overlay(annotated_frame: np.ndarray, results: dict[str, Any], fusion: dict[str, Any]) -> None:
    palette = {
        "critical": (20, 20, 230),
        "high": (0, 110, 255),
        "watch": (255, 190, 40),
        "low": (80, 210, 120),
    }
    color = palette.get(fusion["risk_level"], (220, 220, 220))
    cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 88), (12, 16, 24), -1)
    cv2.putText(annotated_frame, fusion["summary"], (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    score_line = (
        f"Wpn {max(results['weapon_score_yolo'], results['gun_score_police'], results['knife_score_police']):.2f}  "
        f"Viol {max(results['violence_score_lstm'], results['violence_score_police']):.2f}  "
        f"Police {results['police_score']:.2f}  Accident {results['accident_conf']:.2f}"
    )
    cv2.putText(annotated_frame, score_line, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 235, 240), 2)


def detect_objects(
    frame: np.ndarray,
    *,
    is_video: bool = False,
    violence_input: np.ndarray | None = None,
    threshold_police: float = DEFAULT_POLICE_THRESHOLD,
    threshold_weapon: float = DEFAULT_WEAPON_THRESHOLD,
    threshold_violence: float = DEFAULT_VIOLENCE_THRESHOLD,
    threshold_accident: float = DEFAULT_ACCIDENT_THRESHOLD,
    detection_enabled: bool = True,
    source_type: str | None = None,
    frame_index: int = 0,
    frame_skip: int = DEFAULT_FRAME_SKIP,
) -> dict[str, Any]:
    frame = validate_frame(frame)
    source = source_type or (INPUT_VIDEO if is_video else INPUT_IMAGE)
    config = InferenceConfig(
        threshold_police=_clamp_threshold(threshold_police, DEFAULT_POLICE_THRESHOLD),
        threshold_weapon=_clamp_threshold(threshold_weapon, DEFAULT_WEAPON_THRESHOLD),
        threshold_violence=_clamp_threshold(threshold_violence, DEFAULT_VIOLENCE_THRESHOLD),
        threshold_accident=_clamp_threshold(threshold_accident, DEFAULT_ACCIDENT_THRESHOLD),
        detection_enabled=detection_enabled,
        frame_index=max(int(frame_index), 0),
        frame_skip=max(int(frame_skip), 1),
        source_type=source,
    )

    annotated_frame = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_count = 16 if violence_input is not None else 1
    results: dict[str, Any] = {
        "is_video": source in {INPUT_VIDEO, INPUT_STREAM},
        "source_type": source,
        "title": "Normal scene",
        "summary": "Normal scene",
        "scene": "Normal",
        "risk_level": "low",
        "fusion_confidence": 0.0,
        "weapons_detected": 0,
        "weapon_detections": [],
        "has_weapon_yolo": False,
        "weapon_score_yolo": 0.0,
        "gun_score_police": 0.0,
        "knife_score_police": 0.0,
        "has_police": False,
        "police_score": 0.0,
        "police_top_label": None,
        "police_top_score": 0.0,
        "police_scores": {},
        "violence_score_lstm": 0.0,
        "violence_score_police": 0.0,
        "accident_class": None,
        "accident_conf": 0.0,
        "component_errors": {},
        "component_latency_ms": {},
        "signals": [],
        "image": None,
    }

    if not detection_enabled:
        _draw_overlay(annotated_frame, results, {"summary": "Detection paused", "risk_level": "low"})
    else:
        components: list[tuple[str, Callable[[], dict[str, Any]]]] = []
        if should_run_model("weapon", config, frame_count):
            components.append(("weapon", lambda: _detect_weapon(frame_rgb, annotated_frame, config.threshold_weapon)))
        if should_run_model("police", config, frame_count):
            components.append(("police", lambda: police_detect(frame, config.threshold_police)))
        if should_run_model("violence", config, frame_count):
            components.append(("violence", lambda: {"violence_score_lstm": _predict_violence_score(frame_rgb, violence_input)}))
        if should_run_model("accident", config, frame_count):
            components.append(("accident", lambda: dict(zip(("accident_class", "accident_conf"), _predict_accident(frame)))))

        for component_name, runner in components:
            output, error, latency = _safe_component(component_name, runner)
            results["component_latency_ms"][component_name] = round(latency, 2)
            if error:
                results["component_errors"][component_name] = error
            else:
                results.update(output)

        fusion = _fuse_scene(results, config)
        results.update(
            {
                "title": fusion["summary"],
                "summary": fusion["summary"],
                "scene": fusion["scene"],
                "risk_level": fusion["risk_level"],
                "fusion_confidence": round(float(fusion["confidence"]), 4),
                "signals": fusion["signals"],
            }
        )
        _draw_overlay(annotated_frame, results, fusion)

    success, buffer = cv2.imencode(".jpg", annotated_frame)
    if not success:
        raise InferenceError("Failed to encode annotated output frame.")
    results["image"] = buffer.tobytes()

    _append_log(
        {
            "source": source,
            "summary": results["summary"],
            "risk_level": results["risk_level"],
            "confidence": results["fusion_confidence"],
            "errors": results["component_errors"],
            "latency_ms": results["component_latency_ms"],
        }
    )
    LOGGER.info(
        "Fusion decision source=%s summary=%s risk=%s confidence=%.3f errors=%s",
        source,
        results["summary"],
        results["risk_level"],
        results["fusion_confidence"],
        list(results["component_errors"].keys()),
    )
    return results

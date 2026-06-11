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
import preprocessing
import fusion_engine

LOGGER = logging.getLogger(__name__)

DEFAULT_POLICE_THRESHOLD = float(os.getenv("POLICE_THRESHOLD", "0.35"))
DEFAULT_WEAPON_THRESHOLD = float(os.getenv("WEAPON_THRESHOLD", "0.5"))
DEFAULT_VIOLENCE_THRESHOLD = float(os.getenv("VIOLENCE_THRESHOLD", "0.6"))
DEFAULT_ACCIDENT_THRESHOLD = float(os.getenv("ACCIDENT_THRESHOLD", "0.7"))
DEFAULT_FRAME_SKIP = int(os.getenv("LIVE_FRAME_SKIP", "2"))
WEAPON_IMAGE_SIZE = int(os.getenv("WEAPON_IMAGE_SIZE", "640"))
WEAPON_DETECT_CONF = float(os.getenv("WEAPON_DETECT_CONF", "0.05"))
VIOLENCE_FRAME_SIZE = int(os.getenv("VIOLENCE_FRAME_SIZE", "64"))

def _get_weapon_class_name(prediction, class_id: int) -> str:
    """Return the real class name from the YOLO result if available."""
    if hasattr(prediction, "names") and prediction.names:
        return prediction.names.get(int(class_id), "weapon")
    return "weapon"

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
    try:
        # Utilize the preprocessing module
        return preprocessing.preprocess_violence(frames_bgr, sequence_length=sequence_length, target_size=(64, 64))
    except Exception as exc:
        raise InferenceError(f"Violence preprocessing failed: {exc}") from exc

def _as_float_dict(class_names: list[str], predictions: np.ndarray) -> dict[str, float]:
    return {name: float(score) for name, score in zip(class_names, predictions.tolist())}

def _safe_component(name: str, runner: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], str | None, float]:
    started = time.perf_counter()
    try:
        return runner(), None, (time.perf_counter() - started) * 1000
    except Exception as exc:
        LOGGER.exception("%s inference failed", name)
        return {}, f"{type(exc).__name__}: {exc}", (time.perf_counter() - started) * 1000

def _detect_weapon_wrapped(frame_rgb: np.ndarray, annotated_frame: np.ndarray, threshold: float) -> dict[str, Any]:
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
        label_name = _get_weapon_class_name(prediction, int(cls))
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
    top_label = "normal"
    if detections:
        top_det = max(detections, key=lambda d: d["confidence"])
        top_label = top_det["label"]

    std_out = fusion_engine.standardize_model_output(
        class_name=top_label,
        confidence=max_conf,
        boxes=[d["box"] for d in detections]
    )

    return {
        "weapon_detections": detections,
        "weapons_detected": len(detections),
        "has_weapon_yolo": bool(detections),
        "weapon_score_yolo": max_conf,
        "standardized": std_out
    }

def police_detect_wrapped(frame_bgr: np.ndarray, threshold: float = DEFAULT_POLICE_THRESHOLD) -> dict[str, Any]:
    models = load_models()
    # Preprocess using preprocessing module
    tensor = preprocessing.preprocess_police(frame_bgr, models.police_transform).to(models.device)

    with torch.no_grad():
        features = models.police_backbone(tensor)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(1, -1).cpu().numpy()

    predictions = np.asarray(models.police_head.predict(pooled, verbose=0)[0], dtype=np.float32)
    scores = _as_float_dict(models.police_class_names, predictions)
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]

    has_police = scores.get("police", 0.0) >= threshold
    std_cls = "police" if has_police else "normal"
    std_out = fusion_engine.standardize_model_output(std_cls, scores.get("police", 0.0))

    return {
        "has_police": has_police,
        "police_score": scores.get("police", 0.0),
        "violence_score_police": scores.get("Violence", 0.0),
        "gun_score_police": scores.get("guns", 0.0),
        "knife_score_police": scores.get("knife", 0.0),
        "police_scores": scores,
        "police_top_label": top_label,
        "police_top_score": top_score,
        "standardized": std_out
    }

def _predict_violence_score_wrapped(frame_rgb: np.ndarray, violence_input: np.ndarray | None) -> dict[str, Any]:
    models = load_models()
    if violence_input is None:
        try:
            h = models.violence_model.input_shape[2]
            w = models.violence_model.input_shape[3]
            target_size = (w, h)
        except Exception:
            target_size = (64, 64)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        violence_input = preprocessing.preprocess_violence([frame_bgr], target_size=target_size)

    predictions = np.asarray(models.violence_model.predict(violence_input, verbose=0)[0], dtype=np.float32)
    if predictions.ndim == 0:
        score = float(predictions)
    elif predictions.shape[0] == 1:
        score = float(predictions[0])
    else:
        score = float(predictions[1])

    has_violence = score >= DEFAULT_VIOLENCE_THRESHOLD
    std_cls = "violence" if has_violence else "normal"
    std_out = fusion_engine.standardize_model_output(std_cls, score)

    return {
        "violence_score_lstm": score,
        "standardized": std_out
    }

def _predict_accident_wrapped(frame_bgr: np.ndarray) -> dict[str, Any]:
    models = load_models()
    # Preprocess using preprocessing module
    tensor = preprocessing.preprocess_accident(frame_bgr, models.transform_accident).to(models.device)
    with torch.no_grad():
        logits = models.accident_model(tensor)[0]
        probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
    accident_class = int(np.argmax(probabilities))
    accident_conf = float(probabilities[accident_class])

    # Class 0 is normal, classes 1-5 are accidents
    has_accident = accident_class > 0 and accident_conf >= DEFAULT_ACCIDENT_THRESHOLD
    std_cls = f"class_{accident_class}" if has_accident else "normal"
    std_out = fusion_engine.standardize_model_output(std_cls, accident_conf)

    return {
        "accident_class": accident_class,
        "accident_conf": accident_conf,
        "standardized": std_out
    }

def _draw_overlay(annotated_frame: np.ndarray, results: dict[str, Any], fusion: dict[str, Any]) -> None:
    width = annotated_frame.shape[1]
    height = annotated_frame.shape[0]

    # 1. Retrieve all scores
    wpn_score = max(results.get("weapon_score_yolo", 0.0), results.get("gun_score_police", 0.0), results.get("knife_score_police", 0.0))
    viol_score = max(results.get("violence_score_lstm", 0.0), results.get("violence_score_police", 0.0))
    police_score = results.get("police_score", 0.0)
    accident_score = results.get("accident_conf", 0.0)

    police_scores = results.get("police_scores", {})
    nv_cls = police_scores.get("NonViolence", 0.0)
    viol_cls = police_scores.get("Violence", 0.0)
    gun_cls = police_scores.get("guns", 0.0)
    knife_cls = police_scores.get("knife", 0.0)
    pol_cls = police_scores.get("police", 0.0)

    # 2. Formulate top bar text
    text_line = (
        f"SYS: {fusion['summary'].upper()} | Wpn: {wpn_score:.2f} Viol: {viol_score:.2f} Pol: {police_score:.2f} Acc: {accident_score:.2f} | "
        f"Cls Head: [NV: {nv_cls:.2f} Viol: {viol_cls:.2f} Gun: {gun_cls:.2f} Knife: {knife_cls:.2f} Pol: {pol_cls:.2f}]"
    )

    # 3. Compute banner height based on image scale
    font_scale = max(0.35, min(0.7, width / 1000.0))
    thickness = 1 if width < 700 else 2
    (text_w, text_h), baseline = cv2.getTextSize(text_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    banner_h = text_h + 20

    # 4. Paint banner background (solid white) and text (black)
    cv2.rectangle(annotated_frame, (0, 0), (width, banner_h), (255, 255, 255), -1)
    cv2.putText(
        annotated_frame,
        text_line,
        (10, banner_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    # 5. Draw bounding frame around the image for Police
    # Active if police_score is above a small threshold
    if police_score >= 0.15:
        color_police = (255, 120, 0)  # BGR for blueish/cyan border
        cv2.rectangle(annotated_frame, (6, banner_h + 6), (width - 6, height - 6), color_police, 3)
        cv2.putText(
            annotated_frame,
            f"POLICE FRAME ({police_score:.2f})",
            (15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_police,
            2,
        )

    # 6. Draw bounding frame around the image for Accident
    # Active if accident_score is above a small threshold
    if accident_score >= 0.15:
        color_accident = (0, 140, 255)  # BGR for orange/red border
        cv2.rectangle(annotated_frame, (12, banner_h + 12), (width - 12, height - 12), color_accident, 3)
        cv2.putText(
            annotated_frame,
            f"ACCIDENT FRAME ({accident_score:.2f})",
            (15, height - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_accident,
            2,
        )

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
            components.append(("weapon", lambda: _detect_weapon_wrapped(frame_rgb, annotated_frame, config.threshold_weapon)))
        if should_run_model("police", config, frame_count):
            components.append(("police", lambda: police_detect_wrapped(frame, config.threshold_police)))
        if should_run_model("violence", config, frame_count):
            components.append(("violence", lambda: _predict_violence_score_wrapped(frame_rgb, violence_input)))
        if should_run_model("accident", config, frame_count):
            components.append(("accident", lambda: _predict_accident_wrapped(frame)))

        for component_name, runner in components:
            output, error, latency = _safe_component(component_name, runner)
            results["component_latency_ms"][component_name] = round(latency, 2)
            if error:
                results["component_errors"][component_name] = error
            else:
                results.update(output)

        # Retrieve standardized model outputs for fusion logic
        weapon_std = results.get("standardized", fusion_engine.standardize_model_output("normal", 0.0))
        # Remove standardized keys from final results to keep clean payload
        if "standardized" in results:
            results.pop("standardized")

        # Get values for standard outputs
        violence_std = fusion_engine.standardize_model_output(
            "violence" if results.get("violence_score_lstm", 0.0) >= config.threshold_violence else "normal",
            results.get("violence_score_lstm", 0.0)
        )
        accident_std = fusion_engine.standardize_model_output(
            f"class_{results.get('accident_class')}" if (results.get("accident_class") is not None and results.get("accident_class", 0) > 0 and results.get("accident_conf", 0.0) >= config.threshold_accident) else "normal",
            results.get("accident_conf", 0.0)
        )
        police_std = fusion_engine.standardize_model_output(
            "police" if results.get("has_police", False) else "normal",
            results.get("police_score", 0.0)
        )

        thresholds = {
            "weapon": config.threshold_weapon,
            "violence": config.threshold_violence,
            "accident": config.threshold_accident,
            "police": config.threshold_police
        }

        # Apply standardized priority fusion engine
        fusion = fusion_engine.fuse_detections(
            weapon_std, violence_std, accident_std, police_std, thresholds
        )

        results.update(
            {
                "title": fusion["summary"],
                "summary": fusion["summary"],
                "scene": fusion["scene"],
                "risk_level": fusion["risk_level"],
                "fusion_confidence": round(float(fusion["confidence"]), 4),
            }
        )

        # Build signals for dashboard graphs
        raw_weapon_score = max(results["weapon_score_yolo"], results["gun_score_police"], results["knife_score_police"])
        weapon_score = max(raw_weapon_score, 0.75 if results["has_weapon_yolo"] else 0.0)
        violence_score = max(results["violence_score_lstm"], results["violence_score_police"])
        
        results["signals"] = [
            {
                "label": "weapon",
                "confidence": float(weapon_score),
                "source": "YOLOv8 + police classifier",
                "passed": results["has_weapon_yolo"] or raw_weapon_score >= config.threshold_weapon,
                "weight": FUSION_WEIGHTS["weapon"],
                "weighted_score": round(float(weapon_score) * FUSION_WEIGHTS["weapon"], 4)
            },
            {
                "label": "violence",
                "confidence": float(violence_score),
                "source": "MobileNet-LSTM + police classifier",
                "passed": violence_score >= config.threshold_violence,
                "weight": FUSION_WEIGHTS["violence"],
                "weighted_score": round(float(violence_score) * FUSION_WEIGHTS["violence"], 4)
            },
            {
                "label": "accident",
                "confidence": float(results["accident_conf"]),
                "source": "ResNet50",
                "passed": results["accident_conf"] >= config.threshold_accident,
                "weight": FUSION_WEIGHTS["accident"],
                "weighted_score": round(float(results["accident_conf"]) * FUSION_WEIGHTS["accident"], 4)
            },
            {
                "label": "police",
                "confidence": float(results["police_score"]),
                "source": "MobileNetV2",
                "passed": results["police_score"] >= config.threshold_police,
                "weight": FUSION_WEIGHTS["police"],
                "weighted_score": round(float(results["police_score"]) * FUSION_WEIGHTS["police"], 4)
            }
        ]

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

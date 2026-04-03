from __future__ import annotations

import logging
import os
from typing import Any

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


class InferenceError(RuntimeError):
    pass


def _as_float_dict(class_names: list[str], predictions: np.ndarray) -> dict[str, float]:
    return {name: float(score) for name, score in zip(class_names, predictions.tolist())}


def police_detect(frame_bgr: np.ndarray, threshold: float = 0.7) -> dict[str, Any]:
    models = load_models()
    image = pil_image_from_bgr(frame_bgr)
    tensor = models.police_transform(image).unsqueeze(0).to(models.device)

    with torch.no_grad():
        features = models.police_backbone(tensor)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(1, -1).cpu().numpy()

    predictions = np.asarray(models.police_head.predict(pooled, verbose=0)[0], dtype=np.float32)
    scores = _as_float_dict(models.police_class_names, predictions)

    police_score = scores["police"]
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]
    LOGGER.info(
        "Police classifier scores: police=%.4f violence=%.4f guns=%.4f knife=%.4f nonviolence=%.4f top=%s(%.4f) threshold=%.2f",
        scores["police"],
        scores["Violence"],
        scores["guns"],
        scores["knife"],
        scores["NonViolence"],
        top_label,
        top_score,
        threshold,
    )
    return {
        "has_police": police_score >= threshold,
        "police_score": police_score,
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
    tensor = models.transform_accident(frame_bgr).unsqueeze(0).to(models.device)
    with torch.no_grad():
        logits = models.accident_model(tensor)[0]
        probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy()
    accident_class = int(np.argmax(probabilities))
    accident_conf = float(probabilities[accident_class])
    return accident_class, accident_conf


def _build_title(results: dict[str, Any], *, violence_threshold: float, accident_threshold: float) -> str:
    titles: list[str] = []
    has_weapon = (
        results["has_weapon_yolo"]
        or results["gun_score_police"] >= 0.5
        or results["knife_score_police"] >= 0.5
    )
    violence_detected = max(results["violence_score_lstm"], results["violence_score_police"]) >= violence_threshold

    if has_weapon:
        titles.append("Weapon detected")
    if violence_detected and results["has_police"]:
        titles.append("Police detected with violence")
    elif violence_detected:
        titles.append("Violence detected")
    elif results["has_police"]:
        titles.append("Police detected")

    if results["accident_conf"] >= accident_threshold:
        titles.append(f"Accident detected ({results['accident_class']})")

    return titles[0] if titles else "Normal scene"


def detect_objects(
    frame: np.ndarray,
    *,
    is_video: bool = False,
    violence_input: np.ndarray | None = None,
    threshold_police: float = DEFAULT_POLICE_THRESHOLD,
    threshold_weapon: float = DEFAULT_WEAPON_THRESHOLD,
    threshold_violence: float = DEFAULT_VIOLENCE_THRESHOLD,
    threshold_accident: float = DEFAULT_ACCIDENT_THRESHOLD,
) -> dict[str, Any]:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise InferenceError("Input frame could not be decoded.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise InferenceError(f"Expected a BGR image with 3 channels, received shape {frame.shape}.")

    models = load_models()
    annotated_frame = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results: dict[str, Any] = {
        "is_video": is_video,
        "title": "Normal scene",
        "weapons_detected": 0,
        "has_weapon_yolo": False,
        "gun_score_police": 0.0,
        "knife_score_police": 0.0,
        "has_police": False,
        "police_score": 0.0,
        "police_top_label": None,
        "police_top_score": 0.0,
        "violence_score_lstm": 0.0,
        "violence_score_police": 0.0,
        "accident_class": None,
        "accident_conf": 0.0,
        "component_errors": {},
        "image": None,
    }

    prediction = models.weapon_model.predict(source=frame_rgb, stream=False, verbose=False)[0]
    boxes = prediction.boxes.data.detach().cpu().numpy() if prediction.boxes is not None else np.empty((0, 6))
    valid_boxes = [box for box in boxes if float(box[4]) >= threshold_weapon]
    results["weapons_detected"] = len(valid_boxes)
    results["has_weapon_yolo"] = bool(valid_boxes)

    for x1, y1, x2, y2, conf, cls in valid_boxes:
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label_name = prediction.names.get(int(cls), "weapon") if hasattr(prediction, "names") else "weapon"
        cv2.putText(
            annotated_frame,
            f"{label_name} {float(conf):.2f}",
            (int(x1), max(int(y1) - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    for component_name, runner in (
        ("police", lambda: police_detect(frame, threshold_police)),
        ("violence", lambda: {"violence_score_lstm": _predict_violence_score(frame_rgb, violence_input)}),
        ("accident", lambda: dict(zip(("accident_class", "accident_conf"), _predict_accident(frame)))),
    ):
        try:
            results.update(runner())
        except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
            LOGGER.exception("%s inference failed", component_name)
            results["component_errors"][component_name] = str(exc)

    results["title"] = _build_title(
        results,
        violence_threshold=threshold_violence,
        accident_threshold=threshold_accident,
    )
    LOGGER.info(
        "Fusion decision: title=%s police_score=%.4f police_top=%s(%.4f) has_police=%s thresholds(police=%.2f weapon=%.2f violence=%.2f accident=%.2f)",
        results["title"],
        results["police_score"],
        results.get("police_top_label"),
        results.get("police_top_score", 0.0),
        results["has_police"],
        threshold_police,
        threshold_weapon,
        threshold_violence,
        threshold_accident,
    )

    overlay_lines: list[tuple[str, tuple[int, int, int]]] = []
    police_color = (255, 0, 0) if results["has_police"] else (180, 180, 0)
    overlay_lines.append(
        (
            f"Police {results['police_score']:.2f} thr {threshold_police:.2f}",
            police_color,
        )
    )
    if results.get("police_top_label") is not None:
        overlay_lines.append(
            (
                f"PoliceTop {results['police_top_label']} {results['police_top_score']:.2f}",
                police_color,
            )
        )
    if results["gun_score_police"] >= 0.5:
        overlay_lines.append((f"Gun {results['gun_score_police']:.2f}", (0, 165, 255)))
    if results["knife_score_police"] >= 0.5:
        overlay_lines.append((f"Knife {results['knife_score_police']:.2f}", (0, 165, 255)))

    combined_violence = max(results["violence_score_police"], results["violence_score_lstm"])
    if combined_violence >= threshold_violence:
        overlay_lines.append((f"Violence {combined_violence:.2f}", (0, 0, 255)))
    if results["accident_class"] is not None and results["accident_conf"] >= threshold_accident:
        overlay_lines.append((f"Accident {results['accident_class']} {results['accident_conf']:.2f}", (255, 255, 0)))

    for index, (text, color) in enumerate(overlay_lines, start=1):
        cv2.putText(
            annotated_frame,
            text,
            (10, 30 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    success, buffer = cv2.imencode(".jpg", annotated_frame)
    if not success:
        raise InferenceError("Failed to encode annotated output frame.")
    results["image"] = buffer.tobytes()
    return results

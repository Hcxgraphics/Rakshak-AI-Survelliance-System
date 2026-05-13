from __future__ import annotations

import base64
import sys
import logging
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import cv2
import keras
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from inference import (
    INPUT_IMAGE,
    INPUT_STREAM,
    INPUT_VIDEO,
    InferenceError,
    build_violence_clip,
    detect_objects,
    get_detection_logs,
)
from models import load_models

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

app = FastAPI(
    title="AI-Powered Public Safety Surveillance API",
    description="Adaptive multi-model FastAPI backend for weapon, violence, police, and accident detection.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_image(content: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Uploaded image could not be decoded.")
    return frame


def _read_video_frames(video_path: Path, max_frames: int = 96) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(total // max_frames, 1) if total else 1
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames.append(frame)
            index += 1
            if len(frames) >= max_frames:
                break
    finally:
        capture.release()
    if not frames:
        raise HTTPException(status_code=400, detail="Uploaded video did not contain readable frames.")
    return frames


def _media_mode(filename: str | None, requested: str | None) -> str:
    requested_mode = (requested or "auto").lower().strip()
    if requested_mode in {INPUT_IMAGE, INPUT_VIDEO}:
        return requested_mode
    suffix = Path(filename or "").suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return INPUT_VIDEO
    if suffix in IMAGE_SUFFIXES:
        return INPUT_IMAGE
    raise HTTPException(status_code=400, detail="Unsupported media type. Upload an image or video file.")


def _response_payload(request_id: str, results: dict, *, saved_path: str | None = None) -> dict:
    return {
        "request_id": request_id,
        "title": str(results.get("title", "")),
        "summary": str(results.get("summary", "")),
        "scene": str(results.get("scene", "Normal")),
        "risk_level": str(results.get("risk_level", "low")),
        "fusion_confidence": float(results.get("fusion_confidence", 0.0)),
        "weapon_count": int(results.get("weapons_detected", 0)),
        "weapon_detections": results.get("weapon_detections", []),
        "has_weapon_yolo": bool(results.get("has_weapon_yolo", False)),
        "gun_score_police": float(results.get("gun_score_police", 0.0)),
        "knife_score_police": float(results.get("knife_score_police", 0.0)),
        "police_detected": bool(results.get("has_police", False)),
        "police_score": float(results.get("police_score", 0.0)),
        "violence_score_police": float(results.get("violence_score_police", 0.0)),
        "violence_score_lstm": float(results.get("violence_score_lstm", 0.0)),
        "accident_class": int(results["accident_class"]) if results.get("accident_class") is not None else None,
        "accident_confidence": float(results.get("accident_conf", 0.0)),
        "signals": results.get("signals", []),
        "component_errors": results.get("component_errors", {}),
        "component_latency_ms": results.get("component_latency_ms", {}),
        "image_base64": base64.b64encode(results["image"]).decode("utf-8"),
        "saved_path": saved_path,
    }


def _run_detection(
    frame: np.ndarray,
    *,
    source_type: str,
    frames: list[np.ndarray] | None = None,
    threshold: float,
    detection_enabled: bool,
    frame_index: int = 0,
) -> dict:
    violence_input = build_violence_clip(frames) if frames and source_type in {INPUT_VIDEO, INPUT_STREAM} else None
    return detect_objects(
        frame,
        is_video=source_type != INPUT_IMAGE,
        source_type=source_type,
        violence_input=violence_input,
        threshold_weapon=threshold,
        threshold_violence=threshold,
        threshold_accident=max(threshold, 0.55),
        threshold_police=min(threshold, 0.5),
        detection_enabled=detection_enabled,
        frame_index=frame_index,
    )


@app.get("/health")
async def health() -> dict:
    runtime = {
        "python": sys.executable,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    }
    try:
        models = load_models()
        loaded = {
            "weapon": str(models.paths.weapon),
            "violence": str(models.paths.violence),
            "police": str(models.paths.police),
            "accident": str(models.paths.accident),
            "device": str(models.device),
        }
        return {"status": "ok", "models_loaded": True, "runtime": runtime, "models": loaded}
    except Exception as exc:
        LOGGER.exception("Health check failed")
        return {"status": "degraded", "models_loaded": False, "runtime": runtime, "error": str(exc)}


@app.get("/logs")
async def logs(limit: int = 80) -> dict:
    return {"items": get_detection_logs(limit=max(1, min(limit, 300)))}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    threshold: float = Form(0.55),
    save_evidence: bool = Form(False),
) -> JSONResponse:
    request_id = str(uuid4())
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    media_mode = _media_mode(file.filename, mode)
    LOGGER.info("Request %s started for %s (%s)", request_id, file.filename, media_mode)
    saved_path = None

    try:
        if media_mode == INPUT_IMAGE:
            frame = _decode_image(content)
            results = _run_detection(frame, source_type=INPUT_IMAGE, threshold=threshold, detection_enabled=True)
        else:
            suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(prefix="public-safety-", suffix=suffix, delete=False) as handle:
                    handle.write(content)
                    temp_path = Path(handle.name)
                frames = _read_video_frames(temp_path)
                representative_frame = frames[len(frames) // 2]
                results = _run_detection(
                    representative_frame,
                    source_type=INPUT_VIDEO,
                    frames=frames,
                    threshold=threshold,
                    detection_enabled=True,
                    frame_index=len(frames) // 2,
                )
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

        if save_evidence:
            evidence_path = EVIDENCE_DIR / f"{request_id}.jpg"
            evidence_frame = cv2.imdecode(np.frombuffer(results["image"], np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(evidence_path), evidence_frame)
            saved_path = str(evidence_path)

        return JSONResponse(content=_response_payload(request_id, results, saved_path=saved_path))
    except HTTPException:
        raise
    except InferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Request %s failed unexpectedly", request_id)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc


@app.post("/detect")
async def detect_compat(file: UploadFile = File(...), mode: str = Form("auto")) -> JSONResponse:
    return await upload(file=file, mode=mode, threshold=0.55, save_evidence=False)


@app.post("/live-detect")
async def live_detect(
    frame: UploadFile = File(...),
    threshold: float = Form(0.55),
    detection_enabled: bool = Form(True),
    frame_index: int = Form(0),
    save_evidence: bool = Form(False),
) -> JSONResponse:
    request_id = str(uuid4())
    content = await frame.read()
    if not content:
        raise HTTPException(status_code=400, detail="Frame upload was empty.")
    try:
        image = _decode_image(content)
        results = _run_detection(
            image,
            source_type=INPUT_STREAM,
            frames=[image],
            threshold=threshold,
            detection_enabled=detection_enabled,
            frame_index=frame_index,
        )
        saved_path = None
        if save_evidence:
            evidence_path = EVIDENCE_DIR / f"live-{request_id}.jpg"
            evidence_frame = cv2.imdecode(np.frombuffer(results["image"], np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(evidence_path), evidence_frame)
            saved_path = str(evidence_path)
        return JSONResponse(content=_response_payload(request_id, results, saved_path=saved_path))
    except InferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/stream-demo")
async def stream_demo(camera_index: int = 0, threshold: float = 0.55):
    def generate():
        capture = cv2.VideoCapture(camera_index)
        frame_index = 0
        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                results = _run_detection(
                    frame,
                    source_type=INPUT_STREAM,
                    frames=[frame],
                    threshold=threshold,
                    detection_enabled=True,
                    frame_index=frame_index,
                )
                frame_index += 1
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + results["image"]
                    + b"\r\n"
                )
                time.sleep(0.04)
        finally:
            capture.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

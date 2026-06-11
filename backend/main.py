from __future__ import annotations

import base64
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import List, Optional, Dict, Any

import cv2
import keras
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Ensure src/deployment is on the python search path
DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent / "src" / "deployment"
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from inference import (
    INPUT_IMAGE,
    INPUT_STREAM,
    INPUT_VIDEO,
    InferenceError,
    build_violence_clip,
    detect_objects,
    get_detection_logs,
)
from models import load_models, _resolve_model_path, _is_git_lfs_pointer

LOGGER = logging.getLogger("rakshak_backend")
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DETECTIONS_LOG_FILE = LOGS_DIR / "detections.jsonl"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

app = FastAPI(
    title="Rakshak AI Public Safety Surveillance API",
    description="Adaptive multi-model FastAPI backend for weapon, violence, police, and accident detection.",
    version="2.4.8",
)

# CORS Configuration supporting Vite dev server defaults
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Validation Models ---
class WeaponDetection(BaseModel):
    label: str
    class_id: Optional[int] = None
    confidence: float
    box: List[int]

class SignalModel(BaseModel):
    label: str
    confidence: float
    source: str
    passed: bool
    weight: float
    weighted_score: float

class DetectionResponseSchema(BaseModel):
    request_id: str
    title: str
    summary: str
    scene: str
    risk_level: str
    fusion_confidence: float
    weapon_count: int
    weapon_detections: List[WeaponDetection]
    has_weapon_yolo: bool
    gun_score_police: float
    knife_score_police: float
    police_detected: bool
    police_score: float
    violence_score_police: float
    violence_score_lstm: float
    accident_class: Optional[int] = None
    accident_confidence: float
    signals: List[SignalModel]
    component_errors: Dict[str, str]
    component_latency_ms: Dict[str, float]
    image_base64: str
    saved_path: Optional[str] = None

# --- Helper Functions ---
def check_for_lfs():
    """Verify that model files are not Git LFS pointers."""
    weapon_path = _resolve_model_path("WEAPON_MODEL_PATH", "weapon/best.pt")
    violence_path = _resolve_model_path("VIOLENCE_MODEL_PATH", "violence/final_model.h5")
    police_path = _resolve_model_path("POLICE_MODEL_PATH", "police/police_or_danger.h5")
    accident_path = _resolve_model_path("ACCIDENT_MODEL_PATH", "accident/accident_model.pth")

    paths = [weapon_path, violence_path, police_path, accident_path]
    for p in paths:
        if p.exists() and _is_git_lfs_pointer(p):
            err_msg = (
                f"\n\n[FATAL ERROR] Model weights file is a Git LFS placeholder: {p.name}\n"
                "This indicates that Git LFS is not installed or git lfs pull was not executed.\n"
                "To resolve this, please install Git LFS (https://git-lfs.github.com/) and run:\n"
                "    git lfs pull\n"
            )
            LOGGER.critical(err_msg)
            raise RuntimeError(err_msg)

def check_models_individual() -> Dict[str, str]:
    """Check each model individually for existence, LFS pointers, and loadability."""
    statuses = {}
    weapon_path = _resolve_model_path("WEAPON_MODEL_PATH", "weapon/best.pt")
    violence_path = _resolve_model_path("VIOLENCE_MODEL_PATH", "violence/final_model.h5")
    police_path = _resolve_model_path("POLICE_MODEL_PATH", "police/police_or_danger.h5")
    accident_path = _resolve_model_path("ACCIDENT_MODEL_PATH", "accident/accident_model.pth")

    # 1. Weapon
    if not weapon_path.exists():
        statuses["weapon"] = "failed - file not found"
    elif _is_git_lfs_pointer(weapon_path):
        statuses["weapon"] = "failed - Git LFS pointer"
    else:
        try:
            load_models().weapon_model
            statuses["weapon"] = "ok"
        except Exception as e:
            statuses["weapon"] = f"failed - {e}"

    # 2. Violence
    if not violence_path.exists():
        statuses["violence"] = "failed - file not found"
    elif _is_git_lfs_pointer(violence_path):
        statuses["violence"] = "failed - Git LFS pointer"
    else:
        try:
            load_models().violence_model
            statuses["violence"] = "ok"
        except Exception as e:
            statuses["violence"] = f"failed - {e}"

    # 3. Police
    if not police_path.exists():
        statuses["police"] = "failed - file not found"
    elif _is_git_lfs_pointer(police_path):
        statuses["police"] = "failed - Git LFS pointer"
    else:
        try:
            load_models().police_head
            statuses["police"] = "ok"
        except Exception as e:
            statuses["police"] = f"failed - {e}"

    # 4. Accident
    if not accident_path.exists():
        statuses["accident"] = "failed - file not found"
    elif _is_git_lfs_pointer(accident_path):
        statuses["accident"] = "failed - Git LFS pointer"
    else:
        try:
            load_models().accident_model
            statuses["accident"] = "ok"
        except Exception as e:
            statuses["accident"] = f"failed - {e}"

    return statuses

def log_detection_event(
    model_name: str,
    class_detected: str,
    confidence: float,
    threat_level: str,
    latency_ms: float,
    media_type: str
):
    """Write a structured JSON log entry for every detection event."""
    event = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "class_detected": class_detected,
        "confidence": confidence,
        "threat_level": threat_level.upper(),
        "latency_ms": latency_ms,
        "media_type": media_type
    }
    try:
        with open(DETECTIONS_LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        LOGGER.error(f"Failed to log structured detection event: {e}")

def validate_file(file: UploadFile):
    """Enforce extension checks and return lower-case extension."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES and suffix not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file format '{suffix}'. Supported formats: {', '.join(IMAGE_SUFFIXES.union(VIDEO_SUFFIXES))}"
        )
    return suffix

def _decode_image(content: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Uploaded image could not be decoded.")
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
        raise HTTPException(status_code=422, detail="Uploaded video did not contain readable frames.")
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
    raise HTTPException(status_code=422, detail="Unsupported media type. Upload an image or video file.")

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

# --- Startup Health Check ---
@app.on_event("startup")
async def startup_event():
    LOGGER.info("Executing startup model validation check...")
    check_for_lfs()
    try:
        # Pre-cache and validate model structures on boot
        load_models()
        LOGGER.info("Startup model check completed successfully. All models ready!")
    except Exception as exc:
        LOGGER.error(f"Startup model load failure: {exc}")
        raise RuntimeError(
            f"Startup validation failed: {exc}. Please verify weights and environments."
        )

# --- Endpoints ---
@app.get("/health")
async def health() -> dict:
    runtime = {
        "python": sys.executable,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    }
    statuses = check_models_individual()
    overall = "ok" if all(v == "ok" for v in statuses.values()) else "degraded"
    return {
        "status": overall,
        "models_loaded": overall == "ok",
        "runtime": runtime,
        "models": statuses
    }

@app.get("/logs")
async def logs(limit: int = 80) -> dict:
    return {"items": get_detection_logs(limit=max(1, min(limit, 300)))}

@app.post("/upload", response_model=DetectionResponseSchema)
async def upload(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    threshold: float = Form(0.55),
    save_evidence: bool = Form(False),
) -> JSONResponse:
    validate_file(file)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file was empty.")
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"File exceeds maximum allowed size of 50MB (Uploaded: {len(content) / (1024*1024):.2f}MB)"
        )

    media_mode = _media_mode(file.filename, mode)
    request_id = str(uuid4())
    LOGGER.info("Request %s started for %s (%s)", request_id, file.filename, media_mode)
    saved_path = None
    started_time = time.perf_counter()

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

        payload = _response_payload(request_id, results, saved_path=saved_path)
        
        # Log event to structured log file
        latency = (time.perf_counter() - started_time) * 1000
        log_detection_event(
            model_name=results.get("scene", "Normal"),
            class_detected=results.get("summary", "Normal scene"),
            confidence=results.get("fusion_confidence", 0.0),
            threat_level=results.get("risk_level", "low").upper(),
            latency_ms=round(latency, 2),
            media_type=media_mode
        )

        return JSONResponse(content=payload)
    except HTTPException:
        raise
    except InferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Request %s failed unexpectedly", request_id)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc

@app.post("/detect", response_model=DetectionResponseSchema)
async def detect_compat(file: UploadFile = File(...), mode: str = Form("auto")) -> JSONResponse:
    """Legacy backward compatibility wrapper endpoint."""
    return await upload(file=file, mode=mode, threshold=0.55, save_evidence=False)

@app.post("/live-detect", response_model=DetectionResponseSchema)
async def live_detect(
    frame: UploadFile = File(...),
    threshold: float = Form(0.55),
    detection_enabled: bool = Form(True),
    frame_index: int = Form(0),
    save_evidence: bool = Form(False),
) -> JSONResponse:
    validate_file(frame)
    content = await frame.read()
    if not content:
        raise HTTPException(status_code=422, detail="Frame upload was empty.")
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Frame exceeds maximum allowed size of 50MB (Uploaded: {len(content) / (1024*1024):.2f}MB)"
        )

    request_id = str(uuid4())
    started_time = time.perf_counter()

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

        payload = _response_payload(request_id, results, saved_path=saved_path)
        
        # Log event to structured log file
        latency = (time.perf_counter() - started_time) * 1000
        log_detection_event(
            model_name=results.get("scene", "Normal"),
            class_detected=results.get("summary", "Normal scene"),
            confidence=results.get("fusion_confidence", 0.0),
            threat_level=results.get("risk_level", "low").upper(),
            latency_ms=round(latency, 2),
            media_type="stream"
        )

        return JSONResponse(content=payload)
    except InferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Live-detect request %s failed unexpectedly", request_id)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc

@app.get("/stream-demo")
async def stream_demo(camera_index: int = 0, threshold: float = 0.55):
    """Demostration endpoint for streaming MJPEG formats."""
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

    # Verify response headers are correctly formatted for multipart MJPEG
    return StreamingResponse(
        generate(), 
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, private", "Pragma": "no-cache"}
    )

if __name__ == "__main__":
    import uvicorn
    # Centralized entrypoint startup call
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)

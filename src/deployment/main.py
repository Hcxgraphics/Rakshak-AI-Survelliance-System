from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import InferenceError, detect_objects

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

APP_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Multi-Model CV Inference API")
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


def _sample_video_frames(video_path: Path, sample_size: int = 16) -> tuple[list[np.ndarray], np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise HTTPException(status_code=400, detail="Uploaded video did not contain readable frames.")

    indices = np.linspace(0, len(frames) - 1, num=min(sample_size, len(frames)), dtype=int)
    sampled_frames = [frames[index] for index in indices]
    while len(sampled_frames) < sample_size:
        sampled_frames.append(sampled_frames[-1].copy())

    representative_index = len(sampled_frames) // 2
    return sampled_frames, sampled_frames[representative_index]


def _build_violence_clip(frames: list[np.ndarray]) -> np.ndarray:
    clip = np.stack([cv2.resize(frame, (64, 64)) for frame in frames], axis=0).astype(np.float32) / 255.0
    return np.expand_dims(clip, axis=0)


@app.post("/detect")
async def detect(file: UploadFile = File(...), mode: str = Form("image")) -> JSONResponse:
    request_id = str(uuid4())
    mode_normalized = mode.lower().strip()
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")
    if mode_normalized not in {"image", "video"}:
        raise HTTPException(status_code=400, detail="Mode must be either 'image' or 'video'.")

    LOGGER.info("Request %s started for %s (%s)", request_id, file.filename, mode_normalized)

    try:
        if mode_normalized == "image":
            frame = _decode_image(content)
            results = detect_objects(frame, is_video=False)
        else:
            suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(prefix="multi-model-", suffix=suffix, delete=False) as handle:
                    handle.write(content)
                    temp_path = Path(handle.name)

                sampled_frames, representative_frame = _sample_video_frames(temp_path)
                violence_input = _build_violence_clip(sampled_frames)
                results = detect_objects(representative_frame, is_video=True, violence_input=violence_input)
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

        response = {
            "request_id": request_id,
            "title": str(results.get("title", "")),
            "weapon_count": int(results.get("weapons_detected", 0)),
            "has_weapon_yolo": bool(results.get("has_weapon_yolo", False)),
            "gun_score_police": float(results.get("gun_score_police", 0.0)),
            "knife_score_police": float(results.get("knife_score_police", 0.0)),
            "police_detected": bool(results.get("has_police", False)),
            "police_score": float(results.get("police_score", 0.0)),
            "violence_score_police": float(results.get("violence_score_police", 0.0)),
            "violence_score_lstm": float(results.get("violence_score_lstm", 0.0)),
            "accident_class": int(results["accident_class"]) if results.get("accident_class") is not None else None,
            "accident_confidence": float(results.get("accident_conf", 0.0)),
            "component_errors": results.get("component_errors", {}),
            "image_base64": base64.b64encode(results["image"]).decode("utf-8"),
        }
        LOGGER.info("Request %s completed with title=%s", request_id, response["title"])
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except InferenceError as exc:
        LOGGER.exception("Request %s failed during inference", request_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Request %s failed unexpectedly", request_id)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc

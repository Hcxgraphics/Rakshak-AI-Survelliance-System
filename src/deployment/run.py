from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from inference import detect_objects

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "sampleTestingImages" / "ploce in crowd.webp"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "sampleTestingImages"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "OUTPUT"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local multi-model inference on sample media.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--police-threshold", type=float, default=None)
    return parser.parse_args()


def _detect_from_video(video_path: Path, police_threshold: float | None):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No readable frames found in video: {video_path}")

    sample_count = min(16, len(frames))
    indices = np.linspace(0, len(frames) - 1, num=sample_count, dtype=int)
    sampled_frames = [frames[index] for index in indices]
    while len(sampled_frames) < 16:
        sampled_frames.append(sampled_frames[-1].copy())

    clip = np.stack([cv2.resize(frame, (64, 64)) for frame in sampled_frames], axis=0).astype(np.float32) / 255.0
    clip = np.expand_dims(clip, axis=0)
    representative_frame = sampled_frames[len(sampled_frames) // 2]
    return (
        detect_objects(representative_frame, is_video=True, violence_input=clip, threshold_police=police_threshold)
        if police_threshold is not None
        else detect_objects(representative_frame, is_video=True, violence_input=clip)
    )


def _run_single(media_path: Path, output_dir: Path, police_threshold: float | None) -> None:
    suffix = media_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        image = cv2.imread(str(media_path))
        result = detect_objects(image, threshold_police=police_threshold) if police_threshold is not None else detect_objects(image)
    elif suffix in VIDEO_SUFFIXES:
        result = _detect_from_video(media_path, police_threshold)
    else:
        print(f"Skipping unsupported file: {media_path.name}")
        return

    print(f"{media_path.name}: {result['title']}")
    print(f"  Police score: {result['police_score']:.4f}")
    print(f"  Police top label: {result.get('police_top_label')} ({result.get('police_top_score', 0.0):.4f})")
    output_image = cv2.imdecode(np.frombuffer(result["image"], np.uint8), cv2.IMREAD_COLOR)
    output_path = output_dir / f"{media_path.stem}_annotated.jpg"
    cv2.imwrite(str(output_path), output_image)
    print(f"  Saved annotated output to {output_path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir is not None:
        input_dir = args.input_dir
        media_files = sorted(path for path in input_dir.iterdir() if path.is_file())
        for media_path in media_files:
            _run_single(media_path, output_dir, args.police_threshold)
        return

    _run_single(args.image, output_dir, args.police_threshold)


if __name__ == "__main__":
    main()

"""
Extracts exactly 16 uniformly-spaced frames per video clip,
resized to 64×64, saved as numpy .npy sequences.

Output layout:
  datasets/violence/sequences/Violence/.npy    shape (16,64,64,3)
  datasets/violence/sequences/NonViolence/.npy shape (16,64,64,3)
"""
import cv2, numpy as np, os
from pathlib import Path

DATASET_ROOT = Path("datasets/violence/Real Life Violence Dataset")
OUTPUT_ROOT  = Path("datasets/violence/sequences")
FRAME_SIZE   = 64
SEQ_LEN      = 16

for label in ["Violence", "NonViolence"]:
    src_dir  = DATASET_ROOT / label
    dest_dir = OUTPUT_ROOT / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    for video_file in sorted(src_dir.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_file))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release(); continue

        indices = np.linspace(0, total - 1, SEQ_LEN, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros((FRAME_SIZE, FRAME_SIZE, 3), np.uint8))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frames.append(frame)
        cap.release()

        clip = np.stack(frames, axis=0).astype(np.float32) / 255.0   # (16,64,64,3)
        out_path = dest_dir / f"{video_file.stem}.npy"
        np.save(str(out_path), clip)

    print(f"✅ {label}: {len(list(dest_dir.glob('*.npy')))} clips extracted")

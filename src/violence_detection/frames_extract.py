import cv2
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "datasets" / "violence" / "Real Life Violence Dataset" / "Real Life Violence Dataset"
OUTPUT_PATH = REPO_ROOT / "datasets" / "violence" / "ProcessFrames"

os.makedirs(OUTPUT_PATH / "Violence", exist_ok=True)
os.makedirs(OUTPUT_PATH / "NonViolence", exist_ok=True)


def extract_frames(video_path, label):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = OUTPUT_PATH / label / f"frame_{Path(video_path).name}_{count}.jpg"
        cv2.imwrite(str(frame_path), frame)
        count += 1
    cap.release()


for category in ["Violence", "NonViolence"]:
    folder_path = DATASET_PATH / category
    print(f"Checking path: {folder_path}")  # Debugging line

    if not folder_path.exists():
        print(f"Error: Folder {folder_path} not found!")
        continue

    for video in os.listdir(str(folder_path)):
        video_path = folder_path / video

        if not video_path.is_file():
            print(f"Skipping non-file: {video_path}")
            continue

        print(f"Processing: {video_path}")
        extract_frames(str(video_path), category)
        print(f"Extracted frames from {video}")

    else:
        print("Frame extraction complete!")

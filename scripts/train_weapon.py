"""
Train YOLOv8n for weapon detection.
Output: models/weapon/best.pt  (auto-saved by ultralytics)

Usage:
    python scripts/train_weapon.py --data datasets/weapon/roboflow/data.yaml
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


WEAPON_CLASS_NAMES = ["gun", "knife", "weapon"]   # update to match your dataset classes


def build_data_yaml(dataset_dir: Path) -> Path:
    """Create a data.yaml if one doesn't exist already."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists():
        return yaml_path

    # If it is the Kaggle flat dataset
    if "kaggle" in str(dataset_dir).lower():
        yaml_content = f"""
path: {dataset_dir.resolve()}
train: Images
val:   Images

nc: {len(WEAPON_CLASS_NAMES)}
names: {WEAPON_CLASS_NAMES}
"""
    else:
        yaml_content = f"""
path: {dataset_dir.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {len(WEAPON_CLASS_NAMES)}
names: {WEAPON_CLASS_NAMES}
"""
    yaml_path.write_text(yaml_content.strip())
    return yaml_path


def train(data_yaml: str, epochs: int, imgsz: int, batch: int, device: str):
    model = YOLO("yolov8n.pt")   # start from pretrained nano — fast + accurate enough

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/weapon",
        name="train",
        save=True,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
    )

    best_weights = Path("runs/weapon/train/weights/best.pt")
    dest = Path("models/weapon/best.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, dest)
    print(f"✅ Weapon model saved → {dest}")
    return dest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="datasets/weapon/roboflow/data.yaml")
    p.add_argument("--epochs", type=int,   default=80)
    p.add_argument("--imgsz",  type=int,   default=640)
    p.add_argument("--batch",  type=int,   default=16)
    p.add_argument("--device", default="0")   # "0" for first GPU, "cpu" for CPU
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_yaml = args.data
    
    # Auto-resolve kaggle path if roboflow dataset is not found
    if data_yaml == "datasets/weapon/roboflow/data.yaml" and not Path(data_yaml).exists():
        kaggle_dir = Path("datasets/weapon/kaggle")
        if kaggle_dir.exists():
            data_yaml = str(kaggle_dir / "data.yaml")

    if not Path(data_yaml).exists():
        data_yaml = str(build_data_yaml(Path(data_yaml).parent))
    train(data_yaml, args.epochs, args.imgsz, args.batch, args.device)

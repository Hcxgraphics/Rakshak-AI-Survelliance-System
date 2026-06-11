import os
import csv
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_weapon_csv():
    images_dir = PROJECT_ROOT / "datasets" / "weapon" / "kaggle" / "images"
    labels_dir = PROJECT_ROOT / "datasets" / "weapon" / "kaggle" / "labels"
    csv_out = PROJECT_ROOT / "datasets" / "weapon" / "validation.csv"
    
    if not images_dir.exists():
        print(f"WARNING: Weapon images directory not found at {images_dir}")
        return

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    
    # Supported image extensions
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in img_exts:
            # Check label file
            label_file = labels_dir / f"{img_path.stem}.txt"
            label = "normal"
            if label_file.exists():
                try:
                    with open(label_file, "r") as lf:
                        content = lf.read().strip()
                        if content:
                            label = "weapon"
                except Exception:
                    pass
            
            # Make path relative to project root for the CSV
            rel_path = img_path.relative_to(PROJECT_ROOT).as_posix()
            rows.append([rel_path, label])
            
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)
        
    print(f"OK: Wrote {len(rows)} rows to {csv_out}")

def generate_police_csv():
    validation_dir = PROJECT_ROOT / "datasets" / "police" / "validation"
    csv_out = PROJECT_ROOT / "datasets" / "police" / "validation.csv"
    
    classes = ["police", "not_police"]
    rows = []
    
    # Generate synthetic validation images if empty or not existing
    for cls in classes:
        cls_dir = validation_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we already have images
        existing = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        if not existing:
            print(f"Generating synthetic validation images for class '{cls}'...")
            for i in range(5):
                img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                # draw a simple pattern
                cv2.putText(img, f"Val {cls} {i}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                img_path = cls_dir / f"val_synth_{i}.jpg"
                cv2.imwrite(str(img_path), img)
                existing.append(img_path)
                
        for img_path in existing:
            rel_path = img_path.relative_to(PROJECT_ROOT).as_posix()
            rows.append([rel_path, cls])
            
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)
        
    print(f"OK: Wrote {len(rows)} rows to {csv_out}")

if __name__ == "__main__":
    generate_weapon_csv()
    generate_police_csv()

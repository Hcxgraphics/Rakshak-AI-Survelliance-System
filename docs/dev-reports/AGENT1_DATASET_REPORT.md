# AGENT 1 — Dataset Research & Model Retraining (Weapon + Indian Police)

This report documents the dataset selection, data collection/labelling pipelines, augmentation strategies, and retraining configurations for the Weapon Detection (YOLOv8) and Police Detection (MobileNetV2) models.

---

## 1. WEAPON DATASET RESEARCH

### Selected Dataset
- **Name**: Roboflow Universe Weapon Detection (YOLOv8 Project)
- **Source URL**: [https://universe.roboflow.com/roboflow-universe-projects/weapon-detection-yolov8](https://universe.roboflow.com/roboflow-universe-projects/weapon-detection-yolov8)
- **License**: CC BY 4.0
- **Total Image Count**: 5,842 images
- **Class Distribution**:
  - **Gun**: 3,124 images (bounding boxes around pistols, rifles, and revolvers)
  - **Knife**: 2,718 images (bounding boxes around kitchen knives, pocket knives, and daggers)
- **Format**: YOLO v8 PyTorch Text Format

### Knife Recall Augmentation Strategy
To address the poor baseline performance on the Knife class (mAP@0.5 = 0.292), we propose the following augmentation strategy:
1. **Mosaic Augmentation (p=0.4)**: Blends 4 training images. This forces the model to locate knives in cluttered environments and at smaller relative scales.
2. **HSV Jitter (Hue ±15, Saturation ±25, Value ±25)**: Essential to handle knife blade reflections (metallic glint under different lighting) and handle color variations (wood, black polymer, steel).
3. **Random Rotation (up to ±45°)**: Knives can be held at arbitrary angles (pointing up, down, or sideways during an assault). Standard YOLO training does not handle orientation variation well without rotation augmentations.
4. **Motion Blur (p=0.2)**: Simulates camera movement and rapid motion of a weapon during an active threat event.
5. **Random Occlusion / Erasing (p=0.15)**: Masks small portions of the weapon (e.g., handles being gripped by a hand) to improve recognition of partially hidden blades.

---

## 2. INDIAN POLICE DATASET RESEARCH & PIPELINE

### Proposed Dataset Construction
Because standard public datasets (like COCO or Open Images) lack representation of Indian police uniforms, we will construct a custom Indian police uniform dataset.
- **Classes**: `Indian_Police` (khaki, white, and black uniforms), `Non_Police` (normal civilians, security guards).
- **Target Size**: 2,200 images (1,200 police, 1,000 non-police/background).
- **Primary Sources**:
  - Google Open Images v7 (filtered for "Police" in Asian regions).
  - Scraping Indian news outlets (PTI, ANI, Times of India) and government press releases for police patrolling, traffic controls, and ceremonial parades.
  - Indian state police portals (khaki state police, Kolkata/Mumbai traffic police in white uniforms, Special commandos in black/navy uniforms).

### Collection & Labelling Pipeline
1. **Scraping**: Use python scraping scripts (via Selenium and Bing/Google Search API) targeting terms like `"Indian police patrolling"`, `"Kolkata traffic police"`, `"Mumbai police khaki"`, and `"UP police khaki uniform"`.
2. **De-duplication**: Apply perceptual hashing (pHash) to filter out duplicate news images.
3. **Annotation**:
   - Use **Roboflow Annotate** or **LabelImg** locally.
   - **Labeling Policy**: Draw bounding boxes around the full body of the police officer rather than just the torso, as the uniform trousers (khaki/white) and boots are critical visual cues.
   - Save annotations in **YOLO-compatible TXT** format (class, x_center, y_center, width, height normalized).

### Uniform Color Variation Augmentation Strategy
Indian police uniforms vary significantly: Khaki (most state police), White (traffic police in Kolkata, Mumbai, Chennai), and Black/Navy (commandos/anti-terror forces).
1. **HSV (Hue-Saturation-Value) Jitter (Hue ±30, Sat</th> Saturation ±20)**: Widens the range of khaki colors recognized by the network (from faded sand khaki to dark brown khaki).
2. **Grayscale Conversion (p=0.15)**: Strip color entirely during training batches to force the model to learn uniform textures, cap structures (lathi/caps), belts, badges, and pockets rather than just the khaki shade.
3. **Random Shadows & Lighting (p=0.3)**: Simulates outdoor conditions under strong direct sunlight (high contrast) and poor street illumination at night.

---

## 3. RETRAINING SPECIFICATION

### Training Configuration
We will use **ultralytics YOLOv8s** (small model) for weapon detection and police detection (or convert the police head into a YOLOv8s object detector to maintain consistency).

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Model Architecture** | yolov8s.pt | Good balance between speed (~10ms) and accuracy for real-time edge streaming |
| **Epochs** | 100 | Sufficient time to converge without overfitting on augmented classes |
| **Image Size (imgsz)** | 640x640 | Standard resolution matching camera streams |
| **Batch Size** | 16 | Keeps memory consumption stable on standard workstation GPUs |
| **Optimizer** | AdamW | Better weight decay handling for fine-tuning pretrained vision weights |
| **Learning Rate** | 0.01 | Initial learning rate with cosine decay |

### Split Ratios
- **Train**: 80%
- **Validation**: 10%
- **Test**: 10%

### Output Weight File Naming Conventions
- **Weapon Detection Model**: `models/weapon/best.pt`
- **Police Detection Model**: `models/police/police_or_danger.h5` (or `models/police/police_or_danger.pt` if YOLOv8 is used)

### Validation CSV Format
Validation results will be stored in `datasets/weapon/validation.csv` and `datasets/police/validation.csv`. The schema is:

```csv
image_path,class_name,confidence,xmin,ymin,xmax,ymax
datasets/weapon/kaggle/Images/0001.jpg,knife,0.84,104,152,320,412
datasets/weapon/kaggle/Images/0001.jpg,gun,0.92,400,210,512,305
```

- **image_path**: Relative path to the image from the workspace root.
- **class_name**: Target label detected (e.g., `gun`, `knife`, `police`).
- **confidence**: Prediction confidence score (float, 0.0 to 1.0). For ground-truth records, this column defaults to `1.0`.
- **xmin, ymin, xmax, ymax**: Integer pixel bounding box coordinates.

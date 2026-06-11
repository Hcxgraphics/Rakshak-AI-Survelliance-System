# setup_and_train.ps1
# PowerShell script to run the entire pipeline: Datasets -> Weights -> Tests -> Start Server

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

Write-Host "════════════ 1. Environment Setup ════════════" -ForegroundColor Cyan
if (Test-Path ".venv") {
    Write-Host "Activating existing virtual environment..." -ForegroundColor Green
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    python -m venv .venv
    & .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    pip install roboflow kaggle gdown albumentations tqdm ultralytics tensorflow torch torchvision Pillow -q
}

Write-Host "════════════ 2. Datasets Acquisition ════════════" -ForegroundColor Cyan
Write-Host "Downloading weapon dataset..." -ForegroundColor Yellow
python scripts/download_weapon_dataset.py

Write-Host "Downloading violence dataset..." -ForegroundColor Yellow
python scripts/download_violence_dataset.py

Write-Host "Extracting violence frames..." -ForegroundColor Yellow
python scripts/extract_violence_frames.py

Write-Host "Downloading police/danger dataset..." -ForegroundColor Yellow
python scripts/download_police_dataset.py

Write-Host "Downloading accident dataset..." -ForegroundColor Yellow
python scripts/download_accident_dataset.py

Write-Host "════════════ 3. Model Training ════════════" -ForegroundColor Cyan
Write-Host "Training YOLOv8 weapon model..." -ForegroundColor Yellow
if (-not (Test-Path "models/weapon/best.pt")) {
    python scripts/train_weapon.py --epochs 80 --imgsz 640
} else {
    Write-Host "Skipping weapon training (models/weapon/best.pt already exists)." -ForegroundColor Green
}

Write-Host "Training MobileNetV2-LSTM violence model..." -ForegroundColor Yellow
if (-not (Test-Path "models/violence/final_model.h5")) {
    python scripts/train_violence.py --epochs 30
} else {
    Write-Host "Skipping violence training (models/violence/final_model.h5 already exists)." -ForegroundColor Green
}

Write-Host "Training police classifier Dense head..." -ForegroundColor Yellow
if (-not (Test-Path "models/police/police_or_danger.h5")) {
    if (Test-Path "datasets/police/DangePolice_coco/annotations/instances_train2017_normalized.json") {
        python src/police_detection/police_training_pipeline.py `
          --train-annotations datasets/police/DangePolice_coco/annotations/instances_train2017_normalized.json `
          --val-annotations datasets/police/DangePolice_coco/annotations/instances_val2017_normalized.json `
          --train-images datasets/police/DangePolice_coco/train2017 `
          --val-images datasets/police/DangePolice_coco/val2017 `
          --output-model models/police/police_or_danger.h5 `
          --output-metadata models/police/police_or_danger.labels.json `
          --epochs 25 --batch-size 32
    } else {
        Write-Host "Skipping police training (DangePolice COCO annotations not found)." -ForegroundColor Magenta
    }
} else {
    Write-Host "Skipping police training (models/police/police_or_danger.h5 already exists)." -ForegroundColor Green
}

Write-Host "Training accident classifier ResNet50 model..." -ForegroundColor Yellow
if (-not (Test-Path "models/accident/accident_model.pth")) {
    python scripts/train_accident.py --epochs 15
} else {
    Write-Host "Skipping accident training (models/accident/accident_model.pth already exists)." -ForegroundColor Green
}

Write-Host "════════════ 4. Pipeline & Unit Testing ════════════" -ForegroundColor Cyan
python scripts/run_all_tests.py

Write-Host "════════════ 5. Launch FastAPI Server ════════════" -ForegroundColor Cyan
cd src/deployment
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

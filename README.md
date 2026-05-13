# AI-Powered Public Safety Surveillance System

Modern local surveillance dashboard for multi-model public safety detection.

## What It Detects

- Weapon detection with YOLOv8
- Violence detection with MobileNet + LSTM
- Police presence with MobileNetV2 features + classifier head
- Accident detection with ResNet50 PyTorch
- Final scene classification with confidence fusion and priority rules

## Project Structure

```text
AI-Powered-Public-Safety-Surveillance-System/
  backend/                 FastAPI entrypoint wrapper
  frontend/                React + Tailwind dashboard
  models/                  Local model weights
  notebooks/               Training and evaluation notebooks
  src/deployment/          FastAPI app, model loading, smart inference engine
  datasets/                Validation/training manifests and data
  utils/                   Reserved for shared utilities
  evidence/                Saved annotated evidence images
  logs/                    Runtime logs
```

## Smart Inference Pipeline

The deployment pipeline now includes:

- Input validation for images, videos, and live frames
- Adaptive routing by source type
- Model-safe execution with graceful component fallbacks
- Confidence filtering per model
- Weighted fusion across weapon, violence, police, and accident signals
- Priority ordering: weapon > violence > accident > police > normal
- Structured latency, error, and prediction logs

## Backend Endpoints

- `GET /health` checks model loading and runtime status
- `POST /upload` accepts image or video media
- `POST /live-detect` accepts webcam frames from the dashboard
- `GET /logs` returns recent detection events
- `POST /detect` remains available for older clients
- `GET /stream-demo` streams annotated frames from a local camera

## Local Setup

Create and activate a Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the FastAPI backend:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Install and run the React dashboard:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Dashboard Features

- Dark glassmorphism UI
- Upload image/video mode with preview
- Live camera mode with frame-by-frame inference
- Detection ON/OFF toggle
- Confidence threshold slider
- Annotated output panel
- Detection badges and confidence bars
- Timeline and logs dashboard
- Alert popup for high-risk scenes
- Save Evidence option

## Evaluation Notebook

Open:

```text
notebooks/model_evaluation_dashboard.ipynb
```

Create validation CSVs with this format:

```csv
path,label
assets/sample_images/example.jpg,weapon
```

Default expected manifests:

- `datasets/weapon/validation.csv`
- `datasets/violence/validation.csv`
- `datasets/police/validation.csv`
- `datasets/accident/validation.csv`

The notebook reports confusion matrices, precision, recall, F1-score, accuracy, sample predictions, error analysis, and side-by-side model comparison.

## Model Path Overrides

The backend uses local weights from `models/` by default. You can override paths with environment variables:

```bash
set WEAPON_MODEL_PATH=models/weapon/best.pt
set VIOLENCE_MODEL_PATH=models/violence/final_model.h5
set POLICE_MODEL_PATH=models/police/police_or_danger.h5
set ACCIDENT_MODEL_PATH=models/accident/accident_model.pth
```

For fully offline MobileNetV2 police feature extraction, keep the pretrained Torch cache populated or set:

```bash
set POLICE_BACKBONE_WEIGHTS=C:\path\to\mobilenet_v2-b0353104.pth
```

#!/bin/bash
# Startup script for the Rakshak AI Surveillance System FastAPI backend

# Navigate to the repository root directory
cd "$(dirname "$0")/.."

echo "🚀 Starting Rakshak AI Surveillance System Backend..."

# Activate virtual environment if present
if [ -d ".venv" ]; then
    echo "✔ Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "✔ Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "⚠ No local virtual environment found. Running with global python interpreter."
fi

# Configure python path to recognize deployment module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/src/deployment:$(pwd)/backend

# Execute Git LFS pointer validation test before boot
echo "✔ Running model weights validation..."
python -c "
import os
import sys
from pathlib import Path

# Add directories to search path
sys.path.insert(0, 'src/deployment')

def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open('rb') as file:
            return file.read(64).startswith(b'version https://git-lfs')
    except OSError:
        return False

# Model paths check
models = [
    Path('models/weapon/best.pt'),
    Path('models/violence/final_model.h5'),
    Path('models/police/police_or_danger.h5'),
    Path('models/accident/accident_model.pth')
]

for p in models:
    if p.exists() and _is_git_lfs_pointer(p):
        print(f'\n[FATAL ERROR] Model weights file is a Git LFS placeholder pointer: {p}')
        print('Please install Git LFS (https://git-lfs.github.com/) and pull the real binaries: git lfs pull\n')
        sys.exit(1)
"
if [ $? -ne 0 ]; then
    echo "❌ Startup validation failed. Server boot aborted."
    exit 1
fi

echo "✔ Verification successful. Launching Uvicorn server..."
# Boot server using python entrypoint (which uses host 127.0.0.1 and port 8000)
python backend/main.py

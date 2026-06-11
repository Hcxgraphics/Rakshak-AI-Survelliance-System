from __future__ import annotations

import sys
from pathlib import Path

# Insert backend directory on system path so main refers to backend/main.py
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Import the centralized app for backward compatibility
from main import app  # noqa: E402, F401

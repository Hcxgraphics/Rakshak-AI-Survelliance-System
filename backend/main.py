from __future__ import annotations

import sys
from pathlib import Path

DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent / "src" / "deployment"
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from main import app  # noqa: E402,F401

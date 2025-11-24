"""Ensure the project package is importable during tests without installation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

root_path = str(ROOT)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

src_path = str(SRC)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

"""Remove Python cache directories under repo root.

Currently removes:
- `__pycache__`
- `.pytest_cache`
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def remove_pycache(root: Path) -> None:
    patterns = ["__pycache__", ".pytest_cache", ".coverage", ".ruff_cache"]
    for pat in patterns:
        for path in root.rglob(pat):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    remove_pycache(ROOT)

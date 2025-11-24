from __future__ import annotations

import shutil
from pathlib import Path


def _remove_tree(path: Path) -> None:
    if not path.exists():
        print(f"Skipping {path} (not found)")
        return
    if not path.is_dir():
        raise NotADirectoryError(f"Refusing to delete non-directory path: {path}")
    print(f"Removing {path}")
    shutil.rmtree(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    _remove_tree(root / "results")


if __name__ == "__main__":
    main()

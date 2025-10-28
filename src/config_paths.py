"""Helpers to resolve configuration paths with optional run-specific results subdirectories."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sanitize_run_directory(value: str) -> str | None:
    """Return a safe run-directory component (no absolutes, no parent traversals)."""

    candidate = value.strip()
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_absolute():
        raise ValueError("results.run_directory must be a relative path.")
    parts = []
    for part in path.parts:
        if part in ("", ".", ".."):
            continue
        parts.append(part)
    if not parts:
        return None
    return "/".join(parts)


def get_results_run_directory(config: Mapping[str, object] | None) -> str | None:
    """Extract the ``results.run_directory`` value from the root configuration mapping."""

    if not isinstance(config, Mapping):
        return None
    results_cfg = config.get("results")
    if not isinstance(results_cfg, Mapping):
        return None
    raw_value = results_cfg.get("run_directory")
    if raw_value is None:
        return None
    return _sanitize_run_directory(str(raw_value))


def apply_results_run_directory(
    path: Path,
    run_directory: str | None,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Insert the run directory between ``results/`` and the remainder of the path."""

    if not run_directory:
        return path

    root = repo_root or REPO_ROOT
    absolute = path if path.is_absolute() else (root / path)

    try:
        rel = absolute.relative_to(root)
    except ValueError:
        return absolute

    parts = rel.parts
    if not parts:
        return absolute
    if parts[0] != "results":
        return absolute

    new_rel = Path("results") / run_directory / Path(*parts[1:])
    return (root / new_rel).resolve()


def override_config_results_directory(
    config: MutableMapping[str, object],
    run_directory: str | None,
) -> None:
    """Mutate in-place paths under config['results'] to include the run directory."""

    if not run_directory:
        return
    results_cfg = config.get("results")
    if not isinstance(results_cfg, MutableMapping):
        return
    for key, value in list(results_cfg.items()):
        if key == "run_directory":
            continue
        if isinstance(value, str) and value.startswith("results/"):
            new_value = Path("results") / run_directory / Path(value[len("results/") :])
            results_cfg[key] = new_value.as_posix()

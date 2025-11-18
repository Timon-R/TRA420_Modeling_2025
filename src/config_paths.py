"""Helpers to resolve configuration paths with optional run-specific results subdirectories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, MutableMapping

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_ENV_VAR = "TRA420_CONFIG_PATH"
CONFIG_ROOT_KEY = "_config_root"


def get_config_path(default: Path | None = None) -> Path:
    """Return the configuration path, honouring TRA420_CONFIG_PATH when set."""

    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    if default is not None:
        return default.resolve()
    return (REPO_ROOT / "config.yaml").resolve()


def sanitize_run_directory(value: str | None) -> str | None:
    """Return a safe run-directory component (no absolutes, no parent traversals)."""

    if value is None:
        return None
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


def join_run_directory(*parts: str | None) -> str | None:
    """Combine multiple run-directory components into a single sanitised path."""

    merged: list[str] = []
    for part in parts:
        sanitized = sanitize_run_directory(part) if part is not None else None
        if sanitized:
            merged.append(sanitized)
    if not merged:
        return None
    return "/".join(merged)


def set_config_root(config: MutableMapping[str, object], root: Path) -> None:
    """Annotate a config mapping with its filesystem root for relative paths."""

    if not isinstance(config, MutableMapping):
        return
    config[CONFIG_ROOT_KEY] = str(root.resolve())


def get_config_root(config: Mapping[str, object], fallback: Path | None = None) -> Path:
    """Return the base directory that relative paths should resolve against."""

    if isinstance(config, Mapping):
        value = config.get(CONFIG_ROOT_KEY)
        if isinstance(value, str):
            try:
                return Path(value).expanduser().resolve()
            except OSError:
                pass
    return (fallback or REPO_ROOT).resolve()


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
    return sanitize_run_directory(str(raw_value))


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

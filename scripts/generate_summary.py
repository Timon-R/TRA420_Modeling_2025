"""Generate summary tables and plots across module outputs based on config."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config_paths import get_config_path  # noqa: E402
from results_summary import (  # noqa: E402
    build_summary,
    write_plots,
    write_summary_json,
    write_summary_text,
)


def _load_config() -> dict:
    path = get_config_path(ROOT / "config.yaml")
    if not path.exists():
        raise FileNotFoundError("config.yaml not found at repository root.")
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("generate_summary")

    try:
        config = _load_config()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    if "results" not in config or "summary" not in config.get("results", {}):
        logger.info("results.summary not configured; skipping summary generation.")
        return

    try:
        settings, methods, metrics_map = build_summary(ROOT, config)
    except ValueError as exc:
        logger.error("Unable to build summary: %s", exc)
        return

    summary_txt = write_summary_text(settings, methods, metrics_map)
    logger.info("Summary written to %s", summary_txt.relative_to(ROOT))

    summary_json = write_summary_json(settings, methods, metrics_map)
    logger.info("JSON summary written to %s", summary_json.relative_to(ROOT))

    write_plots(settings, methods, metrics_map)
    if settings.include_plots:
        logger.info(
            "Plots written under %s",
            (settings.output_directory / "plots").relative_to(ROOT),
        )


if __name__ == "__main__":
    main()

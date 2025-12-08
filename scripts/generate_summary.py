"""Generate summary tables and plots across module outputs based on config."""

from __future__ import annotations

import argparse
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
    write_socioeconomic_tables,
    write_summary_csv,
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

    parser = argparse.ArgumentParser(description="Generate summary tables and plots.")
    parser.add_argument(
        "--run-directory",
        "--run-subdir",
        dest="run_directory",
        default=None,
        help="Override results.run_directory to target a specific run (e.g., 'global').",
    )
    args = parser.parse_args()

    try:
        config = _load_config()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    if args.run_directory:
        results_cfg = config.setdefault("results", {})
        results_cfg["run_directory"] = args.run_directory

    if "results" not in config or "summary" not in config.get("results", {}):
        logger.info("results.summary not configured; skipping summary generation.")
        return

    try:
        settings, methods, metrics_map = build_summary(ROOT, config)
    except ValueError as exc:
        logger.error("Unable to build summary: %s", exc)
        return

    summary_csv = write_summary_csv(settings, methods, metrics_map)
    logger.info("Summary CSV written to %s", summary_csv.relative_to(ROOT))

    socio_outputs = write_socioeconomic_tables(settings)
    if socio_outputs:
        for path in socio_outputs:
            logger.info("Socioeconomics written to %s", path.relative_to(ROOT))

    write_plots(settings, methods, metrics_map)
    if settings.include_plots:
        logger.info(
            "Plots written under %s",
            (settings.output_directory / "plots").relative_to(ROOT),
        )


if __name__ == "__main__":
    main()

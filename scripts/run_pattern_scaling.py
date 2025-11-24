"""Apply pattern scaling factors to climate module results."""

from __future__ import annotations

import logging
from pathlib import Path

from config_paths import apply_results_run_directory, get_results_run_directory
from pattern_scaling import (
    DEFAULT_ROOT,
    get_scaling_factors,
    load_config,
    scale_results,
)

LOGGER = logging.getLogger("pattern_scaling.run")


def main() -> None:
    LOGGER.info("Loading pattern-scaling configuration")
    config = load_config()
    ps_cfg = config.get("pattern_scaling", {})
    run_directory = get_results_run_directory(config)
    countries = ps_cfg.get("countries", [])
    weighting = ps_cfg.get("scaling_weighting", "area")
    scaling_factors = get_scaling_factors(config)
    LOGGER.info("Scaling %d countries using weighting '%s'", len(countries), weighting)
    scale_results(config, scaling_factors)
    output_dir_cfg = ps_cfg.get("output_directory", "results/climate_scaled")
    output_dir_path = Path(output_dir_cfg)
    if not output_dir_path.is_absolute():
        output_dir_path = (DEFAULT_ROOT / output_dir_path).resolve()
    output_dir = apply_results_run_directory(
        output_dir_path,
        run_directory,
        repo_root=DEFAULT_ROOT,
    )
    LOGGER.info("Pattern-scaled outputs written to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    main()

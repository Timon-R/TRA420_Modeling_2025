"""Run the calc_emissions workflow for a specific configuration or country.

The script now requires either ``--country`` (matching a ``config_<name>.yaml``
file under ``data/calc_emissions/countries``) or an explicit ``--config`` path.
All logging is routed through the standard logging module so output integrates
with larger pipelines.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from calc_emissions import run_from_config  # noqa: E402
from config_paths import get_results_run_directory  # noqa: E402

LOGGER = logging.getLogger("calc_emissions.run")


def _load_country_settings() -> tuple[Path, str, dict | None, str | None]:
    config_path = ROOT / "config.yaml"
    config = {}
    if config_path.exists():
        with config_path.open() as handle:
            config = yaml.safe_load(handle) or {}
    module_cfg = config.get("calc_emissions", {})
    countries_cfg = module_cfg.get("countries", {})
    directory = ROOT / countries_cfg.get("directory", "data/calc_emissions/countries")
    pattern = countries_cfg.get("pattern", "config_{name}.yaml")
    run_directory = get_results_run_directory(config)
    return directory, pattern, config.get("time_horizon"), run_directory


def _available_countries(directory: Path, pattern: str) -> str:
    glob_pattern = pattern.replace("{name}", "*")
    names = sorted(p.stem.replace("config_", "") for p in directory.glob(glob_pattern))
    return ", ".join(names) or "none"


def _resolve_config_path(args: argparse.Namespace, directory: Path, pattern: str) -> Path:
    if args.config:
        return Path(args.config)

    if args.country:
        country_key = args.country.replace(" ", "_")
        candidate = pattern.replace("{name}", country_key)
        path = directory / candidate
        if not path.exists():
            available = _available_countries(directory, pattern)
            raise FileNotFoundError(
                f"Country config '{country_key}' not found in {directory}. "
                f"Available: {available}."
            )
        return path

    raise ValueError("Either --country or --config must be supplied")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate electricity emission deltas from calc_emissions configs"
    )
    parser.add_argument("--config", help="Explicit path to a calc_emissions config file")
    parser.add_argument(
        "--country",
        help=(
            "Country name matching a config_<name>.yaml entry under calc_emissions.countries "
            "(spaces replaced with underscores)."
        ),
    )
    args = parser.parse_args()

    directory, pattern, global_horizon, run_directory = _load_country_settings()

    if not args.config and not args.country:
        available = _available_countries(directory, pattern)
        parser.error(
            "Specify --country or --config. Known countries: "
            f"{available} (configured under calc_emissions.countries)."
        )

    config_path = _resolve_config_path(args, directory, pattern)
    LOGGER.info("Running calc_emissions for %s", config_path)

    results = run_from_config(
        config_path,
        default_years=global_horizon,
        results_run_directory=run_directory,
    )
    for name, result in results.items():
        if name == "baseline":
            continue
        LOGGER.info(
            "Scenario '%s' ΔCO₂ (Mt/year):\n%s",
            name,
            result.delta_mtco2.to_frame(name="delta_mtco2").to_string(),
        )
        totals = {
            pollutant: result.total_emissions_mt[pollutant]
            for pollutant in sorted(result.total_emissions_mt)
        }
        summary_years = [y for y in [2030, 2050, 2100] if y in result.delta_mtco2.index]
        if summary_years:
            summary_df = pd.DataFrame({k: v.loc[summary_years] for k, v in totals.items()})
            LOGGER.info(
                "Scenario '%s' totals (Mt) for key years:\n%s", name, summary_df.to_string()
            )


if __name__ == "__main__":
    main()

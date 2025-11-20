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

from calc_emissions import BASE_DEMAND_CASE, EmissionScenarioResult, run_from_config  # noqa: E402
from calc_emissions.writers import write_per_country_results  # noqa: E402
from config_paths import apply_results_run_directory, get_results_run_directory  # noqa: E402

LOGGER = logging.getLogger("calc_emissions.run")


def _load_country_settings() -> (
    tuple[
        Path,
        str,
        dict | None,
        str | None,
        list[str],
        list[str],
        str,
        Path,
    ]
):
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
    mix_cases = countries_cfg.get("mix_scenarios", []) or []
    demand_cases = countries_cfg.get("demand_scenarios", []) or []
    baseline_case = (
        str(countries_cfg.get("baseline_demand_case", BASE_DEMAND_CASE)).strip() or BASE_DEMAND_CASE
    )
    per_country_root_cfg = Path(countries_cfg.get("resources_root", "results/emissions"))
    if per_country_root_cfg.is_absolute():
        per_country_root = per_country_root_cfg
    else:
        per_country_root = (ROOT / per_country_root_cfg).resolve()
    per_country_root = apply_results_run_directory(
        per_country_root,
        run_directory,
        repo_root=ROOT,
    )
    per_country_root = _ensure_run_directory(per_country_root, run_directory)
    return (
        directory,
        pattern,
        config.get("time_horizon"),
        run_directory,
        mix_cases,
        demand_cases,
        baseline_case,
        per_country_root,
    )


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

    (
        directory,
        pattern,
        global_horizon,
        run_directory,
        configured_mix_cases,
        configured_demand_cases,
        baseline_case,
        per_country_root,
    ) = _load_country_settings()

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
        allowed_demand_cases=configured_demand_cases or None,
        allowed_mix_cases=configured_mix_cases or None,
    )

    country_label = (
        args.country.replace(" ", "_")
        if args.country
        else Path(config_path).stem.replace("config_", "")
    )
    write_per_country_results({country_label: results}, per_country_root, baseline_case)
    LOGGER.info("Per-country results written under %s", per_country_root)

    mixes: dict[str, dict[str, EmissionScenarioResult]] = {}
    for res in results.values():
        mixes.setdefault(res.mix_case, {})[res.demand_case] = res

    for mix, demand_map in mixes.items():
        LOGGER.info(
            "Mix '%s' computed; demand cases: %s", mix, ", ".join(sorted(demand_map.keys()))
        )
        for demand_name, result in sorted(demand_map.items()):
            if demand_name == baseline_case:
                continue
            LOGGER.info(
                "Mix '%s' — demand '%s' ΔCO₂ (Mt/year):\n%s",
                mix,
                demand_name,
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
                    "Mix '%s' demand '%s' totals (Mt) for key years:\n%s",
                    mix,
                    demand_name,
                    summary_df.to_string(),
                )


if __name__ == "__main__":
    main()


def _ensure_run_directory(path: Path, run_directory: str | None) -> Path:
    if not run_directory:
        return path
    run_directory = run_directory.strip().strip("/")
    if not run_directory:
        return path
    parts = path.resolve().parts
    try:
        idx = parts.index("results")
    except ValueError:
        return path
    if idx + 1 < len(parts) and parts[idx + 1] == run_directory:
        return path
    return Path(*parts[: idx + 1]) / run_directory / Path(*parts[idx + 1 :])

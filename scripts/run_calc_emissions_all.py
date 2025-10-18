"""Run emission calculations for all countries and aggregate deltas.

This script discovers all country config files in ``country_data/config_*.yaml``,
executes the emissions calculation for each country, and then sums the per-year
emission deltas (relative to each country's baseline) across all countries.

Outputs are written under ``resources/All_countries/<scenario>/`` as CSV files
named like the country-specific outputs: ``co2.csv``, ``sox.csv``, ``nox.csv``,
``pm25.csv``, and ``gwp100.csv`` where available. Each CSV has two columns:
``year`` and ``delta`` (Mt/year), representing the sum of deltas across all
included countries for that scenario and pollutant.

Usage:
  - Default: aggregate all countries found in country_data/
  - Optional: pass --countries to restrict the set (names should match config
    filenames, e.g., "Albania", "Bosnia-Herzegovina", "North_Macedonia").
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calc_emissions import run_from_config  # noqa: E402


POLLUTANTS = ["co2", "sox", "nox", "pm25", "gwp100"]


def list_country_configs(countries_filter: List[str] | None = None) -> Dict[str, Path]:
    base = Path(__file__).resolve().parents[1]
    cfg_dir = base / "country_data"
    configs: Dict[str, Path] = {}
    for path in sorted(cfg_dir.glob("config_*.yaml")):
        # Skip backup or nested configs if any
        if "Backup_config" in str(path):
            continue
        name = path.stem.replace("config_", "")
        if countries_filter and name not in countries_filter:
            continue
        configs[name] = path
    return configs


def aggregate_deltas(country_results_dirs: Dict[str, Path]) -> Dict[str, Dict[str, pd.Series]]:
    """Read per-country delta CSVs and sum across countries.

    Returns a nested mapping: scenario -> pollutant -> Series(year -> delta_sum)
    """
    aggregated: Dict[str, Dict[str, pd.Series]] = {}

    # Collect all scenario names by inspecting directories under each country's resources
    all_scenarios: set[str] = set()
    for country, outdir in country_results_dirs.items():
        if not outdir.exists():
            continue
        for scen_dir in outdir.iterdir():
            if scen_dir.is_dir():
                all_scenarios.add(scen_dir.name)

    # For each scenario and pollutant, sum the deltas across countries
    for scenario in sorted(all_scenarios):
        aggregated.setdefault(scenario, {})
        for pollutant in POLLUTANTS:
            sum_series: pd.Series | None = None
            for country, outdir in country_results_dirs.items():
                scen_dir = outdir / scenario
                csv_path = scen_dir / f"{pollutant}.csv"
                if not csv_path.exists():
                    continue
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                if "year" not in df.columns or "delta" not in df.columns:
                    continue
                series = pd.Series(df["delta"].values, index=df["year"].astype(int))
                sum_series = series if sum_series is None else sum_series.add(series, fill_value=0.0)
            if sum_series is not None:
                # Ensure integer year index sorted
                sum_series.index = sum_series.index.astype(int)
                sum_series = sum_series.sort_index()
                aggregated[scenario][pollutant] = sum_series

    return aggregated


def write_aggregated_outputs(aggregated: Dict[str, Dict[str, pd.Series]], base_output: Path) -> None:
    for scenario, pol_map in aggregated.items():
        scen_dir = base_output / scenario
        scen_dir.mkdir(parents=True, exist_ok=True)
        for pollutant, series in pol_map.items():
            df = pd.DataFrame({"year": series.index.astype(int), "delta": series.values})
            df.to_csv(scen_dir / f"{pollutant}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run emissions for all countries and aggregate deltas.")
    parser.add_argument(
        "--countries",
        nargs="*",
        help=(
            "Optional list of country names to include (matching config filenames without prefix). "
            "Examples: Albania Bosnia-Herzegovina North_Macedonia Serbia Kosovo Montenegro"
        ),
    )
    parser.add_argument(
        "--output",
        default="resources/All_countries",
        help="Aggregate output directory (default: resources/All_countries)",
    )
    args = parser.parse_args()

    # Prepare list of configs
    countries_filter = [c.replace(" ", "_") for c in args.countries] if args.countries else None
    configs = list_country_configs(countries_filter)
    if not configs:
        print("No country config files found matching the selection.")
        sys.exit(1)

    # Run each country's calculation and collect their output directories
    country_outdirs: Dict[str, Path] = {}
    for name, cfg_path in configs.items():
        # Execute calculations to ensure outputs are fresh
        results = run_from_config(cfg_path)
        # Determine output directory from config by reading the same key used by run_from_config
        # Re-open config minimally to get output path
        import yaml  # local import to avoid global dependency at import time

        with Path(cfg_path).open() as handle:
            cfg = yaml.safe_load(handle) or {}
        mod_cfg = cfg.get("calc_emissions", {})
        outdir = Path(mod_cfg.get("output_directory", "resources"))
        country_outdirs[name] = outdir

    # Aggregate deltas from on-disk scenario CSVs for all pollutants
    aggregated = aggregate_deltas(country_outdirs)

    # Write aggregated outputs to a consolidated folder
    base_output = Path(args.output)
    base_output.mkdir(parents=True, exist_ok=True)
    write_aggregated_outputs(aggregated, base_output)

    # Brief console summary
    print(f"Aggregated results written to: {base_output}")
    if aggregated:
        example_scen = next(iter(sorted(aggregated)))
        if "co2" in aggregated[example_scen]:
            series = aggregated[example_scen]["co2"]
            years_to_show = [y for y in [2030, 2050, 2100] if y in series.index]
            if years_to_show:
                print("\nExample (scenario: {}, pollutant: co2) — selected years:".format(example_scen))
                print(series.loc[years_to_show].to_frame(name="delta_mt").to_string())


if __name__ == "__main__":
    main()

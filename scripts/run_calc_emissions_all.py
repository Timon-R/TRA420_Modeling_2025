"""Aggregate electricity-emission scenarios across all configured countries.

The helper discovers per-country configuration files referenced under
``calc_emissions.countries`` in ``config.yaml`` (defaulting to
``data/calc_emissions/countries/config_*.yaml``). For each country the standard
calc-emissions workflow is executed, producing per-scenario deltas. The script
then sums baseline and scenario totals across countries, writes aggregated
results (resources + final results), and optionally mirrors the aggregated
scenarios at the root of ``resources/`` so the climate module can consume them
without extra configuration.

When imported, the ``run_all_countries`` function can be reused by higher-level
pipelines. Running the module as a script retains the previous CLI behaviour
with additional ``--results-output`` support.
"""

from __future__ import annotations

import argparse
import logging

# Ensure src/ is importable before importing the calculator
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calc_emissions import EmissionScenarioResult, run_from_config  # noqa: E402
from config_paths import apply_results_run_directory, get_results_run_directory  # noqa: E402
from calc_emissions.writers import write_per_country_results  # noqa: E402

LOGGER = logging.getLogger("calc_emissions.aggregate")

POLLUTANTS = ["co2", "sox", "nox", "pm25", "gwp100"]


def _load_country_settings() -> (
    tuple[
        Path,
        str,
        Path,
        Path | None,
        Path,
        list[str],
        dict[str, int] | None,
        str | None,
    ]
):
    config_path = ROOT / "config.yaml"
    config = {}
    if config_path.exists():
        with config_path.open() as handle:
            config = yaml.safe_load(handle) or {}
    run_directory = get_results_run_directory(config)
    module_cfg = config.get("calc_emissions", {})
    countries_cfg = module_cfg.get("countries", {})
    directory = ROOT / countries_cfg.get("directory", "data/calc_emissions/countries")
    pattern = countries_cfg.get("pattern", "config_{name}.yaml")
    aggregate_output = ROOT / countries_cfg.get(
        "aggregate_output_directory", "resources/All_countries"
    )
    aggregate_results = countries_cfg.get("aggregate_results_directory")
    results_path = None
    if aggregate_results is not None:
        results_candidate = ROOT / aggregate_results
        results_path = apply_results_run_directory(
            results_candidate,
            run_directory,
            repo_root=ROOT,
        )
    resources_root = ROOT / countries_cfg.get("resources_root", "resources")
    scenarios = countries_cfg.get("scenarios", [])
    global_horizon = config.get("time_horizon")
    return (
        directory,
        pattern,
        aggregate_output,
        results_path,
        resources_root,
        scenarios,
        global_horizon,
        run_directory,
    )


def list_country_configs(
    directory: Path, pattern: str, countries_filter: Iterable[str] | None = None
) -> Dict[str, Path]:
    configs: Dict[str, Path] = {}
    glob_pattern = pattern.replace("{name}", "*")
    for path in sorted(directory.glob(glob_pattern)):
        if not path.is_file():
            continue
        if "Backup" in path.name:
            continue
        name = path.stem.replace("config_", "")
        if countries_filter and name not in countries_filter:
            continue
        configs[name] = path
    return configs


def _clone_result(result: EmissionScenarioResult) -> EmissionScenarioResult:
    return EmissionScenarioResult(
        name=result.name,
        years=list(result.years),
        demand_twh=result.demand_twh.copy(),
        generation_twh=result.generation_twh.copy(),
        technology_emissions_mt={k: df.copy() for k, df in result.technology_emissions_mt.items()},
        total_emissions_mt={k: series.copy() for k, series in result.total_emissions_mt.items()},
        delta_mtco2=result.delta_mtco2.copy(),
    )


def _add_series(target: pd.Series | None, addition: pd.Series) -> pd.Series:
    if target is None:
        return addition.copy()
    idx = target.index.union(addition.index)
    target = target.reindex(idx, fill_value=0.0)
    addition = addition.reindex(idx, fill_value=0.0)
    return target + addition


def _add_frame(target: pd.DataFrame | None, addition: pd.DataFrame) -> pd.DataFrame:
    if target is None:
        return addition.copy()
    index = target.index.union(addition.index)
    columns = target.columns.union(addition.columns)
    target = target.reindex(index=index, columns=columns, fill_value=0.0)
    addition = addition.reindex(index=index, columns=columns, fill_value=0.0)
    return target + addition


def _accumulate_result(target: EmissionScenarioResult, addition: EmissionScenarioResult) -> None:
    target.demand_twh = _add_series(target.demand_twh, addition.demand_twh)
    target.generation_twh = _add_frame(target.generation_twh, addition.generation_twh)

    for pollutant, df in addition.technology_emissions_mt.items():
        target_df = target.technology_emissions_mt.get(pollutant)
        target.technology_emissions_mt[pollutant] = _add_frame(target_df, df)
    for pollutant, series in addition.total_emissions_mt.items():
        target_series = target.total_emissions_mt.get(pollutant)
        target.total_emissions_mt[pollutant] = _add_series(target_series, series)

    target.delta_mtco2 = _add_series(target.delta_mtco2, addition.delta_mtco2)


def _build_aggregated_results(
    per_country_results: List[dict[str, EmissionScenarioResult]],
) -> dict[str, EmissionScenarioResult]:
    # New aggregation: input per_country_results is a list where each element is a
    # mapping of scenario keys of the form '<mix>/<demand>' -> EmissionScenarioResult.
    # We aggregate per '<mix>' and compute deltas for each '<mix>/<demand>' vs
    # the '<mix>/baseline' aggregated totals.
    baseline_aggs: Dict[str, EmissionScenarioResult] = {}
    scenario_aggs: Dict[tuple[str, str], EmissionScenarioResult] = {}

    for result_map in per_country_results:
        for scenario_name, scenario_res in result_map.items():
            if "/" not in scenario_name:
                # Skip unexpected keys
                continue
            mix, demand = scenario_name.split("/", 1)
            if demand == "baseline":
                agg = baseline_aggs.get(mix)
                if agg is None:
                    baseline_aggs[mix] = _clone_result(scenario_res)
                else:
                    _accumulate_result(agg, scenario_res)
            else:
                key = (mix, demand)
                agg = scenario_aggs.get(key)
                if agg is None:
                    scenario_aggs[key] = _clone_result(scenario_res)
                else:
                    _accumulate_result(agg, scenario_res)

    if not baseline_aggs:
        raise ValueError("No baseline results aggregated; ensure country configs include a 'baseline' demand scenario.")

    # Recompute deltas for each aggregated scenario against the corresponding mix baseline
    aggregated: Dict[str, EmissionScenarioResult] = {}
    for (mix, demand), agg_res in scenario_aggs.items():
        baseline_res = baseline_aggs.get(mix)
        if baseline_res is None:
            # No baseline for this mix; keep totals as-is and zero deltas
            agg_res.delta_mtco2 = agg_res.delta_mtco2 * 0.0
        else:
            baseline_co2 = baseline_res.total_emissions_mt.get("co2")
            totals = agg_res.total_emissions_mt.get("co2")
            if totals is not None and baseline_co2 is not None:
                totals = totals.reindex(baseline_co2.index.union(totals.index), fill_value=0.0)
                baseline_aligned = baseline_co2.reindex(totals.index, fill_value=0.0)
                agg_res.delta_mtco2 = totals - baseline_aligned

        aggregated[f"{mix}/{demand}"] = agg_res

    # Include aggregated baselines for each mix (with zero deltas)
    for mix, base in baseline_aggs.items():
        base.delta_mtco2 = base.delta_mtco2 * 0.0
        aggregated[f"{mix}/baseline"] = base

    return aggregated


def _write_outputs(
    aggregated: dict[str, EmissionScenarioResult],
    resources_dir: Path,
    results_dir: Path | None,
    resources_root: Path | None,
) -> None:
    # Aggregated keys are '<mix>/<demand>'. For each key, find the corresponding
    # mix baseline at '<mix>/baseline' to compute deltas. CSVs contain columns
    # 'year', 'absolute', and 'delta'. Files are written to dest/<mix>/<demand>/<pollutant>.csv
    for scenario_name, scenario_res in aggregated.items():
        destinations = [resources_dir]
        if results_dir is not None:
            destinations.append(results_dir)
        if resources_root is not None:
            destinations.append(resources_root)

        mix = scenario_name.split("/", 1)[0] if "/" in scenario_name else None
        baseline_name = f"{mix}/baseline" if mix is not None else None

        for dest in destinations:
            scenario_dir = dest / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            for pollutant, totals in scenario_res.total_emissions_mt.items():
                baseline_totals = None
                if baseline_name is not None:
                    baseline_res = aggregated.get(baseline_name)
                    if baseline_res is not None:
                        baseline_totals = baseline_res.total_emissions_mt.get(pollutant)

                if baseline_totals is None:
                    delta = totals
                else:
                    delta = totals - baseline_totals.reindex(totals.index, fill_value=0.0)

                df = pd.DataFrame({
                    "year": totals.index.astype(int),
                    "absolute": totals.values,
                    "delta": delta.values,
                })
                (scenario_dir / f"{pollutant}.csv").parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(scenario_dir / f"{pollutant}.csv", index=False)


# The per-country writer implementation has been moved to
# `src/calc_emissions/writers.py` and is imported as
# `write_per_country_results` at module top-level. We no longer expose the
# underscored compatibility wrapper here.


def run_all_countries(
    countries: Iterable[str] | None = None,
    output: Path | None = None,
    results_output: Path | None = None,
    mirror_to_root: bool = True,
    scenarios: Iterable[str] | None = None,
) -> dict[str, EmissionScenarioResult]:
    (
        directory,
        pattern,
        default_output,
        default_results,
        resources_root,
        configured_scenarios,
        global_horizon,
        run_directory,
    ) = _load_country_settings()

    countries_filter = [c.replace(" ", "_") for c in countries] if countries else None
    configs = list_country_configs(directory, pattern, countries_filter)
    if not configs:
        raise FileNotFoundError("No country config files found for aggregation.")

    if scenarios is None or not list(scenarios):
        scenario_filter = [s for s in configured_scenarios if s]
    else:
        scenario_filter = [s for s in scenarios if s]

    per_country_results: List[dict[str, EmissionScenarioResult]] = []
    per_country_map: dict[str, dict[str, EmissionScenarioResult]] = {}
    LOGGER.info("Running calc_emissions for %d countries", len(configs))
    for country, cfg_path in configs.items():
        LOGGER.info("  • %s", country)
        results = run_from_config(
            cfg_path,
            default_years=global_horizon,
            results_run_directory=run_directory,
        )

        # `results` now contains keys of the form '<mix>/<demand>'. Determine
        # which mix scenarios to include (either provided by scenario_filter or
        # all mixes present in the results) and collect all matching keys.
        available_mixes = {name.split("/", 1)[0] for name in results.keys() if "/" in name}
        if scenario_filter:
            selected_mixes = [s for s in scenario_filter if s]
            missing = set(selected_mixes) - available_mixes
            if missing:
                raise KeyError(f"Country '{country}' missing mix scenarios: {sorted(missing)}")
        else:
            selected_mixes = sorted(available_mixes)

        filtered_results: dict[str, EmissionScenarioResult] = {}
        for name, res in results.items():
            if "/" not in name:
                continue
            mix, demand = name.split("/", 1)
            if mix in selected_mixes:
                filtered_results[name] = res
        per_country_results.append(filtered_results)
        per_country_map[country] = filtered_results

        with cfg_path.open() as handle:
            cfg = yaml.safe_load(handle) or {}
        mod_cfg = cfg.get("calc_emissions", {})
        outdir = Path(mod_cfg.get("output_directory", "resources"))
        if not outdir.is_absolute():
            outdir = (cfg_path.parent / outdir).resolve()
        if global_horizon and isinstance(global_horizon, dict):
            years_cfg = mod_cfg.get("years")
            if years_cfg:
                mismatch = any(
                    int(years_cfg.get(key, global_horizon[key])) != int(global_horizon[key])
                    for key in ("start", "end", "step")
                    if key in global_horizon
                )
                if mismatch:
                    LOGGER.warning(
                        "Country '%s' years %s differ from global time_horizon %s",
                        country,
                        years_cfg,
                        global_horizon,
                    )
    aggregated_results = _build_aggregated_results(per_country_results)

    resources_dest = output or default_output
    resources_dest.mkdir(parents=True, exist_ok=True)

    results_dest = results_output if results_output is not None else default_results
    if results_dest is not None:
        results_dest.mkdir(parents=True, exist_ok=True)

    mirror_dest = resources_root if mirror_to_root else None
    _write_outputs(aggregated_results, resources_dest, results_dest, mirror_dest)

    # Write per-country CSVs to resources/<Country>/<scenario> so the aggregator
    # can fully replace the per-country runner.
    if per_country_map:
        # resources_root is a Path returned by _load_country_settings
        write_per_country_results(per_country_map, resources_root)

    LOGGER.info("Aggregated deltas written to %s", resources_dest)
    if results_dest is not None:
        LOGGER.info("Aggregated results copied to %s", results_dest)

    return aggregated_results


def _parse_args() -> argparse.Namespace:
    (
        _directory,
        _pattern,
        default_output,
        default_results,
        _resources_root,
        configured_scenarios,
        _global_horizon,
        _run_directory,
    ) = _load_country_settings()
    parser = argparse.ArgumentParser(
        description="Run emissions for all countries and aggregate deltas."
    )
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
        default=str(
            default_output.relative_to(ROOT)
            if default_output.is_relative_to(ROOT)
            else default_output
        ),
        help=(
            "Aggregate output directory "
            "(defaults to calc_emissions.countries.aggregate_output_directory)"
        ),
    )
    parser.add_argument(
        "--results-output",
        default=(
            None
            if default_results is None
            else str(
                default_results.relative_to(ROOT)
                if default_results.is_relative_to(ROOT)
                else default_results
            )
        ),
        help=(
            "Optional directory for final aggregated results "
            "(defaults to calc_emissions.countries.aggregate_results_directory)"
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        help=(
            "Scenario names to aggregate (must exist in every country config). "
            f"Defaults to config list: {configured_scenarios or 'all'}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    countries = args.countries if args.countries else None
    output = Path(args.output)
    results_output = Path(args.results_output) if args.results_output else None
    scenario_filter = args.scenarios if args.scenarios else None

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

    aggregated = run_all_countries(
        countries=countries,
        output=output,
        results_output=results_output,
        scenarios=scenario_filter,
    )
    example = next((name for name in sorted(aggregated) if name != "baseline"), None)
    if example:
        co2_delta = aggregated[example].delta_mtco2
        years_to_show = [y for y in [2030, 2050, 2100] if y in co2_delta.index]
        if years_to_show:
            LOGGER.info(
                "Example scenario '%s' CO₂ deltas:\n%s",
                example,
                co2_delta.loc[years_to_show].to_frame(name="delta_mt").to_string(),
            )


if __name__ == "__main__":
    main()

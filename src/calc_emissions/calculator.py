"""Calculate electricity-sector emissions from demand/mix scenarios.

Results include per-pollutant (CO₂, SO₂, NOₓ, PM₂.₅) emissions in Mt/year. The
climate module consumes the CO₂ deltas, while the additional pollutants support
air-quality analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger("calc_emissions")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


POLLUTANTS = {
    "co2": {"column": "co2_mt_per_twh", "to_mt": 1.0, "unit": "Mt"},
    "so2": {"column": "so2_kt_per_twh", "to_mt": 1e-3, "unit": "Mt"},
    "nox": {"column": "nox_kt_per_twh", "to_mt": 1e-3, "unit": "Mt"},
    "pm25": {"column": "pm25_kt_per_twh", "to_mt": 1e-3, "unit": "Mt"},
}


@dataclass
class EmissionScenarioResult:
    """Result container for an emission scenario."""

    name: str
    years: list[int]
    demand_twh: pd.Series
    generation_twh: pd.DataFrame
    technology_emissions_mt: dict[str, pd.DataFrame]
    total_emissions_mt: dict[str, pd.Series]
    delta_mtco2: pd.Series


def run_from_config(config_path: Path | str = "config.yaml") -> dict[str, EmissionScenarioResult]:
    """Run emission calculations based on ``config.yaml`` and write delta CSVs."""
    config_path = Path(config_path)
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}

    module_cfg = config.get("calc_emissions", {})
    if not module_cfg:
        raise ValueError("'calc_emissions' section missing from config.yaml")

    years = _generate_years(module_cfg.get("years", {}))
    emission_factors = _load_emission_factors(
        Path(module_cfg.get("emission_factors_file", "data/emission_factors.csv"))
    )

    demand_scenarios = module_cfg.get("demand_scenarios", {})
    mix_scenarios = module_cfg.get("mix_scenarios", {})

    baseline_cfg = module_cfg.get("baseline", {})
    baseline_demand = _resolve_demand_series(
        years, demand_scenarios, baseline_cfg.get("demand_scenario"), baseline_cfg.get("demand_custom")
    )
    baseline_mix = _resolve_mix_shares(
        years,
        mix_scenarios,
        emission_factors.index,
        baseline_cfg.get("mix_scenario"),
        baseline_cfg.get("mix_custom"),
    )

    baseline = calculate_emissions(
        name="baseline",
        demand_series=baseline_demand,
        mix_shares=baseline_mix,
        emission_factors=emission_factors,
    )

    output_dir = Path(module_cfg.get("output_directory", "resources"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(module_cfg.get("results_directory", "results/emissions"))
    results_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, EmissionScenarioResult] = {}
    for scenario_cfg in module_cfg.get("scenarios", []):
        name = scenario_cfg.get("name")
        if not name:
            raise ValueError("Each calc_emissions scenario requires a 'name'.")
        LOGGER.info("Calculating emissions for scenario '%s'", name)

        demand_series = _resolve_demand_series(
            years,
            demand_scenarios,
            scenario_cfg.get("demand_scenario"),
            scenario_cfg.get("demand_custom"),
        )
        mix_shares = _resolve_mix_shares(
            years,
            mix_scenarios,
            emission_factors.index,
            scenario_cfg.get("mix_scenario"),
            scenario_cfg.get("mix_custom"),
        )

        scenario_result = calculate_emissions(
            name=name,
            demand_series=demand_series,
            mix_shares=mix_shares,
            emission_factors=emission_factors,
        )

        scenario_dir = output_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        results_scenario_dir = results_dir / name
        results_scenario_dir.mkdir(parents=True, exist_ok=True)

        for pollutant, totals in scenario_result.total_emissions_mt.items():
            baseline_totals = baseline.total_emissions_mt.get(pollutant)
            if baseline_totals is None:
                delta = totals.copy()
            else:
                delta = totals - baseline_totals

            if pollutant == "co2":
                scenario_result.delta_mtco2 = delta

            df = pd.DataFrame({"year": years, "delta": delta})
            file_path = scenario_dir / f"{pollutant}.csv"
            df.to_csv(file_path, index=False)

            results_file_path = results_scenario_dir / f"{pollutant}.csv"
            df.to_csv(results_file_path, index=False)

        results[name] = scenario_result

    return results


def calculate_emissions(
    name: str,
    demand_series: pd.Series,
    mix_shares: pd.DataFrame,
    emission_factors: pd.DataFrame,
) -> EmissionScenarioResult:
    """Calculate generation and emissions for a single scenario."""
    if not isinstance(demand_series.index, pd.Index):
        raise TypeError("demand_series must be a pandas Series with year index")

    generation = mix_shares.multiply(demand_series, axis=0)

    technology_emissions: dict[str, pd.DataFrame] = {}
    total_emissions: dict[str, pd.Series] = {}

    for pollutant, meta in POLLUTANTS.items():
        column = meta["column"]
        if column not in emission_factors.columns:
            continue
        factors = emission_factors[column]
        emissions = generation.mul(factors, axis=1)
        emissions_mt = emissions * meta["to_mt"]
        technology_emissions[pollutant] = emissions_mt
        total_emissions[pollutant] = emissions_mt.sum(axis=1)

    if "co2" not in total_emissions:
        raise ValueError("Emission factors must include a 'co2_mt_per_twh' column.")

    co2_series = total_emissions["co2"]

    return EmissionScenarioResult(
        name=name,
        years=list(demand_series.index.astype(int)),
        demand_twh=demand_series,
        generation_twh=generation,
        technology_emissions_mt=technology_emissions,
        total_emissions_mt=total_emissions,
        delta_mtco2=pd.Series(np.zeros_like(co2_series.values), index=co2_series.index, dtype=float),
    )


def _generate_years(cfg: Mapping[str, float | int]) -> list[int]:
    start = int(cfg.get("start", 2025))
    end = int(cfg.get("end", 2100))
    step = int(cfg.get("step", 5))
    if start >= end:
        raise ValueError("'start' year must be less than 'end' year")
    if step <= 0:
        raise ValueError("'step' must be positive")
    years = list(range(start, end + 1, step))
    return years


def _load_emission_factors(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Emission factors file not found: {path}")
    df = pd.read_csv(path, comment="#")
    if "technology" not in df.columns:
        raise ValueError("Emission factors CSV must contain a 'technology' column.")
    df["technology"] = df["technology"].str.strip().str.lower()
    required_columns = [meta["column"] for meta in POLLUTANTS.values()]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Emission factors CSV is missing required columns: " + ", ".join(missing)
        )
    factors = df.set_index("technology")[required_columns].astype(float)
    return factors


def _resolve_demand_series(
    years: Sequence[int],
    scenarios: Mapping[str, Mapping[str, Mapping]],
    scenario_name: str | None,
    custom: Mapping[int, float] | None,
) -> pd.Series:
    if scenario_name:
        scenario = scenarios.get(scenario_name)
        if scenario is None:
            raise ValueError(f"Demand scenario '{scenario_name}' not found in config.")
        values = scenario.get("values")
        if not values:
            raise ValueError(f"Demand scenario '{scenario_name}' must define 'values'.")
    elif custom:
        values = custom
    else:
        raise ValueError("Provide either 'demand_scenario' or 'demand_custom'.")

    series = _values_to_series(values, years, "demand")
    return series.rename("demand_twh")


def _resolve_mix_shares(
    years: Sequence[int],
    scenarios: Mapping[str, Mapping[str, Mapping]],
    technologies: Iterable[str],
    scenario_name: str | None,
    custom: Mapping[str, Mapping] | None,
) -> pd.DataFrame:
    tech_list = [tech.lower() for tech in technologies]
    if scenario_name:
        scenario = scenarios.get(scenario_name)
        if scenario is None:
            raise ValueError(f"Mix scenario '{scenario_name}' not found in config.")
        shares_cfg = scenario.get("shares")
        if shares_cfg is None:
            raise ValueError(f"Mix scenario '{scenario_name}' must define 'shares'.")
    elif custom:
        shares_cfg = custom.get("shares", {})
    else:
        raise ValueError("Provide either 'mix_scenario' or 'mix_custom'.")

    data = {}
    for tech in tech_list:
        entry = shares_cfg.get(tech)
        if entry is None:
            data[tech] = np.zeros(len(years))
            continue
        if isinstance(entry, Mapping):
            series = _values_to_series(entry, years, f"mix share for {tech}")
        else:
            series = pd.Series([float(entry)] * len(years), index=years)
        data[tech] = series.values

    mix_df = pd.DataFrame(data, index=years)
    totals = mix_df.sum(axis=1)
    if np.any(totals == 0):
        raise ValueError("Mix shares sum to zero for at least one year; cannot normalise.")
    mix_df = mix_df.divide(totals, axis=0)
    return mix_df


def _values_to_series(
    values: Mapping[int | str, float],
    years: Sequence[int],
    label: str,
) -> pd.Series:
    mapping = {}
    for key, value in values.items():
        mapping[int(key)] = float(value)
    if not mapping:
        raise ValueError(f"No values provided for {label}.")
    series = pd.Series(mapping, dtype=float).sort_index()
    idx = pd.Index(sorted(set(years)))
    series = series.reindex(idx.union(series.index), method=None)
    series = series.interpolate(method="index").ffill().bfill()
    return series.reindex(idx)

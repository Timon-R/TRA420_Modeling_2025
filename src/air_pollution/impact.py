"""Compute health impacts from non-CO₂ emission changes.

This module links the electricity emission scenarios produced by
``calc_emissions`` to concentration-response functions for air pollutants.
High-level workflow:

1. Load baseline concentration statistics per country.
2. Retrieve baseline and scenario emissions from ``calc_emissions``.
3. Assume concentration changes scale linearly with emission ratios.
4. Apply a log-linear relative-risk model to estimate mortality response.

Outputs are written as CSV files per scenario/pollutant containing the
percentage change in mortality for each country and year.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from calc_emissions.calculator import EmissionScenarioResult, run_from_config as run_emissions

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_POLLUTANT_FILES = {
    "pm25": "data/air_pollution/PM25_country_stats.csv",
    "nox": "data/air_pollution/NOx_country_stats.csv",
}

DEFAULT_RELATIVE_RISK = {"pm25": 1.08, "nox": 1.03}
DEFAULT_REFERENCE_DELTA = 10.0  # µg/m³ corresponding to the RR default.
DEFAULT_FALLBACK_MEASURES = ["median", "mean", "min", "max"]


@dataclass
class PollutantImpact:
    """Container for pollutant-specific health impacts."""

    pollutant: str
    concentration_metric: str
    beta: float
    concentrations: pd.Series
    emission_ratio: pd.Series
    impacts: pd.DataFrame


@dataclass
class AirPollutionResult:
    """Scenario-level health impact results."""

    scenario: str
    pollutant_results: dict[str, PollutantImpact]


def run_from_config(
    config_path: Path | str = "config.yaml",
    emission_results: Mapping[str, EmissionScenarioResult] | None = None,
) -> dict[str, AirPollutionResult]:
    """Run air-pollution health calculations defined in ``config.yaml``.

    Parameters
    ----------
    config_path:
        Path to the configuration file containing ``air_pollution`` settings.
    emission_results:
        Optional results returned by :func:`calc_emissions.run_from_config`.
        When omitted the function will execute the emissions module first.

    Returns
    -------
    dict[str, AirPollutionResult]
        Mapping of scenario name to impact results.
    """

    config_path = Path(config_path)
    with config_path.open() as handle:
        full_config = yaml.safe_load(handle) or {}

    module_cfg = full_config.get("air_pollution")
    if not module_cfg:
        raise ValueError("'air_pollution' section missing from config.yaml")

    output_dir = Path(module_cfg.get("output_directory", "results/air_pollution"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fallback_measures = _normalise_measures(
        module_cfg.get("concentration_fallback_order", DEFAULT_FALLBACK_MEASURES)
    )
    default_measure = module_cfg.get("concentration_measure", fallback_measures[0])
    countries_filter = module_cfg.get("countries")
    selection = _normalise_scenario_selection(module_cfg.get("scenarios", "all"))

    if emission_results is None:
        emission_results = run_emissions(config_path=config_path)
    if "baseline" not in emission_results:
        raise ValueError("calc_emissions results must include a 'baseline' entry.")

    baseline = emission_results["baseline"]
    pollutant_cfg = module_cfg.get("pollutants", {})
    pollutants = set(pollutant_cfg) if pollutant_cfg else set(DEFAULT_POLLUTANT_FILES)

    results: dict[str, AirPollutionResult] = {}

    for scenario_name, scenario_result in emission_results.items():
        if scenario_name == "baseline":
            continue
        if selection and scenario_name not in selection:
            continue

        pollutant_results: dict[str, PollutantImpact] = {}
        for pollutant in sorted(pollutants):
            scenario_series = scenario_result.total_emissions_mt.get(pollutant)
            baseline_series = baseline.total_emissions_mt.get(pollutant)
            if scenario_series is None or baseline_series is None:
                continue

            cfg = pollutant_cfg.get(pollutant, {})
            concentrations = _load_concentrations(
                config_path.parent,
                cfg.get("stats_file", DEFAULT_POLLUTANT_FILES.get(pollutant)),
                cfg.get("concentration_measure", default_measure),
                fallback_measures,
                countries_filter,
            )

            if concentrations.empty:
                continue

            beta = _derive_beta(cfg, pollutant)
            emission_ratio = _compute_emission_ratio(scenario_series, baseline_series)
            impacts = _build_impacts(concentrations, emission_ratio, beta)

            if impacts.empty:
                continue

            pollutant_results[pollutant] = PollutantImpact(
                pollutant=pollutant,
                concentration_metric=cfg.get("concentration_measure", default_measure),
                beta=beta,
                concentrations=concentrations,
                emission_ratio=emission_ratio,
                impacts=impacts,
            )

            _write_output(output_dir, scenario_name, pollutant, impacts)

        if pollutant_results:
            results[scenario_name] = AirPollutionResult(
                scenario=scenario_name, pollutant_results=pollutant_results
            )

    return results


def _normalise_measures(measures: Iterable[str]) -> list[str]:
    ordered = []
    for measure in measures:
        if not isinstance(measure, str):
            continue
        key = measure.strip().lower()
        if key and key not in ordered:
            ordered.append(key)
    if not ordered:
        ordered = DEFAULT_FALLBACK_MEASURES.copy()
    return ordered


def _normalise_scenario_selection(selection: object) -> set[str]:
    if selection in (None, "all"):
        return set()
    if isinstance(selection, str):
        return {selection}
    if isinstance(selection, Iterable):
        return {str(item) for item in selection}
    return set()


def _load_concentrations(
    config_dir: Path,
    stats_path: str | None,
    preferred_measure: str,
    fallback_measures: Iterable[str],
    countries_filter: Iterable[str] | None,
) -> pd.Series:
    if not stats_path:
        return pd.Series(dtype=float)

    path = Path(stats_path)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Air-pollution statistics file not found: '{path}'.")

    df = pd.read_csv(path)
    if "country" not in df.columns:
        raise ValueError(f"'country' column missing from air-pollution file: '{path}'.")

    measures = [preferred_measure.lower(), *[m.lower() for m in fallback_measures]]
    numeric_cols = ["mean", "median", "min", "max"]
    available = {col.lower(): col for col in df.columns if col.lower() in numeric_cols}

    chosen_col = None
    for candidate in measures:
        if candidate in available:
            chosen_col = available[candidate]
            break

    if chosen_col is None:
        raise ValueError(
            f"No usable concentration column found in '{path}'. " f"Available: {sorted(available)}"
        )

    series = df.set_index("country")[chosen_col].astype(float)
    if countries_filter:
        countries = [c for c in countries_filter if c in series.index]
        series = series.loc[countries]

    series = series.dropna()
    return series


def _derive_beta(cfg: Mapping[str, object], pollutant: str) -> float:
    if "beta" in cfg:
        return float(cfg["beta"])
    rr = cfg.get("relative_risk", DEFAULT_RELATIVE_RISK.get(pollutant))
    reference = float(cfg.get("reference_delta", DEFAULT_REFERENCE_DELTA))
    if rr is None:
        raise ValueError(f"Pollutant '{pollutant}' requires either 'beta' or 'relative_risk'.")
    if reference == 0:
        raise ValueError("reference_delta must be non-zero.")
    return math.log(float(rr)) / reference


def _compute_emission_ratio(
    scenario_series: pd.Series,
    baseline_series: pd.Series,
) -> pd.Series:
    aligned = pd.concat([scenario_series, baseline_series], axis=1, keys=["scenario", "baseline"])
    aligned = aligned.sort_index()
    ratio = np.divide(
        aligned["scenario"],
        aligned["baseline"].replace(0.0, np.nan),
    )
    zero_mask = aligned["baseline"] == 0.0
    ratio = ratio.astype(float)
    ratio.loc[zero_mask & (aligned["scenario"] == 0.0)] = 1.0
    ratio.name = "emission_ratio"
    return ratio


def _build_impacts(
    concentrations: pd.Series,
    emission_ratio: pd.Series,
    beta: float,
) -> pd.DataFrame:
    records = []
    for year, ratio in emission_ratio.items():
        if pd.isna(ratio):
            for country, baseline_conc in concentrations.items():
                records.append(
                    {
                        "country": country,
                        "year": int(year),
                        "baseline_concentration": baseline_conc,
                        "emission_ratio": np.nan,
                        "new_concentration": np.nan,
                        "delta_concentration": np.nan,
                        "percent_change_mortality": np.nan,
                    }
                )
            continue

        multiplier = float(ratio)
        for country, baseline_conc in concentrations.items():
            new_conc = baseline_conc * multiplier
            delta_conc = new_conc - baseline_conc
            perc_change = math.exp(beta * delta_conc) - 1.0
            records.append(
                {
                    "country": country,
                    "year": int(year),
                    "baseline_concentration": baseline_conc,
                    "emission_ratio": multiplier,
                    "new_concentration": new_conc,
                    "delta_concentration": delta_conc,
                    "percent_change_mortality": perc_change,
                }
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    df.sort_values(["country", "year"], inplace=True)
    return df


def _write_output(output_dir: Path, scenario: str, pollutant: str, impacts: pd.DataFrame) -> None:
    scenario_dir = output_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    path = scenario_dir / f"{pollutant}_health_impact.csv"
    impacts.to_csv(path, index=False)

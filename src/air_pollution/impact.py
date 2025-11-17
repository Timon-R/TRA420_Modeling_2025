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
from config_paths import apply_results_run_directory, get_results_run_directory

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
    country_weights: pd.Series
    weighted_percent_change: pd.Series
    baseline_deaths_per_year: float | None = None
    deaths_summary: pd.DataFrame | None = None
    baseline_deaths_by_country: pd.Series | None = None


@dataclass
class AirPollutionResult:
    """Scenario-level health impact results."""

    scenario: str
    pollutant_results: dict[str, PollutantImpact]
    total_mortality_summary: pd.DataFrame | None = None


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

    run_directory = get_results_run_directory(full_config)
    output_dir = Path(module_cfg.get("output_directory", "results/air_pollution"))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir = apply_results_run_directory(
        output_dir,
        run_directory,
        repo_root=config_path.parent.resolve(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fallback_measures = _normalise_measures(
        module_cfg.get("concentration_fallback_order", DEFAULT_FALLBACK_MEASURES)
    )
    default_measure = module_cfg.get("concentration_measure", fallback_measures[0])
    countries_filter = module_cfg.get("countries")
    selection = _normalise_scenario_selection(module_cfg.get("scenarios", "all"))
    module_country_weights_cfg = module_cfg.get("country_weights")
    module_baseline_cfg = module_cfg.get("baseline_deaths")
    module_baseline_deaths = _parse_baseline_deaths(module_baseline_cfg)
    module_vsl_cfg = module_cfg.get("value_of_statistical_life_usd")
    module_vsl_value = float(module_vsl_cfg) if module_vsl_cfg is not None else None
    module_weights_cfg = (
        module_baseline_cfg.get("weights") if isinstance(module_baseline_cfg, Mapping) else None
    )

    if emission_results is None:
        emission_results = run_emissions(
            config_path=config_path,
            results_run_directory=run_directory,
        )
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
            concentrations, baseline_deaths_by_country = _load_concentrations(
                config_path.parent,
                cfg.get("stats_file", DEFAULT_POLLUTANT_FILES.get(pollutant)),
                cfg.get("concentration_measure", default_measure),
                fallback_measures,
                countries_filter,
            )

            if concentrations.empty:
                continue

            country_weights = _resolve_country_weights(
                concentrations.index,
                module_country_weights_cfg,
                cfg.get("country_weights"),
            )
            if baseline_deaths_by_country is not None and not baseline_deaths_by_country.empty:
                normalized = _weights_from_baseline(
                    baseline_deaths_by_country, concentrations.index
                )
                if normalized is not None:
                    country_weights = normalized

            beta = _derive_beta(cfg, pollutant)
            emission_ratio = _compute_emission_ratio(scenario_series, baseline_series)
            impacts = _build_impacts(concentrations, emission_ratio, beta)

            if impacts.empty:
                continue

            weighted_percent_change = _weighted_percent_change(impacts, country_weights)

            baseline_deaths = _parse_baseline_deaths(cfg.get("baseline_deaths"))
            if baseline_deaths is None and baseline_deaths_by_country is not None:
                baseline_deaths = float(baseline_deaths_by_country.sum())
            deaths_summary = None
            if baseline_deaths is not None:
                deaths_summary = _build_death_summary(
                    weighted_percent_change,
                    baseline_deaths,
                    value_of_statistical_life=module_vsl_value,
                )
                if deaths_summary.empty:
                    deaths_summary = None

            pollutant_results[pollutant] = PollutantImpact(
                pollutant=pollutant,
                concentration_metric=cfg.get("concentration_measure", default_measure),
                beta=beta,
                concentrations=concentrations,
                emission_ratio=emission_ratio,
                impacts=impacts,
                country_weights=country_weights,
                weighted_percent_change=weighted_percent_change,
                baseline_deaths_per_year=baseline_deaths,
                deaths_summary=deaths_summary,
                baseline_deaths_by_country=baseline_deaths_by_country,
            )

            _write_output(output_dir, scenario_name, pollutant, impacts)
            if deaths_summary is not None:
                _write_mortality_summary(output_dir, scenario_name, pollutant, deaths_summary)

        if pollutant_results:
            baseline_for_total = module_baseline_deaths
            if baseline_for_total is None:
                derived = sum(
                    impact.baseline_deaths_per_year or 0.0 for impact in pollutant_results.values()
                )
                if derived > 0.0:
                    baseline_for_total = float(derived)
                    module_baseline_deaths = baseline_for_total
            total_summary = None
            if baseline_for_total is not None:
                total_summary = _build_total_mortality_summary(
                    pollutant_results,
                    baseline_for_total,
                    module_weights_cfg,
                    module_vsl_value,
                )
                if total_summary is not None:
                    _write_total_mortality_summary(output_dir, scenario_name, total_summary)
            results[scenario_name] = AirPollutionResult(
                scenario=scenario_name,
                pollutant_results=pollutant_results,
                total_mortality_summary=total_summary,
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
) -> tuple[pd.Series, pd.Series | None]:
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
    baseline_deaths = None
    if "baseline_deaths_per_year" in df.columns:
        baseline_series = df.set_index("country")["baseline_deaths_per_year"].astype(float)
        baseline_deaths = baseline_series
    if countries_filter:
        countries = [c for c in countries_filter if c in series.index]
        series = series.loc[countries]

    series = series.dropna()
    if baseline_deaths is not None:
        baseline_deaths = baseline_deaths.reindex(series.index).dropna()
    return series, baseline_deaths


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


def _parse_baseline_deaths(baseline_cfg: Mapping[str, object] | None) -> float | None:
    if not baseline_cfg:
        return None
    if not isinstance(baseline_cfg, Mapping):
        raise TypeError("'baseline_deaths' must be a mapping with configuration values.")

    if "per_year" in baseline_cfg:
        return float(baseline_cfg["per_year"])

    total = baseline_cfg.get("total")
    if total is None:
        raise ValueError(
            "Provide either 'per_year' or 'total' inside 'baseline_deaths' configuration."
        )

    years = baseline_cfg.get("years")
    span = baseline_cfg.get("span")
    if years is not None and span is not None:
        raise ValueError("Use either 'years' (list) or 'span' (start/end) to define the period.")

    if years is not None:
        if isinstance(years, str) or not isinstance(years, Iterable):
            raise TypeError("'years' must be an iterable of years.")
        years_list = [int(year) for year in years]
        count = len(years_list)
    elif span is not None:
        if not isinstance(span, Mapping):
            raise TypeError("'span' must be a mapping with 'start' and 'end' years.")
        if "start" not in span or "end" not in span:
            raise ValueError("'span' must include 'start' and 'end' keys.")
        start = int(span["start"])
        end = int(span["end"])
        if start > end:
            raise ValueError("'span.start' must be <= 'span.end'.")
        count = end - start + 1
    else:
        raise ValueError("Specify either 'years' or 'span' when using 'baseline_deaths.total'.")

    if count <= 0:
        raise ValueError("The baseline period must cover at least one year.")

    return float(total) / float(count)


def _parse_weight_config(
    cfg: Mapping[str, object] | str | None, items: Iterable[str]
) -> pd.Series | None:
    items = [str(item) for item in items]
    if not items:
        return None

    if cfg is None:
        return None
    if isinstance(cfg, str):
        if cfg.lower() == "equal":
            return pd.Series(1.0, index=items, dtype=float)
        raise ValueError(f"Unsupported weight shorthand '{cfg}'. Use 'equal' or a mapping.")
    if not isinstance(cfg, Mapping):
        raise TypeError("Weights configuration must be a mapping or the string 'equal'.")

    weights: dict[str, float] = {}
    for key, value in cfg.items():
        if key is None:
            continue
        key_str = str(key)
        if key_str not in items:
            continue
        weights[key_str] = float(value)
    if not weights:
        return None
    series = pd.Series(weights, dtype=float)
    series = series.reindex(items, fill_value=0.0)
    return series


def _resolve_country_weights(
    countries: Iterable[str],
    module_cfg: Mapping[str, object] | str | None,
    pollutant_cfg: Mapping[str, object] | str | None,
) -> pd.Series:
    countries = [str(country) for country in countries]
    if not countries:
        return pd.Series(dtype=float)

    weights = _parse_weight_config(pollutant_cfg, countries)
    if weights is None:
        weights = _parse_weight_config(module_cfg, countries)
    if weights is None:
        weights = pd.Series(1.0, index=countries, dtype=float)

    weights = weights.astype(float).fillna(0.0)
    total = weights.sum()
    if total <= 0.0:
        weights = pd.Series(1.0, index=countries, dtype=float)
        total = weights.sum()
    return weights / total


def _resolve_weights(
    weights_cfg: Mapping[str, object] | str | None, items: Iterable[str]
) -> pd.Series:
    items = [str(item) for item in items]
    if not items:
        return pd.Series(dtype=float)

    weights = _parse_weight_config(weights_cfg, items)
    if weights is None:
        weights = pd.Series(1.0, index=items, dtype=float)

    weights = weights.astype(float).fillna(0.0)
    total = weights.sum()
    if total <= 0.0:
        weights = pd.Series(1.0, index=items, dtype=float)
        total = weights.sum()
    return weights / total


def _weights_from_baseline(
    baseline_series: pd.Series, countries: Iterable[str]
) -> pd.Series | None:
    filtered = baseline_series.reindex(list(countries)).dropna()
    if filtered.empty:
        return None
    total = filtered.sum()
    if not math.isfinite(total) or total <= 0.0:
        return None
    return (filtered / total).astype(float)


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


def _weighted_percent_change(impacts: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if impacts.empty:
        return pd.Series(dtype=float)

    weights = weights.astype(float).fillna(0.0)
    grouped = impacts.groupby("year")
    results: list[tuple[int, float]] = []

    for year, group in grouped:
        group = group.dropna(subset=["percent_change_mortality"])
        if group.empty:
            continue
        values = group.set_index("country")["percent_change_mortality"].astype(float)
        available_weights = weights.reindex(values.index).fillna(0.0)
        if available_weights.sum() <= 0.0:
            available_weights = pd.Series(1.0, index=values.index, dtype=float)
        total = available_weights.sum()
        if total <= 0.0:
            continue
        normalized = available_weights / total
        percent_change = float(np.dot(values.loc[normalized.index], normalized))
        results.append((int(year), percent_change))

    if not results:
        return pd.Series(dtype=float)

    results.sort(key=lambda item: item[0])
    years, values = zip(*results, strict=False)
    return pd.Series(values, index=years, dtype=float)


def _build_death_summary(
    percent_change: pd.Series,
    baseline_deaths_per_year: float,
    *,
    value_of_statistical_life: float | None = None,
) -> pd.DataFrame:
    if percent_change.empty:
        return pd.DataFrame()

    df = (
        percent_change.sort_index()
        .astype(float, copy=True)
        .to_frame(name="percent_change_mortality")
    )
    df["baseline_deaths_per_year"] = baseline_deaths_per_year
    df["delta_deaths_per_year"] = df["baseline_deaths_per_year"] * df["percent_change_mortality"]
    df["new_deaths_per_year"] = df["baseline_deaths_per_year"] + df["delta_deaths_per_year"]
    if value_of_statistical_life is not None:
        vsl = float(value_of_statistical_life)
        df["value_of_statistical_life_usd"] = vsl
        df["baseline_value_usd"] = df["baseline_deaths_per_year"] * vsl
        df["delta_value_usd"] = df["delta_deaths_per_year"] * vsl
        df["new_value_usd"] = df["new_deaths_per_year"] * vsl
    df.reset_index(inplace=True)
    return df


def _build_total_mortality_summary(
    pollutant_results: Mapping[str, PollutantImpact],
    baseline_deaths_per_year: float | None,
    weights_cfg: Mapping[str, object] | str | None,
    value_of_statistical_life: float | None,
) -> pd.DataFrame | None:
    if baseline_deaths_per_year is None:
        return None

    series_map: dict[str, pd.Series] = {}
    for impact in pollutant_results.values():
        if impact.weighted_percent_change.empty:
            continue
        series_map[impact.pollutant] = impact.weighted_percent_change

    if not series_map:
        return None

    combined = pd.DataFrame(series_map).sort_index()
    combined = combined.dropna(how="all")

    rows: list[dict[str, float]] = []
    for year, row in combined.iterrows():
        available_pollutants = [col for col, value in row.items() if pd.notna(value)]
        if not available_pollutants:
            continue
        weights = _resolve_weights(weights_cfg, available_pollutants)
        values = row[weights.index].to_numpy(dtype=float, copy=True)
        percent_change = float(np.dot(values, weights.to_numpy(dtype=float, copy=True)))
        delta = baseline_deaths_per_year * percent_change
        entry = {
            "year": int(year),
            "percent_change_mortality": percent_change,
            "baseline_deaths_per_year": baseline_deaths_per_year,
            "delta_deaths_per_year": delta,
            "new_deaths_per_year": baseline_deaths_per_year + delta,
        }
        if value_of_statistical_life is not None:
            vsl = float(value_of_statistical_life)
            entry["value_of_statistical_life_usd"] = vsl
            entry["baseline_value_usd"] = baseline_deaths_per_year * vsl
            entry["delta_value_usd"] = delta * vsl
            entry["new_value_usd"] = entry["new_deaths_per_year"] * vsl
        rows.append(entry)

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _write_output(output_dir: Path, scenario: str, pollutant: str, impacts: pd.DataFrame) -> None:
    scenario_dir = output_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    path = scenario_dir / f"{pollutant}_health_impact.csv"
    impacts.to_csv(path, index=False)


def _write_mortality_summary(
    output_dir: Path, scenario: str, pollutant: str, summary: pd.DataFrame
) -> None:
    scenario_dir = output_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    path = scenario_dir / f"{pollutant}_mortality_summary.csv"
    summary.to_csv(path, index=False)


def _write_total_mortality_summary(output_dir: Path, scenario: str, summary: pd.DataFrame) -> None:
    scenario_dir = output_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    path = scenario_dir / "total_mortality_summary.csv"
    summary.to_csv(path, index=False)

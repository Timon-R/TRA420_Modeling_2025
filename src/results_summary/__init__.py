"""Generate configurable cross-module summaries (metrics + plots)."""

from __future__ import annotations

import logging
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from calc_emissions.constants import BASE_DEMAND_CASE
from calc_emissions.scenario_io import (
    list_available_scenarios,
    load_scenario_absolute,
    load_scenario_delta,
    split_scenario_name,
)
from config_paths import apply_results_run_directory, get_results_run_directory
from economic_module.scc import _load_ssp_economic_data
from economic_module.socioeconomics import DiceSocioeconomics

LOGGER = logging.getLogger("results.summary")

AVAILABLE_METHODS: tuple[str, ...] = ("constant_discount", "ramsey_discount")


@dataclass(slots=True)
class SummarySettings:
    years: list[int]
    output_directory: Path
    include_plots: bool = True
    plot_format: str = "png"
    year_periods: list[tuple[int, int]] = field(default_factory=list)
    aggregation_mode: str = "per_year"
    aggregation_horizon: tuple[int, int] | None = None
    plot_start: int = 2025
    plot_end: int = 2100
    climate_labels: list[str] = field(default_factory=list)
    run_method: str = "kernel"
    scc_output_directory: Path | None = None
    base_year: int = 2025
    climate_output_directory: Path | None = None
    gdp_series: dict[str, dict[int, float]] | None = None
    population_series: dict[str, dict[int, float]] | None = None
    per_country_emission_root: Path | None = None
    pattern_output_directory: Path | None = None
    pattern_countries: list[str] | None = None


@dataclass(slots=True)
class ScenarioMetrics:
    emission_delta_mt: dict[int, float] = field(default_factory=dict)
    temperature_delta_c: dict[int, float] = field(default_factory=dict)
    mortality_delta: dict[int, float] = field(default_factory=dict)
    mortality_percent: dict[int, float] = field(default_factory=dict)
    mortality_baseline: dict[int, float] = field(default_factory=dict)
    mortality_value_delta: dict[int, float] | None = None
    mortality_sum: dict[str, float] = field(default_factory=dict)
    mortality_value_sum: dict[str, float] = field(default_factory=dict)
    scc_usd_per_tco2: dict[str, dict[int, float]] = field(default_factory=dict)
    damages_usd: dict[str, dict[int, float]] = field(default_factory=dict)
    damages_sum: dict[str, dict[str, float]] = field(default_factory=dict)
    scc_average: dict[str, float] = field(default_factory=dict)
    damage_total_usd: dict[str, float] = field(default_factory=dict)
    emission_timeseries: dict[int, float] | None = None
    temperature_timeseries: dict[int, float] | None = None


def _ensure_years(summary_cfg: Mapping[str, object]) -> list[int]:
    years_cfg = summary_cfg.get("years")
    if not years_cfg:
        return [2030, 2050]
    years: list[int] = []
    for entry in years_cfg:
        try:
            years.append(int(entry))
        except (TypeError, ValueError):
            LOGGER.warning("Ignoring invalid summary year '%s'", entry)
    return sorted(set(years))


def _resolve_methods(methods_cfg: Mapping[str, object]) -> list[str]:
    run = methods_cfg.get("run")
    if isinstance(run, str):
        if run.lower() == "all":
            return list(AVAILABLE_METHODS)
        run_list = [run]
    elif isinstance(run, Iterable):
        run_list = [str(item) for item in run]
    else:
        run_list = []
    selected = [method for method in run_list if method in AVAILABLE_METHODS]
    if selected:
        return selected
    return list(AVAILABLE_METHODS)


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_") or "scenario"


def _parse_climate_labels(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        labels: list[str] = []
        for item in value:
            if item is None:
                continue
            label = str(item).strip()
            if label:
                labels.append(label)
        return labels
    return []


def _split_climate_suffix(name: str, climate_labels: Sequence[str]) -> tuple[str, str | None]:
    for label in sorted({cl.strip() for cl in climate_labels if cl}, key=len, reverse=True):
        suffix = f"_{label}"
        if name.endswith(suffix):
            return name[: -len(suffix)], label
    return name, None


def _scenario_climate_label(scenario: str, climate_labels: Sequence[str]) -> str | None:
    if not climate_labels:
        return None
    _, climate_label = _split_climate_suffix(scenario, climate_labels)
    return climate_label


def _infer_ssp_family_from_label(label: str | None) -> str:
    if not label:
        return "SSP2"
    match = re.match(r"ssp\s*([0-9])", label, re.IGNORECASE)
    if match:
        return f"SSP{match.group(1)}"
    return "SSP2"


def _collapse_by_climate_label(
    data: Mapping[str, dict[int, float]] | None, climate_labels: Sequence[str]
) -> dict[str, dict[int, float]]:
    if not data:
        return {}
    if not climate_labels:
        return dict(data)
    collapsed: dict[str, dict[int, float]] = {}
    for scenario, series in data.items():
        base, label = _split_climate_suffix(scenario, climate_labels)
        key = label or base or scenario
        if key not in collapsed:
            collapsed[key] = dict(series)
    return collapsed


def _resolve_directory(path: Path, label: str) -> Path:
    candidate = path.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"{label} directory not found: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {candidate}")
    return candidate


def _default_emission_root(
    root_cfg: Mapping[str, object],
    root: Path,
    run_directory: str | None,
) -> Path | None:
    countries_cfg = root_cfg.get("calc_emissions", {}).get("countries", {})
    path_str = countries_cfg.get("aggregate_output_directory") or countries_cfg.get(
        "aggregate_results_directory"
    )
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (root / path).resolve()
    return apply_results_run_directory(path, run_directory, repo_root=root)


def _default_per_country_root(
    root_cfg: Mapping[str, object],
    root: Path,
    run_directory: str | None,
) -> Path | None:
    countries_cfg = root_cfg.get("calc_emissions", {}).get("countries", {})
    path_str = countries_cfg.get("resources_root")
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (root / path).resolve()
    return apply_results_run_directory(path, run_directory, repo_root=root)


def _default_temperature_root(
    root_cfg: Mapping[str, object],
    root: Path,
    run_directory: str | None,
) -> Path | None:
    climate_cfg = root_cfg.get("climate_module", {})
    path_str = climate_cfg.get("resource_directory") or climate_cfg.get("output_directory")
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (root / path).resolve()
    return apply_results_run_directory(path, run_directory, repo_root=root)


def _resolve_path(
    root: Path,
    value: object | None,
    run_directory: str | None,
) -> Path | None:
    if value in (None, "", False):
        return None
    path = Path(str(value))
    absolute = path if path.is_absolute() else (root / path).resolve()
    return apply_results_run_directory(absolute, run_directory, repo_root=root)


def _load_full_series(path_or_frame: Path | pd.DataFrame, value_column: str) -> dict[int, float]:
    if isinstance(path_or_frame, pd.DataFrame):
        frame = path_or_frame.copy()
    else:
        path = Path(path_or_frame)
        if not path.exists():
            LOGGER.debug("Data file missing for full series load: %s", path)
            return {}
        frame = pd.read_csv(path)
        if "year" not in frame.columns or value_column not in frame.columns:
            LOGGER.warning(
                "File %s missing required columns 'year'/'%s' for full series load",
                path,
                value_column,
            )
            return {}
    frame["year"] = frame["year"].astype(int)
    series = frame.set_index("year")[value_column].astype(float)
    return {int(year): float(series.get(int(year))) for year in series.index}


def _load_emission_frame_from_mix(
    root: Path,
    scenario: str,
    value_column: str,
    baseline_case: str,
) -> pd.DataFrame:
    if value_column == "delta":
        frame = load_scenario_delta(root, scenario, baseline_case=baseline_case)
    elif value_column == "absolute":
        frame = load_scenario_absolute(root, scenario)
    else:
        raise ValueError(
            "Unsupported emission_column '%s'. Use 'delta' or 'absolute'." % value_column
        )
    return frame


def _load_scc_summary_table(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        LOGGER.debug("SCC summary table missing: %s", path)
        return {}
    frame = pd.read_csv(path)
    required = {"scenario", "method", "scc_usd_per_tco2"}
    if not required.issubset(frame.columns):
        LOGGER.warning("SCC summary table %s missing required columns", path)
        return {}
    summary: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        scenario = str(row["scenario"])
        method = str(row["method"])
        try:
            value = float(row["scc_usd_per_tco2"])
        except (TypeError, ValueError):
            value = math.nan
        store = summary.setdefault(scenario, {})
        store[method] = value
        safe = _safe_name(scenario)
        if safe not in summary:
            summary[safe] = store
    return summary


def _collect_socio_series(
    root: Path,
    config: Mapping[str, object],
    scenario_labels: Iterable[str],
    climate_labels: Sequence[str],
    run_directory: str | None,
    plot_end: int,
) -> tuple[dict[str, dict[int, float]] | None, dict[str, dict[int, float]] | None]:
    economic_cfg = config.get("economic_module", {}) or {}
    socio_cfg = config.get("socioeconomics", {}) or {}
    scenario_list = list(scenario_labels)
    if not scenario_list:
        return None, None

    gdp_series_path = _resolve_path(root, economic_cfg.get("gdp_series"), run_directory)
    gdp_population_dir = economic_cfg.get("gdp_population_directory")
    if gdp_population_dir is not None:
        gdp_population_dir = _resolve_path(root, gdp_population_dir, run_directory)
    if gdp_population_dir is None:
        gdp_population_dir = (root / "data" / "GDP_and_Population_data").resolve()

    gdp_map: dict[str, dict[int, float]] = {}
    pop_map: dict[str, dict[int, float]] = {}

    try:
        if gdp_series_path and gdp_series_path.exists():
            frame = pd.read_csv(gdp_series_path)
            if "year" not in frame.columns or "gdp_trillion_usd" not in frame.columns:
                raise ValueError(
                    f"GDP series file {gdp_series_path} missing required columns "
                    "'year'/'gdp_trillion_usd'"
                )
            frame["year"] = frame["year"].astype(int)
            gdp_base = {
                int(year): float(value)
                for year, value in frame.set_index("year")["gdp_trillion_usd"].astype(float).items()
            }
            pop_base: dict[int, float] | None = None
            if "population_million" in frame.columns:
                pop_base = {
                    int(year): float(value)
                    for year, value in frame.set_index("year")["population_million"]
                    .astype(float)
                    .items()
                }
            for scenario in scenario_list:
                gdp_map[scenario] = dict(gdp_base)
                if pop_base is not None:
                    pop_map[scenario] = dict(pop_base)
            return gdp_map or None, pop_map or None

        socio_mode = str(socio_cfg.get("mode", "")).strip().lower()
        if socio_mode == "dice":
            dice_cfg = socio_cfg.get("dice", {}) or {}
            for scenario in scenario_list:
                climate_label = _scenario_climate_label(scenario, climate_labels)
                gdp_series, population_series = _project_dice_series(
                    root,
                    dice_cfg,
                    climate_label,
                    plot_end,
                )
                if gdp_series:
                    gdp_map[scenario] = gdp_series
                if population_series:
                    pop_map[scenario] = population_series
            return gdp_map or None, pop_map or None

        for scenario in scenario_list:
            climate_label = _scenario_climate_label(scenario, climate_labels)
            family = _infer_ssp_family_from_label(climate_label)
            gdp_series, population_series = _load_ssp_economic_data(family, gdp_population_dir)
            gdp_map[scenario] = {
                int(year): float(value) for year, value in gdp_series.sort_index().items()
            }
            if population_series is not None:
                pop_map[scenario] = {
                    int(year): float(value)
                    for year, value in population_series.sort_index().items()
                }
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Unable to load socioeconomics series: %s", exc)
        return None, None

    return gdp_map or None, pop_map or None


def _project_dice_series(
    root: Path,
    dice_cfg: Mapping[str, object],
    climate_label: str | None,
    end_year: int,
) -> tuple[dict[int, float] | None, dict[int, float] | None]:
    cfg = dict(dice_cfg)
    scenario_setting = str(cfg.get("scenario", "SSP2")).strip()
    if scenario_setting.lower() == "as_climate_scenario":
        resolved = _infer_ssp_family_from_label(climate_label)
    else:
        resolved = scenario_setting.upper()
    cfg["scenario"] = resolved
    model = DiceSocioeconomics.from_config(cfg, base_path=root)
    frame = model.project(int(end_year))
    gdp_series = {
        int(year): float(value)
        for year, value in frame.set_index("year")["gdp_trillion_usd"].astype(float).items()
    }
    population_series = {
        int(year): float(value)
        for year, value in frame.set_index("year")["population_million"].astype(float).items()
    }
    return gdp_series, population_series


def _read_series(
    source: Path | pd.DataFrame,
    value_column: str,
    years: Iterable[int],
    *,
    transform: callable | None = None,
) -> dict[int, float]:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            LOGGER.debug("Data file missing: %s", path)
            return {year: math.nan for year in years}
        frame = pd.read_csv(path)
        if "year" not in frame.columns or value_column not in frame.columns:
            LOGGER.warning("File %s missing required columns 'year'/'%s'", path, value_column)
            return {year: math.nan for year in years}
    frame["year"] = frame["year"].astype(int)
    series = frame.set_index("year")[value_column].astype(float)
    if transform is not None:
        series = transform(series)
    return {int(year): float(series.get(int(year), np.nan)) for year in years}


def _read_temperature(path: Path, years: Iterable[int]) -> dict[int, float]:
    if not path.exists():
        LOGGER.debug("Temperature series missing: %s", path)
        return {year: math.nan for year in years}
    frame = pd.read_csv(path)
    if "year" not in frame.columns:
        LOGGER.warning("Temperature file %s missing 'year' column", path)
        return {year: math.nan for year in years}
    frame["year"] = frame["year"].astype(int)
    if "temperature_delta" in frame.columns:
        series = frame.set_index("year")["temperature_delta"].astype(float)
    elif {"temperature_baseline", "temperature_adjusted"}.issubset(frame.columns):
        base = frame.set_index("year")["temperature_baseline"].astype(float)
        adj = frame.set_index("year")["temperature_adjusted"].astype(float)
        series = adj - base
    else:
        LOGGER.warning("Temperature file %s missing delta information", path)
        return {year: math.nan for year in years}
    return {int(year): float(series.get(int(year), np.nan)) for year in years}


def _load_temperature_series(path: Path) -> dict[int, float] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "year" not in frame.columns:
        return None
    frame["year"] = frame["year"].astype(int)
    if "temperature_delta" in frame.columns:
        series = frame.set_index("year")["temperature_delta"].astype(float)
    elif {"temperature_baseline", "temperature_adjusted"}.issubset(frame.columns):
        base = frame.set_index("year")["temperature_baseline"].astype(float)
        adj = frame.set_index("year")["temperature_adjusted"].astype(float)
        series = adj - base
    else:
        return None
    return {int(year): float(value) for year, value in series.items()}


def _read_mortality(
    path: Path, years: Iterable[int]
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float] | None]:
    if not path.exists():
        LOGGER.debug("Mortality summary missing: %s", path)
        nan_map = {year: math.nan for year in years}
        return nan_map, nan_map.copy(), nan_map.copy()
    frame = pd.read_csv(path)
    expected = {
        "year",
        "delta_deaths_per_year",
        "percent_change_mortality",
        "baseline_deaths_per_year",
    }
    value_col = frame.columns.intersection(["delta_value_usd"])
    if not expected.issubset(frame.columns):
        LOGGER.warning("Mortality file %s missing required columns", path)
        nan_map = {year: math.nan for year in years}
        return nan_map, nan_map.copy(), nan_map.copy(), None
    frame["year"] = frame["year"].astype(int)
    frame = frame.set_index("year")
    delta = frame["delta_deaths_per_year"].astype(float)
    percent = frame["percent_change_mortality"].astype(float)
    baseline = frame["baseline_deaths_per_year"].astype(float)
    value_series = frame[value_col[0]].astype(float) if len(value_col) else None
    return (
        {int(year): float(delta.get(int(year), np.nan)) for year in years},
        {int(year): float(percent.get(int(year), np.nan)) for year in years},
        {int(year): float(baseline.get(int(year), np.nan)) for year in years},
        (
            {int(year): float(value_series.get(int(year), np.nan)) for year in years}
            if value_series is not None
            else None
        ),
    )


def _sum_over_periods(
    frame: pd.DataFrame, value_col: str, periods: Iterable[tuple[int, int]]
) -> dict[str, float]:
    """Interpolate to annual steps and sum over configured periods."""
    if "year" not in frame.columns or value_col not in frame.columns:
        return {}
    series = frame.set_index(frame["year"].astype(int))[value_col].astype(float)
    if series.empty:
        return {}
    full_index = pd.Index(range(int(series.index.min()), int(series.index.max()) + 1))
    series = series.reindex(full_index).interpolate(method="index").ffill().bfill()
    sums: dict[str, float] = {}
    for start, end in periods:
        idx = pd.Index(range(start, end + 1))
        if idx.empty:
            continue
        window = series.reindex(idx).interpolate(method="index").ffill().bfill()
        sums[f"{start}_to_{end}"] = float(window.sum())
    return sums


def _read_scc_timeseries(
    output_dir: Path,
    scenario: str,
    methods: Iterable[str],
    years: Iterable[int],
) -> tuple[
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
    dict[str, float],
]:
    scenario_key = _safe_name(scenario)
    damages: dict[str, dict[int, float]] = {}
    scc_values: dict[str, dict[int, float]] = {}
    totals: dict[str, float] = {}
    for method in methods:
        candidates = [
            output_dir / f"scc_timeseries_{method}_{scenario}.csv",
            output_dir / f"scc_timeseries_{method}_{scenario_key}.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if not path.exists():
            LOGGER.debug("SCC timeseries missing for %s (%s)", scenario, method)
            damages[method] = {year: math.nan for year in years}
            scc_values[method] = {year: math.nan for year in years}
            totals[method] = math.nan
            continue
        frame = pd.read_csv(path)
        if "year" not in frame.columns or "delta_damage_usd" not in frame.columns:
            LOGGER.warning("SCC timeseries %s missing damage data", path)
            damages[method] = {year: math.nan for year in years}
            totals[method] = math.nan
            scc_values[method] = {year: math.nan for year in years}
            continue
        frame["year"] = frame["year"].astype(int)
        # Prefer emission-year attributed NPV damages when available; fall back
        # to per-calendar-year damage deltas to preserve backward compatibility.
        damages_series = frame.set_index("year")["delta_damage_usd"].astype(float)
        damages[method] = {
            int(year): float(damages_series.get(int(year), np.nan)) for year in years
        }
        if "discounted_delta_usd" in frame.columns:
            totals[method] = float(frame["discounted_delta_usd"].astype(float).sum(skipna=True))
        else:
            totals[method] = math.nan
        if "scc_usd_per_tco2" in frame.columns:
            scc_series = frame.set_index("year")["scc_usd_per_tco2"].astype(float)
            scc_values[method] = {
                int(year): float(scc_series.get(int(year), np.nan)) for year in years
            }
        else:
            scc_values[method] = {year: math.nan for year in years}
    return damages, scc_values, totals


def _compute_damage_period_sums(
    output_dir: Path,
    scenario: str,
    methods: Iterable[str],
    periods: Iterable[tuple[int, int]],
) -> dict[str, dict[str, float]]:
    if not periods:
        return {}
    sums: dict[str, dict[str, float]] = {}
    scenario_key = _safe_name(scenario)
    for method in methods:
        method_sum: dict[str, float] = {}
        candidates = [
            output_dir / f"pulse_emission_damages_{method}_{scenario}.csv",
            output_dir / f"pulse_emission_damages_{method}_{scenario_key}.csv",
            output_dir / f"scc_timeseries_{method}_{scenario}.csv",
            output_dir / f"scc_timeseries_{method}_{scenario_key}.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            sums[method] = method_sum
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            sums[method] = method_sum
            continue
        value_col = None
        for candidate in ("discounted_delta_usd", "delta_damage_usd"):
            if candidate in frame.columns:
                value_col = candidate
                break
        if value_col is None:
            sums[method] = method_sum
            continue
        method_sum = _sum_over_periods(frame, value_col, periods)
        sums[method] = method_sum
    return sums


def build_summary(
    root: Path, config: Mapping[str, object]
) -> tuple[SummarySettings, list[str], dict[str, ScenarioMetrics]]:
    root = Path(root)
    try:
        economic_cfg = config["economic_module"]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(
            "economic_module configuration is required for summary generation."
        ) from exc
    if not isinstance(economic_cfg, Mapping):
        raise ValueError("economic_module configuration must be a mapping.")
    try:
        emission_to_tonnes = float(economic_cfg.get("emission_to_tonnes", 1e6))
    except (TypeError, ValueError):
        emission_to_tonnes = 1e6
    try:
        base_year_value = int(economic_cfg.get("base_year", 2025))
    except (TypeError, ValueError):
        base_year_value = 2025

    countries_cfg = config.get("calc_emissions", {}).get("countries", {}) or {}
    baseline_case = (
        str(countries_cfg.get("baseline_demand_case", BASE_DEMAND_CASE)).strip() or BASE_DEMAND_CASE
    )

    summary_cfg = (
        config.get("results", {}).get("summary", {})
        if isinstance(config.get("results"), Mapping)
        else {}
    )
    run_directory = get_results_run_directory(config)

    years = _ensure_years(summary_cfg)
    if not years:
        raise ValueError("results.summary.years must contain at least one reporting year.")
    year_periods = _parse_year_periods(summary_cfg)

    output_directory = _resolve_path(root, summary_cfg.get("output_directory"), run_directory)
    if output_directory is None:
        default_summary_dir = (root / "results" / "summary").resolve()
        output_directory = apply_results_run_directory(
            default_summary_dir,
            run_directory,
            repo_root=root,
        )
    include_plots = bool(summary_cfg.get("include_plots", True))
    plot_format = str(summary_cfg.get("plot_format", "png"))
    plot_start = int(summary_cfg.get("plot_start", 2025))
    plot_end = int(summary_cfg.get("plot_end", 2100))

    aggregation_mode = str(economic_cfg.get("aggregation", "per_year"))
    aggregation_horizon_cfg = economic_cfg.get("aggregation_horizon")
    aggregation_horizon: tuple[int, int] | None = None
    if isinstance(aggregation_horizon_cfg, Mapping):
        start = aggregation_horizon_cfg.get("start")
        end = aggregation_horizon_cfg.get("end")
        if start is not None and end is not None:
            aggregation_horizon = (int(start), int(end))

    # settings will be created after climate label resolution

    methods = _resolve_methods(economic_cfg.get("methods", {}))
    run_cfg = economic_cfg.get("run", {}) if isinstance(economic_cfg, Mapping) else {}
    run_method = str(run_cfg.get("method", "kernel")) if isinstance(run_cfg, Mapping) else "kernel"

    data_sources = economic_cfg.get("data_sources", {})
    if not isinstance(data_sources, Mapping):
        data_sources = {}
    emission_root = _resolve_path(root, data_sources.get("emission_root"), run_directory)
    if emission_root is None:
        emission_root = _default_emission_root(config, root, run_directory)
    if emission_root is None:
        raise ValueError("Unable to resolve emission data root directory.")
    try:
        emission_root = _resolve_directory(emission_root, "Emission data")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError(str(exc)) from exc

    temperature_root = _resolve_path(root, data_sources.get("temperature_root"), run_directory)
    if temperature_root is None:
        temperature_root = _default_temperature_root(config, root, run_directory)
    if temperature_root is None:
        raise ValueError("Unable to resolve temperature data directory.")
    try:
        temperature_root = _resolve_directory(temperature_root, "Temperature data")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError(str(exc)) from exc

    scc_output_dir = _resolve_path(root, economic_cfg.get("output_directory"), run_directory)
    if scc_output_dir is None:
        default_scc_dir = (root / "results" / "economic").resolve()
        scc_output_dir = apply_results_run_directory(
            default_scc_dir,
            run_directory,
            repo_root=root,
        )
    air_pollution_cfg = config.get("air_pollution", {})
    air_cfg = air_pollution_cfg if isinstance(air_pollution_cfg, Mapping) else {}
    air_output_dir = _resolve_path(root, air_cfg.get("output_directory"), run_directory)
    if air_output_dir is None:
        default_air_dir = (root / "results" / "air_pollution").resolve()
        air_output_dir = apply_results_run_directory(
            default_air_dir,
            run_directory,
            repo_root=root,
        )

    emission_column = str(economic_cfg.get("emission_column", "delta"))

    reference_scenario = str(economic_cfg.get("reference_scenario", "")).strip()
    evaluation = economic_cfg.get("evaluation_scenarios", [])

    def _all_emission_scenarios() -> list[str]:
        countries_cfg = (
            config.get("calc_emissions", {}).get("countries", {})
            if isinstance(config, Mapping)
            else {}
        )
        demand_cases_cfg = countries_cfg.get("demand_scenarios", [])
        if isinstance(demand_cases_cfg, Mapping):
            demand_cases = list(demand_cases_cfg.keys())
        elif isinstance(demand_cases_cfg, Iterable) and not isinstance(
            demand_cases_cfg, (str, bytes)
        ):
            demand_cases = [str(case) for case in demand_cases_cfg]
        else:
            demand_cases = []
        baseline = (
            str(countries_cfg.get("baseline_demand_case", baseline_case)).strip() or baseline_case
        )
        try:
            return list_available_scenarios(
                emission_root,
                demand_cases or [baseline_case],
                include_baseline=False,
                baseline_case=baseline,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            LOGGER.warning("Unable to enumerate emission scenarios; using empty list: %s", exc)
            return []

    include_all = False
    if isinstance(evaluation, str) and evaluation.strip().lower() == "all":
        include_all = True
    elif isinstance(evaluation, Iterable) and not isinstance(evaluation, (str, bytes)):
        eval_list = [item for item in evaluation if item is not None]
        include_all = len(eval_list) == 0
    else:
        include_all = not evaluation

    if include_all:
        evaluation_scenarios = _all_emission_scenarios()
    else:
        evaluation_scenarios = [str(label) for label in evaluation]
        evaluation_scenarios.extend(_all_emission_scenarios())
        evaluation_scenarios = sorted(set(evaluation_scenarios))

    if not evaluation_scenarios:
        raise ValueError(
            "economic_module.evaluation_scenarios must list at least one scenario or 'all'."
        )

    climate_inputs = _parse_climate_labels(data_sources.get("climate_scenarios"))
    climate_labels: list[str | None]
    if climate_inputs:
        climate_labels = climate_inputs
    else:
        derived: list[str] = []
        if reference_scenario:
            prefix = f"{reference_scenario}_"
            for path in temperature_root.glob(f"{prefix}*.csv"):
                suffix = path.stem[len(prefix) :]
                if suffix:
                    derived.append(suffix)
        climate_labels = sorted(set(derived)) or [None]

    climate_labels_str = [cl for cl in climate_labels if cl is not None]

    scc_summary_table = _load_scc_summary_table(scc_output_dir / "scc_summary.csv")

    metrics_map: dict[str, ScenarioMetrics] = {}

    for scenario in evaluation_scenarios:
        if reference_scenario and scenario == reference_scenario:
            continue
        emission_frame = _load_emission_frame_from_mix(
            emission_root, scenario, emission_column, baseline_case
        )
        emission_delta = _read_series(emission_frame, emission_column, years)
        emission_series = _load_full_series(emission_frame, emission_column)
        if not emission_series:
            emission_series_dict: dict[int, float] | None = None
        else:
            emission_series_dict = emission_series

        mortality_path = (air_output_dir / scenario / "total_mortality_summary.csv").resolve()
        (
            mortality_delta,
            mortality_percent,
            mortality_baseline,
            mortality_value_delta,
        ) = _read_mortality(mortality_path, years)
        mortality_frame = pd.read_csv(mortality_path) if mortality_path.exists() else pd.DataFrame()
        mortality_sums = _sum_over_periods(mortality_frame, "delta_deaths_per_year", year_periods)
        mortality_value_sums = _sum_over_periods(mortality_frame, "delta_value_usd", year_periods)

        for climate_label in climate_labels:
            scenario_label = f"{scenario}_{climate_label}" if climate_label else scenario
            temperature_path = (temperature_root / f"{scenario_label}.csv").resolve()
            temperature_delta = _read_temperature(temperature_path, years)
            temperature_series = _load_temperature_series(temperature_path)

            damages, scc_values, damage_totals = _read_scc_timeseries(
                scc_output_dir, scenario_label, methods, years
            )
            damage_sums = _compute_damage_period_sums(
                scc_output_dir, scenario_label, methods, year_periods
            )

            average_lookup = scc_summary_table.get(scenario_label) or scc_summary_table.get(
                _safe_name(scenario_label), {}
            )
            scc_average = dict(average_lookup)
            if aggregation_mode != "average":
                scc_average = {}

            damages_adjusted = {method: dict(series) for method, series in damages.items()}
            ramsey_key = "ramsey_discount"
            if ramsey_key in methods:
                ramsey_scc = scc_values.get(ramsey_key, {})
                computed: dict[int, float] = {}
                for year in years:
                    emission_value = emission_delta.get(year, math.nan)
                    scc_value = ramsey_scc.get(year, math.nan)
                    if any(math.isnan(value) for value in (emission_value, scc_value)):
                        computed[year] = math.nan
                    else:
                        computed[year] = emission_value * emission_to_tonnes * scc_value
                damages_adjusted[ramsey_key] = computed

            scc_values_copy = {method: dict(series) for method, series in scc_values.items()}

            metrics_map[scenario_label] = ScenarioMetrics(
                emission_delta_mt=emission_delta,
                temperature_delta_c=temperature_delta,
                mortality_delta=mortality_delta,
                mortality_percent=mortality_percent,
                mortality_baseline=mortality_baseline,
                mortality_value_delta=mortality_value_delta,
                mortality_sum=mortality_sums,
                mortality_value_sum=mortality_value_sums,
                scc_usd_per_tco2=scc_values_copy,
                damages_usd=damages_adjusted,
                damages_sum=damage_sums,
                scc_average=scc_average,
                damage_total_usd=damage_totals,
                emission_timeseries=emission_series_dict,
                temperature_timeseries=temperature_series,
            )

    gdp_series_map, population_series_map = _collect_socio_series(
        root,
        config,
        metrics_map.keys(),
        climate_labels_str,
        run_directory,
        plot_end,
    )
    pattern_cfg = config.get("pattern_scaling", {}) if isinstance(config, Mapping) else {}
    pattern_output_directory: Path | None = None
    pattern_countries: list[str] | None = None
    if isinstance(pattern_cfg, Mapping):
        pattern_output = pattern_cfg.get("output_directory")
        if pattern_output:
            candidate = _resolve_path(root, pattern_output, run_directory)
            try:
                pattern_output_directory = _resolve_directory(candidate, "Pattern-scaling output")
            except (FileNotFoundError, NotADirectoryError):
                pattern_output_directory = None
        countries_cfg = pattern_cfg.get("countries")
        if isinstance(countries_cfg, Iterable) and not isinstance(countries_cfg, (str, bytes)):
            pattern_countries = [str(c) for c in countries_cfg if c]

    # Construct summary settings after climate label resolution
    settings = SummarySettings(
        years=years,
        output_directory=output_directory,
        include_plots=include_plots,
        plot_format=plot_format,
        aggregation_mode=aggregation_mode,
        aggregation_horizon=aggregation_horizon,
        plot_start=plot_start,
        plot_end=plot_end,
        climate_labels=climate_labels_str,
        run_method=run_method,
        scc_output_directory=scc_output_dir,
        base_year=base_year_value,
        climate_output_directory=temperature_root,
        gdp_series=gdp_series_map,
        population_series=population_series_map,
        pattern_output_directory=pattern_output_directory,
        pattern_countries=pattern_countries,
        year_periods=year_periods,
    )
    settings.per_country_emission_root = _default_per_country_root(config, root, run_directory)

    if not metrics_map:
        LOGGER.warning(
            "No metrics collected for summary; check input directories and configuration."
        )

    return settings, methods, metrics_map


def write_summary_csv(
    settings: SummarySettings,
    methods: Iterable[str],
    metrics_map: Mapping[str, ScenarioMetrics],
    *,
    output_path: Path | None = None,
) -> Path:
    summary_dir = output_path or settings.output_directory
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "summary.csv"

    rows: list[dict[str, object]] = []
    per_country_root = settings.per_country_emission_root
    pattern_dir = settings.pattern_output_directory
    pattern_countries = settings.pattern_countries or []
    cached_country_dirs: dict[str, list[Path]] = {}

    def _country_dirs(mix_case: str) -> list[Path]:
        if per_country_root is None:
            return []
        if mix_case in cached_country_dirs:
            return cached_country_dirs[mix_case]
        mix_dir = per_country_root / mix_case
        if not mix_dir.exists():
            cached_country_dirs[mix_case] = []
            return []
        dirs = sorted(child for child in mix_dir.iterdir() if child.is_dir())
        cached_country_dirs[mix_case] = dirs
        return dirs

    for scenario in sorted(metrics_map):
        metrics = metrics_map[scenario]
        base_label, climate_label = _split_climate_suffix(scenario, settings.climate_labels)
        try:
            mix_case, demand_case = split_scenario_name(base_label)
        except ValueError:
            mix_case, demand_case = base_label, BASE_DEMAND_CASE
        row: dict[str, object] = {
            "energy_mix": mix_case,
            "climate_scenario": climate_label or "",
            "demand_case": demand_case,
        }
        for year in settings.years:
            row[f"delta_co2_Mt_all_countries_{year}"] = metrics.emission_delta_mt.get(
                year, math.nan
            )
            row[f"delta_T_C_{year}"] = metrics.temperature_delta_c.get(year, math.nan)
            row[f"air_pollution_mortality_difference_{year}"] = metrics.mortality_delta.get(
                year, math.nan
            )
            row[f"air_pollution_mortality_percent_change_{year}"] = metrics.mortality_percent.get(
                year, math.nan
            )
            if metrics.mortality_value_delta is not None:
                row[f"air_pollution_monetary_benefit_usd_{year}"] = (
                    metrics.mortality_value_delta.get(year, math.nan)
                )
        for label, value in metrics.mortality_sum.items():
            row[f"air_pollution_mortality_difference_sum_{label}"] = value
        for label, value in metrics.mortality_value_sum.items():
            row[f"air_pollution_monetary_benefit_sum_usd_{label}"] = value
        if settings.aggregation_mode == "average":
            for method in methods:
                row[f"scc_average_{method}"] = metrics.scc_average.get(method, math.nan)
        else:
            for method in methods:
                for year in settings.years:
                    key = f"SCC_{method}_{year}_usd_per_tco2"
                    value = metrics.scc_usd_per_tco2.get(method, {}).get(year, math.nan)
                    row[key] = value
        for method in methods:
            series = metrics.damages_usd.get(method, {})
            for year in settings.years:
                row[f"damages_PPP2020_usd_baseyear_{settings.base_year}_{method}_{year}"] = (
                    series.get(year, math.nan)
                )
            for label, value in metrics.damages_sum.get(method, {}).items():
                row[f"damages_PPP2020_usd_baseyear_{settings.base_year}_sum_{method}_{label}"] = (
                    value
                )

        if per_country_root is not None:
            for country_dir in _country_dirs(mix_case):
                country = country_dir.name
                for pollutant_file in sorted(country_dir.glob("*.csv")):
                    pollutant = pollutant_file.stem
                    try:
                        df = pd.read_csv(pollutant_file, comment="#")
                    except FileNotFoundError:
                        continue
                    delta_col = f"delta_{demand_case}"
                    if delta_col not in df.columns or "year" not in df.columns:
                        continue
                    for year in settings.years:
                        matches = df.loc[df["year"] == year, delta_col]
                        if matches.empty:
                            continue
                        row[f"delta_{pollutant}_{country}_{year}"] = float(matches.iloc[0])

        if pattern_dir is not None and climate_label:
            for iso3 in pattern_countries:
                pattern_file = pattern_dir / f"{iso3}_{base_label}_{climate_label}.csv"
                if not pattern_file.exists():
                    continue
                try:
                    pdf = pd.read_csv(pattern_file)
                except Exception:
                    continue
                if "year" not in pdf.columns or "temperature_delta" not in pdf.columns:
                    continue
                for year in settings.years:
                    match = pdf.loc[pdf["year"] == year, "temperature_delta"]
                    if match.empty:
                        continue
                    row[f"delta_T_{iso3}_{year}"] = float(match.iloc[0])

        rows.append(row)

    df = pd.DataFrame(rows)
    base_columns = ["energy_mix", "climate_scenario", "demand_case"]
    ordered: list[str] = []
    for year in settings.years:
        ordered.append(f"delta_co2_Mt_all_countries_{year}")
        ordered.append(f"delta_T_C_{year}")
        for method in methods:
            ordered.append(f"SCC_{method}_{year}_usd_per_tco2")
        for method in methods:
            ordered.append(f"damages_PPP2020_usd_baseyear_{settings.base_year}_{method}_{year}")
        ordered.append(f"air_pollution_mortality_difference_{year}")
        ordered.append(f"air_pollution_mortality_percent_change_{year}")
        ordered.append(f"air_pollution_monetary_benefit_usd_{year}")
    if metrics_map:
        sample_metrics = next(iter(metrics_map.values()))
        for label in sample_metrics.mortality_sum:
            ordered.append(f"air_pollution_mortality_difference_sum_{label}")
        for label in sample_metrics.mortality_value_sum:
            ordered.append(f"air_pollution_monetary_benefit_sum_usd_{label}")
        for method in methods:
            for label in sample_metrics.damages_sum.get(method, {}):
                ordered.append(
                    f"damages_PPP2020_usd_baseyear_{settings.base_year}_sum_{method}_{label}"
                )
    other_columns = [col for col in df.columns if col not in base_columns + ordered]
    final_columns = (
        base_columns + [col for col in ordered if col in df.columns] + sorted(other_columns)
    )
    df = df[final_columns]
    df.to_csv(csv_path, index=False)
    return csv_path


def _pivot_metric(metrics_map: Mapping[str, ScenarioMetrics], attr: str, method: str | None = None):
    data = {}
    for scenario, metrics in metrics_map.items():
        metric_value = getattr(metrics, attr)
        if method is not None:
            metric_value = metric_value.get(method, {})
        data[scenario] = metric_value
    return data


def _plot_grouped_bars(
    data: Mapping[str, Mapping[int, float]],
    years: list[int],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    file_suffix: str,
    plot_format: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("matplotlib not available; skipping plots for %s", file_suffix)
        return

    scenarios = list(data.keys())
    x = np.arange(len(scenarios))
    if len(years) == 0:
        return
    width = 0.8 / max(len(years), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 1.5), 4.5))
    for idx, year in enumerate(years):
        values = [data[scenario].get(year, np.nan) for scenario in scenarios]
        offset = (idx - (len(years) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=str(year))
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path / f"{file_suffix}.{plot_format}", format=plot_format)
    plt.close(fig)


def _plot_emission_timeseries(
    data: Mapping[str, Mapping[int, float]],
    *,
    output_path: Path,
    file_name: str,
    plot_format: str,
    window: tuple[int, int],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("matplotlib not available; skipping emission timeseries plot")
        return

    valid_series = {scenario: series for scenario, series in data.items() if series}
    if not valid_series:
        LOGGER.info("No emission timeseries data available; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    start_year, end_year = window
    for scenario, series in valid_series.items():
        years_sorted = sorted(year for year in series if start_year <= year <= end_year)
        if not years_sorted:
            continue
        values = [series[year] for year in years_sorted]
        ax.plot(years_sorted, values, marker="o", label=scenario)

    ax.set_xlabel("Year")
    ax.set_ylabel("Emission delta (Mt CO₂)")
    ax.set_title("Emission Delta Over Time")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    output_path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path / f"{file_name}.{plot_format}", format=plot_format)
    plt.close(fig)


def _plot_temperature_timeseries(
    data: Mapping[str, Mapping[int, float] | None],
    *,
    output_path: Path,
    file_name: str,
    plot_format: str,
    window: tuple[int, int],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("matplotlib not available; skipping temperature timeseries plot")
        return

    valid_series = {
        scenario: series
        for scenario, series in data.items()
        if series is not None and len(series) > 0
    }
    if not valid_series:
        LOGGER.info("No temperature timeseries data available; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    start_year, end_year = window
    for scenario, series in valid_series.items():
        years_sorted = sorted(year for year in series if start_year <= year <= end_year)
        if not years_sorted:
            continue
        values = [series[year] for year in years_sorted]
        ax.plot(years_sorted, values, marker="o", label=scenario)

    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature delta (°C)")
    ax.set_title("Temperature Delta Over Time")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    output_path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path / f"{file_name}.{plot_format}", format=plot_format)
    plt.close(fig)


def _plot_scc_timeseries(
    settings: SummarySettings,
    methods: Iterable[str],
    metrics_map: Mapping[str, ScenarioMetrics],
) -> None:
    if settings.run_method != "pulse":
        return
    if settings.scc_output_directory is None:
        LOGGER.info("SCC output directory not available; skipping SCC timeseries plot.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("matplotlib not available; skipping SCC timeseries plot")
        return

    output_dir = settings.output_directory / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        curves: list[tuple[list[int], list[float], str]] = []
        plotted_labels: set[str] = set()
        for scenario in sorted(metrics_map):
            safe_name = _safe_name(scenario)
            path = settings.scc_output_directory / f"scc_timeseries_{method}_{safe_name}.csv"
            if not path.exists():
                continue

            if settings.climate_labels:
                _, climate_label = _split_climate_suffix(scenario, settings.climate_labels)
                legend_label = climate_label or scenario
            else:
                legend_label = scenario

            if legend_label in plotted_labels:
                continue

            df = pd.read_csv(path)
            if "year" not in df.columns or "scc_usd_per_tco2" not in df.columns:
                LOGGER.debug("SCC timeseries %s missing required columns", path)
                continue
            df = df.dropna(subset=["scc_usd_per_tco2"])
            if df.empty:
                continue
            df = df.sort_values("year")
            mask = (df["year"] >= settings.plot_start) & (df["year"] <= settings.plot_end)
            df = df.loc[mask]
            if df.empty:
                continue
            years = df["year"].astype(int).tolist()
            values = df["scc_usd_per_tco2"].astype(float).tolist()
            curves.append((years, values, legend_label))
            plotted_labels.add(legend_label)

        if not curves:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for years, values, label in curves:
            ax.plot(years, values, marker="o", linewidth=1.6, label=label)
        ax.set_xlabel("Year")
        ax.set_ylabel("SCC (USD/tCO₂)")
        ax.set_title(f"SCC Timeseries ({method})")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"scc_timeseries_{method}.{settings.plot_format}",
            format=settings.plot_format,
        )
        plt.close(fig)


def write_plots(
    settings: SummarySettings,
    methods: Iterable[str],
    metrics_map: Mapping[str, ScenarioMetrics],
) -> None:
    if not settings.include_plots:
        LOGGER.info("Plot generation disabled via configuration.")
        return

    years = settings.years
    output_dir = settings.output_directory / "plots"

    # Emission delta (Mt CO2)
    emission_data = _pivot_metric(metrics_map, "emission_delta_mt")
    _plot_grouped_bars(
        emission_data,
        years,
        title="Emission Delta (Mt CO₂)",
        ylabel="Mt CO₂",
        output_path=output_dir,
        file_suffix="emission_delta",
        plot_format=settings.plot_format,
    )

    # Temperature delta (°C)
    temp_data = _pivot_metric(metrics_map, "temperature_delta_c")
    _plot_grouped_bars(
        temp_data,
        years,
        title="Temperature Delta (°C)",
        ylabel="°C",
        output_path=output_dir,
        file_suffix="temperature_delta",
        plot_format=settings.plot_format,
    )

    # Mortality delta (deaths/year)
    mortality_data = _pivot_metric(metrics_map, "mortality_delta")
    _plot_grouped_bars(
        mortality_data,
        years,
        title="Mortality Delta (deaths/year)",
        ylabel="deaths/year",
        output_path=output_dir,
        file_suffix="mortality_delta",
        plot_format=settings.plot_format,
    )

    # Mortality percent (%)
    mortality_percent_data = {
        scenario: {
            year: value * 100 if not math.isnan(value) else value for year, value in values.items()
        }
        for scenario, values in _pivot_metric(metrics_map, "mortality_percent").items()
    }
    _plot_grouped_bars(
        mortality_percent_data,
        years,
        title="Mortality Change (%)",
        ylabel="Percent change",
        output_path=output_dir,
        file_suffix="mortality_percent",
        plot_format=settings.plot_format,
    )

    # Mortality value (USD/year)
    mortality_value_data = {
        scenario: values
        for scenario, values in _pivot_metric(metrics_map, "mortality_value_delta").items()
        if values
    }
    if mortality_value_data:
        mortality_value_scaled = {
            scenario: {
                year: value / 1e9 if not math.isnan(value) else value
                for year, value in series.items()
            }
            for scenario, series in mortality_value_data.items()
        }
        _plot_grouped_bars(
            mortality_value_scaled,
            years,
            title="Mortality Value (Billion USD/year)",
            ylabel="Billion USD/year",
            output_path=output_dir,
            file_suffix="mortality_value",
            plot_format=settings.plot_format,
        )

    for method in methods:
        damages_data = _pivot_metric(metrics_map, "damages_usd", method)
        damages_scaled = {
            scenario: {
                year: value / 1e9 if not math.isnan(value) else value
                for year, value in values.items()
            }
            for scenario, values in damages_data.items()
        }
        _plot_grouped_bars(
            damages_scaled,
            years,
            title=f"Climate Damages ({method}) [Billion USD]",
            ylabel="Billion USD",
            output_path=output_dir,
            file_suffix=f"damages_{method}",
            plot_format=settings.plot_format,
        )

        scc_data = _pivot_metric(metrics_map, "scc_usd_per_tco2", method)
        _plot_grouped_bars(
            scc_data,
            years,
            title=f"SCC ({method}) [USD/tCO₂]",
            ylabel="USD/tCO₂",
            output_path=output_dir,
            file_suffix=f"scc_{method}",
            plot_format=settings.plot_format,
        )

    emission_ts = {
        scenario: metrics.emission_timeseries or {}
        for scenario, metrics in metrics_map.items()
        if metrics.emission_timeseries is not None
    }
    _plot_emission_timeseries(
        emission_ts,
        output_path=output_dir,
        file_name="emission_delta_timeseries",
        plot_format=settings.plot_format,
        window=(settings.plot_start, settings.plot_end),
    )

    temperature_ts = {
        scenario: metrics.temperature_timeseries for scenario, metrics in metrics_map.items()
    }
    _plot_temperature_timeseries(
        temperature_ts,
        output_path=output_dir,
        file_name="temperature_delta_timeseries",
        plot_format=settings.plot_format,
        window=(settings.plot_start, settings.plot_end),
    )

    _plot_scc_timeseries(settings, methods, metrics_map)
    _plot_socioeconomic_timeseries(settings)
    _include_background_plots(settings, output_dir)


def _include_background_plots(settings: SummarySettings, plots_dir: Path) -> None:
    climate_dir = settings.climate_output_directory
    if climate_dir is None:
        return
    for stem in ("background_climate_full", "background_climate_horizon"):
        source = climate_dir / f"{stem}.png"
        if not source.exists():
            continue
        destination = plots_dir / source.name
        try:
            shutil.copyfile(source, destination)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            LOGGER.warning("Unable to copy %s into summary plots: %s", source, exc)


def _plot_socioeconomic_timeseries(settings: SummarySettings) -> None:
    if not settings.gdp_series:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("matplotlib not available; skipping socioeconomics plot")
        return

    plots_dir = settings.output_directory / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(settings.plot_start, settings.plot_end + 1))
    if not years:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    gdp_series = _collapse_by_climate_label(settings.gdp_series, settings.climate_labels)
    pop_series = (
        _collapse_by_climate_label(settings.population_series, settings.climate_labels)
        if settings.population_series
        else None
    )

    scenarios = sorted(gdp_series.keys())
    for scenario in scenarios:
        series = gdp_series[scenario]
        values = [series.get(year, math.nan) for year in years]
        axes[0].plot(years, values, label=scenario)
    axes[0].set_ylabel("GDP (trillion USD)")
    axes[0].set_title("Socioeconomic trajectories")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    if pop_series:
        for scenario in sorted(pop_series.keys()):
            series = pop_series.get(scenario)
            if not series:
                continue
            values = [series.get(year, math.nan) for year in years]
            axes[1].plot(years, values, label=scenario)
        axes[1].set_ylabel("Population (million)")
        axes[1].legend(loc="best", fontsize=8)
    else:
        axes[1].set_visible(False)
    axes[-1].set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(
        plots_dir / f"socioeconomics.{settings.plot_format}",
        format=settings.plot_format,
    )
    plt.close(fig)


def _parse_year_periods(summary_cfg: Mapping[str, object]) -> list[tuple[int, int]]:
    periods_cfg = summary_cfg.get("year_period", []) if isinstance(summary_cfg, Mapping) else []
    periods: list[tuple[int, int]] = []
    if isinstance(periods_cfg, Mapping):
        periods_cfg = [periods_cfg]
    if isinstance(periods_cfg, Iterable) and not isinstance(periods_cfg, (str, bytes)):
        for entry in periods_cfg:
            if not isinstance(entry, Mapping):
                continue
            try:
                start = int(entry.get("start"))
                end = int(entry.get("end"))
            except Exception:
                continue
            if start >= end:
                continue
            periods.append((start, end))
    return periods

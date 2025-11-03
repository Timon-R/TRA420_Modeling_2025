"""Generate configurable cross-module summaries (metrics + plots)."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config_paths import apply_results_run_directory, get_results_run_directory

LOGGER = logging.getLogger("results.summary")

AVAILABLE_METHODS: tuple[str, ...] = ("constant_discount", "ramsey_discount")


@dataclass(slots=True)
class SummarySettings:
    years: list[int]
    output_directory: Path
    include_plots: bool = True
    plot_format: str = "png"
    aggregation_mode: str = "per_year"
    aggregation_horizon: tuple[int, int] | None = None
    plot_start: int = 2025
    plot_end: int = 2100
    climate_labels: list[str] = field(default_factory=list)
    run_method: str = "kernel"
    scc_output_directory: Path | None = None
    base_year: int = 2025


@dataclass(slots=True)
class ScenarioMetrics:
    emission_delta_mt: dict[int, float] = field(default_factory=dict)
    temperature_delta_c: dict[int, float] = field(default_factory=dict)
    mortality_delta: dict[int, float] = field(default_factory=dict)
    mortality_percent: dict[int, float] = field(default_factory=dict)
    mortality_baseline: dict[int, float] = field(default_factory=dict)
    scc_usd_per_tco2: dict[str, dict[int, float]] = field(default_factory=dict)
    damages_usd: dict[str, dict[int, float]] = field(default_factory=dict)
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


def _load_full_series(path: Path, value_column: str) -> dict[int, float]:
    if not path.exists():
        LOGGER.debug("Data file missing for full series load: %s", path)
        return {}
    frame = pd.read_csv(path)
    if "year" not in frame.columns or value_column not in frame.columns:
        LOGGER.warning(
            "File %s missing required columns 'year'/'%s' for full series load", path, value_column
        )
        return {}
    frame["year"] = frame["year"].astype(int)
    series = frame.set_index("year")[value_column].astype(float)
    return {int(year): float(series.get(int(year))) for year in series.index}


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


def _read_series(
    path: Path,
    value_column: str,
    years: Iterable[int],
    *,
    transform: callable | None = None,
) -> dict[int, float]:
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
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
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
    if not expected.issubset(frame.columns):
        LOGGER.warning("Mortality file %s missing required columns", path)
        nan_map = {year: math.nan for year in years}
        return nan_map, nan_map.copy(), nan_map.copy()
    frame["year"] = frame["year"].astype(int)
    frame = frame.set_index("year")
    delta = frame["delta_deaths_per_year"].astype(float)
    percent = frame["percent_change_mortality"].astype(float)
    baseline = frame["baseline_deaths_per_year"].astype(float)
    return (
        {int(year): float(delta.get(int(year), np.nan)) for year in years},
        {int(year): float(percent.get(int(year), np.nan)) for year in years},
        {int(year): float(baseline.get(int(year), np.nan)) for year in years},
    )


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
        path = output_dir / f"scc_timeseries_{method}_{scenario_key}.csv"
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

    summary_cfg = (
        config.get("results", {}).get("summary", {})
        if isinstance(config.get("results"), Mapping)
        else {}
    )
    run_directory = get_results_run_directory(config)

    years = _ensure_years(summary_cfg)
    if not years:
        raise ValueError("results.summary.years must contain at least one reporting year.")

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
    if not evaluation:
        raise ValueError("economic_module.evaluation_scenarios must list at least one scenario.")
    evaluation_scenarios = [str(label) for label in evaluation]

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
        emission_path = (emission_root / scenario / "co2.csv").resolve()
        emission_delta = _read_series(emission_path, emission_column, years)
        emission_series = _load_full_series(emission_path, emission_column)
        if not emission_series:
            emission_series_dict: dict[int, float] | None = None
        else:
            emission_series_dict = emission_series

        mortality_path = (air_output_dir / scenario / "total_mortality_summary.csv").resolve()
        mortality_delta, mortality_percent, mortality_baseline = _read_mortality(
            mortality_path, years
        )

        for climate_label in climate_labels:
            scenario_label = f"{scenario}_{climate_label}" if climate_label else scenario
            temperature_path = (temperature_root / f"{scenario_label}.csv").resolve()
            temperature_delta = _read_temperature(temperature_path, years)
            temperature_series = _load_temperature_series(temperature_path)

            damages, scc_values, damage_totals = _read_scc_timeseries(
                scc_output_dir, scenario_label, methods, years
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
                scc_usd_per_tco2=scc_values_copy,
                damages_usd=damages_adjusted,
                scc_average=scc_average,
                damage_total_usd=damage_totals,
                emission_timeseries=emission_series_dict,
                temperature_timeseries=temperature_series,
            )

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
    )

    if not metrics_map:
        LOGGER.warning(
            "No metrics collected for summary; check input directories and configuration."
        )

    return settings, methods, metrics_map


def write_summary_json(
    settings: SummarySettings,
    methods: Iterable[str],
    metrics_map: Mapping[str, ScenarioMetrics],
    *,
    output_path: Path | None = None,
) -> Path:
    summary_dir = output_path or settings.output_directory
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "years": settings.years,
        "methods": list(methods),
        "scenarios": {},
    }
    for scenario, metrics in metrics_map.items():
        payload["scenarios"][scenario] = {
            "emission_delta_mt": metrics.emission_delta_mt,
            "temperature_delta_c": metrics.temperature_delta_c,
            "mortality_delta": metrics.mortality_delta,
            "mortality_percent": metrics.mortality_percent,
            "mortality_baseline": metrics.mortality_baseline,
            "scc_usd_per_tco2": metrics.scc_usd_per_tco2,
            "damages_usd": metrics.damages_usd,
            "scc_average": metrics.scc_average,
            "damage_total_usd": metrics.damage_total_usd,
            "emission_timeseries": metrics.emission_timeseries,
            "temperature_timeseries": metrics.temperature_timeseries,
        }
    json_path = summary_dir / "summary.json"
    json_path.write_text(json.dumps(payload, indent=2))
    return json_path


def write_summary_text(
    settings: SummarySettings,
    methods: Iterable[str],
    metrics_map: Mapping[str, ScenarioMetrics],
    *,
    output_path: Path | None = None,
) -> Path:
    output_dir = output_path or settings.output_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "summary.txt"

    def _format_value(value: object, precision: int = 2) -> str:
        if value is None:
            return "NaN"
        if isinstance(value, float) and math.isnan(value):
            return "NaN"
        if isinstance(value, (int, float)):
            return f"{value:,.{precision}f}"
        return str(value)

    lines: list[str] = []
    lines.append("Summary Overview")
    lines.append("=" * len("Summary Overview"))
    lines.append("")
    lines.append(
        f"Damages are per reporting year and expressed as present-value {settings.base_year} USD."
    )
    lines.append("")

    for scenario in sorted(metrics_map):
        metrics = metrics_map[scenario]
        lines.append(f"Scenario: {scenario}")

        lines.append("  Emission delta (Mt CO₂):")
        for year in settings.years:
            value = metrics.emission_delta_mt.get(year, math.nan)
            lines.append(f"    {year}: {_format_value(value)}")

        lines.append("  Temperature delta (°C):")
        for year in settings.years:
            value = metrics.temperature_delta_c.get(year, math.nan)
            lines.append(f"    {year}: {_format_value(value)}")

        lines.append("  Mortality change (deaths/year):")
        for year in settings.years:
            value = metrics.mortality_delta.get(year, math.nan)
            lines.append(f"    {year}: {_format_value(value, precision=1)}")

        if settings.aggregation_mode == "average":
            horizon_text = ""
            if settings.aggregation_horizon is not None:
                start, end = settings.aggregation_horizon
                horizon_text = f" ({start}-{end})"
            lines.append(f"  SCC average{horizon_text}:")
            for method in methods:
                value = metrics.scc_average.get(method, math.nan)
                lines.append(f"    {method}: {_format_value(value)} USD/tCO₂")
        else:
            lines.append("  SCC (USD/tCO₂):")
            for method in methods:
                lines.append(f"    {method}:")
                series = metrics.scc_usd_per_tco2.get(method, {})
                for year in settings.years:
                    value = series.get(year, math.nan)
                    lines.append(f"      {year}: {_format_value(value)}")

        lines.append(f"  Damages (Billion USD, PV {settings.base_year}):")
        for method in methods:
            lines.append(f"    {method}:")
            series = metrics.damages_usd.get(method, {})
            for year in settings.years:
                value = series.get(year, math.nan)
                if isinstance(value, float) and math.isnan(value):
                    lines.append(f"      {year}: NaN")
                else:
                    lines.append(f"      {year}: {_format_value(value / 1e9)}")

        lines.append("")

    text_path.write_text("\n".join(lines).rstrip() + "\n")
    return text_path


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
    # Collapse by base scenario (strip climate suffix), since emissions are
    # identical across SSPs for a given policy scenario.
    if settings.climate_labels:
        collapsed: dict[str, Mapping[int, float]] = {}
        for scenario in sorted(emission_data.keys()):
            base, _ = _split_climate_suffix(scenario, settings.climate_labels)
            if base not in collapsed:
                collapsed[base] = emission_data[scenario]
        emission_data = collapsed
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
    if settings.climate_labels:
        collapsed_mortality: dict[str, Mapping[int, float]] = {}
        for scenario in sorted(mortality_data.keys()):
            base, _ = _split_climate_suffix(scenario, settings.climate_labels)
            if base not in collapsed_mortality:
                collapsed_mortality[base] = mortality_data[scenario]
        mortality_data = collapsed_mortality
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
    if settings.climate_labels:
        collapsed_percent: dict[str, Mapping[int, float]] = {}
        for scenario in sorted(mortality_percent_data.keys()):
            base, _ = _split_climate_suffix(scenario, settings.climate_labels)
            if base not in collapsed_percent:
                collapsed_percent[base] = mortality_percent_data[scenario]
        mortality_percent_data = collapsed_percent
    _plot_grouped_bars(
        mortality_percent_data,
        years,
        title="Mortality Change (%)",
        ylabel="Percent change",
        output_path=output_dir,
        file_suffix="mortality_percent",
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
    if settings.climate_labels:
        collapsed_ts: dict[str, Mapping[int, float]] = {}
        for scenario, series in emission_ts.items():
            base, _ = _split_climate_suffix(scenario, settings.climate_labels)
            if base not in collapsed_ts:
                collapsed_ts[base] = series
        emission_ts = collapsed_ts
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

"""Compute the social cost of carbon using precomputed temperature and emission scenarios.

Usage
-----
```bash
python scripts/run_scc.py --method constant_discount \
    --temperature baseline=results/climate/baseline_t.csv \
    --temperature policy=results/climate/policy_t.csv \
    --emission baseline=data/baseline_e.csv --emission policy=data/policy_e.csv

python scripts/run_scc.py --method ramsey_discount --rho 0.01 --eta 1.5 --aggregation per_year
python scripts/run_scc.py --method all
```

Inputs
------
- Temperature CSVs must expose ``year`` and the column named via
  ``--temperature-column`` (default ``temperature_adjusted``).
- Emission CSVs supply annual CO₂ deltas with ``year`` plus
  ``--emission-column`` (default ``delta``); values are converted to tonnes
  using ``--emission-unit-multiplier``.
- Temperature CSVs exported by the climate module now include
  ``climate_scenario`` to signal which SSP family should drive GDP/population
  selection.
- Scenario labels supplied for temperatures and emissions must match exactly.
- Aggregation can be ``average`` (policy-wide USD/tCO₂) or ``per_year`` (time-varying SCC series).
- Damage function parameters (DICE ``delta1`` / ``delta2`` and optional threshold, saturation,
  and catastrophe extensions) can be set in ``config.yaml`` or overridden via the
  ``--damage-*`` CLI options.

The script loads GDP, temperature, and emission difference series, feeds them into
:mod:`economic_module`, and writes per-method detail tables plus SCC time paths to
``results/economic``. Methods can be configured via ``config.yaml`` under the
``economic_module`` section or overridden through CLI options.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Mapping, cast

import numpy as np
import pandas as pd
import yaml

from calc_emissions import BASE_DEMAND_CASE
from calc_emissions.scenario_io import (
    load_scenario_absolute,
    load_scenario_delta,
    split_scenario_name,
)
from config_paths import (
    apply_results_run_directory,
    get_config_path,
    get_results_run_directory,
)
from economic_module import EconomicInputs, SCCAggregation, SCCResult, compute_scc
from economic_module.socioeconomics import DiceSocioeconomics

ROOT = Path(__file__).resolve().parents[1]


def _discover_emission_scenarios(emission_root: Path, baseline_case: str) -> list[str]:
    """Infer available emission scenario names from aggregated emission files."""
    scenarios: set[str] = set()
    if not emission_root.exists():
        return []
    for mix_dir in emission_root.iterdir():
        if not mix_dir.is_dir():
            continue
        co2_path = mix_dir / "co2.csv"
        if not co2_path.exists():
            continue
        try:
            df = pd.read_csv(co2_path, comment="#")
        except Exception:
            continue
        demand_cases: set[str] = set()
        for col in df.columns:
            if col.startswith("delta_"):
                demand_cases.add(col.removeprefix("delta_"))
            if col.startswith("absolute_"):
                demand_cases.add(col.removeprefix("absolute_"))
        for demand in demand_cases:
            scenarios.add(f"{mix_dir.name}__{demand}")
    if baseline_case:
        scenarios.add(baseline_case)
    return sorted(scenarios)


AVAILABLE_DISCOUNT_METHODS = ["constant_discount", "ramsey_discount"]
AVAILABLE_AGGREGATIONS: tuple[SCCAggregation, ...] = ("average", "per_year")

RUN_DIRECTORY: str | None = None


def _format_path(path: Path) -> Path:
    abs_path = path.resolve()
    try:
        return abs_path.relative_to(ROOT)
    except ValueError:  # pragma: no cover - filesystem safety
        return abs_path


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    return cleaned.strip("_") or "scenario"


def _write_scc_outputs(result: SCCResult, method: str, climate_key: str, output_dir: Path) -> None:
    """Export pulse SCC timeseries for a given climate pathway/method."""
    safe_climate = _safe_name(climate_key)
    per_year = result.per_year.copy()
    required = [
        "year",
        "discount_factor",
        "discounted_damage_attributed_usd",
        "scc_usd_per_tco2",
    ]
    missing = [col for col in required if col not in per_year.columns]
    if missing:
        raise KeyError(f"Per-year table missing required columns: {missing}")
    frame = pd.DataFrame(
        {
            "year": per_year["year"].astype(int),
            "discount_factor": per_year["discount_factor"].astype(float),
            "pv_damage_per_pulse_usd": per_year["discounted_damage_attributed_usd"].astype(float),
            "scc_usd_per_tco2": per_year["scc_usd_per_tco2"].astype(float),
        }
    )
    if "pulse_size_tco2" in per_year.columns:
        frame["pulse_size_tco2"] = per_year["pulse_size_tco2"].astype(float)
    else:
        frame["pulse_size_tco2"] = np.nan
    if "delta_emissions_tco2" in per_year.columns:
        frame["delta_emissions_tco2"] = per_year["delta_emissions_tco2"].astype(float)
    else:
        frame["delta_emissions_tco2"] = np.nan
    if "discounted_delta_usd" in per_year.columns:
        frame["discounted_delta_usd"] = per_year["discounted_delta_usd"].astype(float)
    else:
        frame["discounted_delta_usd"] = np.nan
    pulse_path = output_dir / f"pulse_scc_timeseries_{method}_{safe_climate}.csv"
    frame.to_csv(pulse_path, index=False)


def _build_damage_table(
    scenario_label: str,
    scc_result: SCCResult,
    emission_to_tonnes: float,
) -> pd.DataFrame:
    per_year = scc_result.per_year.copy()
    years = per_year["year"].astype(int)
    delta_tonnes = per_year["delta_emissions_tco2"].astype(float)
    scc_series = per_year["scc_usd_per_tco2"].astype(float)
    discount_series = per_year.get("discount_factor")
    if discount_series is not None:
        discount = discount_series.astype(float)
    else:
        discount = pd.Series(np.nan, index=per_year.index)

    divisor = emission_to_tonnes if emission_to_tonnes not in (0, None) else 1.0
    damage_df = pd.DataFrame(
        {
            "year": years,
            "delta_emissions_mtco2": delta_tonnes / divisor,
            "delta_emissions_tco2": delta_tonnes,
            "scc_usd_per_tco2": scc_series,
            "discount_factor": discount,
        }
    )
    damage_df["damage_usd"] = damage_df["delta_emissions_tco2"] * damage_df["scc_usd_per_tco2"]
    damage_df["discounted_damage_usd"] = damage_df["damage_usd"] * damage_df[
        "discount_factor"
    ].fillna(0.0)

    details = scc_result.details
    if "consumption_per_capita_usd" in details.columns:
        consumption_lookup = dict(
            zip(details["year"], details["consumption_per_capita_usd"], strict=False)
        )
        damage_df["consumption_per_capita_usd"] = damage_df["year"].map(consumption_lookup)
    if "consumption_growth" in details.columns:
        growth_lookup = dict(zip(details["year"], details["consumption_growth"], strict=False))
        damage_df["consumption_growth"] = damage_df["year"].map(growth_lookup)
    return damage_df


def _resolve_path(path_like: str | Path, *, apply_run_directory: bool = True) -> Path:
    path = Path(path_like)
    absolute = path if path.is_absolute() else (ROOT / path)
    absolute = absolute.resolve()
    if apply_run_directory:
        absolute = apply_results_run_directory(
            absolute,
            RUN_DIRECTORY,
            repo_root=ROOT,
        )
    return absolute


def _load_emission_frame_from_mix(
    root: Path,
    scenario: str,
    column: str,
    baseline_case: str,
) -> pd.DataFrame:
    if column == "delta":
        return load_scenario_delta(root, scenario, baseline_case=baseline_case)
    if column == "absolute":
        return load_scenario_absolute(root, scenario)
    raise ValueError(
        f"Unsupported emission column '{column}'. Use 'delta' or 'absolute' for mix-based data."
    )


def _max_required_year(time_cfg: Mapping[str, object], economic_cfg: Mapping[str, object]) -> int:
    start = int(time_cfg.get("start", economic_cfg.get("base_year", 2025)))
    end_candidates: list[int] = [int(time_cfg.get("end", start))]
    damage_duration = economic_cfg.get("damage_duration_years")
    if isinstance(damage_duration, (int, float)):
        end_candidates.append(start + int(damage_duration) - 1)
    agg_cfg = economic_cfg.get("aggregation_horizon", {}) or {}
    agg_end = agg_cfg.get("end")
    if agg_end is not None:
        end_candidates.append(int(agg_end))
    return max(end_candidates)


def _build_socioeconomic_projection(
    root_cfg: Mapping[str, object],
    *,
    end_year: int,
    climate_label: str | None,
) -> pd.DataFrame | None:
    socio_cfg = root_cfg.get("socioeconomics", {}) or {}
    mode = str(socio_cfg.get("mode", "")).strip().lower()
    if mode != "dice":
        return None
    dice_cfg = dict(socio_cfg.get("dice", {}) or {})
    scenario_setting = str(dice_cfg.get("scenario", "SSP2")).strip()
    if scenario_setting.lower() == "as_climate_scenario":
        if not climate_label:
            raise ValueError(
                "socioeconomics.dice.scenario='as_climate_scenario' "
                "requires a climate scenario context."
            )
        match = re.match(r"ssp\s*([0-9])", climate_label, re.IGNORECASE)
        if not match:
            raise ValueError(
                "Unable to infer SSP family from climate scenario '" + str(climate_label) + "'."
            )
        resolved = f"SSP{match.group(1)}"
    else:
        resolved = scenario_setting.upper()
    dice_cfg["scenario"] = resolved
    model = DiceSocioeconomics.from_config(dice_cfg, base_path=ROOT)
    return model.project(end_year)


def _load_config() -> dict:
    path = get_config_path(ROOT / "config.yaml")
    if not path.exists():
        return {}
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _ensure_directory(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {path}")
    return path


def _prompt_for_directory(label: str) -> Path:
    while True:  # pragma: no cover - interactive fallback
        response = input(f"Enter path to {label}: ").strip()
        if not response:
            continue
        candidate = _resolve_path(response)
        if candidate.exists() and candidate.is_dir():
            return candidate
        print(f"'{candidate}' is not a valid directory. Please try again.")


def _default_emission_root(root_cfg: dict) -> Path | None:
    countries_cfg = root_cfg.get("calc_emissions", {}).get("countries", {})
    path_str = countries_cfg.get("aggregate_output_directory") or countries_cfg.get(
        "aggregate_results_directory"
    )
    if not path_str:
        return None
    return _resolve_path(path_str)


def _default_temperature_root(root_cfg: dict) -> Path | None:
    climate_cfg = root_cfg.get("climate_module", {})
    path_str = climate_cfg.get("resource_directory") or climate_cfg.get("output_directory")
    if not path_str:
        return None
    return _resolve_path(path_str)


def _parse_climate_labels(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        labels = [item.strip() for item in value.split(",") if item.strip()]
        return labels
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


def _climate_labels_from_module(root_cfg: Mapping[str, object]) -> list[str]:
    climate_cfg = root_cfg.get("climate_module", {}).get("climate_scenarios", {}) or {}
    definitions = climate_cfg.get("definitions") or []
    label_map: dict[str, str] = {}
    for entry in definitions:
        if not isinstance(entry, Mapping):
            continue
        label = str(entry.get("label") or entry.get("id") or "").strip()
        if label:
            label_map[label] = label

    def _normalize(value: object) -> str | None:
        if value is None:
            return None
        label = str(value).strip()
        if not label:
            return None
        if label_map:
            return label_map.get(label) or (label if label in label_map else None)
        return label

    run_cfg = climate_cfg.get("run")
    resolved: list[str] = []
    if isinstance(run_cfg, str):
        if run_cfg.strip().lower() == "all":
            resolved = list(label_map.values()) if label_map else []
        else:
            candidate = _normalize(run_cfg)
            if candidate:
                resolved = [candidate]
    elif isinstance(run_cfg, Iterable):
        for entry in run_cfg:
            candidate = _normalize(entry)
            if candidate:
                resolved.append(candidate)

    if not resolved and label_map:
        resolved = list(label_map.values())

    return [label for label in resolved if label]


def _restrict_inputs_window(
    inputs: EconomicInputs, start_year: int, end_year: int
) -> EconomicInputs:
    years = inputs.years.astype(int)
    current_start = int(years.min())
    current_end = int(years.max())

    if start_year < current_start:
        raise ValueError(
            f"SCC inputs start at {current_start} but requested window begins at {start_year}."
        )

    gdp = inputs.gdp_trillion_usd
    population = inputs.population_million
    temp_series = dict(inputs.temperature_scenarios_c)
    emission_series = dict(inputs.emission_scenarios_tco2)

    if end_year > current_end:
        extra_years = np.arange(current_end + 1, end_year + 1, dtype=int)
        if extra_years.size:
            years = np.concatenate([years, extra_years])
            gdp_extension = np.full(extra_years.shape, gdp[-1], dtype=float)
            gdp = np.concatenate([gdp, gdp_extension])

            if population is not None:
                pop_extension = np.full(extra_years.shape, population[-1], dtype=float)
                population = np.concatenate([population, pop_extension])

            for name, series in temp_series.items():
                extension = np.full(extra_years.shape, series[-1], dtype=float)
                temp_series[name] = np.concatenate([series, extension])

            for name, series in emission_series.items():
                extension = np.full(extra_years.shape, series[-1], dtype=float)
                emission_series[name] = np.concatenate([series, extension])

    mask = (years >= start_year) & (years <= end_year)
    if not mask.any():
        raise ValueError(
            f"No data available between {start_year} and {end_year} for SCC calculation."
        )

    selected_years = years[mask]
    gdp_selected = gdp[mask]
    population_selected = population[mask] if population is not None else None
    temperatures = {label: series[mask] for label, series in temp_series.items()}
    emissions = {label: series[mask] for label, series in emission_series.items()}

    climate_scenarios = None
    if inputs.climate_scenarios:
        climate_scenarios = dict(inputs.climate_scenarios)

    return EconomicInputs(
        years=selected_years,
        gdp_trillion_usd=gdp_selected,
        temperature_scenarios_c=temperatures,
        emission_scenarios_tco2=emissions,
        population_million=population_selected,
        climate_scenarios=climate_scenarios,
        ssp_family=inputs.ssp_family,
    )


def _default_discount_methods(cfg: dict) -> list[str]:
    methods_cfg = cfg.get("methods", {})
    run = methods_cfg.get("run")
    if isinstance(run, str):
        if run.lower() == "all":
            return AVAILABLE_DISCOUNT_METHODS
        return [run]
    if isinstance(run, Iterable):
        selected = [str(item) for item in run]
        valid = [m for m in selected if m in AVAILABLE_DISCOUNT_METHODS]
        return valid or AVAILABLE_DISCOUNT_METHODS
    return AVAILABLE_DISCOUNT_METHODS


def _build_parser(cfg: dict, root_cfg: dict) -> argparse.ArgumentParser:
    gdp_default = cfg.get("gdp_series")
    gdp_pop_dir_default = cfg.get("gdp_population_directory", "data/GDP_and_Population_data")
    base_year_default = int(cfg.get("base_year", 2025))
    aggregation_default = str(cfg.get("aggregation", "average")).lower()
    if aggregation_default not in AVAILABLE_AGGREGATIONS:
        aggregation_default = "average"
    damage_cfg = cfg.get("damage_function", {})
    damage_delta1_default = float(damage_cfg.get("delta1", 0.0))
    damage_delta2_default = float(damage_cfg.get("delta2", 0.002))
    damage_use_threshold_default = bool(damage_cfg.get("use_threshold", False))
    damage_threshold_temp_default = float(damage_cfg.get("threshold_temperature", 3.0))
    damage_threshold_scale_default = float(damage_cfg.get("threshold_scale", 0.2))
    damage_threshold_power_default = float(damage_cfg.get("threshold_power", 2.0))
    damage_use_saturation_default = bool(damage_cfg.get("use_saturation", False))
    damage_max_fraction_default = float(damage_cfg.get("max_fraction", 0.99))
    damage_saturation_mode_default = str(damage_cfg.get("saturation_mode", "rational")).lower()
    damage_use_catastrophic_default = bool(damage_cfg.get("use_catastrophic", False))
    damage_catastrophic_temp_default = float(damage_cfg.get("catastrophic_temperature", 5.0))
    damage_disaster_fraction_default = float(damage_cfg.get("disaster_fraction", 0.75))
    damage_disaster_gamma_default = float(damage_cfg.get("disaster_gamma", 1.0))
    damage_disaster_mode_default = str(damage_cfg.get("disaster_mode", "prob")).lower()
    aggregation_horizon_cfg = cfg.get("aggregation_horizon", {}) or {}
    aggregation_start_default = aggregation_horizon_cfg.get("start")
    aggregation_end_default = aggregation_horizon_cfg.get("end")

    data_sources_cfg = cfg.get("data_sources", {}) or {}
    emission_root_default = data_sources_cfg.get("emission_root")
    temperature_root_default = data_sources_cfg.get("temperature_root")

    climate_defaults_cfg = data_sources_cfg.get("climate_scenarios")
    if climate_defaults_cfg:
        climate_label_default = ",".join(
            str(item).strip() for item in climate_defaults_cfg if str(item).strip()
        )
    else:
        climate_single_default = data_sources_cfg.get("climate_scenario")
        if climate_single_default is not None:
            climate_label_default = str(climate_single_default).strip()
        else:
            climate_label_default = ""

    parser = argparse.ArgumentParser(
        description=(
            "Calculate SCC for temperature/emission scenarios relative to a reference pathway."
        )
    )
    parser.add_argument(
        "--temperature",
        action="append",
        help=(
            "Temperature series specification as 'label=path/to.csv'. Provide at least two entries."
        ),
    )
    parser.add_argument(
        "--emission",
        action="append",
        help=(
            "Emission delta series specification as 'label=path/to.csv'. Provide "
            "matching labels to temperature."
        ),
    )
    parser.add_argument(
        "--reference-scenario",
        default=cfg.get("reference_scenario"),
        help=(
            "Scenario label treated as the reference/baseline "
            "(default from config or first listed)."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help=(
            "Scenario label to evaluate (repeat flag for multiple). Defaults to "
            "config evaluation list or all non-reference scenarios."
        ),
    )
    parser.add_argument(
        "--gdp-csv",
        default=gdp_default,
        help=(
            "Optional override CSV with 'year', 'gdp_trillion_usd', and "
            "optionally 'population_million'."
        ),
    )
    parser.add_argument(
        "--gdp-population-directory",
        default=gdp_pop_dir_default,
        help=(
            "Directory with SSP GDP/Population data. Defaults to the IIASA Scenario "
            "Explorer extracts (`IIASA/GDP.csv`, `IIASA/Population.csv`) under "
            "data/GDP_and_Population_data."
        ),
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=base_year_default,
        help="Year used as present value reference.",
    )
    parser.add_argument(
        "--aggregation",
        choices=AVAILABLE_AGGREGATIONS,
        default=aggregation_default,
        help="Return per-year SCC values or the aggregated average (default from config).",
    )
    parser.add_argument(
        "--discount-methods",
        default=None,
        help=(
            "Comma-separated discounting methods to run "
            "(constant_discount, ramsey_discount, or 'all'; default from config)."
        ),
    )
    parser.add_argument(
        "--emission-root",
        help=(
            "Root directory containing <scenario>/co2.csv emission deltas "
            "(defaults to economic_module.data_sources.emission_root)."
        ),
        default=emission_root_default,
    )
    parser.add_argument(
        "--temperature-root",
        help=(
            "Directory containing climate CSVs named <scenario>_<climate>.csv "
            "(defaults to economic_module.data_sources.temperature_root)."
        ),
        default=temperature_root_default,
    )
    parser.add_argument(
        "--climate-scenario",
        help=(
            "Climate scenario label(s) to pair with emission deltas (comma-separated). "
            "Defaults to economic_module.data_sources.climate_scenarios or the first "
            "climate definition."
        ),
        default=climate_label_default,
    )
    parser.add_argument(
        "--aggregation-start",
        type=int,
        default=aggregation_start_default,
        help=(
            "First year included when aggregation=='average'. "
            "Overrides economic_module.aggregation_horizon.start."
        ),
    )
    parser.add_argument(
        "--aggregation-end",
        type=int,
        default=aggregation_end_default,
        help=(
            "Final year (inclusive) when aggregation=='average'. "
            "Overrides economic_module.aggregation_horizon.end."
        ),
    )
    parser.add_argument(
        "--discount-rate",
        type=float,
        default=float(
            cfg.get("methods", {}).get("constant_discount", {}).get("discount_rate", 0.03)
        ),
        help="Constant annual discount rate (decimal).",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=float(cfg.get("methods", {}).get("ramsey_discount", {}).get("rho", 0.005)),
        help="Ramsey: pure rate of time preference (decimal).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=float(cfg.get("methods", {}).get("ramsey_discount", {}).get("eta", 1.0)),
        help="Ramsey: elasticity of marginal utility of consumption.",
    )
    parser.add_argument(
        "--add-tco2",
        type=float,
        default=cfg.get("add_tco2"),
        help=(
            "Optional override for total emission delta (t CO₂). If omitted, "
            "derived from emission series."
        ),
    )
    parser.add_argument(
        "--temperature-column",
        default=cfg.get("temperature_column", "temperature_adjusted"),
        help="Column name for temperatures inside the CSV files (default 'temperature_adjusted').",
    )
    parser.add_argument(
        "--emission-column",
        default=cfg.get("emission_column", "delta"),
        help="Column name for emission deltas inside the CSV files (default 'delta').",
    )
    parser.add_argument(
        "--emission-unit-multiplier",
        type=float,
        default=float(cfg.get("emission_to_tonnes", 1e6)),
        help="Factor converting emission column units to t CO₂ (default converts Mt to tonnes).",
    )
    parser.add_argument(
        "--damage-delta1",
        type=float,
        default=damage_delta1_default,
        help="Linear term (delta1) for the DICE quadratic damage function.",
    )
    parser.add_argument(
        "--damage-delta2",
        type=float,
        default=damage_delta2_default,
        help="Quadratic term (delta2) for the DICE damage function.",
    )
    parser.add_argument(
        "--damage-threshold-temperature",
        type=float,
        default=damage_threshold_temp_default,
        help="Temperature (°C) above which threshold amplification begins.",
    )
    parser.add_argument(
        "--damage-threshold-scale",
        type=float,
        default=damage_threshold_scale_default,
        help="Scale factor applied when threshold amplification is enabled.",
    )
    parser.add_argument(
        "--damage-threshold-power",
        type=float,
        default=damage_threshold_power_default,
        help="Power used in the threshold amplification curve.",
    )
    parser.add_argument(
        "--damage-max-fraction",
        type=float,
        default=damage_max_fraction_default,
        help="Upper bound on GDP loss fraction (used for saturation and final cap).",
    )
    parser.add_argument(
        "--damage-saturation-mode",
        choices=("rational", "clamp"),
        default=damage_saturation_mode_default,
        help="Saturation style when enabled: 'rational' curve or hard clamp.",
    )
    parser.add_argument(
        "--damage-disaster-mode",
        choices=("prob", "step"),
        default=damage_disaster_mode_default,
        help="Catastrophic add-on style: probabilistic ('prob') or step change.",
    )
    parser.set_defaults(
        damage_use_threshold=damage_use_threshold_default,
        damage_use_saturation=damage_use_saturation_default,
        damage_use_catastrophic=damage_use_catastrophic_default,
    )
    parser.add_argument(
        "--damage-use-threshold",
        dest="damage_use_threshold",
        action="store_true",
        help="Enable temperature threshold amplification in the damage function.",
    )
    parser.add_argument(
        "--damage-no-threshold",
        dest="damage_use_threshold",
        action="store_false",
        help="Disable temperature threshold amplification.",
    )
    parser.add_argument(
        "--damage-use-saturation",
        dest="damage_use_saturation",
        action="store_true",
        help="Enable saturation of damages to stay below the configured max fraction.",
    )
    parser.add_argument(
        "--damage-no-saturation",
        dest="damage_use_saturation",
        action="store_false",
        help="Disable saturation; damages still capped at the max fraction.",
    )
    parser.add_argument(
        "--damage-use-catastrophic",
        dest="damage_use_catastrophic",
        action="store_true",
        help="Enable catastrophic damages above the catastrophic temperature.",
    )
    parser.add_argument(
        "--damage-no-catastrophic",
        dest="damage_use_catastrophic",
        action="store_false",
        help="Disable catastrophic damage add-ons.",
    )
    parser.add_argument(
        "--damage-catastrophic-temperature",
        type=float,
        default=damage_catastrophic_temp_default,
        help="Temperature (°C) at which catastrophic damages begin.",
    )
    parser.add_argument(
        "--damage-disaster-fraction",
        type=float,
        default=damage_disaster_fraction_default,
        help="Fractional GDP loss added during catastrophic events.",
    )
    parser.add_argument(
        "--damage-disaster-gamma",
        type=float,
        default=damage_disaster_gamma_default,
        help="Controls the steepness of the probabilistic catastrophic response.",
    )
    parser.add_argument(
        "--output-directory",
        default=cfg.get("output_directory", "results/economic"),
        help="Directory for per-method detail tables.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Display available SCC calculation methods and exit.",
    )
    return parser


def _collect_sources(
    arg_entries: list[str] | None,
    cfg_map: Mapping[str, str] | None,
    *,
    minimum: int,
    label: str,
) -> dict[str, Path]:
    entries: dict[str, str] = {}
    if isinstance(cfg_map, Mapping):
        for key, value in cfg_map.items():
            entries[str(key)] = str(value)

    for spec in arg_entries or []:
        if "=" not in spec:
            raise ValueError(f"{label} arguments must use the form name=path.")
        name, path = spec.split("=", 1)
        entries[name.strip()] = path.strip()

    if len(entries) < minimum:
        raise ValueError(f"Provide at least {minimum} {label} series via config or CLI.")

    return {name: _resolve_path(path) for name, path in entries.items()}


def _select_reference(args: argparse.Namespace, cfg: dict, available: Iterable[str]) -> str:
    available = list(available)
    if not available:
        raise ValueError("No temperature scenarios available.")
    candidate = args.reference_scenario or cfg.get("reference_scenario")
    if candidate and candidate in available:
        return candidate
    return available[0]


def _select_targets(
    args: argparse.Namespace, cfg: dict, available: Iterable[str], reference: str
) -> list[str]:
    available_set = set(available)
    available_set.discard(reference)
    if not available_set:
        raise ValueError("Reference scenario is the only series provided.")

    if args.scenario:
        targets = [label for label in args.scenario if label in available_set]
        if targets:
            return targets
    cfg_targets = cfg.get("evaluation_scenarios")
    if isinstance(cfg_targets, Iterable):
        targets = [str(label) for label in cfg_targets if str(label) in available_set]
        if targets:
            return targets
    return sorted(available_set)


def main() -> None:
    root_cfg = _load_config()
    global RUN_DIRECTORY
    RUN_DIRECTORY = get_results_run_directory(root_cfg)
    config = root_cfg.get("economic_module", {})
    countries_cfg = root_cfg.get("calc_emissions", {}).get("countries", {}) or {}
    baseline_case = (
        str(countries_cfg.get("baseline_demand_case", BASE_DEMAND_CASE)).strip() or BASE_DEMAND_CASE
    )

    parser = _build_parser(config, root_cfg)
    args = parser.parse_args()

    if args.list_methods:
        msg_lines = [
            "This tool always uses the FaIR pulse workflow.",
            "",
            "Available discounting methods:",
            "  constant_discount - fixed annual discount rate (e.g., 3%).",
            "  ramsey_discount   - Ramsey rule with rho (time preference) and eta (risk aversion).",
        ]
        print("\n".join(msg_lines))
        return

    discount_methods = _default_discount_methods(config)
    if args.discount_methods:
        spec = args.discount_methods.strip()
        if spec.lower() == "all":
            discount_methods = AVAILABLE_DISCOUNT_METHODS
        else:
            requested = [item.strip() for item in spec.split(",") if item.strip()]
            valid = [m for m in requested if m in AVAILABLE_DISCOUNT_METHODS]
            if valid:
                discount_methods = valid
    if not discount_methods:
        parser.error("No valid discounting methods selected.")

    time_horizon_cfg = root_cfg.get("time_horizon", {})
    horizon_start = int(time_horizon_cfg.get("start", args.base_year))
    if args.base_year != horizon_start:
        parser.error(
            "economic_module.base_year must match time_horizon.start "
            f"(currently {args.base_year} vs {horizon_start})."
        )

    required_year = _max_required_year(time_horizon_cfg, config)
    gdp_path = _resolve_path(args.gdp_csv) if args.gdp_csv else None
    socio_cfg = root_cfg.get("socioeconomics", {}) or {}
    socio_mode = str(socio_cfg.get("mode", "")).strip().lower()
    use_socioeconomics = gdp_path is None and socio_mode == "dice"
    gdp_population_directory = None
    if gdp_path is None and not use_socioeconomics and args.gdp_population_directory:
        gdp_population_directory = _resolve_path(args.gdp_population_directory)
    socio_cache: dict[str, pd.DataFrame | None] = {}

    try:
        manual_temperature = _collect_sources(
            args.temperature, config.get("temperature_series"), minimum=0, label="temperature"
        )
        manual_emission = _collect_sources(
            args.emission, config.get("emission_series"), minimum=0, label="emission"
        )
    except ValueError as exc:
        parser.error(str(exc))

    if manual_temperature or manual_emission:
        parser.error(
            "Manual --temperature/--emission overrides are no longer supported; "
            "configure the economic_module.data_sources section instead."
        )
    else:
        data_sources_cfg = config.get("data_sources", {}) or {}
        emission_root_input = args.emission_root or data_sources_cfg.get("emission_root")
        emission_root_path: Path | None = None
        if emission_root_input:
            emission_candidate = _resolve_path(emission_root_input)
            if emission_candidate.exists():
                emission_root_path = emission_candidate
        if emission_root_path is None:
            default_emission_root = _default_emission_root(root_cfg)
            if default_emission_root and default_emission_root.exists():
                emission_root_path = default_emission_root
        if emission_root_path is None:
            emission_root_path = _prompt_for_directory("emission delta root")  # pragma: no cover
        emission_root_path = _ensure_directory(emission_root_path, "emission delta root")

        reference_default = str(
            args.reference_scenario or config.get("reference_scenario") or f"{BASE_DEMAND_CASE}"
        )
        evaluation_cfg = config.get("evaluation_scenarios") or []
        if isinstance(evaluation_cfg, str):
            evaluation_cfg = [evaluation_cfg]
        evaluation_cfg = [entry for entry in evaluation_cfg if str(entry).strip()]

        climate_inputs = _parse_climate_labels(args.climate_scenario)
        if not climate_inputs:
            climate_inputs = _parse_climate_labels(data_sources_cfg.get("climate_scenarios"))
        if not climate_inputs:
            single_climate = data_sources_cfg.get("climate_scenario")
            if single_climate:
                climate_inputs = [str(single_climate).strip()]
        if not climate_inputs:
            climate_inputs = _climate_labels_from_module(root_cfg)
        if not climate_inputs:
            climate_defs = (
                root_cfg.get("climate_module", {})
                .get("climate_scenarios", {})
                .get("definitions", [])
            )
            if climate_defs:
                first_def = climate_defs[0]
                fallback = first_def.get("label") or first_def.get("id")
                if fallback:
                    climate_inputs = [str(fallback).strip()]
        climate_inputs = [label for label in climate_inputs if label]
        if not climate_inputs:
            parser.error("Unable to determine climate scenario labels.")

        temperature_root_input = args.temperature_root or data_sources_cfg.get("temperature_root")
        temperature_root_path: Path | None = None
        if temperature_root_input:
            temperature_candidate = _resolve_path(temperature_root_input)
            if temperature_candidate.exists():
                temperature_root_path = temperature_candidate
        if temperature_root_path is None:
            default_temperature_root = _default_temperature_root(root_cfg)
            if default_temperature_root and default_temperature_root.exists():
                temperature_root_path = default_temperature_root
        if temperature_root_path is None:
            temperature_root_path = _prompt_for_directory(
                "temperature results directory"
            )  # pragma: no cover
        temperature_root_path = _ensure_directory(temperature_root_path, "temperature root")

        available_scenarios: set[str] = set()
        scenario_temperature: dict[tuple[str, str], Path] = {}
        emission_cache: dict[str, pd.DataFrame] = {}
        damage_entries: list[dict[str, str]] = []

        for mix_dir in sorted(p for p in emission_root_path.iterdir() if p.is_dir()):
            co2_path = mix_dir / "co2.csv"
            if not co2_path.exists():
                continue
            try:
                df = pd.read_csv(co2_path, comment="#")
            except Exception:
                continue
            demand_cases: list[str] = []
            for col in df.columns:
                if col.startswith("delta_") or col.startswith("absolute_"):
                    demand_cases.append(col.split("_", 1)[1])
            demand_cases = sorted(set(demand_cases))
            if not demand_cases:
                continue
            mix_scenarios = {f"{mix_dir.name}__{case}" for case in demand_cases}
            for case in demand_cases:
                scenario_name = f"{mix_dir.name}__{case}"
                available_scenarios.add(scenario_name)
                if scenario_name not in emission_cache:
                    try:
                        emission_cache[scenario_name] = _load_emission_frame_from_mix(
                            emission_root_path,
                            scenario_name,
                            args.emission_column,
                            baseline_case,
                        )
                    except (FileNotFoundError, KeyError, ValueError):
                        continue
            for climate_label in climate_inputs:
                for scenario_name in sorted(mix_scenarios):
                    temp_file = temperature_root_path / f"{scenario_name}_{climate_label}.csv"
                    if not temp_file.exists():
                        continue
                    scenario_temperature[(scenario_name, climate_label)] = temp_file
                    _, demand_case = split_scenario_name(scenario_name)
                    if demand_case != baseline_case:
                        damage_entries.append(
                            {
                                "scenario": scenario_name,
                                "mix": mix_dir.name,
                                "climate": climate_label,
                            }
                        )
    output_dir = _resolve_path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregation = cast(SCCAggregation, args.aggregation)
    agg_start = args.aggregation_start
    agg_end = args.aggregation_end

    run_cfg = config.get("run", {}) or {}
    pulse_cfg = run_cfg.get("pulse", {}) or {}
    damage_kwargs = {
        "delta1": args.damage_delta1,
        "delta2": args.damage_delta2,
        "use_threshold": args.damage_use_threshold,
        "threshold_temperature": args.damage_threshold_temperature,
        "threshold_scale": args.damage_threshold_scale,
        "threshold_power": args.damage_threshold_power,
        "use_saturation": args.damage_use_saturation,
        "max_fraction": args.damage_max_fraction,
        "saturation_mode": args.damage_saturation_mode,
        "use_catastrophic": args.damage_use_catastrophic,
        "catastrophic_temperature": args.damage_catastrophic_temperature,
        "disaster_fraction": args.damage_disaster_fraction,
        "disaster_gamma": args.damage_disaster_gamma,
        "disaster_mode": args.damage_disaster_mode,
    }
    if pulse_cfg:
        damage_kwargs["_pulse_size_tco2__"] = float(pulse_cfg.get("pulse_size_tco2", 1.0e6))
    scc_cache: dict[tuple[str, str], SCCResult] = {}

    if not damage_entries:
        parser.error("No demand scenarios available for SCC damage calculations.")
    if reference_default not in available_scenarios:
        parser.error(f"Reference scenario '{reference_default}' not found in emission outputs.")

    driver_candidates = [
        scenario
        for scenario in evaluation_cfg
        if scenario in available_scenarios and scenario != reference_default
    ]
    driver_scenario = driver_candidates[0] if driver_candidates else None
    if driver_scenario is None:
        fallback = [s for s in sorted(available_scenarios) if s != reference_default]
        if fallback:
            driver_scenario = fallback[0]
    if driver_scenario is None:
        parser.error("Need at least one non-reference scenario to drive SCC computation.")

    for climate_label in climate_inputs:
        reference_label = f"{reference_default}_{climate_label}"
        driver_label = f"{driver_scenario}_{climate_label}"
        temp_reference = scenario_temperature.get((reference_default, climate_label))
        temp_driver = scenario_temperature.get((driver_scenario, climate_label))
        if temp_reference is None or temp_driver is None:
            print(
                f"[SCC] Skipping climate={climate_label} "
                f"(missing temperature data for {reference_default} or {driver_scenario})."
            )
            continue
        emission_reference = emission_cache.get(reference_default)
        emission_driver = emission_cache.get(driver_scenario)
        if emission_reference is None or emission_driver is None:
            parser.error("Emission data missing for reference or driver scenario.")

        temp_map = {
            reference_label: temp_reference,
            driver_label: temp_driver,
        }
        emission_map = {
            reference_label: emission_reference,
            driver_label: emission_driver,
        }

        gdp_frame_override: pd.DataFrame | None = None
        if use_socioeconomics:
            cache_key = climate_label or "__default__"
            if cache_key not in socio_cache:
                try:
                    socio_cache[cache_key] = _build_socioeconomic_projection(
                        root_cfg,
                        end_year=required_year,
                        climate_label=climate_label,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    parser.error(str(exc))
            gdp_frame_override = socio_cache.get(cache_key)

        inputs = EconomicInputs.from_csv(
            temp_map,
            emission_map,
            gdp_path,
            temperature_column=args.temperature_column,
            emission_column=args.emission_column,
            emission_to_tonnes=args.emission_unit_multiplier,
            gdp_frame=gdp_frame_override,
            gdp_population_directory=gdp_population_directory,
        )

        working_inputs = inputs
        damage_duration_years = config.get("damage_duration_years")
        if damage_duration_years is not None:
            try:
                damage_duration = int(damage_duration_years)
            except (TypeError, ValueError):
                parser.error("damage_duration_years must be an integer number of years.")
            if damage_duration <= 0:
                parser.error("damage_duration_years must be positive.")
            start_year = int(time_horizon_cfg.get("start", int(working_inputs.years.min())))
            if "start" in time_horizon_cfg and "end" in time_horizon_cfg:
                horizon_span = int(time_horizon_cfg["end"]) - int(time_horizon_cfg["start"])
                if damage_duration <= horizon_span:
                    parser.error("damage_duration_years must exceed the time_horizon span.")
            target_end = start_year + damage_duration - 1
            try:
                working_inputs = _restrict_inputs_window(working_inputs, start_year, target_end)
            except ValueError as exc:
                parser.error(str(exc))

        pulse_max_year = int(working_inputs.years.max())
        horizon_end_cfg = time_horizon_cfg.get("end")
        if horizon_end_cfg is not None:
            pulse_max_year = min(pulse_max_year, int(horizon_end_cfg))

        if aggregation == "average":
            if agg_start is None or agg_end is None:
                parser.error(
                    "When aggregation=='average', provide aggregation_horizon start and end."
                )
            try:
                agg_start_int = int(agg_start)
                agg_end_int = int(agg_end)
            except (TypeError, ValueError):
                parser.error("Aggregation horizon years must be integers.")
            if agg_start_int > agg_end_int:
                parser.error("aggregation_horizon.start must be <= aggregation_horizon.end.")
            if agg_start_int < int(working_inputs.years.min()) or agg_end_int > int(
                working_inputs.years.max()
            ):
                parser.error(
                    "Aggregation horizon "
                    f"{agg_start_int}-{agg_end_int} falls outside available data "
                    f"({working_inputs.years[0]}-{working_inputs.years[-1]})."
                )
            if not (agg_start_int <= args.base_year <= agg_end_int):
                parser.error(
                    f"Base year {args.base_year} must lie within aggregation "
                    f"horizon {agg_start_int}-{agg_end_int}."
                )
            try:
                working_inputs = _restrict_inputs_window(working_inputs, agg_start_int, agg_end_int)
            except ValueError as exc:
                parser.error(str(exc))

            if args.base_year not in set(working_inputs.years.tolist()):
                parser.error(
                    f"Base year {args.base_year} is outside the SCC evaluation window "
                    f"({working_inputs.years[0]}-{working_inputs.years[-1]})."
                )

            if agg_end is not None:
                pulse_max_year = min(pulse_max_year, int(agg_end))

        for discount_method in discount_methods:
            run_kwargs = {
                "scenario": driver_label,
                "reference": reference_label,
                "base_year": args.base_year,
                "aggregation": aggregation,
                "add_tco2": args.add_tco2,
                "damage_kwargs": dict(damage_kwargs),
            }
            if discount_method == "constant_discount":
                run_kwargs["discount_rate"] = args.discount_rate
            elif discount_method == "ramsey_discount":
                run_kwargs["rho"] = args.rho
                run_kwargs["eta"] = args.eta

            result = compute_scc(
                working_inputs,
                "pulse",
                discount_method=discount_method,
                pulse_max_year=pulse_max_year,
                **run_kwargs,
            )

            scc_cache[(climate_label, discount_method)] = result
            _write_scc_outputs(result, discount_method, climate_label, output_dir)

            print(
                f"[SCC pulse:{discount_method}] SSP={climate_label} driver={driver_scenario} "
                f"vs {reference_default} (base year {result.base_year})"
            )

    if not scc_cache:
        parser.error("No SCC results generated; check climate scenario availability.")

    for entry in damage_entries:
        climate_label = entry["climate"]
        scenario_name = entry["scenario"]
        mix_case = entry["mix"]
        scenario_label = f"{scenario_name}_{climate_label}"
        mix_dir = output_dir / mix_case
        mix_dir.mkdir(parents=True, exist_ok=True)
        for discount_method in discount_methods:
            scc_result = scc_cache.get((climate_label, discount_method))
            if scc_result is None:
                continue
            damage_df = _build_damage_table(
                scenario_label,
                scc_result,
                args.emission_unit_multiplier,
            )
            damage_df["climate_scenario"] = climate_label
            damage_df["method"] = discount_method
            damage_path = mix_dir / f"damages_{discount_method}_{scenario_label}.csv"
            damage_df.to_csv(damage_path, index=False)


if __name__ == "__main__":
    main()

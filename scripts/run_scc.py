"""Compute the social cost of carbon using precomputed temperature and emission scenarios.

Usage
-----
```bash
python scripts/run_scc.py --method constant_discount \
    --temperature baseline=resources/baseline_t.csv --temperature policy=resources/policy_t.csv \
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
from typing import Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
import yaml

from config_paths import (
    apply_results_run_directory,
    get_config_path,
    get_results_run_directory,
)
from economic_module import EconomicInputs, SCCAggregation, compute_scc
from economic_module.socioeconomics import DiceSocioeconomics

ROOT = Path(__file__).resolve().parents[1]

RUN_METHODS = ["kernel", "pulse"]
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


def _split_climate_suffix(name: str, climate_labels: Sequence[str]) -> tuple[str, str | None]:
    for label in sorted({cl.strip() for cl in climate_labels if cl}, key=len, reverse=True):
        suffix = f"_{label}"
        if name.endswith(suffix):
            return name[: -len(suffix)], label
    return name, None


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


def _default_run_method(cfg: dict) -> str:
    run_cfg = cfg.get("run", {})
    method = run_cfg.get("method", "kernel")
    if method not in RUN_METHODS:
        method = "kernel"
    return method


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
    run_method_default = _default_run_method(cfg)
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
            climate_defs = (
                root_cfg.get("climate_module", {})
                .get("climate_scenarios", {})
                .get("definitions", [])
            )
            if climate_defs:
                first_def = climate_defs[0]
                climate_label_default = (
                    first_def.get("label") or first_def.get("id") or ""
                ).strip()
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
            "Directory with SSP GDP/Population data. Prefers long-format CSVs "
            "(SSP_gdp_long_2020ppp.csv, SSP_population_long.csv); falls back to "
            "Excel (GDP_SSP1_5.xlsx, POP_SSP1_5.xlsx)."
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
        "--run-method",
        choices=RUN_METHODS,
        default=run_method_default,
        help="Kernel-based SCC ('kernel') or FaIR pulse SCC ('pulse').",
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

    parser = _build_parser(config, root_cfg)
    args = parser.parse_args()

    if args.list_methods:
        msg_lines = [
            "Available run methods:",
            "  kernel - Kernel-based SCC allocation (fast).",
            "  pulse  - Definition-faithful FaIR pulse runs (slower).",
            "",
            "Available discounting methods:",
            "  constant_discount - fixed annual discount rate (e.g., 3%).",
            "  ramsey_discount   - Ramsey rule with rho (time preference) and eta (risk aversion).",
        ]
        print("\n".join(msg_lines))
        return

    run_method = args.run_method
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
    if run_method == "pulse" and len(discount_methods) != 1:
        parser.error("Pulse run method supports exactly one discounting method at a time.")

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

    temperature_sources: dict[str, Path]
    emission_sources: dict[str, Path]
    reference_lookup: dict[str, str] = {}
    scenario_groups: list[dict[str, object]] = []

    if manual_temperature or manual_emission:
        if not manual_temperature or not manual_emission:
            parser.error(
                "Provide both --temperature and --emission mappings when specifying manually."
            )
        if set(manual_temperature) != set(manual_emission):
            parser.error("Temperature and emission scenario labels must match.")
        temperature_sources = manual_temperature
        emission_sources = manual_emission
        reference = _select_reference(args, config, temperature_sources.keys())
        targets = _select_targets(args, config, temperature_sources.keys(), reference)
        if not targets:
            parser.error("No target scenarios selected for evaluation.")
        reference_lookup[reference] = reference
        for scenario in targets:
            reference_lookup[scenario] = reference
        scenario_groups.append(
            {
                "reference": reference,
                "targets": targets,
                "temperature": temperature_sources,
                "emission": emission_sources,
                "ref_map": dict(reference_lookup),
                "climate_label": None,
            }
        )
    else:
        data_sources_cfg = config.get("data_sources", {}) or {}
        reference_default = str(
            args.reference_scenario or config.get("reference_scenario") or "baseline"
        )
        evaluation_cfg = config.get("evaluation_scenarios") or []
        if isinstance(evaluation_cfg, str):
            evaluation_cfg = [evaluation_cfg]
        base_scenarios = [reference_default, *map(str, evaluation_cfg)]
        base_scenarios = [name for name in base_scenarios if name]
        base_scenarios = list(dict.fromkeys(base_scenarios))
        if len(base_scenarios) < 2:
            parser.error("Provide at least one evaluation scenario besides the reference.")

        climate_inputs = _parse_climate_labels(args.climate_scenario)
        if not climate_inputs:
            climate_inputs = _parse_climate_labels(data_sources_cfg.get("climate_scenarios"))
        if not climate_inputs:
            single_climate = data_sources_cfg.get("climate_scenario")
            if single_climate:
                climate_inputs = [str(single_climate).strip()]
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

        emission_sources = {}
        temperature_sources = {}
        for climate_label in climate_inputs:
            reference_label = f"{reference_default}_{climate_label}"
            reference_lookup[reference_label] = reference_label
            emission_path = (emission_root_path / reference_default / "co2.csv").resolve()
            if not emission_path.exists():
                parser.error(
                    "Emission delta file not found for scenario "
                    f"'{reference_default}' under {emission_root_path}."
                )
            emission_sources[reference_label] = emission_path
            temp_filename = f"{reference_default}_{climate_label}.csv"
            temp_path = (temperature_root_path / temp_filename).resolve()
            if not temp_path.exists():
                parser.error(
                    f"Temperature file '{temp_filename}' not found in {temperature_root_path}."
                )
            temperature_sources[reference_label] = temp_path

            for base in base_scenarios:
                if base == reference_default:
                    continue
                combo_label = f"{base}_{climate_label}"
                emission_path = (emission_root_path / base / "co2.csv").resolve()
                if not emission_path.exists():
                    parser.error(
                        "Emission delta file not found for scenario "
                        f"'{base}' under {emission_root_path}."
                    )
                emission_sources[combo_label] = emission_path
                temp_path = (temperature_root_path / f"{base}_{climate_label}.csv").resolve()
                if not temp_path.exists():
                    parser.error(
                        "Temperature file "
                        f"'{base}_{climate_label}.csv' not found in {temperature_root_path}."
                    )
                temperature_sources[combo_label] = temp_path
                reference_lookup[combo_label] = reference_label
        for climate_label in climate_inputs:
            suffix = f"_{climate_label}"
            reference_label = f"{reference_default}_{climate_label}"
            temp_subset = {
                name: path for name, path in temperature_sources.items() if name.endswith(suffix)
            }
            if not temp_subset:
                continue
            if reference_label not in temp_subset:
                parser.error(
                    f"Temperature data missing for reference scenario '{reference_label}'."
                )
            emission_subset = {name: emission_sources[name] for name in temp_subset}
            targets = [name for name in temp_subset if name != reference_label]
            if not targets:
                parser.error(f"No evaluation scenarios found for climate '{climate_label}'.")
            ref_map = {name: reference_lookup.get(name, reference_label) for name in temp_subset}
            scenario_groups.append(
                {
                    "reference": reference_label,
                    "targets": targets,
                    "temperature": temp_subset,
                    "emission": emission_subset,
                    "ref_map": ref_map,
                    "climate_label": climate_label,
                }
            )

    if not scenario_groups:
        parser.error("No target scenarios selected for evaluation.")

    output_dir = _resolve_path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregation = cast(SCCAggregation, args.aggregation)
    agg_start = args.aggregation_start
    agg_end = args.aggregation_end

    summary_rows = []
    run_cfg = config.get("run", {}) or {}
    kernel_cfg = run_cfg.get("kernel", {}) or {}
    allocation_cfg = kernel_cfg.get("allocation", {}) or {}
    pulse_cfg = run_cfg.get("pulse", {}) or {}
    kernel_horizon = kernel_cfg.get("horizon")
    kernel_alpha = float(kernel_cfg.get("regularization_alpha", 1.0e-6))
    kernel_nonneg = bool(kernel_cfg.get("nonnegativity", False))
    kernel_smooth = float(kernel_cfg.get("smoothing_lambda", 0.0))
    linearized_damage = bool(allocation_cfg.get("linearized_damage", False))
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

    for group in scenario_groups:
        temp_subset = cast(dict[str, Path], group["temperature"])
        emission_subset = cast(dict[str, Path], group["emission"])
        reference_label = cast(str, group["reference"])
        targets = cast(list[str], group["targets"])
        ref_map = cast(dict[str, str], group["ref_map"])
        climate_label = cast(str | None, group.get("climate_label"))

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
            temp_subset,
            emission_subset,
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

        pulse_max_year = None
        if run_method == "pulse":
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

            if run_method == "pulse" and agg_end is not None:
                pulse_max_year = min(pulse_max_year, int(agg_end))

        for scenario in targets:
            scenario_reference = ref_map.get(scenario, reference_label)
            safe_name = _safe_name(scenario)

            for discount_method in discount_methods:
                run_kwargs = {
                    "scenario": scenario,
                    "reference": scenario_reference,
                    "base_year": args.base_year,
                    "aggregation": aggregation,
                    "add_tco2": args.add_tco2,
                }
                damage_kw = dict(damage_kwargs)
                damage_kw["_kernel_horizon__"] = (
                    int(kernel_horizon) if kernel_horizon is not None else 0
                )
                damage_kw["_kernel_alpha__"] = kernel_alpha
                damage_kw["_kernel_nonneg__"] = kernel_nonneg
                damage_kw["_kernel_smooth__"] = kernel_smooth
                damage_kw["_linearized_flag__"] = linearized_damage
                if run_method == "pulse":
                    damage_kw["_pulse_size_tco2__"] = float(pulse_cfg.get("pulse_size_tco2", 1.0e6))
                run_kwargs["damage_kwargs"] = damage_kw

                if discount_method == "constant_discount":
                    run_kwargs["discount_rate"] = args.discount_rate
                elif discount_method == "ramsey_discount":
                    run_kwargs["rho"] = args.rho
                    run_kwargs["eta"] = args.eta

                if run_method == "kernel":
                    method_key = discount_method
                    result = compute_scc(working_inputs, method_key, **run_kwargs)
                    run_label = "kernel"
                else:  # pulse
                    method_key = "pulse"
                    result = compute_scc(
                        working_inputs,
                        method_key,
                        discount_method=discount_method,
                        pulse_max_year=pulse_max_year,
                        **run_kwargs,
                    )
                    run_label = "pulse"

                method_label = result.method
                file_tag = method_label
                details_path = output_dir / f"scc_{file_tag}_{safe_name}.csv"
                result.details.to_csv(details_path, index=False)

                timeseries_path = output_dir / f"scc_timeseries_{file_tag}_{safe_name}.csv"
                result.per_year.to_csv(timeseries_path, index=False)
                extra_paths: list[Path] = [timeseries_path]

                if run_label == "pulse":
                    pulse_columns = {
                        "year": "year",
                        "discount_factor": "discount_factor",
                        "discounted_damage_attributed_usd": "pv_damage_per_pulse_usd",
                        "scc_usd_per_tco2": "scc_usd_per_tco2",
                        "pulse_size_tco2": "pulse_size_tco2",
                    }
                    pulse_df = result.per_year[list(pulse_columns.keys())].rename(
                        columns=pulse_columns
                    )
                    pulse_path = output_dir / f"pulse_scc_timeseries_{file_tag}_{safe_name}.csv"
                    pulse_df.to_csv(pulse_path, index=False)
                    extra_paths.append(pulse_path)

                    damage_columns = {
                        "year": "year",
                        "delta_emissions_tco2": "delta_emissions_tco2",
                        "delta_damage_usd": "delta_damage_usd",
                        "discounted_delta_usd": "pv_delta_damage_usd",
                    }
                    damage_df = result.per_year[list(damage_columns.keys())].rename(
                        columns=damage_columns
                    )
                    damage_path = output_dir / f"pulse_emission_damages_{file_tag}_{safe_name}.csv"
                    damage_df.to_csv(damage_path, index=False)
                    extra_paths.append(damage_path)

                summary_rows.append(
                    {
                        "scenario": scenario,
                        "reference": scenario_reference,
                        "method": method_label,
                        "run_method": run_label,
                        "aggregation": aggregation,
                        "scc_usd_per_tco2": result.scc_usd_per_tco2,
                        "base_year": result.base_year,
                        "total_delta_emissions_tco2": result.add_tco2,
                        "discount_rate": args.discount_rate
                        if method_label == "constant_discount"
                        else np.nan,
                        "rho": args.rho if method_label == "ramsey_discount" else np.nan,
                        "eta": args.eta if method_label == "ramsey_discount" else np.nan,
                    }
                )

                print(
                    f"[{run_label}:{method_label}] {scenario} vs {scenario_reference}: "
                    f"{result.scc_usd_per_tco2:,.2f} USD/tCO2 (aggregation {aggregation}, "
                    f"base year {result.base_year})"
                )
                print(f"  Details written to {_format_path(details_path)}")
                for path in extra_paths:
                    print(f"  Output written to {_format_path(path)}")

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "scc_summary.csv"
    summary.to_csv(summary_path, index=False)

    summary_display = _format_path(summary_path)
    print(f"Summary table written to {summary_display}")


if __name__ == "__main__":
    main()

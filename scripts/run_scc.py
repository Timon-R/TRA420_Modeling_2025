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
from pathlib import Path
from typing import Iterable, Mapping, cast

import numpy as np
import pandas as pd
import yaml

from economic_module import EconomicInputs, SCCAggregation, compute_scc

ROOT = Path(__file__).resolve().parents[1]

AVAILABLE_METHODS = ["constant_discount", "ramsey_discount"]
AVAILABLE_AGGREGATIONS: tuple[SCCAggregation, ...] = ("average", "per_year")


def _format_path(path: Path) -> Path:
    abs_path = path.resolve()
    try:
        return abs_path.relative_to(ROOT)
    except ValueError:  # pragma: no cover - filesystem safety
        return abs_path


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    return cleaned.strip("_") or "scenario"


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _load_config() -> dict:
    path = ROOT / "config.yaml"
    if not path.exists():
        return {}
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _default_methods(cfg: dict) -> list[str]:
    methods_cfg = cfg.get("methods", {})
    run = methods_cfg.get("run")
    if isinstance(run, str):
        if run.lower() == "all":
            return AVAILABLE_METHODS
        return [run]
    if isinstance(run, Iterable):
        selected = [str(item) for item in run]
        valid = [m for m in selected if m in AVAILABLE_METHODS]
        return valid or AVAILABLE_METHODS
    return AVAILABLE_METHODS


def _build_parser(cfg: dict) -> argparse.ArgumentParser:
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

    parser = argparse.ArgumentParser(
        description=(
            "Calculate SCC for temperature/emission scenarios relative to a " "reference pathway."
        )
    )
    parser.add_argument(
        "--temperature",
        action="append",
        help=(
            "Temperature series specification as 'label=path/to.csv'. Provide at "
            "least two entries."
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
            "Directory containing SSP GDP/Population workbooks "
            "(GDP_SSP1_5.xlsx, POP_SSP1_5.xlsx)."
        ),
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=base_year_default,
        help="Year used as present value reference.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=AVAILABLE_METHODS + ["all"],
        help=(
            "Discounting approach: constant rate or Ramsey rule. Use multiple "
            "--method flags or 'all'."
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=AVAILABLE_AGGREGATIONS,
        default=aggregation_default,
        help="Return per-year SCC values or the aggregated average (default from config).",
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


def _resolve_methods(args: argparse.Namespace, cfg: dict) -> list[str]:
    if args.method:
        if "all" in args.method:
            return AVAILABLE_METHODS
        return [m for m in args.method if m in AVAILABLE_METHODS]
    return _default_methods(cfg)


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
    config = _load_config().get("economic_module", {})

    parser = _build_parser(config)
    args = parser.parse_args()

    if args.list_methods:
        msg_lines = [
            "Available methods:",
            "  constant_discount - fixed annual discount rate (e.g., 3%).",
            "  ramsey_discount   - Ramsey rule with rho (time preference) and eta (risk aversion).",
        ]
        print("\n".join(msg_lines))
        return

    methods = _resolve_methods(args, config)
    if not methods:
        parser.error("No valid SCC methods selected.")

    try:
        temperature_sources = _collect_sources(
            args.temperature, config.get("temperature_series"), minimum=2, label="temperature"
        )
        emission_sources = _collect_sources(
            args.emission, config.get("emission_series"), minimum=2, label="emission"
        )
    except ValueError as exc:
        parser.error(str(exc))

    if set(temperature_sources) != set(emission_sources):
        parser.error("Temperature and emission scenario labels must match.")

    reference = _select_reference(args, config, temperature_sources.keys())
    targets = _select_targets(args, config, temperature_sources.keys(), reference)
    if not targets:
        parser.error("No target scenarios selected for evaluation.")

    gdp_path = _resolve_path(args.gdp_csv) if args.gdp_csv else None
    gdp_population_directory = (
        _resolve_path(args.gdp_population_directory) if args.gdp_population_directory else None
    )

    inputs = EconomicInputs.from_csv(
        temperature_sources,
        emission_sources,
        gdp_path,
        temperature_column=args.temperature_column,
        emission_column=args.emission_column,
        emission_to_tonnes=args.emission_unit_multiplier,
        gdp_population_directory=gdp_population_directory,
    )

    output_dir = _resolve_path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    aggregation = cast(SCCAggregation, args.aggregation)
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

    for scenario in targets:
        for method in methods:
            kwargs = {
                "scenario": scenario,
                "reference": reference,
                "base_year": args.base_year,
                "aggregation": aggregation,
                "add_tco2": args.add_tco2,
                "damage_kwargs": damage_kwargs,
            }
            if method == "constant_discount":
                kwargs["discount_rate"] = args.discount_rate
            elif method == "ramsey_discount":
                kwargs["rho"] = args.rho
                kwargs["eta"] = args.eta

            result = compute_scc(inputs, method, **kwargs)

            safe_name = _safe_name(scenario)
            details_path = output_dir / f"scc_{method}_{safe_name}.csv"
            result.details.to_csv(details_path, index=False)

            timeseries_path = output_dir / f"scc_timeseries_{method}_{safe_name}.csv"
            result.per_year.to_csv(timeseries_path, index=False)

            summary_rows.append(
                {
                    "scenario": scenario,
                    "reference": reference,
                    "method": method,
                    "aggregation": aggregation,
                    "scc_usd_per_tco2": result.scc_usd_per_tco2,
                    "base_year": result.base_year,
                    "total_delta_emissions_tco2": result.add_tco2,
                    "discount_rate": args.discount_rate
                    if method == "constant_discount"
                    else np.nan,
                    "rho": args.rho if method == "ramsey_discount" else np.nan,
                    "eta": args.eta if method == "ramsey_discount" else np.nan,
                }
            )

            print(
                f"[{method}] {scenario} vs {reference}: {result.scc_usd_per_tco2:,.2f} USD/tCO2 "
                f"(aggregation {aggregation}, base year {result.base_year})"
            )
            print(f"  Details written to {_format_path(details_path)}")
            print(f"  SCC time series written to {_format_path(timeseries_path)}")

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "scc_summary.csv"
    summary.to_csv(summary_path, index=False)

    summary_display = _format_path(summary_path)
    print(f"Summary table written to {summary_display}")


if __name__ == "__main__":
    main()

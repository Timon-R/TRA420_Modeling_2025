"""Run FaIR temperature scenarios based on ``config.yaml``.

Usage
-----
```bash
python scripts/run_fair_scenarios.py
```

Configuration lives in ``config.yaml`` under the ``climate_module`` section:

- ``emission_timeseries_directory`` – directory containing subfolders like
  ``resources/<scenario>/co2.csv`` (values in Mt CO₂/yr) produced by the
  ``calc_emissions`` workflow. The runner automatically detects each folder
  unless ``emission_scenarios.run`` restricts the list.
- ``climate_scenarios`` – SSP pathways to evaluate (``run: all`` or explicit list).
- ``parameters`` – global FaIR options (time grid, climate setup, overrides).
- ``sample_years_option`` – determines which years are written to output CSVs.

Output
------
For every combination of emission scenario and climate pathway the script writes a
CSV to ``results/climate/<emission>_<climate>.csv`` (archival result) and to
``resources/climate/<emission>_<climate>.csv`` (shared input for downstream modules).
It also prints a summary table with 2030, 2050 and 2100 temperatures/deltas.

Emission time series
--------------------
Emission differences are expressed in **Mt CO₂ per year** (derived from the CSVs in
``resources/``). The runner converts them to Gt CO₂ before passing them to FaIR. If
you want to prescribe a custom trajectory inside the config instead of using the
detected CSV, pass an array with the same length (and ordering) as the FaIR
timepoints returned by
:func:`climate_module.compute_temperature_change`. A helper in
:mod:`climate_module.scenario_runner` already prepares the timepoints for you, so a
callable adjustment can look like this:

>>> def custom_series(timepoints, cfg):
...     years = [2025.5, 2030.5, 2040.5]
...     values = [-1.0, -3.0, -5.0]
...     return np.interp(timepoints, years, values, left=0.0, right=-5.0)
...
>>> ScenarioSpec(..., emission_adjustments={"CO2 FFI": custom_series})

The callable receives the FaIR timepoints (mid-year values) and the time
configuration. Return a NumPy array matching ``timepoints`` and the runner will
inject it into the adjusted configuration automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from climate_module import DEFAULT_TIME_CONFIG, ScenarioSpec, run_scenarios, step_change

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("climate.run")


def _load_config() -> dict:
    path = ROOT / "config.yaml"
    if not path.exists():
        return {}
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


ROOT_CONFIG = _load_config()
CONFIG = ROOT_CONFIG.get("climate_module", {})
GLOBAL_HORIZON = ROOT_CONFIG.get("time_horizon", {})
ECONOMIC_CFG = ROOT_CONFIG.get("economic_module", {})

PARAMETERS = CONFIG.get("parameters", {})

GLOBAL_START = float(GLOBAL_HORIZON.get("start", DEFAULT_TIME_CONFIG["start_year"]))
GLOBAL_END = float(GLOBAL_HORIZON.get("end", DEFAULT_TIME_CONFIG["end_year"]))
GLOBAL_STEP = float(GLOBAL_HORIZON.get("step", 5.0))
DAMAGE_DURATION = ECONOMIC_CFG.get("damage_duration_years")
if isinstance(DAMAGE_DURATION, (int, float)) and DAMAGE_DURATION > 0:
    extended_end = GLOBAL_START + int(DAMAGE_DURATION) - 1
    CLIMATE_END = float(max(GLOBAL_END, extended_end))
else:
    CLIMATE_END = GLOBAL_END
CLIMATE_TIMESTEP = 1.0
BASE_CLIMATE_SETUP = PARAMETERS.get("climate_setup", "ar6")

DEFAULT_TIME_CONFIG.update(
    {
        "start_year": GLOBAL_START,
        "end_year": CLIMATE_END,
        "timestep": CLIMATE_TIMESTEP,
    }
)

BASE_START = GLOBAL_START
BASE_END = CLIMATE_END
BASE_TIMESTEP = CLIMATE_TIMESTEP

OVERRIDE_KEYS = {
    "ocean_heat_capacity",
    "ocean_heat_transfer",
    "deep_ocean_efficacy",
    "forcing_4co2",
    "equilibrium_climate_sensitivity",
}

BASE_OVERRIDES = {key: PARAMETERS[key] for key in OVERRIDE_KEYS if key in PARAMETERS}

DEFAULT_ADJUSTMENT_SPECIE = PARAMETERS.get("adjustment_specie", "CO2 FFI")
DEFAULT_ADJUSTMENT_DELTA = float(PARAMETERS.get("adjustment_delta", -2.0))
DEFAULT_ADJUSTMENT_START = float(PARAMETERS.get("adjustment_start_year", 2025))
ADJUSTMENT_CSV = PARAMETERS.get("adjustment_timeseries_csv")

SUMMARY_CANDIDATES = {2030, 2050, 2100, int(round(CLIMATE_END))}
SUMMARY_YEARS = sorted(year for year in SUMMARY_CANDIDATES if GLOBAL_START <= year <= CLIMATE_END)

# Emission label used as the reference/baseline in downstream modules. We keep
# the baseline CSVs in resources/ (consumed by SCC), but avoid duplicating them
# in results/climate since each scenario CSV already contains a baseline column.
REFERENCE_EMISSION_LABEL = str(ECONOMIC_CFG.get("reference_scenario", "baseline"))


def _derive_sample_years(option: str) -> list[int]:
    option = option.lower()
    if option == "full":
        start = int(GLOBAL_START)
        end = int(CLIMATE_END)
        return list(range(start, end + 1))

    start = int(GLOBAL_START)
    end = int(CLIMATE_END)
    years: list[int] = [start]
    threshold = max(2050, start)
    current = start
    while True:
        step = 5 if current < threshold else 10
        current += step
        if current > end:
            break
        years.append(current)
    if years[-1] != end:
        years.append(end)
    # Ensure reasonable spacing when the horizon is coarse (e.g., 5-year steps)
    fallback_step = max(1, int(round(GLOBAL_STEP)))
    filled = list(range(start, end + 1, fallback_step))
    combined = sorted(set(years) | set(filled))
    return [year for year in combined if start <= year <= end]


SAMPLE_YEARS = _derive_sample_years(CONFIG.get("sample_years_option", "default"))

OUTPUT_DIR = ROOT / CONFIG.get("output_directory", "results/climate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = ROOT / CONFIG.get("resource_directory", "resources/climate")
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

EMISSION_DIR = ROOT / CONFIG.get("emission_timeseries_directory", "resources")
EMISSION_DIR.mkdir(parents=True, exist_ok=True)


def _discover_emission_map() -> dict[str, Path]:
    return {
        d.name: d for d in sorted(EMISSION_DIR.iterdir()) if d.is_dir() and (d / "co2.csv").exists()
    }


def _selected_emissions(emission_map: dict[str, Path]) -> list[str]:
    emission_cfg = CONFIG.get("emission_scenarios", {})
    emission_run = emission_cfg.get("run", "all") if isinstance(emission_cfg, dict) else "all"
    if isinstance(emission_run, str) and emission_run.lower() == "all":
        return sorted(emission_map.keys())
    emission_ids = [emission_run] if isinstance(emission_run, str) else list(emission_run)
    return [name for name in emission_ids if name in emission_map]


CLIMATE_CFG = CONFIG.get("climate_scenarios", {})
CLIMATE_DEFS = {
    entry["id"]: entry
    for entry in CLIMATE_CFG.get("definitions", [])
    if isinstance(entry, dict) and "id" in entry
}
climate_run = CLIMATE_CFG.get("run", "all")
if isinstance(climate_run, str) and climate_run.lower() == "all":
    SELECTED_CLIMATE_IDS = list(CLIMATE_DEFS.keys())
else:
    climate_ids = [climate_run] if isinstance(climate_run, str) else list(climate_run)
    SELECTED_CLIMATE_IDS = [cid for cid in climate_ids if cid in CLIMATE_DEFS]


def build_scenarios() -> list[ScenarioSpec]:
    emission_map = _discover_emission_map()
    selected_emissions = _selected_emissions(emission_map)
    if not selected_emissions:
        raise FileNotFoundError(f"No emission difference files found in '{EMISSION_DIR}'.")
    if not SELECTED_CLIMATE_IDS:
        raise ValueError("No climate scenarios selected. Check config.yaml.")

    specs: list[ScenarioSpec] = []
    for emission_name in selected_emissions:
        emission_dir = emission_map[emission_name]
        default_co2 = emission_dir / "co2.csv"
        if not default_co2.exists():
            raise FileNotFoundError(
                f"Missing co2.csv for emission scenario '{emission_name}' in {emission_dir}"
            )

        for climate_id in SELECTED_CLIMATE_IDS:
            definition = CLIMATE_DEFS[climate_id]
            label_suffix = definition.get("label", climate_id)
            label = f"{emission_name}_{label_suffix}"

            apply_adjustment = definition.get("apply_adjustment", True)
            adjustments = None
            if apply_adjustment:
                specie = definition.get("adjustment_specie", DEFAULT_ADJUSTMENT_SPECIE)
                csv_override = definition.get("adjustment_timeseries_csv", None)
                if csv_override:
                    csv_candidate = Path(csv_override)
                    if not csv_candidate.is_absolute():
                        emission_relative = emission_dir / csv_candidate
                        if emission_relative.exists():
                            csv_candidate = emission_relative
                        else:
                            csv_candidate = (ROOT / csv_candidate).resolve()
                    csv_path = csv_candidate
                else:
                    csv_path = default_co2
                adjustments = {specie: _timeseries_adjustment(csv_path)}
            else:
                csv_path = None

            delta = definition.get("adjustment_delta")
            start_delta_year = definition.get("adjustment_start_year", DEFAULT_ADJUSTMENT_START)
            if adjustments is None and delta is not None:
                adjustments = {
                    definition.get("adjustment_specie", DEFAULT_ADJUSTMENT_SPECIE): step_change(
                        float(delta), start_year=float(start_delta_year)
                    )
                }

            start_year = float(definition.get("start_year", BASE_START))
            end_year = float(definition.get("end_year", BASE_END))
            timestep = float(definition.get("timestep", BASE_TIMESTEP))
            climate_setup = definition.get("climate_setup", BASE_CLIMATE_SETUP)

            overrides = dict(BASE_OVERRIDES)
            for key in OVERRIDE_KEYS:
                if key in definition:
                    overrides[key] = definition[key]

            compute_kwargs = {"climate_setup": climate_setup}
            if overrides:
                compute_kwargs["climate_overrides"] = overrides

            specs.append(
                ScenarioSpec(
                    label=label,
                    scenario=climate_id,
                    emission_adjustments=adjustments,
                    start_year=start_year,
                    end_year=end_year,
                    timestep=timestep,
                    compute_kwargs=compute_kwargs,
                )
            )

    return specs


def _write_csv(label: str, climate_scenario: str, result) -> None:
    years = result.years
    mask = _sampling_mask(years)
    df = pd.DataFrame(
        {
            "year": years[mask].astype(int),
            "temperature_baseline": result.baseline[mask],
            "temperature_adjusted": result.adjusted[mask],
            "temperature_delta": result.delta[mask],
            "climate_scenario": climate_scenario,
        }
    )
    # Avoid writing redundant baseline CSVs to results/climate; the baseline
    # temperature path is already included as a column in every scenario file.
    # Still mirror the full CSV to resources/climate for downstream consumers.
    if not label.startswith(f"{REFERENCE_EMISSION_LABEL}_"):
        df.to_csv(OUTPUT_DIR / f"{label}.csv", index=False)
    df.to_csv(RESOURCE_DIR / f"{label}.csv", index=False)


def _print_summary(label: str, result) -> None:
    years = result.years
    indices = [int(np.argmin(np.abs(years - year))) for year in SUMMARY_YEARS]
    header = (
        f"\n=== {label} ===\n"
        "  Year | Baseline (°C) | Adjusted (°C) | Delta (°C)\n"
        "  -----+----------------+----------------+-----------"
    )
    rows = [
        f"  {int(years[idx]):4d} | {result.baseline[idx]:14.2f} | "
        f"{result.adjusted[idx]:14.2f} | {result.delta[idx]:9.2f}"
        for idx in indices
    ]
    LOGGER.info("%s\n%s", header, "\n".join(rows))


def _sampling_mask(years: np.ndarray) -> np.ndarray:
    indices = [int(np.argmin(np.abs(years - year))) for year in SAMPLE_YEARS]
    mask = np.zeros_like(years, dtype=bool)
    mask[indices] = True
    return mask


_TIMESERIES_CACHE: dict[Path, tuple[np.ndarray, np.ndarray]] = {}


def _timeseries_adjustment(rel_path: Path):
    path = Path(rel_path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if path not in _TIMESERIES_CACHE:
        df = pd.read_csv(path)
        if not {"year", "delta"}.issubset(df.columns):
            raise ValueError(f"Timeseries file '{path}' must contain 'year' and 'delta' columns.")
        years = df["year"].to_numpy(dtype=float)
        deltas = df["delta"].to_numpy(dtype=float)
        _TIMESERIES_CACHE[path] = (years, deltas)

    years, deltas = _TIMESERIES_CACHE[path]

    def builder(timepoints: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
        offset = cfg.get("timestep", 1.0) / 2
        mt_values = np.interp(timepoints, years + offset, deltas, left=deltas[0], right=deltas[-1])
        LOGGER.debug(
            "Interpolated adjustments from %s with %d rows onto %d timepoints",
            path,
            len(deltas),
            len(timepoints),
        )
        return mt_values / 1000.0

    return builder


def main() -> None:
    specs = build_scenarios()
    results = run_scenarios(specs)
    for spec in specs:
        result = results[spec.label]
        _write_csv(spec.label, spec.scenario, result)
        _print_summary(spec.label, result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    main()

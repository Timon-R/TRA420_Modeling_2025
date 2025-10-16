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
Emission differences are expressed in **Mt CO₂ per year** (derived from the CSVs in `resources/`). The runner converts them to Gt CO₂ before passing them to FaIR. If you want to prescribe a custom
trajectory inside the config instead of using the detected CSV, pass an array with
the same length (and ordering) as the FaIR timepoints returned by
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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from climate_module import DEFAULT_TIME_CONFIG, ScenarioSpec, run_scenarios, step_change


def _load_config() -> dict:
    path = ROOT / "config.yaml"
    if not path.exists():
        return {}
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


CONFIG = _load_config().get("climate_module", {})

PARAMETERS = CONFIG.get("parameters", {})

BASE_START = float(PARAMETERS.get("start_year", DEFAULT_TIME_CONFIG["start_year"]))
BASE_END = float(PARAMETERS.get("end_year", DEFAULT_TIME_CONFIG["end_year"]))
BASE_TIMESTEP = float(PARAMETERS.get("timestep", DEFAULT_TIME_CONFIG["timestep"]))
BASE_CLIMATE_SETUP = PARAMETERS.get("climate_setup", "ar6")

DEFAULT_TIME_CONFIG.update(
    {
        "start_year": BASE_START,
        "end_year": BASE_END,
        "timestep": BASE_TIMESTEP,
    }
)

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

SUMMARY_YEARS = [2030, 2050, 2100]


def _derive_sample_years(option: str) -> list[int]:
    option = option.lower()
    if option == "full":
        return list(range(2025, int(BASE_END) + 1))
    return [2025, 2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080, 2090, 2100]


SAMPLE_YEARS = _derive_sample_years(CONFIG.get("sample_years_option", "default"))

OUTPUT_DIR = ROOT / CONFIG.get("output_directory", "results/climate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESOURCE_DIR = ROOT / CONFIG.get("resource_directory", "resources/climate")
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

EMISSION_DIR = ROOT / CONFIG.get("emission_timeseries_directory", "resources")
EMISSION_DIR.mkdir(parents=True, exist_ok=True)
EMISSION_MAP = {
    d.name: d
    for d in sorted(EMISSION_DIR.iterdir())
    if d.is_dir() and (d / "co2.csv").exists()
}
EMISSION_CFG = CONFIG.get("emission_scenarios", {})
emission_run = EMISSION_CFG.get("run", "all") if isinstance(EMISSION_CFG, dict) else "all"
if isinstance(emission_run, str) and emission_run.lower() == "all":
    SELECTED_EMISSIONS = sorted(EMISSION_MAP.keys())
else:
    emission_ids = [emission_run] if isinstance(emission_run, str) else list(emission_run)
    SELECTED_EMISSIONS = [name for name in emission_ids if name in EMISSION_MAP]

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
    if not SELECTED_EMISSIONS:
        raise FileNotFoundError(f"No emission difference files found in '{EMISSION_DIR}'.")
    if not SELECTED_CLIMATE_IDS:
        raise ValueError("No climate scenarios selected. Check config.yaml.")

    specs: list[ScenarioSpec] = []
    for emission_name in SELECTED_EMISSIONS:
        emission_dir = EMISSION_MAP[emission_name]
        default_co2 = emission_dir / "co2.csv"
        if not default_co2.exists():
            raise FileNotFoundError(f"Missing co2.csv for emission scenario '{emission_name}' in {emission_dir}")

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
    print(header)
    print("\n".join(rows))


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
    main()

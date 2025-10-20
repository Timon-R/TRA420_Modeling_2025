# CLI Scripts Overview

This project includes small wrappers around the core modules to make end‑to‑end runs easy.

## Emissions

- `scripts/run_calc_emissions.py` — per‑country deltas to `resources/<Country>/<scenario>/`.
  - `--country <Name>` select a specific country (matches config filename).
  - Writes `co2.csv` (and pollutant CSVs) with `year,delta` in Mt/year.
- `scripts/run_calc_emissions_all.py` — aggregates across countries to `resources/All_countries/<scenario>/`.
  - `--countries` restrict to a subset.
  - `--output` change aggregate resources folder; `--results-output` mirror to `results/...`.

## Air Pollution

- `scripts/run_air_pollution.py`
  - Loads `air_pollution` block from `config.yaml`.
  - Computes per‑pollutant health impacts and optional mortality summaries.
  - Outputs in `results/air_pollution/<scenario>/`.

## Climate (FaIR)

- `scripts/run_fair_scenarios.py`
  - Uses `climate_module` block.
  - Writes `results/climate/<emission>_<climate>.csv` and mirrors to `resources/climate/`.
  - Baseline CSVs are not duplicated in results; each file contains a baseline column.

## Pattern Scaling

- `scripts/run_pattern_scaling.py`
  - Scales global climate CSVs to country trajectories.
  - Writes to `results/climate_scaled/`.

## Economics (SCC)

- `scripts/run_scc.py`
  - Reads global temperature CSVs and aggregated emissions.
  - Selects the SSP family from the temperature CSVs and auto‑loads GDP/population.
  - Outputs per‑scenario timeseries and summary tables to `results/economic/`.
  - Aggregation:
    - `per_year`: SCC series printed for selected years in the summary tool.
    - `average`: single SCC per method over the configured horizon.

## Summary

- `scripts/generate_summary.py`
  - Gathers emission, climate, mortality, and SCC results.
  - Writes `results/summary/summary.txt`, `summary.json`, and plots.
  - Plots deduplicate emission and mortality across SSP suffixes.

## Full Pipeline

- `scripts/run_full_pipeline.py`
  - Executes the standard sequence using `config.yaml` defaults.

## Cleanup Utilities

- `clean_cache.py` — removes `__pycache__`, `.pytest_cache`, `.ruff_cache`, coverage files.
- `clean_outputs.py` — deletes `resources/` and `results/` (use with caution).


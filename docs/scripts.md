# CLI Scripts Overview

This project includes small wrappers around the core modules to make end‑to‑end runs easy.

> Paths listed below show defaults. When `results.run_directory` is set in `config.yaml`,
> each script automatically writes to `results/<run_directory>/…` instead of the bare `results/`
> tree so you can keep multiple experiments side by side.

## Emissions

- `scripts/run_calc_emissions.py` — per‑country deltas to `results/emissions/<mix>/<Country>/`.
  - `--country <Name>` select a specific country (matches config filename).
  - Writes `co2.csv` (and pollutant CSVs) with `year,delta` in Mt/year.
- `scripts/run_calc_emissions_all.py` — aggregates across countries to `results/emissions/All_countries/<mix>/`.
  - `--countries` restrict to a subset.
  - `--output` change the aggregate output folder; `--results-output` mirror to another directory.

## Air Pollution

- `scripts/run_air_pollution.py`
  - Loads `air_pollution` block from `config.yaml`.
  - Computes per‑pollutant health impacts and optional mortality summaries.
  - Outputs in `results/air_pollution/<scenario>/`.

## Climate (FaIR)

- `scripts/run_fair_scenarios.py`
  - Uses `climate_module` block.
  - Writes `results/climate/<emission>_<climate>.csv`.
  - Baseline CSVs are not duplicated in results; each file contains a baseline column.

## Local Climate Impacts

- `scripts/run_local_climate_impacts.py`
  - Scales global climate CSVs to country temperature/precipitation trajectories and prepares inputs for extreme-weather damages.
  - Writes to `results/climate_scaled/`.

## Economics (SCC)

- `scripts/run_scc.py`
  - Reads global temperature CSVs and aggregated emissions.
  - Selects the SSP family from the temperature CSVs and auto‑loads GDP/population.
  - Outputs per‑SSP SCC timeseries and per‑scenario damage tables under `results/economic/`.
  - Aggregation:
    - `per_year`: SCC series printed for selected years in the summary tool.
    - `average`: single SCC per method over the configured horizon.
  - Discounting methods (`--discount-methods` or `economic_module.methods.run`):
    - `constant_discount`, `ramsey_discount` (or both via `all`).
  - Pulse workflow outputs:
    - `pulse_scc_timeseries_<method>_<ssp>.csv` stores the full SCC(τ) path (per SSP) plus the aggregated SCC value.
    - `results/economic/<mix>/damages_<method>_<scenario>.csv` stores emissions, SCC, and damages
      for each mix/demand/climate combination (used by the summary module).

## Summary

- `scripts/generate_summary.py`
  - Gathers emission, climate, mortality, and SCC results.
  - Writes `results/summary/summary.csv` plus plots (see `docs/results_summary.md` for column details).
  - Plots deduplicate emission and mortality across SSP suffixes.

## Full Pipeline

- `scripts/run_full_pipeline.py`
  - Executes the standard sequence using `config.yaml` defaults.

## Cleanup Utilities

- `clean_cache.py` — removes `__pycache__`, `.pytest_cache`, `.ruff_cache`, coverage files.
- `clean_outputs.py` — deletes `results/` (use with caution).

## References
- [CLI usage patterns]:

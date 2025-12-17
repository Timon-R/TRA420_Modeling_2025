# CLI Scripts Overview

This project includes small wrappers around the core modules to make end‑to‑end runs easy.

> Paths listed below show defaults. Most scripts write under `results/<run>/…` when a run
> directory is configured. The run directory is resolved as:
> 1. `results.run_directory` (explicit override), else
> 2. `run.output_subdir` (convenience flag used by `scripts/run_full_pipeline.py`).
>
> This lets you keep multiple experiments side by side without overwriting outputs.

## Emissions

- `scripts/run_calc_emissions.py` — per‑country emissions to `results/<run>/emissions/<mix>/<Country>/`.
  - `--country <Name>` select a specific country (matches config filename).
  - Writes `co2.csv` (and pollutant CSVs) with `year` plus `absolute_*` / `delta_*` columns (Mt/year).
- `scripts/run_calc_emissions_all.py` — aggregates across countries to `results/<run>/emissions/All_countries/<mix>/`.
  - `--countries` restrict to a subset.
  - `--output` change the aggregate output folder; `--results-output` mirror to another directory.

## Air Pollution

- `scripts/run_air_pollution.py`
  - Loads `air_pollution` block from `config.yaml`.
  - Computes per‑pollutant health impacts and optional mortality summaries.
  - Outputs in `results/<run>/air_pollution/<scenario>/`.

## Climate (FaIR)

- `scripts/run_fair_scenarios.py`
  - Uses `climate_module` block.
  - Writes `results/<run>/climate/<emission>_<climate>.csv`.
  - Baseline CSVs are not duplicated in results; each file contains a baseline column.

## Local Climate Impacts

- `scripts/run_local_climate_impacts.py`
  - Scales global climate CSVs to country temperature/precipitation trajectories (2025–2100 by default) and prepares inputs for extreme-weather damages.
  - Writes per-country files under `results/<run>/climate_scaled/<ISO3>/<scenario>.csv` plus an `AVERAGE/` folder containing the equal-weight mean across configured countries.

## Economics (SCC)

- `scripts/run_scc.py`
  - Reads global temperature CSVs and aggregated emissions.
  - Selects the SSP family from the temperature CSVs and auto‑loads GDP/population.
  - Outputs per‑SSP SCC timeseries and per‑scenario damage tables under `results/<run>/economic/`.
  - Aggregation:
    - `per_year`: SCC series printed for selected years in the summary tool.
    - `average`: single SCC per method over the configured horizon.
  - Discounting methods (`--discount-methods` or `economic_module.methods.run`):
    - `constant_discount`, `ramsey_discount` (or both via `all`).
  - Pulse workflow outputs:
    - `pulse_scc_timeseries_<method>_<ssp>.csv` stores the full SCC(τ) path (per SSP) plus the aggregated SCC value.
    - `results/<run>/economic/<mix>/damages_<method>_<scenario>.csv` stores emissions, SCC, and damages
      for each mix/demand/climate combination (used by the summary module).

## Summary

- `scripts/generate_summary.py`
  - Gathers emission, climate, mortality, and SCC results.
  - Writes `results/<run>/summary/summary.csv` plus plots (see `docs/results_summary.md` for column details).
  - Plots deduplicate emission and mortality across SSP suffixes.
  - Use `--run-directory <name>` / `--run-subdir <name>` to point the summary at `results/<name>/…` without editing `config.yaml`.

## Full Pipeline

- `scripts/run_full_pipeline.py`
  - Executes the standard sequence using `config.yaml` defaults.
  - Use `--run-subdir <name>` to write the whole run under `results/<name>/…`.

## Extra Plot Helpers

- `scripts/additional_plots.py` — generates extra emission-savings plots from existing results (writes under `results/<run>/summary/additional_plots/<mix>/` by default).
- `scripts/additional_plots_standalone.py` — standalone variant that only needs `--results-root` (useful when pointing at a copied `results/` tree).

## Cleanup Utilities

- `clean_cache.py` — removes `__pycache__`, `.pytest_cache`, `.ruff_cache`, coverage files.
- `clean_outputs.py` — deletes `results/` (use with caution).

## References
- [CLI usage patterns]:

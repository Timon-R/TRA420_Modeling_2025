# Calc Emissions Module

The `calc_emissions` package converts electricity demand trajectories and
generation mix assumptions into pollutant emission time series expressed in
megatonnes (Mt) per year. Each country has its own configuration in
``data/calc_emissions/countries/``; utility scripts pick the appropriate YAML
and write per-scenario CSVs consumed by downstream modules.

## Workflow Overview

1. **Demand schedule** – each country YAML prescribes electricity demand in
   terawatt-hours (TWh) for the simulation grid.
2. **Generation mix** – technology shares sum to one for every year. Shares can
   be provided as a single value (flat over time) or as a year-indexed map.
3. **Emission factors** – technology-level emission intensity. Columns such as
   ``*_kg_per_kwh``, ``*_mt_per_twh`` or ``*_kt_per_twh`` are accepted and
   harmonised to Mt/TWh automatically. An optional ``gwp100`` column captures
   the overall forcing in CO₂-equivalent units.
4. **Baseline scenario** – sets the reference demand and mix.
5. **Policy scenarios** – reuse named templates or provide custom demand/mix
   entries. Emission deltas are computed relative to the baseline.
6. **Outputs** – per-scenario CSV files (`co2.csv`, `so2.csv`, `nox.csv`,
   `pm25.csv`) under the configured `output_directory`. Each file contains
   `year, delta` columns with deltas in Mt/year.

## Core Equations

For technology *i* in year *t*:

1. Generation (TWh):  
   $ G_{i,t} = D_t \times s_{i,t} $, where $ D_t $ is annual demand (TWh)
   and $ s_{i,t} $ the technology share.

2. Emissions (Mt):  
   $ E_{i,t} = G_{i,t} \times f_{i} $ with $ f_i $ expressed in Mt/TWh after
   harmonisation (kg/kWh inputs are numerically identical to Mt/TWh).

3. Aggregate totals by pollutant:  
   $ E^{\text{tot}}_{p,t} = \sum_i E_{i,t} $

4. Scenario delta vs baseline:  
   $ \Delta E_{p,t} = E^{\text{scenario}}_{p,t} - E^{\text{baseline}}_{p,t} $

## Configuration Keys

### Per-country YAML (stored under ``data/calc_emissions/countries/``)

Each file (e.g. ``config_Albania.yaml``) wraps a full ``calc_emissions`` block:

- ``emission_factors_file`` – country-specific CSV with a ``technology`` column
  and pollutant intensities.
- ``years`` – `{start, end, step}` covering the model horizon. These values should
  match the repository-level ``time_horizon`` entry to maintain consistency across modules.
- ``demand_scenarios`` / ``mix_scenarios`` – named templates that can be reused
  across the country's scenarios.
- ``baseline`` – references the demand/mix scenario used as the reference when
  calculating deltas.
- ``scenarios`` – list of electricity cases (either referencing a named
  template or providing custom demand/mix mappings). Scenario names must align
  across all countries when the top-level scenario filter is used.
- ``output_directory`` / ``results_directory`` – where per-scenario CSVs are
  written (usually under ``resources/calc_emissions/<country>/`` and
  ``results/emissions/<country>/``).

## Outputs

- Per-country runs write deltas to both `resources/calc_emissions/<country>/<scenario>/` (intermediate)
  and `results/emissions/<country>/<scenario>/` (final archive).
- Aggregated multi-country deltas are written to `resources/All_countries/<scenario>/` and mirrored
  to `results/emissions/All_countries/<scenario>/` when `aggregate_results_directory` is configured.
  Aggregated scenarios are also copied directly under `resources/<scenario>/` so the climate module
  can discover them automatically.

### Repository-level metadata (``config.yaml``)

The top-level ``config.yaml`` now stores only the ``calc_emissions.countries``
metadata, which records the directory, filename pattern, aggregate output
location, optional notes file (purely informational), and the list of scenario
names to aggregate across countries. The CLI helpers read this block to discover
country configs.

## Usage Tips

- Supply dense year/value pairs when possible. Sparse entries are interpolated
  linearly via `_values_to_series`.
- Ensure technology names in mixes match the lowercase `technology` column in
  the emission factor file; missing entries default to zero share.
- The module raises when mix shares sum to zero for any year, preventing divide
  by zero during normalisation.
- CO₂ deltas (`co2.csv`) drive the climate pipeline; other pollutants support
  air-quality analyses and can be ingested by downstream modules.
- Run `python scripts/run_calc_emissions_all.py` to process all configured countries and write aggregated multi-country deltas to `resources/All_countries/<scenario>/` (and `results/emissions/All_countries/<scenario>/` when `--results-output` or the config metadata is set).

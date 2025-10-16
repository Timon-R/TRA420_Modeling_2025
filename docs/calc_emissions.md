# Calc Emissions Module

The `calc_emissions` package converts electricity demand trajectories and
generation mix assumptions into pollutant emission time series expressed in
megatonnes (Mt) per year. The outputs feed the climate and air-quality modules.

## Workflow Overview

1. **Demand schedule** – electricity demand is prescribed in terawatt-hours
   (TWh) for each simulation year.
2. **Generation mix** – technology shares sum to one for every year. Shares can
   be provided as a single value (flat over time) or as a year-indexed map.
3. **Emission factors** – technology-level emission intensity:
   - CO₂ in Mt/TWh.
   - SO₂, NOₓ, PM₂.₅ in kt/TWh (converted to Mt by multiplying 1e-3).
4. **Baseline scenario** – sets the reference demand and mix.
5. **Policy scenarios** – reuse named templates or provide custom demand/mix
   entries. Emission deltas are computed relative to the baseline.
6. **Outputs** – per-scenario CSV files (`co2.csv`, `so2.csv`, `nox.csv`,
   `pm25.csv`) under the configured `output_directory`. Each file contains
   `year, delta` columns with deltas in Mt/year.

## Core Equations

For technology *i* in year *t*:

1. Generation:  
   \\( G_{i,t} = D_t \times s_{i,t} \\)  
   where:
   - \\( D_t \\) is annual demand (TWh).  
   - \\( s_{i,t} \\) is the share of technology *i*.

2. Emissions before unit conversion:  
   \\( E^{\*}_{i,t} = G_{i,t} \times f_{i} \\)  
   - \\( f_i \\) is the emission factor (Mt/TWh for CO₂, kt/TWh for other pollutants).

3. Convert non-CO₂ pollutants to Mt:  
   \\( E_{i,t} = E^{\*}_{i,t} \times 10^{-3} \\) for SO₂/NOₓ/PM₂.₅.

4. Aggregate totals by pollutant:  
   \\( E^{\text{tot}}_{p,t} = \sum_i E_{i,t} \\)

5. Scenario delta vs baseline:  
   \\( \Delta E_{p,t} = E^{\text{scenario}}_{p,t} - E^{\text{baseline}}_{p,t} \\)

## Configuration Keys

Define the module under `calc_emissions` in `config.yaml`:

- `emission_factors_file`: CSV with columns `technology`, `co2_mt_per_twh`,
  `so2_kt_per_twh`, `nox_kt_per_twh`, `pm25_kt_per_twh`.
- `years`: `{start, end, step}` to build the simulation grid (inclusive start/end).
- `demand_scenarios`: named maps of `{year: demand_twh}`.
- `mix_scenarios`: named maps of `shares` per technology.
- `baseline`: references named scenarios or custom entries (`demand_custom`,
  `mix_custom`) to establish the reference emissions.
- `scenarios`: list of cases with `name`, and either scenario references or
  custom demand/mix blocks.
- `output_directory`: where delta CSVs are saved (default `resources`).
- `results_directory`: duplicate outputs for archival/reporting.

## Usage Tips

- Supply dense year/value pairs when possible. Sparse entries are interpolated
  linearly via `_values_to_series`.
- Ensure technology names in mixes match the lowercase `technology` column in
  the emission factor file; missing entries default to zero share.
- The module raises when mix shares sum to zero for any year, preventing divide
  by zero during normalisation.
- CO₂ deltas (`co2.csv`) drive the climate pipeline; other pollutants support
  air-quality analyses and can be ingested by downstream modules.

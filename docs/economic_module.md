# Economic Module

The `economic_module` package quantifies damages and computes the social cost of
carbon (SCC) by combining temperature trajectories, emission deltas, GDP, and
discounting assumptions.

## Data Inputs

`EconomicInputs` stores aligned arrays for:

- `years` (integer vector)
- `gdp_trillion_usd`
- Temperature scenarios (°C) keyed by label
- Emission scenarios (t CO₂) keyed by label
- Optional `population_million` (required for Ramsey discounting)

Use `EconomicInputs.from_csv` to load CSV files (`year` column required). Emission
series are multiplied by `emission_to_tonnes` (default converts Mt → t).

When temperature CSVs include the `climate_scenario` column exported by the climate module,
the loader infers the SSP family (SSP1–SSP5) and automatically imports matching GDP and
population projections from `gdp_population_directory` (expects `GDP_SSP1_5.xlsx` and
`POP_SSP1_5.xlsx`, readable via pandas + openpyxl). Provide `gdp_series` only if you need to override the SSP tables with
custom economic trajectories.

## Damage Functions

Default damage function `damage_dice` implements:

\\[
 D(T) = \delta_1 T + \delta_2 T^2
\\]

with optional extensions:

- **Threshold amplification**:  
  \\( D(T) \times \left[1 + s \max(0, T - T_\text{th})^p \right] \\)
- **Saturation** (smooth rational or clamp) ensuring damages stay below `max_fraction`.
- **Catastrophic risk**:
  - Step: add `disaster_fraction` when \\( T \ge T_\text{cat} \\)
  - Probabilistic: add based on \\( 1 - e^{-\gamma \max(0, T - T_\text{cat})} \\)

Configuration keys in `config.yaml` > `economic_module.damage_function`
activate and tune these features.

## Damage Tables

- `compute_damages`: converts temperature series to damages (USD) and returns a
  table with per-scenario temperature, damage fraction, damage USD, and emissions.
- `compute_damage_difference`: subtracts reference damages/emissions from a
  target scenario, yielding `delta_damage_usd` and `delta_emissions_tco2`.

## Discounting Methods

### Constant Discount

Discount factors:  
\\( F_t = (1 + r)^{-(t - t_0)} \\) for \\( t \ge t_0 \\), else 0.

`compute_scc_constant_discount` applies factors to incremental damages and
divides by cumulative emission deltas (or `add_tco2` override) to obtain SCC.

### Ramsey Discount

Requires population data to compute consumption per capita:

\\[
 C_t = \max(GDP_t - \text{damage}_t, 0) \\
 g_t = \frac{C_t - C_{t-1}}{C_{t-1}}
\\]

Discount factors evolve as:
\\[
 F_{t+1} = \frac{F_t}{1 + \rho + \eta g_{t+1}}
\\]

`compute_scc_ramsey_discount` stores per-year consumption growth, discount
factors, and discounted damages in the result details.

## SCC Outputs

`compute_scc_*` return `SCCResult` containing:

- `scc_usd_per_tco2`: aggregated SCC value.
- `per_year`: table with year, damage/emission deltas, discount factors,
  incremental damage deltas (year-on-year changes), discounted incremental
  damages, and the marginal SCC path (`discounted_incremental_delta_usd` divided
  by the annual emission deltas).
- `details`: richer diagnostics (e.g., consumption metrics for Ramsey).

### SCC Formulas (conceptual)

Let `ΔD_t` be the damage difference (scenario − reference) in year `t` and `ΔE_t`
the emission delta in t CO₂ for year `t`. Let `DF_t` be the discount factor for `t`.

- Per‑year SCC series (`aggregation: per_year`):
  - Incremental damage change: `ΔΔD_t = ΔD_t − ΔD_{t−1}` (with `ΔΔD_{t0} = ΔD_{t0}`)
  - SCC(t) = `DF_t × ΔΔD_t / ΔE_t`

- Aggregated SCC (`aggregation: average`):
  - Numerator: `Σ_t DF_t × ΔD_t`
  - Denominator: `Σ_t ΔE_t`
  - SCC = Numerator / Denominator

## Usage

CLI wrapper `scripts/run_scc.py` orchestrates:

- Loading temperature/emission CSVs (labels must match).
- Selecting reference/target scenarios.
- Choosing methods: `constant_discount`, `ramsey_discount`, or both.
- Applying damage settings from config or CLI overrides (`--damage-*` flags).

## Configuration Summary

Under `economic_module` in `config.yaml`:

- `output_directory`: SCC result tables.
- `gdp_series`: path to GDP (+ optional population) CSV.
- `temperature_series`, `emission_series`: scenario label → CSV path.
- `reference_scenario`, `evaluation_scenarios`: default evaluation set.
- `base_year`: PV reference year (must be in the time grid).
- `aggregation`: `average` or `per_year`.
- `damage_function`: DICE coefficients and optional threshold/saturation/catastrophe settings.
- `methods`: discounting methods and parameters (`discount_rate`, `rho`, `eta`).
- `emission_to_tonnes`: column unit conversion for emission deltas.

## Validation Tips

- Ensure emission and temperature series cover identical years; `EconomicInputs`
  enforces this.
- Provide population data when using Ramsey discounting; otherwise a
  `ValueError` is raised.
- Run the pytest suite (`python -m pytest`) after modifying damage logic or
  discounting to confirm analytic expectations.

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

The loader determines the overlapping `[start, end]` window across GDP, emissions,
and temperature inputs, then resamples everything onto an annual grid within that
window. Missing intermediate years are filled via linear interpolation (with edge
forward/back fills) so downstream SCC calculations always operate on one-year steps
regardless of the reporting cadence in the source CSVs.

When temperature CSVs include the `climate_scenario` column exported by the climate module,
the loader infers the SSP family (SSP1–SSP5) and automatically imports matching GDP and
population projections from `gdp_population_directory` (defaults to the IIASA Scenario Explorer
extracts under `data/GDP_and_Population_data/IIASA/{GDP,Population}.csv`). The IIASA tables end in
2100; the SCC evaluation window is therefore truncated to that year unless you switch to the DICE
mode to synthesize longer trajectories. Provide `gdp_series` only if you need to override the
IIASA tables with custom economic trajectories.

Set `socioeconomics.mode: ssp` (default) to use the IIASA datasets matched to the climate
scenario’s SSP family automatically. Switch to `mode: dice` when you want the endogenous DICE
projections described below.

### DICE socioeconomics mode

Alternatively, populate the top-level `socioeconomics` block in `config.yaml` with
`mode: dice`. The associated `dice` settings use the DICE capital-TFP-population
equations (logistic population, declining TFP growth, Cobb–Douglas output, and a
fixed savings/depreciation pair) to generate a GDP/population table that spans the
full SCC evaluation horizon. `run_scc.py` feeds this synthetic series into
`EconomicInputs.from_csv` via the new `gdp_frame` parameter, so no SSP workbook or
GDP CSV is required when you want a purely DICE-consistent socio-economic baseline.
Set `socioeconomics.dice.scenario` to an explicit SSP (e.g. `SSP2`) or to
`as_climate_scenario` so each climate pathway automatically reuses the matching
SSP family when building its DICE projection.

## Damage Functions

Default damage function `damage_dice` implements:

$$
D(T) = \delta_1 T + \delta_2 T^2
$$

with optional extensions:

- **Threshold amplification**:  
  $ D(T) \times \left[1 + s \max(0, T - T_\text{th})^p \right] $
- **Saturation** (smooth rational or clamp) ensuring damages stay below `max_fraction`.
- **Catastrophic risk**:
  - Step: add `disaster_fraction` when $ T \ge T_\text{cat} $
  - Probabilistic: add based on $ 1 - e^{-\gamma \max(0, T - T_\text{cat})} $

### Custom polynomial coefficients

Set `damage_function.mode` to `dice` (default) to use the quadratic baseline, or to
`custom` to activate bespoke coefficients. In custom mode you supply a list of arbitrary
coefficient/exponent pairs via `damage_function.custom_terms`. Each term is applied as
`coefficient × T^exponent`. For example, to implement `D = 0.01202·T + 0.01724·T^{1.5}`:

```yaml
economic_module:
  damage_function:
    mode: custom
    custom_terms:
      - coefficient: 0.01202
        exponent: 1.0
      - coefficient: 0.01724
        exponent: 1.5
    use_threshold: false
    use_saturation: false
```

All other options (threshold amplification, saturation, catastrophic add-ons) work
in combination with either mode. If `custom_terms` is omitted (or mode stays `dice`),
the baseline reverts to `delta1 * T + delta2 * T^2`.

Configuration keys in `config.yaml` > `economic_module.damage_function`
activate and tune these features.

## Damage Tables

- `compute_damages`: converts temperature series to damages (USD) and returns a
  table with per-scenario temperature, damage fraction, damage USD, and emissions.
- `compute_damage_difference`: subtracts reference damages/emissions from a
  target scenario, yielding `delta_damage_usd` and `delta_emissions_tco2`.

## Discounting Methods

The codebase now supports a single SCC method — the definition-faithful pulse workflow —
but two discounting families can be applied within that workflow:

- **Constant discount** — provide `discount_rate` and the pulse run applies  
  $ F_t = (1 + r)^{-(t - t_0)} $ to each year $ t \ge t_0 $ (else 0). This is the
  familiar fixed-rate PV calculation. The aggregated SCC divides the discounted
  damages by the cumulative emission delta (or `add_tco2` override).
- **Ramsey discount** — provide `rho` and `eta`. Consumption per capita is derived
  from GDP minus damages, growth is  
  $ g_t = (C_t - C_{t-1})/C_{t-1} $, and discount factors evolve via  
  $ F_{t+1} = F_t / (1 + \rho + \eta g_{t+1}) $. Population data is therefore required.

Both schemes share the same pulse machinery; only the discount factors differ.
`SCCResult.details` records the per-year discount factor and discounted damages used in
each calculation so you can audit the PV steps.

## SCC Outputs

`compute_scc` (pulse-only) returns `SCCResult` containing:

- `scc_usd_per_tco2`: aggregated SCC value (PV of Δ damages ÷ cumulative Δ emissions).
- `per_year`: emission-year table with emissions, damages, discount factors, attributed PV
  damages, and the SCC(τ) time series. Column highlights:
  - `delta_damage_usd`: damages realized in the reporting year (undiscounted).
  - `discounted_delta_usd`: the same damages expressed in present-value USD for `base_year`.
- `damage_attributed_usd`: undiscounted damages allocated to the emission year τ via the pulse workflow.
- `discounted_damage_attributed_usd`: present-value (base-year) damages attributable to the emission year τ.
- `scc_usd_per_tco2`: SCC(τ) derived as $ \sum_s β_s \Delta D_{s|\tau} / (β_\tau \Delta E_\tau) $; numerically this equals the emission-year SCC expressed in base-year USD because both numerator and denominator are scaled by the corresponding discount factors.
- `pulse_size_tco2`: size of the perturbation applied in each pulse run (from `run.pulse`).
- `details`: diagnostics (temperature deltas, damages, discount factors).
- `run_method`: `"pulse"` indicating the FaIR pulse workflow produced the result.

Units: GDP and damages originate from the IIASA tables (reported in 2023 PPP USD). The loader applies
the BEA GDPDEF conversion factor (1.05) to express them in PPP USD-2025, so every SCC value is
reported as present-value PPP-2025 USD per tonne CO₂ evaluated at `base_year` (2025 in the default
configuration). The annual SCC series (`scc_usd_per_tco2` column in the timeseries files) is
discounted to the emission year τ; the summary converts those values into PPP_USD_2025 but keeps the
“discounted-to-year” interpretation explicit in the column names. Adjust the GDP inputs if you need
a different currency base or exchange benchmark.

Because emissions are stored in tonnes of CO₂ after applying `emission_to_tonnes`
(default $10^6$ converts Mt → t), any downstream multiplication of SCC × ΔE produces
damages in USD that are already discounted to `base_year`. The results summary uses this
relationship to report per-emission-year damages for each discounting method.

### Pulse Method (definition-faithful)

The SCC pipeline always uses the textbook pulse workflow. It reuses the FaIR configuration identified by the `climate_scenario`
metadata that already backs the temperature inputs—no extra “baseline-only”
FaIR run is issued. Instead, for each emissions year τ the pipeline prepares a
box pulse (a +`pulse_size_tco2` top-hat that lasts exactly one calendar year)
on top of that same baseline scenario and evaluates FaIR once:

1. Build the pulse by taking the difference of two `step_change` adjustments,
   so emissions jump up at τ and drop back down at τ+1.
2. Call FaIR for the adjusted pathway. FaIR internally runs the baseline and
   adjusted configurations together, so this single call yields both the
   unperturbed trajectory and the pulse-perturbed one. There is exactly **one**
   FaIR evaluation per pulse year.
3. Record the incremental temperature response `ΔT_{t|τ}` from the FaIR result.
4. Apply the configured damage function to obtain `ΔD_{t|τ}`.
5. Discount the damages with the selected rule (constant or Ramsey), evaluate
   the PV at the emissions year τ, and divide by the pulse size. SCC(τ) is
   therefore reported in τ-year units; multiply by the discount factor β_τ to
   express it in the base-year currency.

Regardless of how coarsely emissions were reported originally, the pulse workflow
evaluates every calendar year in the overlapping data window; the CSV loader
interpolates intermediate years so SCC(τ) values remain invariant to the reporting
step.

This captures full state dependence and non-linear damages at the cost of one FaIR call
per pulse year (each call bundles the baseline+pulse pair). Pulse size is controlled by
`run.pulse.pulse_size_tco2`. Results are cached across runs with identical pulse settings
so repeat evaluations avoid re-running FaIR.

### Files written by `scripts/run_scc.py`

Each climate pathway / discount method pair produces:

| File | Contents | Consumer |
|------|----------|----------|
| `results/economic/pulse_scc_timeseries_<method>_<ssp>.csv` | `year`, discount factor, PV damages attributed to the pulse, per-year `scc_usd_per_tco2`, `pulse_size_tco2`, emission deltas, and `discounted_delta_usd`. | Results summary (SCC columns + plots) |
| `results/economic/<mix>/damages_<method>_<scenario>.csv` | Scenario-specific emissions, SCC, and monetised damages (Ramsey or constant discount) for every mix/demand/SSP combination. | Results summary (damage columns + plots) |

Legacy `scc_summary.csv` / `scc_timeseries_<method>_<scenario>.csv` files have been removed. Aggregated SCC values come directly from the pulse files (they are identical to `SCCResult.scc_usd_per_tco2`).

When you only need SCC outputs, set `economic_module.write_damages: false` (or run
`scripts/run_scc.py --skip-damages`) to suppress the emission-scenario damage tables. This is
useful when no updated emission deltas are available but you still want fresh SCC series.

### Aggregate SCC

The average SCC (used when `aggregation: average`) retains the familiar ratio:

- Numerator: base-year PV of damages, $ \sum_t β_t \Delta D_t $.
- Denominator: total emission delta, either $ \sum_t \Delta E_t $ or the explicit
  `add_tco2` override.
- SCC = Numerator / Denominator.

## Usage

CLI wrapper `scripts/run_scc.py` orchestrates:

- Loading temperature/emission CSVs (labels must match) and inferring the SSP family.
- Selecting reference/target scenarios (all demand cases of every mix by default).
- Choosing discount methods: `constant_discount` and/or `ramsey_discount` (pulse workflow only).
- Applying damage settings from config or CLI overrides (`--damage-*` flags).

## Configuration Summary

Under `economic_module` in `config.yaml`:

- `output_directory`: SCC result tables.
- `gdp_series`: path to GDP (+ optional population) CSV.
- `temperature_series`, `emission_series`: scenario label → CSV path.
- `reference_scenario`, `evaluation_scenarios`: default evaluation set (overridden when running via the full pipeline, which loops through all mix/demand cases automatically).
- `base_year`: PV reference year (must be in the time grid).
- `aggregation`: `average` or `per_year`.
- `damage_function`: DICE coefficients and optional threshold/saturation/catastrophe settings.
- `run.pulse`: pulse size (tCO₂) and optional pulse horizon; the CLI caches per-SSP results so repeated runs reuse pulse data when the inputs match.
- `run.pulse.pulse_size_tco2`: CO₂ amount applied in each pulse year (tonnes).
- `methods`: discounting configuration (`run` selects constant/Ramsey or list).
- `emission_to_tonnes`: column unit conversion for emission deltas.

## Validation Tips

- Ensure emission and temperature series cover identical years; `EconomicInputs`
  enforces this.
- Provide population data when using Ramsey discounting; otherwise a
  `ValueError` is raised.
- Run the pytest suite (`python -m pytest`) after modifying damage logic or
  discounting to confirm analytic expectations.

## References

- [GDP/Population conversion]: U.S. Bureau of Economic Analysis, Gross Domestic Product: Implicit Price Deflator (GDPDEF), retrieved via Federal Reserve Bank of St. Louis (FRED), used to convert IIASA GDP PPP-2023 to PPP USD-2025 with factor 1.05.
- [Socioeconomics DICE method]: Rickels, W., Meier, F., & Quaas, M. (2023). The historical social cost of fossil and industrial CO2 emissions. Nature Climate Change 2023 13:7, 13(7), 742–747. https://doi.org/10.1038/s41558-023-01709-1
- [Ramsey discount paramters]: Nesje, F., Drupp, M. A., Freeman, M. C., & Groom, B. (2023). Philosophers and economists agree on climate policy paths but for different reasons. Nature Climate Change 2023 13:6, 13(6), 515–522. https://doi.org/10.1038/s41558-023-01681-w
      rho: 0.00625 #average betweeen mean philosopher and economist view

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

- `scc_usd_per_tco2`: aggregated SCC value (PV of Δ damages ÷ cumulative Δ emissions).
- `per_year`: emission-year table with emissions, damages, discount factors, attributed PV
  damages, and the SCC(τ) time series. Column highlights:
  - `delta_damage_usd`: damages realized in the reporting year (undiscounted).
  - `discounted_delta_usd`: the same damages expressed in present-value USD for `base_year`.
  - `damage_attributed_usd`: undiscounted damages allocated to the emission year τ via the kernel/pulse workflow.
  - `discounted_damage_attributed_usd`: present-value (base-year) damages attributable to the emission year τ.
  - `scc_usd_per_tco2`: SCC(τ) derived as \( \sum_s β_s \Delta D_{s|\tau} / (β_\tau \Delta E_\tau) \); numerically this equals the emission-year SCC expressed in base-year USD because both numerator and denominator are scaled by the corresponding discount factors.
  - `pulse_size_tco2`: (pulse mode only) size of the perturbation applied in each pulse run.
- `details`: diagnostics (temperature deltas, damages, discount factors).
- `temperature_kernel`: temperature impulse response when the kernel method is used.
- `run_method`: `"kernel"` or `"pulse"` indicating which workflow produced the result.

Because emissions are stored in tonnes of CO₂ after applying `emission_to_tonnes`
(default \(10^6\) converts Mt → t), any downstream multiplication of SCC × ΔE produces
damages in USD that are already discounted to `base_year`. The results summary uses this
relationship to report per-emission-year damages for each discounting method.

### Pulse Method (definition-faithful)

Set `economic_module.run.method: pulse` when you want the textbook SCC(τ).
The workflow uses the FaIR configuration identified by the `climate_scenario`
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

**Pros**
- Exact by definition (discounted marginal damages for the emission year).
- Handles non-linear damages and FaIR state dependence naturally.

**Cons**
- Slower (one FaIR evaluation per emission year).
- Only one discounting method can be active per run.

### Kernel Method (impulse-response approximation)

Set `economic_module.run.method: kernel` for a fast approximation. In plain language:

1. Treat the observed temperature difference as the result of past emission changes
   filtered through a response curve `g(ℓ)` and solve for that curve via ridge-regularised
   least squares.
2. Use `g(ℓ)` to attribute temperature—and therefore damages—to the year the emissions
   occurred.
3. Discount and scale exactly as in the pulse method to obtain SCC(τ) (which is expressed in τ-year
   units; multiply by β_τ for base-year terms if needed).

Diagnostics (reconstruction identity, welfare identity) are logged so you can gauge the
quality of the fit. If damages are strongly non-linear, consider enabling
`run.kernel.allocation.linearized_damage` or switching to the pulse workflow for audits.

### Tuning Knobs

Kernel estimation and damage allocation are configurable. Defaults honour the full kernel
length, a ridge of 1e-6, no smoothing, and non-linear damages.

```yaml
economic_module:
  kernel:
    horizon: null              # Optional integer L; null = full length
    regularization_alpha: 1.0e-6   # Ridge α added to (CᵀC)
    nonnegativity: false       # Clip kernel to g(ℓ) ≥ 0
    smoothing_lambda: 0.0      # Roughness penalty weight
    allocation:
      linearized_damage: false # Use ∂D/∂T × ΔTτ instead of full damage diff
  run:
    method: kernel             # 'kernel' (fast) or 'pulse' (definition-faithful pulses)
  methods:
    run: ramsey_discount       # Discounting choice (constant / ramsey / list)
    constant_discount:
      discount_rate: 0.03
    ramsey_discount:
      rho: 0.005
      eta: 1.5
    pulse:
      pulse_size_tco2: 1.0e6   # Pulse size when using pulse runs
```

Notes
- `regularization_alpha` corresponds to the ridge added to (CᵀC).
- `horizon` truncates g(ℓ) beyond L lags to reduce variance and noise.
- `nonnegativity`/`smoothing_lambda` provide physical, regularised kernels.
- `linearized_damage` enforces additive damages at the cost of ignoring damage curvature.

### Aggregate SCC

The average SCC (used when `aggregation: average`) retains the familiar ratio:

- Numerator: base-year PV of damages, \( \sum_t β_t \Delta D_t \).
- Denominator: total emission delta, either \( \sum_t \Delta E_t \) or the explicit
  `add_tco2` override.
- SCC = Numerator / Denominator.

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
- `run.method`: `kernel` or `pulse`.
- `run.kernel`: tuning for the kernel estimator (`horizon`, `regularization_alpha`, optional `nonnegativity`, `smoothing_lambda`).
- `run.kernel.allocation.linearized_damage`: toggle numerical ∂D/∂T × ΔTτ allocation.
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

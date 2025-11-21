# Climate Module

The `climate_module` package wraps FaIR to simulate global-mean surface
temperature trajectories for baseline and emission-adjusted scenarios.

## Key Components

- `compute_temperature_change` (in `FaIR.py`): runs FaIR for a selected SSP
  scenario and computes baseline vs adjusted temperature paths.
- `TemperatureResult`: container with `years`, `timepoints` (mid-year),
  `baseline`, `adjusted`, and convenience properties `delta`, `final_delta`,
  `to_frame()`.
- `DEFAULT_SPECIES`: curated subset of FaIR species covering major greenhouse
  gases and aerosol forcings.
- `CLIMATE_PRESETS`: preset parameter bundles (`ar6`, `two_box`) configuring
  ocean heat capacities, transfer coefficients, deep-ocean efficacy, and the
  4×CO₂ forcing strength.
- `scenario_runner.py`: orchestrates batches of scenarios (`ScenarioSpec`,
  `run_scenarios`, `step_change`) while managing time grids and emission
  adjustments.

## Simulation Flow

1. **Time grid** – defined via `start_year`, `end_year`, and `timestep`. FaIR
   uses bounds and mid-year timepoints (`timestep / 2` offset).
2. **Scenarios/configs** – FaIR is configured with two configs: `baseline` and
   `adjusted`. Emission adjustments apply to `adjusted` only.
3. **Species definition** – `_species_properties` extracts the relevant subset
   from FaIR's AR6 catalogue. Emission adjustments are restricted to species
   with `input_mode == "emissions"`.
4. **Climate setup** – `_apply_climate_setup` loads preset or custom climate
   parameters. Optional overrides (e.g., `forcing_4co2`, `equilibrium_climate_sensitivity`)
   are applied afterwards.
5. **Initial state** – `_initialise_model_state` populates concentration,
   cumulative emissions, airborne emissions, and temperature at the first time
   bound.
6. **Emission adjustments** – `_apply_emission_adjustments` transforms scalars,
   arrays, or callables (via `_to_delta_array`) into per-timepoint emission
   deltas added to the `adjusted` configuration.
7. **FaIR run** – `model.run(progress=False)` produces temperature time series.

## Equations & Transformations

- Scalar emission perturbations are expanded as  
  $ \Delta E_t = \text{scalar} $ for all timepoints.
- Callable adjustments `f(timepoints, cfg)` must return an array matching the
  timepoint length; results are interpreted as Gt CO₂/yr.
- The `step_change` helper creates a piecewise constant change:

$$
\Delta E(t) =
    \begin{cases}
      0 & t < T_{\text{start}} \\
      \Delta & t \ge T_{\text{start}}
    \end{cases}
$$

  where `Δ` is specified in Mt CO₂/yr and converted to Gt inside the runner.

- FaIR climate parameters are broadcast to `(config, layer)` matrices using
  `_broadcast_matrix` and `_broadcast_vector`.

## Configuration (`config.yaml`)

Under `climate_module`:

- `output_directory`: location for archived climate CSVs.
- `resource_directory`: copy of outputs consumed by other modules.
- `sample_years_option`: `default` (5-year steps to 2050, then 10-year),
  or `full` (every year).
- `parameters`: global defaults (`start_year`, `end_year`, `timestep`,
  `climate_setup`, and optional overrides such as `deep_ocean_efficacy`,
  `forcing_4co2`, `equilibrium_climate_sensitivity`, and the anomaly
  reference window via `warming_reference_start_year` / `_end_year`). Runs now
  begin in 1750 so calibrated historical drivers are always replayed before
  branching into SSP pathways.
- `climate_scenarios`: list of FaIR/RCMIP pathways with per-scenario options:
  - `id`: scenario identifier (e.g., `ssp245`).
  - `label`: output label suffix.
  - `apply_adjustment`: toggle emission adjustments.
  - `adjustment_specie`, `adjustment_delta`, `adjustment_start_year`,
    `adjustment_timeseries_csv`: control the emission perturbation.
  - Time grid overrides (`start_year`, `end_year`, `timestep`) and climate
    overrides (`ocean_heat_capacity`, etc.).
- `emission_scenarios`: selects which emission folders to load (each must
  contain `co2.csv` with `year, delta` in Mt CO₂/yr).

### Background climate exports

Every execution writes baseline (reference emission) trajectories to
`background_climate_full.csv` (1750 up to the extended climate horizon) and
`background_climate_horizon.csv` (restricted to the configured `time_horizon`).
The corresponding plots are now emitted directly under
`results/summary/plots` (mirrored into run directories when configured) so the
summary module can reference them without extra copying. Temperatures are
expressed as anomalies relative to the `warming_reference_start/end_year`
window (default 1850–1900), ensuring the economic module observes the same
background climate when post-processing SCC results.

### FaIR calibration block

The optional `climate_module.fair.calibration` block activates the IGCC-aligned
parameter set that ships under `data/FaIR_calibration_data/v1.5.0`. Set
`enabled: true` and point `base_path` to the folder containing the CSVs. The
following options customise the run:

- `ensemble_file`, `ensemble_member_id`/`ensemble_member_index`: pick a row from
  `calibrated_constrained_parameters.csv` (defaults to sample `1299`). This row
  replaces FaIR’s ocean heat capacities, transfer coefficients, deep-ocean
  efficacy, forcing scales, F₂×, and CO₂ baseline concentration.
- `species_file`, `co2_species_name`: select the calibrated carbon-cycle pools
  (defaults to row `CO2`).
- `ch4_lifetime_file`, `ch4_lifetime_label`: override CH₄ lifetimes and chemical
  sensitivities (defaults to the `historical_best` row in `CH4_lifetime.csv`).
- `historical_emissions_file`, `solar_forcing_file`, `volcanic_forcing_file`:
  CMIP7 driver tables replayed from 1750 prior to the SSP branch.
- `landuse_scale_file`/`lapsi_scale_file` and labels: provide scalar tweaks for
  land-use and LAPSI forcing streams (optional—defaults include the
  `historical_best` rows).
- `warming_baselines_file`, `warming_baseline_label`,
  `warming_baseline_column`: retain metadata on the IGCC baseline/target
  windows for reporting (no extra computation required in the runner).

All files are consumed directly from disk—no network downloads or `pooch`
helpers are required. Once configured, every `run_fair_scenarios.py` execution
uses the calibrated parameters, ensuring the background climate matches the
historical constraints before adding emission deltas.

## Outputs

`run_fair_scenarios.py` writes per-scenario CSVs containing:

- `year`
- `temperature_baseline`
- `temperature_adjusted`
- `temperature_delta`
- `climate_scenario` – SSP identifier (e.g., `ssp245`) used downstream by the economic module to
  choose matching GDP and population data.

These are written under `results/climate` for downstream use (e.g., the
economic module). To avoid duplication, baseline-only files are no longer
mirrored elsewhere — each scenario CSV already includes the baseline
temperature column.

## Usage

Run the FaIR wrapper after emission deltas exist in `results/emissions/All_countries/<mix>/co2.csv`:

```bash
python scripts/run_fair_scenarios.py
```

Optional flags allow overriding `config.yaml` paths or dumping intermediate
arrays; consult the script docstring for details. When integrating inside
Python code, import `climate_module.scenario_runner.run_scenarios` and provide
`ScenarioSpec` objects.

## Testing Considerations

- The module requires the `fair` Python package. Tests skip if FaIR is not
  installed.
- Unit tests focus on deterministic helpers (`TemperatureResult`, broadcasting,
  delta conversion); full FaIR integrations are best exercised in notebook or
  regression workflows due to run-time cost.

## References
- [FaIR model and calibration files]: Smith, C. (2025). fair calibration data (1.5.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17392386


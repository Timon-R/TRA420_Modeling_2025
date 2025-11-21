# TRA420_Modeling_2025

Modeling pipeline for the TRA420 course project: starting from regional energy demand and elasticity
assumptions, deriving energy mixes, estimating emissions, linking those emissions to global
temperature responses, and evaluating local/global impact metrics such as the Social Cost of Carbon
(SCC).

## Repository Structure

- `src/`
  - `calc_emissions/` — converts electricity demand and mix into emission deltas.
  - `climate_module/` — FaIR wrappers and scenario tools.
  - `air_pollution/` — maps non-CO₂ deltas to concentration-driven health impacts.
  - `economic_module/` — SCC utilities (damages, discounting, reporting).
  - `pattern_scaling/` — pattern-scaling of global responses to country trajectories.
  - *(planned)* `impacts/`, `ui/`, `common/` packages that will divide the workflow into focused modules as they arrive.
- `scripts/` — CLI helpers such as `run_fair_scenarios.py` for quick experiments.
- `data/` — input datasets (raw and processed). Includes `calc_emissions/` (country configs and
  emission factors), air-pollution statistics, GDP/Population tables, and pattern-scaling factors.
  The canonical Excel workbooks for electricity mixes and technology intensities
  (`Electricity_OECD.xlsx`, `Emission_factors_all.xlsx`) now live under
  `data/calc_emissions/`.
- `results/` — generated outputs like emission mixes (`results/emissions/<mix>/<Country>/` and `results/emissions/All_countries/<mix>/`), climate CSVs, and summary tables (ignored by Git).
- `tests/` — pytest suite covering emissions, climate, economic, and pattern-scaling modules.
- `config.yaml` — project-level configuration (scenario metadata, default parameters).
- `environment.yaml` — preferred Python environment specification for reproducibility.
- `pyproject.toml` — project metadata plus Ruff lint/format configuration.
- `.gitignore` — excludes generated results and other artifacts.
- `README.md` — usage guidance and development conventions.
 - `docs/` — module documentation and CLI guides (`economic_module.md`, `climate_module.md`, `pattern_scaling.md`, `air_pollution.md`, `results_summary.md`, `scripts.md`).

## Getting Started

1. **Create the environment**

   ```bash
   mamba env create -f environment.yaml  # or `conda` / `uv` once finalized
   conda activate tra420-modeling
   ```

1. **Install the project in editable mode** (once package metadata is finalized):

   ```bash
   pip install -e .
   ```

## Running the Pipeline

### Electricity emissions per country

Run the calculator for one country (writes deltas under `results/emissions/per_country/<Country>/`). Country names correspond to `config_<name>.yaml` files in `data/calc_emissions/countries/`:

```bash
python scripts/run_calc_emissions.py --country Albania
```

Valid names match config filenames (underscores instead of spaces), e.g. `Serbia`, `Bosnia-Herzegovina`, `North_Macedonia`, `Kosovo`, `Montenegro`.

### Aggregate emissions across all countries

Compute deltas for all countries and write the sum by mix to `results/emissions/All_countries/<mix>/co2.csv` (with per-demand columns):

```bash
python scripts/run_calc_emissions_all.py
```

Options:

- Restrict to specific countries:

  ```bash
  python scripts/run_calc_emissions_all.py --countries Albania Serbia
  ```

- Choose a different output directory or mirror results elsewhere:

  ```bash
  python scripts/run_calc_emissions_all.py --output results/emissions/All_countries_custom --results-output results/emissions/All_countries_custom_copy
  ```

Outputs mirror the per-country structure (`co2.csv`, `nox.csv`, `sox.csv`, `pm25.csv`, and `gwp100.csv` when available) with `absolute_*`/`delta_*` columns for each demand case. The list of country configs, scenario filter, and the default aggregate output directories are configurable via the `calc_emissions.countries` block in `config.yaml`.

### Full end-to-end run

To execute emissions, climate, pattern scaling, air-pollution and SCC modules in a single command (using the defaults from `config.yaml`):

```bash
python scripts/run_full_pipeline.py
```

Typical workflow (driven by `config.yaml`):

1. **Emissions** – `python scripts/run_calc_emissions.py --country <name>` (per-country deltas) or `python scripts/run_calc_emissions_all.py` (aggregated deltas); run before downstream modules so `results/emissions/<mix>/<Country>/` and `results/emissions/All_countries/<mix>/` hold current data.
2. **Air-pollution impacts** – `python scripts/run_air_pollution.py` combines non-CO₂ deltas with concentration stats to estimate mortality percentage changes.
3. **Global climate** – `python scripts/run_fair_scenarios.py` writes `results/climate/*.csv`. Each CSV now includes a `climate_scenario` column. The run also produces background baseline CSVs (`background_climate_full.csv`, `background_climate_horizon.csv`) plus plots written directly to `results/summary/plots` so downstream SCC runs and the final summary share the same reference trajectory.
4. **Pattern scaling (optional)** – `python scripts/run_pattern_scaling.py` consumes the global climate CSVs plus the scaling factors table and produces per-country files under `pattern_scaling.output_directory`.
5. **Economics** – `python scripts/run_scc.py` auto-selects the SSP GDP/population series based on `climate_scenario` and evaluates discounting methods configured in `config.yaml`.
6. **Summary** – `PYTHONPATH=src python scripts/generate_summary.py` compiles key indicators and plots. Emission and mortality plots collapse SSP suffixes (identical across climate pathways), while SCC and temperature remain pathway-specific. The summary plot folder now also mirrors the background climate graphics and adds a `socioeconomics.png` panel showing the GDP/population trajectories used in the SCC calculations.

### Named runs & scenario suites

- Set `run.output_subdir` in `config.yaml` (or pass `--run-subdir <name>` to
  `scripts/run_full_pipeline.py`) to keep outputs under `results/<name>/…`.
- For batch experiments set `run.mode: scenarios` and point
  `run.scenario_file` to a YAML mapping scenario names to overrides. Each entry
  can tweak any part of the base config; the pipeline deep-merges the overrides,
  forces `run.mode` back to `normal`, assigns a per-scenario subdirectory
  (`results/<suite>/<scenario>/…`), and runs the full workflow. After the suite
  finishes an aggregate CSV/JSON plus a copy of the scenario YAML are written to
  `results/<suite>/` for provenance.

## Socioeconomics & FaIR calibration

- The new `socioeconomics` block in `config.yaml` (default mode `dice`) replaces the old SSP lookup with a DICE-style projection. Population follows logistic growth using scenario-specific asymptotes, total factor productivity declines gradually, and capital evolves via DICE’s savings/depreciation rules. These trajectories feed directly into `run_scc.py` without relying on external GDP/POP workbooks.
- The climate module now supports calibrated FaIR runs via `climate_module.fair.calibration`. Provide the local calibration dataset once (this repo ships `data/FaIR_calibration_data/v1.5.0`) and the runner will:
  - pick the requested posterior ensemble member,
  - replace FaIR’s carbon-cycle pools, CH₄ lifetime terms, and F₂× forcing with the calibrated values,
  - replay the CMIP7 historical emissions plus solar/volcanic forcings from 1750 before transitioning to each SSP scenario.
  No `pooch` downloads are required—the files are read directly from the specified `base_path`.

## Testing

Install development dependencies and run the pytest suite from the project root:

```bash
pip install -e '.[dev]'
python -m pytest
```

Individual modules can be exercised via `python -m pytest tests/test_calc_emissions.py` (or similar) when iterating quickly. Continuous integration expects the full suite to pass before changes are submitted.
### Electricity emissions per country

Run the calculator for one country (writes deltas under `results/emissions/per_country/<Country>/`). Country names correspond to `config_<name>.yaml` files in `data/calc_emissions/countries/`:

```bash
python scripts/run_calc_emissions.py --country Albania
```

Valid names match config filenames (underscores instead of spaces), e.g. `Serbia`, `Bosnia-Herzegovina`, `North_Macedonia`, `Kosovo`, `Montenegro`.

Programmatic API
-----------------

If you prefer to call the per-country writer from other Python code (for example
in tests or custom orchestration), import the helper from the `src` package:

```python
from calc_emissions.writers import write_per_country_results

# per_country_map should be: { CountryName: { scenario_name: EmissionScenarioResult } }
write_per_country_results(per_country_map, Path("results/emissions"))
```

This produces the same `results/emissions/<mix>/<Country>/<pollutant>.csv` layout used by the
aggregator script and is a stable import for other modules.

### Aggregate emissions across all countries

Compute deltas for all countries and write the sum by mix to `results/emissions/All_countries/<mix>/co2.csv` (with per-demand columns):

```bash
python scripts/run_calc_emissions_all.py
```

Options:

- Restrict to specific countries:

  ```bash
  python scripts/run_calc_emissions_all.py --countries Albania Serbia
  ```

- Choose a different output directory or mirror results elsewhere:

  ```bash
  python scripts/run_calc_emissions_all.py --output results/emissions/All_countries_custom --results-output results/emissions/All_countries_custom_copy
  ```

Outputs mirror the per-country structure (`co2.csv`, `nox.csv`, `sox.csv`, `pm25.csv`, and `gwp100.csv` when available) with `absolute_*`/`delta_*` columns. The list of country configs, scenario filter, and the default aggregate output directories are configurable via the `calc_emissions.countries` block in `config.yaml`.

### Full end-to-end run

To execute emissions, climate, pattern scaling, air-pollution and SCC modules in a single command (using the defaults from `config.yaml`):

```bash
python scripts/run_full_pipeline.py
```

Typical workflow (driven by `config.yaml`):

1. **Emissions** – `python scripts/run_calc_emissions.py --country <name>` (per-country deltas) or `python scripts/run_calc_emissions_all.py` (aggregated deltas); run before downstream modules so `results/emissions/<mix>/<Country>/` and `results/emissions/All_countries/<mix>/` hold current data.
2. **Air-pollution impacts** – `python scripts/run_air_pollution.py` combines non-CO₂ deltas with concentration stats to estimate mortality percentage changes.
3. **Global climate** – `python scripts/run_fair_scenarios.py` writes `results/climate/*.csv`. Each CSV now includes a `climate_scenario` column.
4. **Pattern scaling (optional)** – `python scripts/run_pattern_scaling.py` consumes the global climate CSVs plus the scaling factors table and produces per-country files under `pattern_scaling.output_directory`.
5. **Economics** – `python scripts/run_scc.py` auto-selects the SSP GDP/population series based on `climate_scenario` and evaluates discounting methods configured in `config.yaml`.
6. **Summary** – `PYTHONPATH=src python scripts/generate_summary.py` compiles key indicators and plots. Emission and mortality plots collapse SSP suffixes (identical across climate pathways), while SCC and temperature remain pathway‑specific.

## Testing

Install development dependencies and run the pytest suite from the project root:

```bash
pip install -e '.[dev]'
python -m pytest
```

Individual modules can be exercised via `python -m pytest tests/test_calc_emissions.py` (or similar) when iterating quickly. Continuous integration expects the full suite to pass before changes are submitted.

## Development Guidelines

- **Module boundaries**: keep each package focused (e.g., `energy` should not depend on UI code).
- **Configuration over constants**: read scenario parameters from `config.yaml` or files in `config/`.
- **Typing & docs**: use type hints and concise docstrings to clarify model interfaces.
- **Testing**: add unit tests alongside new functionality (mirror the structure under `tests/`).
- **Data provenance**: document dataset sources and preprocessing steps in `data/README.md` (create
  the file when data arrives).
- **Version control**: exclude large datasets or exploratory notebooks unless essential; prefer
  lightweight CSV/JSON inputs in Git.

## Contribution Workflow

1. Create a feature branch (e.g., `feature/energy-mix`).
1. Implement changes with tests, documentation updates, and Ruff-clean code.
1. Run `pre-commit run -a` (or at least `ruff check`/`ruff format`) before committing.
1. Open a pull request summarizing scientific assumptions and validation steps.

## Coding Guidelines

- Put reusable library code under `src/` (e.g. `src/climate_module/`). Modules here should expose functions/classes without side effects so they can be imported from notebooks, other scripts, or tests.
- Place runnable entry points or one-off helpers under `scripts/`. These are thin wrappers that import from `src/`, read configuration (like `config.yaml`), and orchestrate the workflow.
- Use `results/emissions/` for intermediate inputs (each mix folder contains `co2/so2/nox/pm25.csv` with per-demand columns). This folder is ignored by Git so you can generate or edit CSVs without polluting commits.

## Ruff Linting & Formatting

- Ruff is configured in `pyproject.toml` to handle both linting and code formatting.

- Install the Git hooks once per clone to enable automatic fixes:

  ```bash
  pre-commit install
  ```

- Run Ruff manually when needed:

  ```bash
  ruff check . --fix
  ruff format .
  ```

- For more comprehensive checks including unsafe fixes:

  ```bash
  ruff check . --fix --unsafe-fixes
  ```

- `pre-commit run -a` will apply the same checks to the whole repository (useful before opening a PR).

## Configuration Overview

All runtime settings live in `config.yaml`.

- `time_horizon`
  - Shared `{start, end, step}` used across calc-emissions, climate, and downstream modules. The climate runner upgrades this window to annual resolution automatically.

- `calc_emissions`
  - Defines electricity demand/mix scenarios and converts them into Mt/year deltas for CO₂, SOₓ, NOₓ, PM₂.₅ (and optional GWP100).
  - Key subsections:
    - `emission_factors_file`: CSV with a `technology` column and pollutant intensities (`*_kg_per_kwh`, `*_mt_per_twh`, etc.); values are harmonised to Mt/TWh.
    - `demand_scenarios` / `mix_scenarios`: named templates used by `baseline` and entries in `scenarios`.
    - `baseline`: reference demand + mix used to compute differences.
    - `scenarios`: list of electricity cases. Each entry can reference a named scenario or supply `*_custom` mappings.
    - `countries`: metadata pointing to per-country configs, aggregate output folders, optional notes file, and the shared scenario filter (names must exist in every country file).
    - Outputs one folder per scenario in the configured directory (default `results/emissions/<mix>/<Country>/`). Files include `co2.csv`, `sox.csv`, `nox.csv`, `pm25.csv`, `gwp100.csv` (when available); the climate module consumes `co2.csv` while the others support air-pollution analysis.

- `climate_module`
  - Consumes emission-difference files and runs FaIR temperature responses.
  - Key options:
    - `output_directory`: where summary CSVs are written (`results/climate` by default).
    - `sample_years_option`: `default` (5-year to 2050, then 10-year) or `full` (every year 2025–2100).
    - `parameters`: global FaIR settings (e.g. `deep_ocean_efficacy`, `forcing_4co2`, `equilibrium_climate_sensitivity`). Start/end years inherit from `time_horizon`, always run at 1-year steps.
    - `climate_scenarios`: SSP pathways to run (use `run: all` or list of IDs) with per-pathway tweaks.
    - `emission_scenarios`: which emission scenario folders in `results/emissions/All_countries/` to process (`all` or list of mix names). Only `co2.csv` feeds FaIR; other pollutant files are optional analytics inputs.
    - When `economic_module.damage_duration_years` exceeds the emission horizon, FaIR extends its run to `start + duration - 1` and holds the terminal emission delta constant.

- `pattern_scaling`
  - Consumes global climate CSVs and applies country-specific pattern-scaling coefficients.
  - Key options:
    - `output_directory`: destination for per-country scaled results.
    - `scaling_factors_file`: path to the scaling table (e.g., `data/cmip6_pattern_scaling_by_country_mean.csv`).
    - `scaling_weighting`: selects which `patterns.*` column to use (e.g., `area`, `gdp.2000`, `pop.2100`).
    - `countries`: ISO3 codes to generate outputs for.
  - Matches climate scenarios using the first four characters of each `climate_module.climate_scenarios.definitions[*].id` or the `climate_scenario` column injected into climate CSVs.

- `air_pollution`
  - Translates emission changes for PM₂.₅ and NOₓ into mortality percentage differences by scaling baseline concentrations with emission ratios.
  - Key options:
    - `output_directory`: where health-impact CSVs are written (`results/air_pollution` by default).
    - `concentration_measure`: preferred statistic (`median`, `mean`, etc.); the module falls back through `concentration_fallback_order` if the field is missing in the data.
    - `country_weights`: weighting used when averaging country-level responses (`equal` or a mapping `{Country: weight}`, normalised automatically; can be overridden per pollutant).
    - `pollutants`: per-pollutant overrides (stats file, `relative_risk` or `beta`, reference concentration delta). Use `baseline_deaths` to convert percentage changes into annual death deltas (`per_year` or `total` plus `years`/`span`); a module-level `baseline_deaths` entry applies to the combined total, optionally weighted by `weights`.
    - `scenarios`: `all` or a list of emission scenario names to evaluate.
  - Outputs include one `*_health_impact.csv` per pollutant plus optional `*_mortality_summary.csv` (if baseline deaths are configured) and a combined `total_mortality_summary.csv` aggregating all pollutants.

- `results`
  - `run_directory`: optional subfolder inserted under `results/` (for example `run_A`). When set, each module automatically writes to `results/<run_directory>/<module>/…` so you can compare runs without overwriting prior outputs.
  - `summary`: configuration for the cross-module report (see `docs/results_summary.md` for details on available fields).

- `economic_module`
  - Computes SCC by combining temperature, emission, and GDP series.
  - Configure discounting under `economic_module.methods` and provide GDP/emission inputs.
  - `damage_function` now supports optional threshold amplification, smooth saturation, and catastrophic add-ons in addition to the DICE quadratic terms (`delta1`, `delta2`). Tune behaviour via keys such as `use_threshold`, `threshold_temperature`, `use_saturation`, `max_fraction`, `use_catastrophic`, and related parameters (see `config.yaml`).
  - Per-year SCC is computed with definition-faithful FaIR pulses (one FaIR evaluation per emission year) so the reported SCC(τ) is exact for the chosen discounting method; see `docs/economic_module.md` for details.
  - Temperature CSVs export a `climate_scenario` column; the SCC runner reads it to select the matching SSP GDP/population series from `gdp_population_directory` (workbooks `GDP_SSP1_5.xlsx` and `POP_SSP1_5.xlsx`). Set `gdp_series` to a custom CSV only when overriding the SSP datasets.
  - `damage_duration_years` extends the SCC damage window beyond the shared time horizon (starting at the global start year); datasets must supply values through the requested end year, and the climate module reuses the last available emission delta during the tail.
  - `data_sources.emission_root` and `data_sources.temperature_root` should target the intermediate `results/` products (`results/emissions/All_countries/<mix>/co2.csv` and `results/climate/<scenario>_<climate>.csv`). Scenario names now use the format `<mix>__<demand>` (e.g. `base_mix__scen1_upper`). Climate pathways default to the SSPs enabled under `climate_module.climate_scenarios.run`; specify `economic_module.data_sources.climate_scenarios` only when you need to override that list.
  - When `aggregation` is set to `average`, provide `aggregation_horizon` (`start`, `end`) to bound the averaging window. The CLI enforces this so you always know which portion of the timeline feeds the aggregate SCC.
- `results`
  - `summary` collects cross-module indicators (SCC, damages, temperature and emission deltas, mortality impacts) for configured years and writes `summary.csv` plus optional comparison bar charts to `output_directory`. See `docs/results_summary.md`.
  - Toggle `include_plots` to disable chart generation (useful on headless systems) or change `plot_format` for publication-ready graphics.
    - When `economic_module.damage_duration_years` exceeds the emission horizon, FaIR extends its run to `start + duration - 1` and holds the terminal emission delta constant.

- `pattern_scaling`
  - Consumes global climate CSVs and applies country-specific pattern-scaling coefficients.
  - Key options:
    - `output_directory`: destination for per-country scaled results.
    - `scaling_factors_file`: path to the scaling table (e.g., `data/cmip6_pattern_scaling_by_country_mean.csv`).
    - `scaling_weighting`: selects which `patterns.*` column to use (e.g., `area`, `gdp.2000`, `pop.2100`).
    - `countries`: ISO3 codes to generate outputs for.
  - Matches climate scenarios using the first four characters of each `climate_module.climate_scenarios.definitions[*].id` or the `climate_scenario` column injected into climate CSVs.

- `air_pollution`
  - Translates emission changes for PM₂.₅ and NOₓ into mortality percentage differences by scaling baseline concentrations with emission ratios.
  - Key options:
    - `output_directory`: where health-impact CSVs are written (`results/air_pollution` by default).
    - `concentration_measure`: preferred statistic (`median`, `mean`, etc.); the module falls back through `concentration_fallback_order` if the field is missing in the data.
    - `country_weights`: weighting used when averaging country-level responses (`equal` or a mapping `{Country: weight}`, normalised automatically; can be overridden per pollutant).
    - `pollutants`: per-pollutant overrides (stats file, `relative_risk` or `beta`, reference concentration delta). Use `baseline_deaths` to convert percentage changes into annual death deltas (`per_year` or `total` plus `years`/`span`); a module-level `baseline_deaths` entry applies to the combined total, optionally weighted by `weights`.
    - `scenarios`: `all` or a list of emission scenario names to evaluate.
  - Outputs include one `*_health_impact.csv` per pollutant plus optional `*_mortality_summary.csv` (if baseline deaths are configured) and a combined `total_mortality_summary.csv` aggregating all pollutants.

- `results`
  - `run_directory`: optional subfolder inserted under `results/` (for example `run_A`). When set, each module automatically writes to `results/<run_directory>/<module>/…` so you can compare runs without overwriting prior outputs.
  - `summary`: configuration for the cross-module report (see `docs/results_summary.md` for details on available fields).

- `economic_module`
  - Computes SCC by combining temperature, emission, and GDP series.
  - Configure discounting under `economic_module.methods` and provide GDP/emission inputs.
  - `damage_function` now supports optional threshold amplification, smooth saturation, and catastrophic add-ons in addition to the DICE quadratic terms (`delta1`, `delta2`). Tune behaviour via keys such as `use_threshold`, `threshold_temperature`, `use_saturation`, `max_fraction`, `use_catastrophic`, and related parameters (see `config.yaml`).
  - Per-year SCC is computed with definition-faithful FaIR pulses (one FaIR evaluation per emission year) so the reported SCC(τ) is exact for the chosen discounting method; see `docs/economic_module.md` for details.
  - Temperature CSVs export a `climate_scenario` column; the SCC runner reads it to select the matching SSP GDP/population series from `gdp_population_directory` (workbooks `GDP_SSP1_5.xlsx` and `POP_SSP1_5.xlsx`). Set `gdp_series` to a custom CSV only when overriding the SSP datasets.
  - `damage_duration_years` extends the SCC damage window beyond the shared time horizon (starting at the global start year); datasets must supply values through the requested end year, and the climate module reuses the last available emission delta during the tail.
  - `data_sources.emission_root` and `data_sources.temperature_root` should target the intermediate `results/` products (`results/emissions/All_countries/<mix>/co2.csv` and `results/climate/<scenario>_<climate>.csv`). Scenario names follow `<mix>__<demand>`. Climate pathways default to the SSPs enabled under `climate_module.climate_scenarios.run`; specify `economic_module.data_sources.climate_scenarios` only when overriding that list.
  - When `aggregation` is set to `average`, provide `aggregation_horizon` (`start`, `end`) to bound the averaging window. The CLI enforces this so you always know which portion of the timeline feeds the aggregate SCC.
- `results`
  - `summary` collects cross-module indicators (SCC, damages, temperature and emission deltas, mortality impacts) for configured years and writes `summary.csv` plus optional comparison bar charts to `output_directory`. See `docs/results_summary.md`.
  - Toggle `include_plots` to disable chart generation (useful on headless systems) or change `plot_format` for publication-ready graphics.

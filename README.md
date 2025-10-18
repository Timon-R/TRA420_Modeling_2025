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
- `data/` — input datasets (raw and processed). Keep large files out of version control when possible.
- `resources/` — emission-difference scenario folders (e.g. `<scenario>/co2.csv`, Mt CO₂/yr); used by the climate runner and ignored by Git.
- `results/` — generated outputs like climate CSVs (ignored by Git).
- `tests/` — pytest suite covering emissions, climate, economic, and pattern-scaling modules.
- `config.yaml` — project-level configuration (scenario metadata, default parameters).
- `environment.yaml` — preferred Python environment specification for reproducibility.
- `pyproject.toml` — project metadata plus Ruff lint/format configuration.
- `.gitignore` — excludes generated results and other artifacts.
- `README.md` — usage guidance and development conventions.

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

Typical workflow (driven by `config.yaml`):

1. **Emissions** – `python scripts/run_calc_emissions.py` (optional; prepares `resources/<scenario>/co2.csv`).
2. **Air-pollution impacts** – `python scripts/run_air_pollution.py` combines non-CO₂ deltas with concentration stats to estimate mortality percentage changes.
3. **Global climate** – `python scripts/run_fair_scenarios.py` writes `results/climate/*.csv` and mirrors to `resources/climate/`. Each CSV now includes a `climate_scenario` column.
4. **Pattern scaling (optional)** – `python scripts/run_pattern_scaling.py` consumes the global climate CSVs plus the scaling factors table and produces per-country files under `pattern_scaling.output_directory`.
5. **Economics** – `python scripts/run_scc.py` auto-selects the SSP GDP/population series based on `climate_scenario` and evaluates discounting methods configured in `config.yaml`.

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
- Use `resources/` for intermediate inputs (each emission scenario has its own folder with `co2/so2/nox/pm25.csv`). This folder is ignored by Git so you can generate or edit CSVs without polluting commits.

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

- `calc_emissions`
  - Defines electricity demand/mix scenarios and converts them into Mt CO₂/yr deltas.
  - Key subsections:
    - `years`: simulation horizon (`start`, `end`, `step`).
    - `emission_factors_file`: CSV with technology factors (CO₂ in Mt/TWh; SO₂/NOₓ/PM₂.₅ in kt/TWh). The calculator converts non-CO₂ pollutants to Mt/year when writing outputs.
    - `demand_scenarios` / `mix_scenarios`: named templates used by `baseline` and entries in `scenarios`.
    - `baseline`: reference demand + mix used to compute differences.
    - `scenarios`: list of electricity cases. Each entry can reference a named scenario or supply `*_custom` mappings.
    - Outputs one folder per scenario in the configured directory (default `resources/`). Files include `co2.csv`, `so2.csv`, `nox.csv`, `pm25.csv` with Mt/year deltas; the climate module consumes `co2.csv` while the others support air-pollution analysis.

- `climate_module`
  - Consumes emission-difference files and runs FaIR temperature responses.
  - Key options:
    - `output_directory`: where summary CSVs are written (`results/climate` by default).
    - `sample_years_option`: `default` (5-year to 2050, then 10-year) or `full` (every year 2025–2100).
    - `parameters`: global FaIR settings; includes time grid (`start_year`, `end_year`, `timestep`) and climate overrides (`deep_ocean_efficacy`, `forcing_4co2`, `equilibrium_climate_sensitivity`).
    - `climate_scenarios`: SSP pathways to run (use `run: all` or list of IDs) with per-pathway tweaks.
    - `emission_scenarios`: which emission scenario folders in `resources/` to process (`all` or list of folder names). Only `co2.csv` feeds FaIR; other pollutant files are optional analytics inputs.

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
    - `pollutants`: per-pollutant overrides (stats file, `relative_risk` or `beta`, reference concentration delta).
    - `scenarios`: `all` or a list of emission scenario names to evaluate.
- `economic_module`
  - Computes SCC by combining temperature, emission, and GDP series.
  - Configure discounting under `economic_module.methods` and provide GDP/emission inputs.
  - `damage_function` now supports optional threshold amplification, smooth saturation, and catastrophic add-ons in addition to the DICE quadratic terms (`delta1`, `delta2`). Tune behaviour via keys such as `use_threshold`, `threshold_temperature`, `use_saturation`, `max_fraction`, `use_catastrophic`, and related parameters (see `config.yaml`).
  - Temperature CSVs export a `climate_scenario` column; the SCC runner reads it to select the matching SSP GDP/population series from `gdp_population_directory` (workbooks `GDP_SSP1_5.xlsx` and `POP_SSP1_5.xlsx`). Set `gdp_series` to a custom CSV only when overriding the SSP datasets.

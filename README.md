# TRA420_Modeling_2025

Modeling pipeline for the TRA420 course project: starting from regional energy demand and elasticity
assumptions, deriving energy mixes, estimating emissions, linking those emissions to global
temperature responses, and evaluating local/global impact metrics such as the Social Cost of Carbon
(SCC).

## Repository Structure

- `src/`
  - `climate_module/` — FaIR wrappers and scenario tools.
  - `calc_emissions/`- Converting electricity demand and mix into emission difference from baseline
  - *(planned)* `impacts/`, `ui/`, `common/` packages that
    will divide the workflow into focused modules as they are implemented.
- `scripts/` — CLI helpers such as `run_fair_scenarios.py` for quick experiments.
- `data/` — input datasets (raw and processed). Keep large files out of version control when possible.
- `resources/` — emission-difference inputs (e.g. `<scenario>_emission_difference.csv`); used by the climate runner and ignored by Git.
- `results/` — generated outputs like climate CSVs (ignored by Git).
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

Implementation is in progress.

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
- Use `resources/` for intermediate inputs (such as `<scenario>_emission_difference.csv` files) that feed the climate runner. This folder is ignored by Git so you can generate or edit CSVs without polluting commits.

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

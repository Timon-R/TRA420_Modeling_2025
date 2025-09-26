# TRA420_Modeling_2025

Modeling pipeline for the TRA420 course project: starting from regional energy demand and elasticity
assumptions, deriving energy mixes, estimating emissions, linking those emissions to global
temperature responses, and evaluating local/global impact metrics such as the Social Cost of Carbon
(SCC).

## Repository Structure

- `src/`
  - `calc_emissions.py` — placeholder for the emission-factor based calculations.
  - `FaIR.py` — placeholder for climate response routines (e.g., linking emissions to temperature).
  - _(planned)_ `energy/`, `emissions/`, `climate/`, `impacts/`, `ui/`, `common/` packages or scripts that will
    divide the end-to-end workflow into focused modules.
- `data/` — input datasets (raw and processed). Keep large files out of version control when possible.
- `config.yaml` — project-level configuration (scenario metadata, default parameters).
- `environment.yaml` — preferred Python environment specification for reproducibility.
- `pyproject.toml` — project metadata plus Ruff lint/format configuration.
- `.pre-commit-config.yaml` — Git pre-commit hooks for automatic linting/formatting.
- `README.md` — usage guidance and development conventions.

## Getting Started

1. **Create the environment**
   ```bash
   mamba env create -f environment.yaml  # or `conda` / `uv` once finalized
   conda activate tra420-modeling
   ```
2. **Install the project in editable mode** (once package metadata is finalized):
   ```bash
   pip install -e .
   ```
3. **Configure data sources** by updating `config.yaml` with pointers to demand datasets, elasticity
   assumptions, and emission-factor catalogues.

## Ruff Linting & Formatting

- Ruff is configured in `pyproject.toml` to handle both linting and code formatting.
- Install the Git hooks once per clone to enable automatic fixes:
  ```bash
  pre-commit install
  ```
- Run Ruff manually when needed:
  ```bash
  ruff check src tests --fix --unsafe-fixes
  ruff format src tests
  ```
- `pre-commit run -a` will apply the same checks to the whole repository (useful before opening a PR).

## Running the Pipeline

Implementation is in progress. Planned modules include:
- Energy mix calculation in `src/energy/` accepting demand trajectories plus elasticity settings.
- Emission estimation in `src/emissions/` leveraging shared emission-factor catalogues.
- Climate response modeling in `src/climate/` (FaIR or similar).
- Impact metrics in `src/impacts/`, covering SCC and other damage metrics.
- GUI elements in `src/ui/` exposing configurable parameters for scenario exploration.

Until the modules are populated, prototype the workflow via simple CLI scripts under `scripts/`.

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
2. Implement changes with tests, documentation updates, and Ruff-clean code.
3. Run `pre-commit run -a` (or at least `ruff check`/`ruff format`) before committing.
4. Open a pull request summarizing scientific assumptions and validation steps.


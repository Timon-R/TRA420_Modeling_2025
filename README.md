# TRA420_Modeling_2025

Modeling pipeline for the TRA420 course project: starting from regional energy demand and elasticity
assumptions, deriving energy mixes, estimating emissions, translating those into global temperature
responses, and evaluating local/global impact metrics such as the Social Cost of Carbon (SCC).

## Repository Structure

- `src/`
  - `calc_emissions.py` — placeholder for the emission-factor based calculations.
  - `FaIR.py` — placeholder for climate response routines (e.g., linking emissions to temperature).
  - _(planned)_ `energy/`, `emissions/`, `climate/`, `impacts/`, `ui/`, `common/` packages or scripts that will
    divide the end-to-end workflow into focused modules.
- `data/` — input datasets (raw and processed). Keep large files out of version control when possible.
- `config.yaml` — project-level configuration (scenario metadata, default parameters).
- `environment.yaml` — preferred Python environment specification for reproducibility.
- `README.md` — usage guidance and development conventions.

## Getting Started

1. **Create the environment**
   ```bash
   mamba env create -f environment.yaml  # or `conda` / `uv` once finalized
   conda activate tra420-modeling
   ```
2. **Install the project in editable mode** (once a `pyproject.toml` or `setup.cfg` is added):
   ```bash
   pip install -e .
   ```
3. **Configure data sources** by updating `config.yaml` with pointers to demand datasets, elasticity
   assumptions, and emission-factor catalogues.

## Running the Pipeline
_to be added, maybe use Snakemake_ 

## Development Guidelines

- **Module boundaries**: keep each package focused (e.g., `energy` should not know about UI details).
- **Configuration over constants**: read scenario parameters from `config.yaml` or dedicated files in
  `config/` rather than hard-coding values.
- **Typing & docs**: use type hints and concise docstrings to clarify model interfaces.
- **Testing**: add unit tests alongside new functionality (target a mirror structure under `tests/`).
- **Data provenance**: document dataset sources and preprocessing steps in `data/README.md` (create
  the file when data arrives).
- **Version control**: keep notebooks and large data files out of git unless essential; prefer
  lightweight CSV/JSON inputs.

## Contribution Workflow

1. Create a feature branch (e.g., `feature/energy-mix`).
2. Implement changes with tests and documentation updates.
3. Run formatting/linting (decide on `ruff`, `black`, etc.) before committing.
4. Open a pull request summarizing the scientific assumptions and validation steps.

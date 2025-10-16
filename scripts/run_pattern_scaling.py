"""Apply pattern scaling factors to climate module results.

This script reads the pattern scaling configuration from ``config.yaml``, loads
country-level scaling factors, and applies them to climate output CSVs to produce
geographically-adjusted temperature trajectories.

Overview of ``config.yaml`` keys used here
------------------------------------------
- ``pattern_scaling`` section:
  - ``scaling_factors_file`` – relative path to CSV containing scaling factors
    (e.g., ``data/pattern_scaling.csv``). Must include columns: ``name``, ``iso3``,
    ``scenario``, and ``patterns.<weighting>``.
  - ``scaling_weighting`` – column suffix that determines which pattern column to
    use (e.g., ``mean`` → ``patterns.mean``).
  - ``countries`` – list of ISO3 country codes to process (e.g., ``[USA, GBR]``).
  - ``output_directory`` – relative path where scaled CSVs will be written
    (e.g., ``outputs/pattern_scaled``).

- ``climate_module`` section:
  - ``output_directory`` – directory containing baseline climate CSVs produced by
    ``run_fair_scenarios.py``.
  - ``climate_scenarios.definitions`` – list of scenario objects; the script uses
    the first four characters of each scenario ``id`` to match against scaling
    factors (e.g., ``rcp45`` → ``rcp4``).

Scaling factors CSV format
--------------------------
The CSV referenced by ``scaling_factors_file`` must contain at minimum:

- ``name`` – descriptive country name
- ``iso3`` – three-letter ISO country code
- ``scenario`` – scenario identifier (first four characters, e.g., ``rcp4``)
- ``patterns.<weighting>`` – area, population, or gdp-weighted scaling factor for years 2000 and 2100

Behavior
--------
1. Loads ``config.yaml`` from the project root.
2. Reads scaling factors and filters by requested countries and scenarios.
3. Iterates over climate result CSVs in ``climate_module.output_directory``.
4. For each CSV matching a scenario in the filename:
   - Reads the result CSV (expects columns: ``year``, ``temperature_baseline``,
     ``temperature_adjusted``).
   - Multiplies both temperature columns by the country-specific scaling factor.
   - Computes ``temperature_delta`` as scaled adjusted \- scaled baseline.
   - Writes a new CSV to ``pattern_scaling.output_directory`` with the naming
     convention ``<ISO3>_<original_filename>.csv``.

Outputs
-------
Per-country CSV files written to ``pattern_scaling.output_directory`` with columns:

- ``year`` – year values from original climate CSV
- ``temperature_baseline`` – scaled baseline temperature (°C)
- ``temperature_adjusted`` – scaled adjusted temperature (°C)
- ``temperature_delta`` – difference between scaled adjusted and baseline (°C)

Each output file is named ``<ISO3>_<original>.csv`` (e.g., ``USA_scenario_rcp45.csv``).

Usage
-----
Run from the project root directory (where ``config.yaml`` is located):

"""

from pattern_scaling import DEFAULT_CONFIG_PATH, get_scaling_factors, load_config, scale_results

config = load_config(DEFAULT_CONFIG_PATH)
sf = get_scaling_factors(config)
scale_results(config, sf)

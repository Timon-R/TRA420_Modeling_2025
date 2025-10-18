## Recent Changes Summary

- Created initial version of the change log file.
- Set up structure for tracking future modifications.
- No content changes have been logged yet.

## 20250905 (manual entry)

- Set up code and data so that emissions are calculated for each of the 5 Balkan countries
  - Set up config files and emission data for each of the 5 countries in 'country/data'
  - Adapted run_calc_emissions.py and calculator.py so that the run_calc_emissions.py script can now be run using the command `run_calc_emissions.py --country COUNTRY`
  - Changed emission factor input units in emission factor tables to kg per kWh, so that they better match data from Ecoinvent. Adapted emission calculation code. Outputs are still in Mt of emissions.
  - Changed name of SO_2 emissions 'so2_kg_per_kWh' to 'sox_kg_kWh' to match emission factors in Ecoinvent.

  ## 20251018
  # Commit 1
  - Added/updated per-country emission factor CSVs from `Emission_factors_all.xlsx`:
    - Wrote country files: `emission_factors_Albania.csv`, `emission_factors_Bosnia-Herzegowina.csv`, `emission_factors_Kosovo.csv`, `emission_factors_Montenegro.csv`, `emission_factors_North_Macedonia.csv`, `emission_factors_Serbia.csv`.
    - Renamed `product` column to `technology` and mapped product strings to technology keys (e.g., lignite->coal, photovoltaic->solar, wood->biomass, etc.).
    - Appended a `storage` technology row (empty numeric fields) to each file and replaced temporary 'NA' placeholders with empty cells for numeric compatibility.

  # Commit 2
  - Removed `calc_emission` input from `config.yaml` file 
  - Standardized country configs (Bosnia-Herzegovina, Kosovo, North_Macedonia, Serbia):
    - Reduced `demand_scenarios.values` to years 2025, 2027, and 2100; introduced `reference`, `upper_bound`, and `lower_bound`.
    - Replaced the `scenarios` list with `scenario_1_lower_bound` and `scenario_1_upper_bound` (using the `reference` mix).
    - Preserved each country's original `custom_example` demand and mix blocks unchanged.
  - Left `config_Albania.yaml` as-is (already in the desired format).
  - Added `country_data/config_Montenegro.yaml` mirroring Serbia's structure:
    - Demand scenarios with years 2025, 2027, 2100 (reference/upper_bound/lower_bound).
    - Baseline uses `reference` demand and `reference` mix.
    - Scenarios include `scenario_1_lower_bound`, `scenario_1_upper_bound`, and a `custom_example` block.
    - Outputs to `resources/Montenegro` and uses `country_data/emission_factors_Montenegro.csv`.
  - Used data from the Excel received by the OECD (`Electricity_OECD.xlsx`) to generate scenarios:
    - 2025 demand corresponds to current demand (2023 in Excel).
    - 2027 corresponds to the demand after the reform (Scenario 1, Upper bound and lower bound). Remains unchanged in baseline
    - 2100 currently reproduces the value from 2027




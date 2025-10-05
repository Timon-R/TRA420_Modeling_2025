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


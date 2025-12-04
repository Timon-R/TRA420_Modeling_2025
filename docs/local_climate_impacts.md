# Local Climate Impacts Module

The `local_climate_impacts` package converts global temperature responses from the climate module
into country-level temperature and precipitation trajectories using pre-computed scaling factors.
These scaled climate signals also anchor the extreme-weather damage estimates (expressed as a
percentage of GDP) derived from published country compendiums.

## Inputs

- **Scaling factors CSV** (`local_climate_impacts.scaling_factors_file`):
  A table such as `data/pattern_scaling/cmip6_pattern_scaling_by_country_mean.csv` with columns:
  - `name` – descriptive country name.
  - `iso3` – ISO3 country code.
  - `scenario` – SSP/RCP identifier (first four characters, e.g., `ssp2`).
  - `patterns.<weighting>` – scaling factors for different weightings (`area`,
    `gdp.2000`, `gdp.2100`, `pop.2000`, `pop.2100`).
- **Optional precipitation scaling CSV**
  (`local_climate_impacts.scaling_factors_file_precipitation`): same schema as the temperature
  scaling file but expressing precipitation response in millimetres per day per degree of global
  temperature change.
- **Climate outputs** (`climate_module.output_directory`): CSV files produced by
  `run_fair_scenarios.py` containing columns `year`, `temperature_baseline`, `temperature_adjusted`,
  `temperature_delta`, and the `climate_scenario` identifier.
- **Extreme-weather baseline costs** (`data/pattern_scaling/extreme_weather_costs.csv`):
  percent-of-GDP damages for each country/scenario combination used as baselines before scaling by
  the adjusted temperature increase.

## Configuration (`config.yaml`)

```yaml
local_climate_impacts:
  output_directory: results/climate_scaled
  scaling_factors_file: data/pattern_scaling/cmip6_pattern_scaling_by_country_mean.csv
  scaling_factors_file_precipitation: data/pattern_scaling/pattern_scaling_precipitation_by_country_mean.csv
  scaling_weighting: area            # controls which patterns.<*> column to use
  countries: [USA, GBR, DEU]         # ISO3 codes to process

climate_module:
  output_directory: results/climate   # location of global climate CSVs
  climate_scenarios:
    definitions:
      - id: ssp245
        ...
```

- `scaling_weighting` selects the suffix appended to `patterns.` in the CSV.
- `countries` enumerates the ISO3 codes that will receive scaled outputs.

## Workflow

1. Load configuration via `local_climate_impacts.load_config` (defaults to project `config.yaml`).
2. Use `local_climate_impacts.get_scaling_factors` to filter the scaling table for the requested
   countries and the scenario prefixes present in the climate module definitions.
3. Run `local_climate_impacts.scale_results` to iterate over climate CSVs, match each file’s
   scenario (from the `climate_scenario` column or filename), apply the scaling factor, and write
   country-specific CSVs to `local_climate_impacts.output_directory`.
4. (Optional) Feed the scaled temperature deltas into the extreme-weather workflow: baseline
   damages from `extreme_weather_costs.csv` are scaled linearly by the adjusted temperature delta,
   yielding `% GDP` damages per country/scenario/year.

Each output file is named `<ISO3>_<original_filename>.csv` and stores:

- `year`
- `temperature_baseline` (scaled)
- `temperature_adjusted` (scaled)
- `temperature_delta` (scaled difference)
- `precipitation_baseline_mm_per_day` (scaled, when precipitation factors are provided)
- `precipitation_adjusted_mm_per_day` (scaled)
- `precipitation_delta_mm_per_day` (scaled difference)
- `climate_scenario`
- `iso3` (country code)
- `scaling_factor`

## Units

- **Precipitation response:** Precipitation scaling factors describe millimetres per day per degree
  of global temperature change. Consequently, the precipitation columns in the scaled CSVs carry
  actual mm/day trajectories for the baseline, adjusted, and delta series.
- **Extreme-weather damages:** Baseline costs in `extreme_weather_costs.csv` use `_pct_gdp`
  suffixes (e.g., `2030_pct_gdp`) so every column is explicitly denoted as a percentage of GDP. The
  damages reported downstream are in `% GDP`.

## Usage

Run the CLI helper after the climate module has produced global results:

```bash
python scripts/run_local_climate_impacts.py
```

The script loads `config.yaml`, computes scaling factors based on the selected weighting and
countries, and writes scaled CSVs to `local_climate_impacts.output_directory`.

## References

- US EPA. Pattern Scaling of Global Climate Variables. 2023.
  (https://www.github.com/USEPA/pattern-scaled-climate-variables)
- World Bank Group. (2024). Western Balkans 6 Country Climate and Development Plan.
  https://doi.org/10.1596/41881
- World Bank Group. (2024). Western Balkans 6 Albania Country Compendium.
- World Bank Group. (2024). Western Balkans 6 Kosovo Country Compendium.
- World Bank Group. (2024). Western Balkans 6 North Macedonia Country Compendium.
- World Bank Group. (2024). Western Balkans 6 Bosnia and Herzegovina Country Compendium.
- World Bank Group. (2024). Western Balkans 6 Montenegro Country Compendium.
- World Bank Group. (2024). Western Balkans 6 Serbia Country Compendium.

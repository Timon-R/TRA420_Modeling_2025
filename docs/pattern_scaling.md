# Pattern Scaling Module

The `pattern_scaling` package converts global temperature responses from the climate
module into country-level trajectories by applying pre-computed scaling factors.

## Inputs

- **Scaling factors CSV** (`pattern_scaling.scaling_factors_file`):
  A table such as `data/cmip6_pattern_scaling_by_country_mean.csv` with columns:
  - `name` – descriptive country name.
  - `iso3` – ISO3 country code.
  - `scenario` – SSP/RCP identifier (first four characters, e.g., `ssp2`).
  - `patterns.<weighting>` – scaling factors for different weightings (`area`,
    `gdp.2000`, `gdp.2100`, `pop.2000`, `pop.2100`).

- **Climate outputs** (`climate_module.output_directory`):
  CSV files produced by `run_fair_scenarios.py` containing columns
  `year`, `temperature_baseline`, `temperature_adjusted`, `temperature_delta`, and
  the new `climate_scenario` identifier.

## Configuration (`config.yaml`)

```yaml
pattern_scaling:
  output_directory: results/climate_scaled
  scaling_factors_file: data/cmip6_pattern_scaling_by_country_mean.csv
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

1. Load configuration via `pattern_scaling.load_config` (defaults to project `config.yaml`).
2. Use `pattern_scaling.get_scaling_factors` to filter the scaling table for the
   requested countries and the scenario prefixes present in the climate module definitions.
3. Run `pattern_scaling.scale_results` to iterate over climate CSVs, match each file’s
   scenario (from `climate_scenario` column or filename), apply the scaling factor, and
   write country-specific CSVs to `pattern_scaling.output_directory`.

Each output file is named `<ISO3>_<original_filename>.csv` and stores:

- `year`
- `temperature_baseline` (scaled)
- `temperature_adjusted` (scaled)
- `temperature_delta` (scaled difference)
- `climate_scenario`
- `iso3` (country code)
- `scaling_factor`

## Main Equation

Pattern scaling assumes a constant multiplier between global and regional temperature
responses:

\\[
  T^{\text{regional}}_{c,t} = S_{c,\text{scenario}} \times T^{\text{global}}_t
\\]

where `S` is the scaling factor drawn from the chosen `patterns.<weighting>` column.
All three temperature columns (`baseline`, `adjusted`, `delta`) are multiplied by the same
factor.

## Usage

Run the CLI helper after the climate module has produced global results:

```bash
python scripts/run_pattern_scaling.py
```

The script loads `config.yaml`, computes scaling factors based on the selected
weighting and countries, and writes scaled CSVs to `pattern_scaling.output_directory`.

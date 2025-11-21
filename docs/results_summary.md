# Results Summary

Collects cross‑module indicators and produces a concise overview (tabular CSV plus optional plots) for every `(energy mix, demand case, climate scenario)` combination. Demand cases are stored as `base_demand`, `scen1_lower`, `scen1_mean`, `scen1_upper`, and the summary iterates over every mix listed in the emissions output directory.

## What It Aggregates

- Emission deltas (Mt CO₂) and other pollutants per selected years, both aggregated across all countries and per country/pollutant combination.
- Temperature deltas (°C) per selected years (global + pattern-scaled per country).
- Air-pollution results when available: mortality differences, mortality percent changes (percentage points), monetary benefits, concentration deltas (µg/m³) averaged across countries, and per-country concentration deltas.
- SCC results by method:
  - Per-year SCC series at configured years when `aggregation: per_year`. Values are
    reported in PPP USD-2025, discounted to the emission year (column name
    `SCC_<method>_<year>_PPP_USD_2025_discounted_to_year_per_tco2`).
  - Average SCC when `aggregation: average` (uses the configured horizon).
- Damages derived from SCC × emission deltas for each discounting method. This multiplies the base-year SCC with the emission-year delta (converted to tonnes), yielding PV damages denominated in PPP-2020 USD (base year reported in the column name).

To reduce duplication, plots that do not vary by climate pathway are de‑duplicated:
- Emission deltas, mortality deltas, mortality values, and non-CO₂ plots collapse SSP suffixes; each mix-demand combination appears once with a shaded envelope for `scen1_lower`/`scen1_upper`.
- Temperature and SCC plots retain SSP suffixes because results vary by pathway.
- A single SCC plot (`plots/scc_timeseries.png`) shows SCC trajectories for each SSP over the entire time horizon.

## Configuration (`results.summary` in `config.yaml`)

```yaml
results:
  run_directory: null        # Optional: set 'experiment_A' to write to results/experiment_A/…
  summary:
    years: [2030, 2050]      # Reported years in the CSV and plots
    year_period:
      - {start: 2025, end: 2050}    # Optional period sums for damages/mortality
    output_directory: results/summary
    include_plots: true
    plot_format: png
```

## Usage

Run the generator after your pipeline steps have produced outputs:

```bash
PYTHONPATH=src python scripts/generate_summary.py
```

If you configure `results.run_directory`, every module (including the summary)
will write to `results/<run_directory>/…`, keeping multi-module runs aligned while
avoiding overwrites between experiments.

It discovers economic results in `results/economic/`, climate CSVs in
`results/climate/`, emissions in `results/emissions/All_countries/<mix>/`, and
air‑pollution summaries in `results/air_pollution/<scenario>/` (all adjusted for the run directory when configured).

## Outputs

- `summary.csv` — a wide table where each row represents a unique `(energy_mix, demand_case, climate pathway)` combination. Column order matches the user's requested priority:
  1. `energy_mix`, `climate_scenario`, `demand_case`.
  2. `delta_co2_Mt_all_countries_<year>` plus per-country pollutant deltas (`delta_<pollutant>_<Country>_<year>`).
  3. `delta_<pollutant>_<unit>_all_countries_<year>` for NOx, SO₂, PM₂.₅, etc.
  4. `delta_T_C_<year>` and `delta_T_<ISO3>_<year>`.
  5. SCC columns `SCC_<method>_<year>_PPP_USD_2025_discounted_to_year_per_tco2` (or `scc_average_<method>`).
  6. Damages columns `damages_PPP2020_usd_baseyear_<base_year>_<method>_<year>` and horizon sums (`damages_PPP2020_usd_baseyear_<base_year>_sum_<method>_<start>_to_<end>`).
  7. Air-pollution metrics:
     - `air_pollution_mortality_difference_all_countries_<year>` (deaths/year),
     - `air_pollution_mortality_percent_change_all_countries_<year>` (percentage points),
     - `air_pollution_monetary_benefit_all_countries_usd_<year>`,
     - concentration deltas averaged across countries
       (`air_pollution_concentration_delta_<pollutant>_microgram_per_m3_all_countries_<year>`),
       and per-country concentration deltas
       (`air_pollution_concentration_delta_<pollutant>_<Country>_microgram_per_m3_<year>`),
     - optional sums `air_pollution_mortality_difference_sum_all_countries_<start>_to_<end>` and
       `air_pollution_monetary_benefit_sum_all_countries_usd_<start>_to_<end>`.
  8. Socioeconomic snapshots for the configured `years`.
  Per-country pollutant deltas and concentration deltas are pulled directly from
  `results/<run>/emissions/<mix>/<Country>/<pollutant>.csv` and
  `results/<run>/air_pollution/<scenario>/<pollutant>_concentration_summary.csv`.
- `plots/` — contains:
  - `scc_timeseries.png` (SCC vs time for each SSP).
  - Background climate and socioeconomics panels.
  - One folder per mix containing emissions, mortality, concentration, and damage plots across the full horizon (annual resolution from the configured time grid) with shaded envelopes for `scen1_lower`/`scen1_upper`.
- SCC results are sourced from the SSP-level `results/economic/pulse_scc_timeseries_<method>_<ssp>.csv`
  files written by `scripts/run_scc.py`; each row combines the SCC(τ) timeseries and the aggregated SCC value for that climate pathway.

## SCC Display Rules

- When `economic_module.aggregation: per_year`, the summary lists SCC values for
  the configured `results.summary.years` by method and scenario.
- When `aggregation: average`, the summary shows one value per method, labeled
  with the averaging horizon `(start‑end)`.

## De‑duplication Rules

- Emission delta, mortality delta, and mortality percent plots are collapsed by
  scenario base name (e.g., `scen1_lower_base_mix`), ignoring climate suffixes
  like `_ssp119`, `_ssp245`, `_ssp370`.
- SCC and temperature plots retain climate suffixes because results vary by
  pathway.

## References
- [Cross-module data aggregation]:

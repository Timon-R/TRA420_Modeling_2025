# Results Summary

Collects cross‑module indicators and produces a concise overview (tabular CSV plus optional plots).

## What It Aggregates

- Emission deltas (Mt CO₂) per selected years.
- Temperature deltas (°C) per selected years.
- Mortality impacts (per‑pollutant totals and combined), if the air‑pollution module ran.
- SCC results by method:
  - Per‑year SCC series at configured years when `aggregation: per_year`.
  - Average SCC when `aggregation: average` (uses the configured horizon).
- Damages derived from SCC × emission deltas for each discounting method. For Ramsey
  discounting this uses the present-value SCC (base-year USD) multiplied by the
  emission-year delta (converted to tonnes), so the reported damages are base-year USD
  PV of the lifetime impact caused by the emissions in that year.

To reduce duplication, plots that do not vary by climate pathway are de‑duplicated:
- Emission deltas and mortality plots collapse SSP suffixes; each base scenario appears once.

## Configuration (`results.summary` in `config.yaml`)

```yaml
results:
  run_directory: null        # Optional: set 'experiment_A' to write to results/experiment_A/…
  summary:
    years: [2030, 2050]     # Reported years in the CSV and plots
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

- `summary.csv` — a wide table where each row represents an emission scenario + climate pathway
  combination. Columns include:
  - `energy_mix`, `climate_scenario`, `demand_case` (`base_demand`, `scen1_lower`, `scen1_mean`, `scen1_upper`).
  - For every configured year: `delta_co2_Mt_all_countries_<year>`, `delta_T_C_<year>`,
    `air_pollution_mortality_difference_<year>`, `air_pollution_mortality_difference_percent_<year>`,
    optional `air_pollution_monetary_benefit_usd_<year>`, `SCC_<method>_<year>_usd_per_tco2` (or
    `scc_average_<method>` when averaging), and `damages_PPP2020_usd_baseyear_<base_year>_<method>_<year>`.
  - For `year_period` ranges, sums are added as `air_pollution_mortality_difference_sum_<start>_to_<end>`,
    `air_pollution_monetary_benefit_sum_usd_<start>_to_<end>`, and
    `damages_PPP2020_usd_baseyear_<base_year>_sum_<method>_<start>_to_<end>`.
  - Per-country pollutant deltas aggregated directly from `results/emissions/<mix>/<Country>/<pollutant>.csv`
    (e.g., `delta_co2_Serbia_2030`, `delta_pm25_Bosnia_and_Herzegovina_2050`) and pattern-scaled
    temperature deltas (`delta_T_<ISO3>_<year>` from `results/climate_scaled`).
  This replaces the previous text/JSON outputs and is consumed by downstream notebooks directly.
- `plots/` — grouped bar charts for emission deltas, temperature deltas,
  damages, SCC (one per method), and mortality metrics; plus emission and
  temperature timeseries charts.
- `scc_summary.csv` now includes a `run_method` column indicating whether SCC
  results came from the kernel or pulse workflow.
- When the pulse method is used, `plots/` also receives `scc_timeseries_<method>.png`
  showing SCC(τ) by scenario (one line per climate pathway).

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

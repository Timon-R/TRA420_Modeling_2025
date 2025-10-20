# Results Summary

Collects cross‑module indicators and produces a concise overview (text, JSON, and plots).

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
  summary:
    years: [2030, 2050]     # Reported years in text/JSON/plots
    output_directory: results/summary
    include_plots: true
    plot_format: png
```

## Usage

Run the generator after your pipeline steps have produced outputs:

```bash
PYTHONPATH=src python scripts/generate_summary.py
```

It discovers economic results in `results/economic/`, climate CSVs in
`resources/climate/`, emissions in `resources/All_countries/<scenario>/`, and
air‑pollution summaries in `results/air_pollution/<scenario>/`.

## Outputs

- `summary.txt` — human‑readable overview for the configured years.
-   Includes an explicit note clarifying that damages are per emission year and
    denominated in present-value USD for `economic_module.base_year`.
- `summary.json` — machine‑readable payload with the same values.
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
  scenario base name (e.g., `scenario_1_lower_bound`), ignoring climate suffixes
  like `_ssp119`, `_ssp245`, `_ssp370`.
- SCC and temperature plots retain climate suffixes because results vary by
  pathway.

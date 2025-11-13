# Air-Pollution Module

The `air_pollution` package links non-CO₂ emission trajectories from the
electricity sector to health impacts by estimating pollutant concentration
changes and applying concentration–response functions.

## Overview

1. **Load emission scenarios** – Imports baseline and policy emission totals
   for PM₂.₅, NOₓ (and any other pollutants present) produced by
   `calc_emissions.run_from_config`.
2. **Read concentration statistics** – Country-level baseline concentration
   summaries (mean/median/min/max) are loaded from CSV files in
   `data/air_pollution/`.
3. **Scale concentrations with emission ratios** – Concentration changes are
   assumed proportional to the ratio of scenario emissions to baseline
   emissions for each pollutant.
4. **Apply health-response coefficients** – Uses log-linear relative risk
   coefficients to estimate the percentage change in mortality.
5. **Aggregate mortality changes** – Optional baseline mortality inputs convert
   percentage changes into absolute deaths per year. Country weights control
   the averaging across countries, and pollutant weights control the combined
   summary.
6. **Write results** – Outputs per-pollutant health impacts, optional
   per-pollutant mortality summaries, and an aggregate summary of the combined
   mortality change for all pollutants with provided baselines.

This module is intentionally lightweight and deterministic: all effects are a
transparent scaling of published concentration statistics by emission ratios,
followed by a standard log‑linear health response. It is designed to be easy to
audit, reproduce, and sensitivity‑test.

## Data Requirements

Provide one CSV per pollutant with country concentration statistics:

- Required columns: `country` and at least one of `mean`, `median`, `min`, `max`.
- Units: concentrations in µg/m³.
- Example (wide format):

  country,median,mean
  Albania,20.4,22.1
  Serbia,25.0,25.9

If both `median` and `mean` exist, the preferred statistic is selected from the
configuration and missing preferences fall back by the configured order.

## Key Equations

- **Emission ratio**  
  $$
r_{p,t} = \frac{E^{\text{scenario}}_{p,t}}{E^{\text{baseline}}_{p,t}}
$$

- **Concentration change**  
  Assuming linear scaling,  
  $$
C^{\text{new}}_{c,p,t} = C^{\text{baseline}}_{c,p} \times r_{p,t}, \qquad
    \Delta C_{c,p,t} = C^{\text{new}}_{c,p,t} - C^{\text{baseline}}_{c,p}
$$

- **Relative-risk slope**  
  With relative risk `RR` specified for a reference delta `Δ_ref`, the slope is
  $$
\beta_p = \frac{\ln(\text{RR}_p)}{\Delta_{\text{ref}, p}}
$$

- **Percentage change in mortality**  
  $$
\Delta m_{c,p,t} = \exp(\beta_p \Delta C_{c,p,t}) - 1
$$

- **Country weighting**  
  Weighted average across countries (weights normalised to sum to 1):
  $$
\overline{\Delta m}_{p,t} = \sum_c w_{c,p} \Delta m_{c,p,t}
$$
  where per-pollutant weights default to module-level weights or equal weighting.

- **Mortality delta (if baseline deaths supplied)**  
  $$
\Delta D_{p,t} = D^{\text{baseline}}_{p} \times \overline{\Delta m}_{p,t}
$$

- **Combined mortality (across pollutants)**  
  Per-year combined mortality uses normalised pollutant weights `w_p` (default
  equal) applied to the weighted percentage changes, multiplied by the
  module-level baseline deaths:
  $$
\Delta m^{\text{total}}_t = \sum_p w_p \overline{\Delta m}_{p,t}, \qquad
    \Delta D^{\text{total}}_t = D^{\text{baseline,total}} \times
    \Delta m^{\text{total}}_t
$$

## Units and Conventions

- Emissions from `calc_emissions` are in megatonnes (Mt) per year. Only ratios
  are used here, so absolute unit scaling cancels out.
- Concentrations are in µg/m³. The RR slope `β` is in (per µg/m³) units.
- Mortality percentage changes are unitless; mortality deltas (deaths/year)
  adopt the baseline deaths’ units and cadence.

## Configuration (`config.yaml`)

```yaml
air_pollution:
  output_directory: results/air_pollution
  concentration_measure: median       # Preferred statistic (median/mean/min/max)
  concentration_fallback_order:
    - median
    - mean
    - min
    - max
  country_weights: equal              # Normalised weights per country; override with {Country: weight}
  scenarios: all                      # Scenario names from calc_emissions (or explicit list)
  pollutants:
    pm25:
      stats_file: data/air_pollution/PM25_country_stats.csv
      relative_risk: 1.08             # RR for the reference concentration delta
      reference_delta: 10.0           # µg/m³ corresponding to the RR value
      # country_weights: {...}        # Optional per-pollutant weighting override
      # baseline_deaths:
      #   per_year: 6000              # Optional pollutant-specific baseline deaths
    nox:
      stats_file: data/air_pollution/NOx_country_stats.csv
      relative_risk: 1.03
      reference_delta: 10.0
  baseline_deaths:
    total: 19000                      # Combined baseline deaths (e.g., multi-year total)
    span:
      start: 2018
      end: 2020
    # weights:                        # Optional pollutant weights for combined summary
    #   pm25: 0.7
    #   nox: 0.3
```

### Configuration Notes

- `country_weights`: accepts `equal` (default) or a mapping `{Country name: weight}`.
  Values are normalised automatically. Per-pollutant `country_weights` override the module-level
  weights.
- `relative_risk` and `reference_delta` can be replaced by `beta` if a slope is
  known directly.
- `baseline_deaths` can be specified per pollutant and/or at the module level.
  Per-pollutant baselines drive per-pollutant mortality summaries, whereas the
  module-level baseline drives the combined `total_mortality_summary.csv`.
- If both `per_year` and `total` are absent a `ValueError` is raised.
- For totals, specify either `span` (`start`/`end`) or an explicit `years` list;
  the module converts totals to an average `per_year` by dividing across the
  number of years in the period.
- `scenarios`: accepts `all`, a single name, or a list; the baseline scenario is
  always required in the emissions results but is not processed as an output.

### Country and Scenario Selection

- Country set defaults to all countries present in the stats file; restrict via
  `air_pollution.countries: ["Serbia", "Albania"]`.
- Scenario selection is controlled by `air_pollution.scenarios`. The module
  processes every non‑baseline scenario produced by `calc_emissions` unless a
  subset is specified.

### Weights

- Country weights: module‑level `country_weights` apply to all pollutants unless
  overridden per pollutant. Use `equal` for uniform weighting or a mapping of
  country → weight; values are normalised automatically each year.
- Combined pollutant weights: `baseline_deaths.weights` defines how per‑pollutant
  percentage changes are blended when computing the total summary (defaults to
  equal weighting across available pollutants).

## Usage

Generate up-to-date emission deltas via `scripts/run_calc_emissions.py`, then
compute health impacts:

```bash
python scripts/run_air_pollution.py
```

The CLI:

1. Runs `calc_emissions.run_from_config()` (unless results are passed in by other code).
2. Invokes `air_pollution.run_from_config()` with the shared `config.yaml`.
3. Prints weighted percentage changes and (if configured) mortality deltas.
4. Writes CSV outputs to `results/air_pollution/<scenario>/`.

Example invocation with a focused scenario set and custom weights:

```yaml
air_pollution:
  scenarios: [scenario_1_lower_bound, scenario_1_upper_bound]
  country_weights:
    Serbia: 2
    Albania: 1
  pollutants:
    pm25:
      stats_file: data/air_pollution/PM25_country_stats.csv
      relative_risk: 1.08
      reference_delta: 10
```

You can also import `run_from_config` from notebooks or other modules to obtain
the structured `AirPollutionResult` objects for further analysis.

## Outputs

For each scenario and pollutant:

- `<pollutant>_health_impact.csv` – columns:
  - `country`
  - `year`
  - `baseline_concentration` (µg/m³)
  - `emission_ratio`
  - `new_concentration`
  - `delta_concentration`
  - `percent_change_mortality`

- `<pollutant>_mortality_summary.csv` (optional) – produced when baseline
  deaths are supplied for that pollutant; columns:
  - `year`
  - `percent_change_mortality` (weighted average)
  - `baseline_deaths_per_year`
  - `delta_deaths_per_year`
  - `new_deaths_per_year`

- `total_mortality_summary.csv` – combined mortality summary when a module-level
  baseline is provided; columns mirror the per-pollutant summary and reflect
  pollutant weights specified in `baseline_deaths.weights`.

File layout per scenario:

```
results/air_pollution/<scenario>/
  pm25_health_impact.csv
  pm25_mortality_summary.csv        # if per-pollutant baseline provided
  nox_health_impact.csv
  nox_mortality_summary.csv         # if per-pollutant baseline provided
  total_mortality_summary.csv       # if module-level baseline provided
```

The `AirPollutionResult` object also exposes:

- `pollutant_results`: mapping of pollutant → `PollutantImpact`
- `PollutantImpact` attributes:
  - `country_weights`: normalised weights used during aggregation
  - `weighted_percent_change`: per-year weighted mortality percentage change
  - `deaths_summary`: per-pollutant mortality deltas (if configured)

## Validation Tips

- Ensure baseline emissions are non-zero; the module assigns an emission ratio
  of 1.0 when both scenario and baseline emissions are zero, and leaves ratios
  undefined (NaN) otherwise.
- Confirm concentration statistics include the requested measure (`median`
  by default) or provide fallbacks in `concentration_fallback_order`.
- When providing custom weights, values need not sum to one—normalisation is
  handled internally.
- Add unit tests when introducing new pollutants or weighting strategies to
  verify that percent-change aggregation and mortality calculations behave as
  expected (see `tests/test_air_pollution.py` for examples).

### Edge Cases and Diagnostics

- If baseline emissions are zero while scenario emissions are non‑zero (or vice
  versa), the emission ratio is undefined; the module skips those entries.
- If both baseline and scenario emissions are zero for a year/pollutant, the
  ratio is treated as 1.0 (no change), yielding a zero mortality delta.
- Outputs are monotone in the emission ratio under the log‑linear model; sanity
  check sign and magnitude by inspecting `*_health_impact.csv` and the aggregate
  summaries.

"""Generate emission-difference tables from config-driven electricity scenarios.

This script reads the ``calc_emissions`` section of ``config.yaml`` and produces a
folder per emission scenario (default location ``resources/<scenario>/``). Each
folder contains pollutant-specific CSVs—``co2.csv``, ``so2.csv``, ``nox.csv``,
``pm25.csv``—with deltas expressed in **Mt per year**. The climate module consumes
``co2.csv`` while the remaining files can be used for air-pollution analysis.

Overview of ``config.yaml`` keys used here
------------------------------------------
- ``emission_factors_file`` – CSV containing technology-level emission factors in
  Mt CO₂ per TWh (see ``data/emission_factors.csv`` for the editable template).
- ``years`` – dictionary with ``start``, ``end`` and ``step`` defining the time
  horizon over which demand/mix trajectories are interpolated.
- ``demand_scenarios`` / ``mix_scenarios`` – named scenario templates. Each
  demand scenario provides TWh values, each mix scenario provides technology
  shares (per year or a flat value). Custom demand/mix entries can be supplied
  on a per-scenario basis using ``*_custom`` blocks.
- ``baseline`` – references the demand/mix scenario used as the reference when
  calculating deltas.
- ``scenarios`` – list of electricity cases. For each entry the script resolves
  the demand and mix (either by name or custom data), calculates technology
  generation/emissions, and stores the difference vs. baseline.

Outputs
-------
For every scenario the script writes CSV files to ``<output_directory>/<scenario>/``
(default ``resources/<scenario>/``) with two columns:

``year``  | ``delta`` (Mt/year)

``co2.csv`` feeds the climate module; ``so2.csv``, ``nox.csv`` and ``pm25.csv`` provide
additional pollutant deltas for air-quality analysis. Run this script before the
climate pipeline so emission deltas stay aligned with the latest demand/mix
assumptions.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd


from calc_emissions import run_from_config  # noqa: E402
import argparse



def main() -> None:
  parser = argparse.ArgumentParser(description="Run electricity emissions calculation for a specific country.")
  parser.add_argument('--country', type=str, required=True,
            help='Country name (e.g., Serbia, Albania, Bosnia-Herzegovina, Kosovo, North_Macedonia)')
  args = parser.parse_args()

  country = args.country.replace(" ", "_")
  config_path = f"country_data/config_{country}.yaml"
  if not Path(config_path).exists():
    print(f"Config file for country '{country}' not found: {config_path}")
    sys.exit(1)

  results = run_from_config(config_path)
  for name, result in results.items():
    print(f"\n=== {name} ===")
    print("ΔCO₂ (Mt/year) relative to baseline:")
    print(result.delta_mtco2.to_frame(name="delta_mtco2").to_string())
    totals = {
      pollutant: result.total_emissions_mt[pollutant]
      for pollutant in sorted(result.total_emissions_mt)
    }
    summary_years = [y for y in [2030, 2050, 2100] if y in result.delta_mtco2.index]
    if summary_years:
      print("\nTotal emissions (Mt) for selected years:")
      summary_df = pd.DataFrame({k: v.loc[summary_years] for k, v in totals.items()})
      print(summary_df.to_string())


if __name__ == "__main__":
    main()

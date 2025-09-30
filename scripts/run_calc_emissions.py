"""Generate emission-difference CSVs from config-driven electricity scenarios.

This script reads the ``calc_emissions`` section of ``config.yaml`` and produces
``<scenario>_emission_difference.csv`` files (values in **Mt CO₂/yr**) which feed
the climate module.

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
For every scenario the script writes ``resources/<scenario>_emission_difference.csv``
(location configurable via ``output_directory``) with two columns:

``year``  | ``delta`` (Mt CO₂/yr)

Run this script before the climate pipeline so the emission deltas are up to
date with the latest demand/mix assumptions.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calc_emissions import run_from_config  # noqa: E402


def main() -> None:
    results = run_from_config()
    for name, result in results.items():
        print(f"\n=== {name} ===")
        print(result.delta_mtco2.to_frame(name="delta_mtco2").to_string())


if __name__ == "__main__":
    main()

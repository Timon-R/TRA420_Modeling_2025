"""Calculate air-pollution health impacts from emission scenarios.

Usage
-----
```bash
python scripts/run_air_pollution.py
```

The script expects ``calc_emissions`` to be configured in ``config.yaml`` and
uses its output directly. Configuration for this module lives in the
``air_pollution`` section of the same file. Results are written to
``results/air_pollution/<scenario>/<pollutant>_health_impact.csv`` with the
following columns:

- ``country`` – country name from the concentration statistics file.
- ``year`` – calendar year.
- ``baseline_concentration`` – reference concentration (µg/m³).
- ``emission_ratio`` – scenario emissions divided by baseline emissions.
- ``new_concentration`` – estimated concentration under the scenario (µg/m³).
- ``delta_concentration`` – change vs. baseline (µg/m³).
- ``percent_change_mortality`` – mortality percentage difference (unitless).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from air_pollution import run_from_config as run_air_pollution  # noqa: E402
from calc_emissions import run_from_config as run_emissions  # noqa: E402


def main() -> None:
    emission_results = run_emissions()
    impact_results = run_air_pollution(emission_results=emission_results)

    for scenario, result in impact_results.items():
        print(f"\n=== Scenario: {scenario} ===")
        for pollutant, impact in result.pollutant_results.items():
            impacts = impact.impacts
            header = (
                f"{pollutant.upper()} "
                f"(metric: {impact.concentration_metric}, beta={impact.beta:.4f})"
            )
            print(header)
            summary = (
                impacts.groupby("year")["percent_change_mortality"]
                .mean()
                .rename("avg_percent_change")
            )
            print(summary.to_frame().to_string())


if __name__ == "__main__":
    main()

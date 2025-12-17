"""Calculate air-pollution health impacts from emission scenarios.

Usage
-----
```bash
python scripts/run_air_pollution.py
```

The script expects ``calc_emissions`` to be configured in ``config.yaml`` and
uses its output directly. Configuration for this module lives in the
``air_pollution`` section of the same file. Results are written to
``results/<run>/air_pollution/<scenario>/<pollutant>_health_impact.csv`` with the
following columns:

- ``country`` – country name from the concentration statistics file.
- ``year`` – calendar year.
- ``baseline_concentration`` – reference concentration (µg/m³).
- ``emission_ratio`` – scenario emissions divided by baseline emissions.
- ``delta_fraction`` – fractional emission change vs baseline (scenario-baseline)/baseline.
- ``new_concentration`` – estimated concentration under the scenario (µg/m³).
- ``delta_concentration`` – change vs. baseline (µg/m³).
- ``percent_change_mortality`` – mortality percentage difference (unitless).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_calc_emissions_all import run_all_countries  # noqa: E402

from air_pollution import run_from_config as run_air_pollution  # noqa: E402

LOGGER = logging.getLogger("air_pollution.run")


def main() -> None:
    emission_results = run_all_countries()
    impact_results = run_air_pollution(emission_results=emission_results)

    for scenario, result in impact_results.items():
        LOGGER.info("Air-pollution results for scenario '%s'", scenario)
        for pollutant, impact in result.pollutant_results.items():
            header = (
                f"{pollutant.upper()} "
                f"(metric: {impact.concentration_metric}, beta={impact.beta:.4f})"
            )
            LOGGER.info(header)
            weights = impact.country_weights
            if not weights.empty:
                if np.allclose(weights.values, weights.values[0]):
                    LOGGER.info("    country weights: equal")
                else:
                    preview = ", ".join(
                        f"{country}:{weights[country]:.2f}" for country in list(weights.index)[:3]
                    )
                    if len(weights.index) > 3:
                        preview = f"{preview}, ..."
                    LOGGER.info("    country weights (normalised): %s", preview)
            summary = impact.weighted_percent_change.rename("weighted_percent_change")
            LOGGER.info("%s", summary.to_frame().to_string())
            if impact.deaths_summary is not None:
                LOGGER.info(
                    "Estimated deaths per year:\n%s", impact.deaths_summary.to_string(index=False)
                )
        if result.total_mortality_summary is not None:
            LOGGER.info(
                "Total (all pollutants) deaths per year:\n%s",
                result.total_mortality_summary.to_string(index=False),
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    main()

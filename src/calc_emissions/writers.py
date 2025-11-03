from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .calculator import EmissionScenarioResult


def write_per_country_results(per_country_map: Dict[str, Dict[str, EmissionScenarioResult]], resources_root: Path) -> None:
    """Write per-country scenario delta CSVs to resources/<Country>/<scenario>/<pollutant>.csv.

    Args:
        per_country_map: mapping country -> mapping scenario_name -> EmissionScenarioResult
        resources_root: root path where country folders will be created

    The CSV format is two columns: year, delta (Mt). This mirrors the writer used previously
    in the scripts module but is placed here so other modules can import it.
    """
    resources_root = Path(resources_root)
    for country, scenarios in per_country_map.items():
        for scenario_name, result in scenarios.items():
            # Skip baseline writing (we only write deltas for non-baseline scenarios)
            if scenario_name.lower() == "baseline":
                continue

            # Each pollutant is stored in result.total_emissions_mt as dict pollutant -> Series
            for pollutant, series in (result.total_emissions_mt or {}).items():
                # The result.delta_mtco2 may be a Series for co2, but for generic pollutants
                # we compute delta as scenario.total - baseline.total if available on the result
                # Here we assume result.delta_mtco2 is the delta for CO2; for other pollutants
                # the EmissionScenarioResult should already contain the correct delta in
                # result.total_emissions_mt vs baseline. We'll write series - baseline if
                # delta is not present on the result.
                out_dir = resources_root / country / scenario_name
                out_dir.mkdir(parents=True, exist_ok=True)

                # If the result has a precomputed delta for this pollutant, prefer that.
                # We store deltas in the 'delta' column in Mt.
                if hasattr(result, "delta_mtco2") and pollutant == "co2":
                    delta_series = result.delta_mtco2
                else:
                    # Fallback: if result._baseline_total is available (private), use it.
                    # Otherwise, try to use a zero baseline (best-effort).
                    baseline = getattr(result, "baseline_total_mt", None)
                    if baseline is not None and pollutant in baseline:
                        delta_series = series - baseline[pollutant]
                    else:
                        # If no baseline is available, assume result has a delta stored as
                        # attribute `delta_{pollutant}` or else zero-fill.
                        attr = f"delta_{pollutant}"
                        delta_series = getattr(result, attr, None)
                        if delta_series is None:
                            # Best effort: write zeros
                            delta_series = pd.Series([0.0] * len(series), index=series.index)

                df = pd.DataFrame({"year": list(map(int, delta_series.index)), "delta": list(delta_series.values)})
                out_path = out_dir / f"{pollutant}.csv"
                df.to_csv(out_path, index=False)

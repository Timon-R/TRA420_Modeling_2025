from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .calculator import EmissionScenarioResult, POLLUTANTS


def write_per_country_results(per_country_map: Dict[str, Dict[str, EmissionScenarioResult]], resources_root: Path) -> None:
    """Write per-country scenario delta CSVs to resources/<Country>/<scenario>/<pollutant>.csv.

    Args:
        per_country_map: mapping country -> mapping scenario_name -> EmissionScenarioResult
        resources_root: root path where country folders will be created

    The CSV format contains three columns: `year`, `absolute` (Mt) and `delta` (Mt).
    `absolute` is the total emissions for the scenario; `delta` is the difference
    relative to the mix-specific demand baseline (or zeros when a baseline is not
    available). Files are written to `resources/<Country>/<mix>/<demand>/<pollutant>.csv`.
    """
    resources_root = Path(resources_root)
    for country, scenarios in per_country_map.items():
        # scenarios keys are expected like '<mix>/<demand>'
        mixes: Dict[str, Dict[str, EmissionScenarioResult]] = {}
        for key, res in scenarios.items():
            if "/" not in key:
                continue
            mix, demand = key.split("/", 1)
            mixes.setdefault(mix, {})[demand] = res

        for mix, demand_map in mixes.items():
            out_dir = resources_root / country / mix
            out_dir.mkdir(parents=True, exist_ok=True)

            for pollutant in set().union(*(r.total_emissions_mt.keys() for r in demand_map.values())):
                unit = POLLUTANTS.get(pollutant, {}).get("unit", "")

                # Determine index (prefer baseline)
                baseline_res = demand_map.get("baseline")
                if baseline_res is not None and pollutant in baseline_res.total_emissions_mt:
                    index = baseline_res.total_emissions_mt[pollutant].index
                else:
                    idxs = [s.total_emissions_mt.get(pollutant).index for s in demand_map.values() if pollutant in s.total_emissions_mt]
                    if not idxs:
                        continue
                    index = idxs[0]
                    for i in idxs[1:]:
                        index = index.union(i)

                data = {"year": list(map(int, index))}
                abs_cols = {}
                for demand_name, res in demand_map.items():
                    abs_name = f"absolute_{demand_name}"
                    abs_cols[demand_name] = abs_name
                    series = res.total_emissions_mt.get(pollutant)
                    if series is None:
                        series_aligned = pd.Series([0.0] * len(index), index=index)
                    else:
                        series_aligned = series.reindex(index, fill_value=0.0)
                    data[abs_name] = list(series_aligned.values)

                # Compute deltas vs baseline
                baseline_series = None
                if baseline_res is not None:
                    baseline_series = baseline_res.total_emissions_mt.get(pollutant)
                    if baseline_series is not None:
                        baseline_series = baseline_series.reindex(index, fill_value=0.0)

                for demand_name in [d for d in demand_map.keys() if d != "baseline"]:
                    abs_series = pd.Series(data[abs_cols[demand_name]], index=index)
                    if baseline_series is None:
                        delta = abs_series * 0.0
                    else:
                        delta = abs_series - baseline_series
                    data[f"delta_{demand_name}"] = list(delta.values)

                # Order columns: year, absolute_baseline, absolute_..., delta_...
                cols = ["year"] + [abs_cols[d] for d in sorted(demand_map.keys())] + [f"delta_{d}" for d in sorted(demand_map.keys()) if d != "baseline"]
                df = pd.DataFrame(data)[cols]

                out_file = out_dir / f"{pollutant}.csv"
                with out_file.open("w", encoding="utf-8") as fh:
                    if unit:
                        fh.write(f"# unit: {unit}\n")
                    df.to_csv(fh, index=False)

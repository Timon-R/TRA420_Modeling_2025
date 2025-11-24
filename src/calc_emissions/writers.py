from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping

import pandas as pd

from .constants import BASE_DEMAND_CASE, POLLUTANTS

if TYPE_CHECKING:  # pragma: no cover
    from .calculator import EmissionScenarioResult


def _build_mix_dataframe(
    demand_map: Mapping[str, EmissionScenarioResult],
    baseline_case: str,
    pollutant: str,
) -> pd.DataFrame:
    baseline = demand_map.get(baseline_case)
    if baseline is None:
        raise ValueError(f"Baseline demand case '{baseline_case}' not found for mix.")

    if pollutant in baseline.total_emissions_mt:
        index = baseline.total_emissions_mt[pollutant].index
    else:
        idxs = [
            res.total_emissions_mt[pollutant].index
            for res in demand_map.values()
            if pollutant in res.total_emissions_mt
        ]
        if not idxs:
            return pd.DataFrame()
        index = idxs[0]
        for more in idxs[1:]:
            index = index.union(more)

    data = {"year": list(map(int, index))}
    baseline_series = None
    for demand_name, res in demand_map.items():
        series = res.total_emissions_mt.get(pollutant)
        if series is None:
            series_aligned = pd.Series([0.0] * len(index), index=index)
        else:
            series_aligned = series.reindex(index, fill_value=0.0)
        abs_col = f"absolute_{demand_name}"
        data[abs_col] = list(series_aligned.values)
        if demand_name == baseline_case:
            baseline_series = series_aligned

    if baseline_series is None:
        baseline_series = pd.Series([0.0] * len(index), index=index)
        data[f"absolute_{baseline_case}"] = list(baseline_series.values)

    for demand_name in demand_map:
        delta_col = f"delta_{demand_name}"
        if demand_name == baseline_case:
            data[delta_col] = [0.0] * len(index)
            continue
        abs_col = f"absolute_{demand_name}"
        abs_series = pd.Series(data[abs_col], index=index, dtype=float)
        delta = abs_series - baseline_series
        data[delta_col] = list(delta.values)

    columns = ["year"]
    columns.extend(sorted([col for col in data if col.startswith("absolute_")], key=lambda c: c))
    columns.extend(sorted([col for col in data if col.startswith("delta_")], key=lambda c: c))
    return pd.DataFrame(data)[columns]


def write_mix_directory(
    mix_name: str,
    demand_map: Mapping[str, EmissionScenarioResult],
    destination: Path,
    baseline_case: str = BASE_DEMAND_CASE,
) -> None:
    dest_dir = Path(destination)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for pollutant in set().union(*(res.total_emissions_mt.keys() for res in demand_map.values())):
        df = _build_mix_dataframe(demand_map, baseline_case, pollutant)
        if df.empty:
            continue
        unit = POLLUTANTS.get(pollutant, {}).get("unit", "")
        with (dest_dir / f"{pollutant}.csv").open("w", encoding="utf-8") as fh:
            if unit:
                fh.write(f"# unit: {unit}\n")
            df.to_csv(fh, index=False)


def write_per_country_results(
    per_country_map: Dict[str, Dict[str, EmissionScenarioResult]],
    destination_root: Path,
    baseline_case: str = BASE_DEMAND_CASE,
) -> None:
    """Write per-country mix CSVs to <dest>/<mix>/<Country>/<pollutant>.csv."""
    destination_root = Path(destination_root)
    for country, scenarios in per_country_map.items():
        if not scenarios:
            continue
        grouped: Dict[str, Dict[str, EmissionScenarioResult]] = {}
        for result in scenarios.values():
            grouped.setdefault(result.mix_case, {})[result.demand_case] = result
        for mix_name, demand_map in grouped.items():
            dest = destination_root / mix_name / country
            write_mix_directory(mix_name, demand_map, dest, baseline_case)

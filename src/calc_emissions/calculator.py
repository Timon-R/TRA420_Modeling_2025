"""Calculate electricity-sector emissions from demand/mix scenarios.

Results include per-pollutant (CO₂, SO₂, NOₓ, PM₂.₅) emissions in Mt/year. The
climate module consumes the CO₂ deltas, while the additional pollutants support
air-quality analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
import os

from config_paths import apply_results_run_directory

LOGGER = logging.getLogger("calc_emissions")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


POLLUTANTS: dict[str, dict[str, object]] = {
    "co2": {
        "aliases": {
            "co2_mt_per_twh": 1.0,
            "co2_kt_per_twh": 1e-3,
            "co2_gwh": 1e-3,  # legacy typo support
            "co2_kg_per_mwh": 1.0,
            "co2_kg_per_kwh": 1.0,
            "co2_g_per_kwh": 1e-6,
        },
        "unit": "Mt",
    },
    "sox": {
        "aliases": {
            "sox_mt_per_twh": 1.0,
            "sox_kt_per_twh": 1e-3,
            "sox_kg_per_mwh": 1.0,
            "sox_kg_per_kwh": 1.0,
            "sox_g_per_kwh": 1e-6,
            "so2_kt_per_twh": 1e-3,
            "so2_kg_per_kwh": 1.0,
        },
        "unit": "Mt",
    },
    "nox": {
        "aliases": {
            "nox_mt_per_twh": 1.0,
            "nox_kt_per_twh": 1e-3,
            "nox_kg_per_mwh": 1.0,
            "nox_kg_per_kwh": 1.0,
            "nox_g_per_kwh": 1e-6,
        },
        "unit": "Mt",
    },
    "pm25": {
        "aliases": {
            "pm25_mt_per_twh": 1.0,
            "pm25_kt_per_twh": 1e-3,
            "pm25_kg_per_mwh": 1.0,
            "pm25_kg_per_kwh": 1.0,
            "pm25_g_per_kwh": 1e-6,
        },
        "unit": "Mt",
    },
    "gwp100": {
        "aliases": {
            "gwp100_mt_per_twh": 1.0,
            "gwp100_kt_per_twh": 1e-3,
            "gwp100_kg_per_mwh": 1.0,
            "gwp100_kg_per_kwh": 1.0,
            "gwp100_g_per_kwh": 1e-6,
        },
        "unit": "Mt CO2eq",
    },
}


@dataclass
class EmissionScenarioResult:
    """Result container for an emission scenario."""

    name: str
    years: list[int]
    demand_twh: pd.Series
    generation_twh: pd.DataFrame
    technology_emissions_mt: dict[str, pd.DataFrame]
    total_emissions_mt: dict[str, pd.Series]
    delta_mtco2: pd.Series


def run_from_config(
    config_path: Path | str = "config.yaml",
    *,
    default_years: Mapping[str, float | int] | None = None,
    results_run_directory: str | None = None,
) -> dict[str, EmissionScenarioResult]:
    """Run emission calculations based on ``config.yaml`` and write delta CSVs."""
    config_path = Path(config_path)
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}

    module_cfg = config.get("calc_emissions", {})
    if not module_cfg:
        raise ValueError("'calc_emissions' section missing from config.yaml")

    fallback_years = default_years
    if fallback_years is None:
        fallback_years = _discover_global_horizon(config_path)
    years = _generate_years(module_cfg.get("years"), fallback_years)
    # Emission factors must be taken from the repository-wide
    # data/calc_emissions/emission_factors directory. Country configs should
    # reference a filename (or relative path) but we will resolve to the
    # canonical directory and load the basename only.
    ef_setting = module_cfg.get("emission_factors_file")
    if not ef_setting:
        raise ValueError("'emission_factors_file' must be set in the country config and point to a file in data/calc_emissions/emission_factors")
    basename = Path(ef_setting).name
    repo_root = Path(__file__).resolve().parents[2]
    ef_path = repo_root / "data" / "calc_emissions" / "emission_factors" / basename
    if not ef_path.exists():
        raise FileNotFoundError(f"Emission factors file not found in data/calc_emissions/emission_factors: {basename}")
    emission_factors = _load_emission_factors(ef_path)

    # Gather demand scenarios. Support both the new map-based `demand_scenarios`
    # *and* the legacy pattern where a top-level `baseline` and a `scenarios`
    # list provide demand_custom/mix_custom entries. Work on a local copy to
    # avoid mutating the original config.
    raw_demand_scenarios = module_cfg.get("demand_scenarios", {}) or {}
    demand_scenarios: dict[str, Mapping] = {}

    # If no explicit demand_scenarios map is provided, try to synthesise one
    # from legacy `baseline` and `scenarios` entries (used in older configs
    # and by some unit tests).
    if not raw_demand_scenarios:
        # baseline block may contain demand_custom or a pointer to a named scenario
        baseline_block = module_cfg.get("baseline") or {}
        if isinstance(baseline_block, Mapping):
            if "demand_custom" in baseline_block:
                demand_scenarios["baseline"] = {"values": baseline_block["demand_custom"]}
            elif "demand_scenario" in baseline_block:
                # leave resolution to _resolve_demand_series later if needed
                pass

        for sc in module_cfg.get("scenarios", []) or []:
            if not isinstance(sc, Mapping):
                continue
            name = sc.get("name")
            if not name:
                continue
            if "demand_custom" in sc:
                demand_scenarios[name] = {"values": sc["demand_custom"]}
            elif "demand_scenario" in sc:
                # reference to an existing demand scenario; leave handling to resolver
                demand_scenarios[name] = {"values": {}}
    else:
        for k, v in raw_demand_scenarios.items():
            key = "baseline" if k == "reference" else k
            if key in demand_scenarios:
                raise ValueError(f"Duplicate demand scenario name after renaming: {key}")
            demand_scenarios[key] = v

    # Mix scenarios: prefer explicit `mix_scenarios` map, but synthesize from
    # legacy `baseline`/`scenarios` mix_custom entries when absent.
    mix_scenarios = module_cfg.get("mix_scenarios") or {}
    if not mix_scenarios:
        synthesized: dict[str, Mapping] = {}
        baseline_block = module_cfg.get("baseline") or {}
        if isinstance(baseline_block, Mapping) and "mix_custom" in baseline_block:
            synthesized["baseline"] = baseline_block["mix_custom"]
        for sc in module_cfg.get("scenarios", []) or []:
            if isinstance(sc, Mapping) and sc.get("name") and "mix_custom" in sc:
                synthesized[sc["name"]] = sc["mix_custom"]
        if synthesized:
            mix_scenarios = synthesized

    output_dir = Path(module_cfg.get("output_directory", "resources"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = apply_results_run_directory(
        Path(module_cfg.get("results_directory", "results/emissions")),
        results_run_directory,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Results keys will be of the form '<mix_name>/<demand_name>' so callers can
    # easily discover emissions grouped by mix. We also write outputs under
    # output_dir/<mix_name>/<demand_name> with CSVs containing both absolute
    # emissions and delta vs the mix-specific demand baseline.
    results: dict[str, EmissionScenarioResult] = {}

    if "baseline" not in demand_scenarios:
        raise ValueError("A demand scenario named 'baseline' (or 'reference') must be provided.")

    for mix_name in mix_scenarios.keys():
        LOGGER.info("Calculating mix scenario '%s'", mix_name)

        # Resolve mix shares once per mix. Support two mix types:
        #  - standard 'shares' mapping (handled by _resolve_mix_shares)
        #  - time-series mixes with 'type: timeseries' and a 'csv' path
        mix_cfg = mix_scenarios.get(mix_name, {})
        if isinstance(mix_cfg, dict) and mix_cfg.get("type") == "timeseries" and "csv" in mix_cfg:
            # Load the timeseries CSV and align to reporting years
            mix_csv_path = _locate_mix_csv(config_path, mix_cfg["csv"])
            mix_ts = _load_mix_timeseries(mix_csv_path)  # indexed by year
            # normalize column names to lower-case technology keys
            mix_ts.columns = [c.strip().lower() for c in mix_ts.columns.astype(str)]
            # Reindex to requested years and ensure all technologies from emission_factors are present
            mix_ts = mix_ts.reindex(years).fillna(0.0)
            for t in emission_factors.index:
                if t not in mix_ts.columns:
                    mix_ts[t] = 0.0
            mix_ts = mix_ts[list(emission_factors.index)]
            # Normalize per-year to sum to 1 (avoid division by zero)
            mix_ts = mix_ts.div(mix_ts.sum(axis=1).replace(0, 1.0), axis=0)
            mix_shares = mix_ts
        else:
            mix_shares = _resolve_mix_shares(
                years,
                mix_scenarios,
                emission_factors.index,
                mix_name,
                None,
            )

        # Compute baseline emissions for this mix (demand baseline)
        baseline_demand = _resolve_demand_series(years, demand_scenarios, "baseline", None)
        baseline_result = calculate_emissions(
            name=f"{mix_name}/baseline",
            demand_series=baseline_demand,
            mix_shares=mix_shares,
            emission_factors=emission_factors,
        )

        # Iterate through all demand scenarios (baseline, upper_bound, lower_bound, ...)
        for demand_name in demand_scenarios.keys():
            LOGGER.info("  • demand case '%s'", demand_name)
            demand_series = _resolve_demand_series(years, demand_scenarios, demand_name, None)
            scenario_result = calculate_emissions(
                name=f"{mix_name}/{demand_name}",
                demand_series=demand_series,
                mix_shares=mix_shares,
                emission_factors=emission_factors,
            )

            results[f"{mix_name}/{demand_name}"] = scenario_result

    # After computing all mix/demand scenario results, write per-mix pollutant CSVs
    # with columns for the baseline absolute and other demand cases plus deltas.
    # Files are written to output_dir/<mix_name>/<pollutant>.csv and likewise to
    # results_dir when configured.
    # Group results by mix
    mixes: dict[str, dict[str, EmissionScenarioResult]] = {}
    for key, res in results.items():
        if "/" not in key:
            continue
        mix, demand = key.split("/", 1)
        mixes.setdefault(mix, {})[demand] = res

    for mix_name, demand_map in mixes.items():
        mix_dir = output_dir / mix_name
        mix_dir.mkdir(parents=True, exist_ok=True)
        mix_results_dir = results_dir / mix_name
        mix_results_dir.mkdir(parents=True, exist_ok=True)

        # Determine baseline series for alignment
        baseline_res = demand_map.get("baseline")

        for pollutant in set().union(*(r.total_emissions_mt.keys() for r in demand_map.values())):
            unit = POLLUTANTS.get(pollutant, {}).get("unit", "")

            # Prepare columns: absolute_baseline, absolute_<demand>, delta_<demand>...
            cols = ["year"]
            abs_cols = {}
            for demand_name, res in demand_map.items():
                abs_name = f"absolute_{demand_name}"
                abs_cols[demand_name] = abs_name
                cols.append(abs_name)
            delta_cols = [f"delta_{d}" for d in demand_map.keys() if d != "baseline"]
            cols.extend(delta_cols)

            # Use baseline index when available, otherwise union of indexes
            if baseline_res is not None and pollutant in baseline_res.total_emissions_mt:
                index = baseline_res.total_emissions_mt[pollutant].index
            else:
                # union of all indices
                idxs = [s.total_emissions_mt.get(pollutant).index for s in demand_map.values() if pollutant in s.total_emissions_mt]
                if not idxs:
                    continue
                index = idxs[0]
                for i in idxs[1:]:
                    index = index.union(i)

            data = {"year": list(map(int, index))}
            # Fill absolute columns
            for demand_name, abs_name in abs_cols.items():
                series = demand_map[demand_name].total_emissions_mt.get(pollutant)
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

            df = pd.DataFrame(data)[cols]

            # Write with a commented unit line
            out_file = mix_dir / f"{pollutant}.csv"
            with out_file.open("w", encoding="utf-8") as fh:
                if unit:
                    fh.write(f"# unit: {unit}\n")
                df.to_csv(fh, index=False)

            results_out_file = mix_results_dir / f"{pollutant}.csv"
            with results_out_file.open("w", encoding="utf-8") as fh:
                if unit:
                    fh.write(f"# unit: {unit}\n")
                df.to_csv(fh, index=False)

    # Back-compatibility: some downstream code and tests expect top-level
    # scenario keys (e.g. 'baseline', 'policy') rather than the new
    # '<mix>/<demand>' convention. If the config used legacy `baseline`/
    # `scenarios` entries (or we synthesised them above), expose flat keys
    # that point to the corresponding '<name>/<name>' result when present.
    if module_cfg.get("baseline") or module_cfg.get("scenarios"):
        for scen in list(demand_scenarios.keys()):
            compound = f"{scen}/{scen}"
            if compound in results and scen not in results:
                results[scen] = results[compound]

    # Also write legacy per-scenario CSVs at output_dir/<scenario>/<pollutant>.csv
    # containing only 'year' and 'delta' columns so older consumers/tests can
    # read the simple delta timeseries.
    if module_cfg.get("baseline") or module_cfg.get("scenarios"):
        baseline_res = results.get("baseline")
        for scen in demand_scenarios.keys():
            res = results.get(scen)
            if res is None:
                continue
            scen_dir = output_dir / scen
            scen_dir.mkdir(parents=True, exist_ok=True)
            for pollutant, totals in res.total_emissions_mt.items():
                if baseline_res is not None:
                    base = baseline_res.total_emissions_mt.get(pollutant)
                    if base is not None:
                        base_aligned = base.reindex(totals.index, fill_value=0.0)
                        delta = totals.reindex(base_aligned.index, fill_value=0.0) - base_aligned
                    else:
                        delta = totals * 0.0
                else:
                    delta = totals * 0.0
                df = pd.DataFrame({"year": totals.index.astype(int), "delta": delta.values})
                df.to_csv(scen_dir / f"{pollutant}.csv", index=False)

    return results
    


def calculate_emissions(
    name: str,
    demand_series: pd.Series,
    mix_shares: pd.DataFrame,
    emission_factors: pd.DataFrame,
) -> EmissionScenarioResult:
    """Calculate generation and emissions for a single scenario."""
    if not isinstance(demand_series.index, pd.Index):
        raise TypeError("demand_series must be a pandas Series with year index")

    # demand_series is in TWh and mix_shares are per-technology fractions
    # (summing to 1 per year). Keep generation in TWh so that multiplying by
    # emission factors (which are expressed per TWh) yields Mt directly.
    generation = mix_shares.multiply(demand_series, axis=0)

    technology_emissions: dict[str, pd.DataFrame] = {}
    total_emissions: dict[str, pd.Series] = {}

    for pollutant, _meta in POLLUTANTS.items():
        if pollutant not in emission_factors.columns:
            continue
        factors = emission_factors[pollutant]
        emissions = generation.mul(factors, axis=1)
        technology_emissions[pollutant] = emissions
        total_emissions[pollutant] = emissions.sum(axis=1)

    if "co2" not in total_emissions:
        raise ValueError("Emission factors must include CO₂ intensities.")

    co2_series = total_emissions["co2"]

    return EmissionScenarioResult(
        name=name,
        years=list(demand_series.index.astype(int)),
        demand_twh=demand_series,
        generation_twh=generation,
        technology_emissions_mt=technology_emissions,
        total_emissions_mt=total_emissions,
        delta_mtco2=pd.Series(
            np.zeros_like(co2_series.values), index=co2_series.index, dtype=float
        ),
    )


def _discover_global_horizon(config_path: Path) -> Mapping[str, float | int] | None:
    """Search parent directories for a root config containing ``time_horizon``."""

    for parent in [config_path.parent, *config_path.parents]:
        candidate = parent / "config.yaml"
        if not candidate.exists():
            continue
        try:
            with candidate.open() as handle:
                config = yaml.safe_load(handle) or {}
        except Exception:  # pragma: no cover - defensive against malformed configs
            continue
        horizon = config.get("time_horizon")
        if isinstance(horizon, Mapping):
            return horizon
    return None


def _generate_years(
    cfg: Mapping[str, float | int] | None,
    fallback: Mapping[str, float | int] | None = None,
) -> list[int]:
    values: dict[str, float | int] = {"start": 2025, "end": 2100, "step": 5}
    for source in (fallback, cfg):
        if not source:
            continue
        for key in ("start", "end", "step"):
            if key in source:
                values[key] = source[key]

    start = int(values["start"])
    end = int(values["end"])
    step = int(values["step"])
    if start >= end:
        raise ValueError("'start' year must be less than 'end' year")
    if step <= 0:
        raise ValueError("'step' must be positive")
    years = list(range(start, end + 1, step))
    return years


def _load_emission_factors(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Emission factors file not found: {path}")
    df = pd.read_csv(path, comment="#")
    if "technology" not in df.columns:
        raise ValueError("Emission factors CSV must contain a 'technology' column.")
    df["technology"] = df["technology"].str.strip().str.lower()
    factors: dict[str, pd.Series] = {}
    for pollutant, meta in POLLUTANTS.items():
        aliases = meta["aliases"]
        selected_series = None
        conversion = None
        for column, scale in aliases.items():
            if column in df.columns:
                selected_series = df.set_index("technology")[column].astype(float)
                conversion = float(scale)
                break
        if selected_series is None:
            continue
        factors[pollutant] = selected_series * conversion

    if "co2" not in factors:
        available = ", ".join(df.columns)
        raise ValueError(
            "Emission factors must provide a CO₂ intensity column. "
            f"Supported aliases include: {list(POLLUTANTS['co2']['aliases'])}. "
            f"Available columns: {available}"
        )

    factor_df = pd.DataFrame(factors)
    factor_df = factor_df.loc[df["technology"].tolist()]
    return factor_df


def _resolve_demand_series(
    years: Sequence[int],
    scenarios: Mapping[str, Mapping[str, Mapping]],
    scenario_name: str | None,
    custom: Mapping[int, float] | None,
) -> pd.Series:
    if scenario_name:
        scenario = scenarios.get(scenario_name)
        if scenario is None:
            raise ValueError(f"Demand scenario '{scenario_name}' not found in config.")
        values = scenario.get("values")
        if not values:
            raise ValueError(f"Demand scenario '{scenario_name}' must define 'values'.")
    elif custom:
        values = custom
    else:
        raise ValueError("Provide either 'demand_scenario' or 'demand_custom'.")

    series = _values_to_series(values, years, "demand")
    return series.rename("demand_twh")


def _resolve_mix_shares(
    years: Sequence[int],
    scenarios: Mapping[str, Mapping[str, Mapping]],
    technologies: Iterable[str],
    scenario_name: str | None,
    custom: Mapping[str, Mapping] | None,
) -> pd.DataFrame:
    tech_list = [tech.lower() for tech in technologies]
    if scenario_name:
        scenario = scenarios.get(scenario_name)
        if scenario is None:
            raise ValueError(f"Mix scenario '{scenario_name}' not found in config.")
        shares_cfg = scenario.get("shares")
        if shares_cfg is None:
            raise ValueError(f"Mix scenario '{scenario_name}' must define 'shares'.")
    elif custom:
        shares_cfg = custom.get("shares", {})
    else:
        raise ValueError("Provide either 'mix_scenario' or 'mix_custom'.")

    data = {}
    for tech in tech_list:
        entry = shares_cfg.get(tech)
        if entry is None:
            data[tech] = np.zeros(len(years))
            continue
        if isinstance(entry, Mapping):
            series = _values_to_series(entry, years, f"mix share for {tech}")
        else:
            series = pd.Series([float(entry)] * len(years), index=years)
        data[tech] = series.values

    mix_df = pd.DataFrame(data, index=years)
    totals = mix_df.sum(axis=1)
    if np.any(totals == 0):
        raise ValueError("Mix shares sum to zero for at least one year; cannot normalise.")
    mix_df = mix_df.divide(totals, axis=0)
    return mix_df


def _load_mix_timeseries(csv_path):
    """
    Load a mix timeseries CSV. Returns a pd.DataFrame indexed by year (int),
    columns are technology keys (strings), values are shares (floats, sum ~1).
    """
    # resolve to an absolute path using pathlib (avoid undefined helper `abspath`)
    csv_path = Path(csv_path).resolve()
    df = pd.read_csv(csv_path)

    # set year index (prefer explicit "year" column)
    if "year" in df.columns:
        df = df.set_index("year")
    else:
        df = df.set_index(df.columns[0])

    # ensure integer year index with clear error if conversion fails
    df.index = pd.to_numeric(df.index, errors="raise").astype(int)

    # normalize column names: strip, lowercase, replace common separators
    def normalize_col(s: str) -> str:
        s = str(s).strip().lower()
        s = s.replace("-", "_").replace(" ", "_")
        return s

    cols = [normalize_col(c) for c in df.columns]

    alias_map = {
        "other_res": "biomass",
        # add other aliases here, e.g. "pv": "solar", "wind_onshore": "wind"
    }

    mapped_cols = [alias_map.get(c, c) for c in cols]
    df.columns = mapped_cols

    # optional validation: ensure shares are numeric and rows sum to ~1
    df = df.apply(pd.to_numeric, errors="raise")
    row_sums = df.sum(axis=1)
    if not ((row_sums - 1.0).abs() <= 1e-2).all():
        # relax threshold or log a warning instead if appropriate
        raise ValueError("Mix timeseries rows must sum to 1.0 (per year).")

    return df


def _locate_mix_csv(config_path: Path, csv_setting: str) -> str:
    """Locate a mix timeseries CSV file.

    Resolution order:
      1. If csv_setting is absolute and exists, return it.
      2. If csv_setting is relative, resolve relative to the country config directory.
      3. Fall back to repo_root/data/calc_emissions/electricity_mixes/<basename>.
    """
    p = Path(csv_setting)
    # 1: absolute
    if p.is_absolute() and p.exists():
        return str(p)
    # 2: relative to config
    candidate = (config_path.parent / p).resolve()
    if candidate.exists():
        return str(candidate)
    # 3: repo data directory
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "data" / "calc_emissions" / "electricity_mixes" / p.name
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Mix timeseries CSV not found: tried {csv_setting} and data/calc_emissions/electricity_mixes/{p.name}")


def _resolve_generation_by_technology(demand_series, mix_cfg, available_years, emission_factors_index):
    """
    demand_series: pd.Series indexed by year (TWh)
    mix_cfg: dict describing the mix (either has 'shares' dict for constant shares
             or has 'csv' path for timeseries)
    available_years: list/int index we want to compute for
    emission_factors_index: list of technology names expected
    Returns: pd.DataFrame generation_twh indexed by year, cols=technologies
    """
    # initialize empty DataFrame
    techs = list(emission_factors_index)
    gen_df = pd.DataFrame(0.0, index=available_years, columns=techs)

    if isinstance(mix_cfg, dict) and mix_cfg.get("type") == "timeseries" and "csv" in mix_cfg:
        mix_ts = _load_mix_timeseries(mix_cfg["csv"])
        # Align years: reindex mix_ts to available_years, forward-fill or nearest? use reindex with fillna=0
        mix_ts = mix_ts.reindex(available_years).fillna(0.0)
        # Ensure columns for all technologies exist (missing -> 0)
        for t in techs:
            if t not in mix_ts.columns:
                mix_ts[t] = 0.0
        # Normalize per-year (if desired)
        mix_ts = mix_ts[techs]
        mix_ts = mix_ts.div(mix_ts.sum(axis=1).replace(0, 1.0), axis=0)
        # Multiply demand for each year by the share row
        for y in available_years:
            demand_val = demand_series.get(y, 0.0)
            gen_df.loc[y] = mix_ts.loc[y].values * demand_val
    else:
        # legacy constant shares handling (existing behavior)
        shares = {}
        if isinstance(mix_cfg, dict) and "shares" in mix_cfg:
            shares = mix_cfg["shares"]
        else:
            # mix_cfg might already be a mapping of shares
            shares = mix_cfg
        # ensure share values for all technologies (missing -> 0)
        for t in techs:
            s = shares.get(t, 0.0)
            gen_df[t] = demand_series * float(s)
    return gen_df


def _values_to_series(
    values: Mapping[int | str, float],
    years: Sequence[int],
    label: str,
) -> pd.Series:
    mapping = {}
    for key, value in values.items():
        mapping[int(key)] = float(value)
    if not mapping:
        raise ValueError(f"No values provided for {label}.")
    series = pd.Series(mapping, dtype=float).sort_index()
    idx = pd.Index(sorted(set(years)))
    series = series.reindex(idx.union(series.index), method=None)
    series = series.interpolate(method="index").ffill().bfill()
    return series.reindex(idx)

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import BASE_DEMAND_CASE

SCENARIO_SEPARATOR = "__"


def build_scenario_name(mix_case: str, demand_case: str) -> str:
    return f"{mix_case}{SCENARIO_SEPARATOR}{demand_case}"


def split_scenario_name(name: str) -> tuple[str, str]:
    if SCENARIO_SEPARATOR not in name:
        raise ValueError(
            f"Scenario '{name}' must use '{SCENARIO_SEPARATOR}' to separate mix and demand cases."
        )
    mix, demand = name.split(SCENARIO_SEPARATOR, 1)
    if not mix or not demand:
        raise ValueError(f"Scenario '{name}' is missing a mix or demand component.")
    return mix, demand


def list_mix_cases(root: Path) -> list[str]:
    root = Path(root)
    return sorted([entry.name for entry in root.iterdir() if entry.is_dir()])


def load_mix_dataframe(root: Path, mix_case: str, pollutant: str = "co2") -> pd.DataFrame:
    root = Path(root)
    path = root / mix_case / f"{pollutant}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Pollutant file not found: {path}")
    df = pd.read_csv(path, comment="#")
    if "year" not in df.columns:
        raise ValueError(f"File '{path}' must contain a 'year' column.")
    return df


def ensure_demand_columns(df: pd.DataFrame, demand_cases: Iterable[str]) -> None:
    missing = [f"absolute_{case}" for case in demand_cases if f"absolute_{case}" not in df.columns]
    if missing:
        raise ValueError(f"Missing absolute columns for demand cases: {missing}")


def load_scenario_delta(
    root: Path,
    scenario_name: str,
    baseline_case: str = BASE_DEMAND_CASE,
    pollutant: str = "co2",
) -> pd.DataFrame:
    mix_case, demand_case = split_scenario_name(scenario_name)
    df = load_mix_dataframe(root, mix_case, pollutant)
    delta_col = f"delta_{demand_case}"
    if delta_col not in df.columns:
        if demand_case == baseline_case:
            df[delta_col] = 0.0
        else:
            available = [col for col in df.columns if col.startswith("delta_")]
            raise KeyError(
                f"Delta column '{delta_col}' not found for mix '{mix_case}'. Available: {available}"
            )
    return df[["year", delta_col]].rename(columns={delta_col: "delta"})


def load_scenario_absolute(
    root: Path,
    scenario_name: str,
    pollutant: str = "co2",
) -> pd.DataFrame:
    mix_case, demand_case = split_scenario_name(scenario_name)
    df = load_mix_dataframe(root, mix_case, pollutant)
    abs_col = f"absolute_{demand_case}"
    if abs_col not in df.columns:
        available = [col for col in df.columns if col.startswith("absolute_")]
        raise KeyError(
            f"Absolute column '{abs_col}' not found for mix '{mix_case}'. Available: {available}"
        )
    return df[["year", abs_col]].rename(columns={abs_col: "absolute"})


def list_available_scenarios(
    root: Path,
    demand_cases: Iterable[str],
    *,
    include_baseline: bool,
    baseline_case: str = BASE_DEMAND_CASE,
) -> list[str]:
    root = Path(root)
    mixes = list_mix_cases(root)
    demand_list = [case for case in demand_cases if case]
    if not include_baseline:
        demand_list = [case for case in demand_list if case != baseline_case]
    scenarios: list[str] = []
    for mix in mixes:
        df = load_mix_dataframe(root, mix)
        for demand in demand_list:
            delta_col = f"delta_{demand}"
            if demand == baseline_case or delta_col in df.columns:
                scenarios.append(build_scenario_name(mix, demand))
    return sorted(scenarios)

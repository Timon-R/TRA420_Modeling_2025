"""Socioeconomic projections derived from DICE-style dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Socioeconomic table not found: {path}")
    return pd.read_csv(path)


def _resolve_path(path_like: str | Path, base: Path | None) -> Path:
    path = Path(path_like)
    if not path.is_absolute() and base is not None:
        path = (base / path).resolve()
    return path


@dataclass(slots=True)
class DiceSocioeconomics:
    """DICE-style growth model for population, TFP, and capital."""

    start_year: int
    scenario: str
    initial_population_million: float
    logistic_growth: float
    population_asymptote_million: float
    tfp_initial_level: float
    tfp_initial_growth: float
    tfp_decline_rate: float
    capital_share: float
    depreciation_rate: float
    savings_rate: float
    capital_output_ratio: float
    initial_capital_trillions: float | None

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, object],
        *,
        base_path: Path | None = None,
    ) -> "DiceSocioeconomics":
        population_cfg = cfg.get("population", {}) or {}
        tfp_cfg = cfg.get("tfp", {}) or {}
        capital_cfg = cfg.get("capital", {}) or {}

        start_year = int(cfg.get("start_year", 2019))
        scenario = str(cfg.get("scenario", "SSP2")).strip().upper()
        initial_population = float(population_cfg.get("initial_population_million", 7713.47))
        logistic_growth = float(population_cfg.get("logistic_growth", 0.028))

        asymptote_path = _resolve_path(
            population_cfg.get(
                "asymptote_table_path", "data/GDP_and_Population_data/DICE/dice_scenarios.csv"
            ),
            base_path,
        )
        scenario_table = _read_table(asymptote_path)
        scenario_table["scenario_upper"] = scenario_table["scenario"].str.upper()
        matches = scenario_table[scenario_table["scenario_upper"] == scenario]
        if matches.empty:
            raise ValueError(f"Scenario '{scenario}' not found in {asymptote_path}.")
        population_asymptote = float(matches.iloc[0]["L_max_million"])

        tfp_growth_table_path = tfp_cfg.get("scenario_growth_table_path")
        if tfp_growth_table_path:
            growth_table = _read_table(_resolve_path(tfp_growth_table_path, base_path))
            growth_table["scenario_upper"] = growth_table["scenario"].str.upper()
            growth_matches = growth_table[growth_table["scenario_upper"] == scenario]
            if growth_matches.empty:
                raise ValueError(f"Scenario '{scenario}' not found in {tfp_growth_table_path}.")
            tfp_growth = float(growth_matches.iloc[0].get("tfp_initial_growth", 0.0))
        else:
            tfp_growth = float(matches.iloc[0].get("tfp_initial_growth", 0.0))

        initial_level_label = str(tfp_cfg.get("initial_level_label", "nordhaus")).strip().lower()
        tfp_levels_path = _resolve_path(
            tfp_cfg.get(
                "initial_levels_table_path",
                "data/GDP_and_Population_data/DICE/dice_tfp_initial_levels.csv",
            ),
            base_path,
        )
        tfp_levels = _read_table(tfp_levels_path)
        tfp_levels["label_lower"] = tfp_levels["label"].str.lower()
        tfp_matches = tfp_levels[tfp_levels["label_lower"] == initial_level_label]
        if tfp_matches.empty:
            raise ValueError(
                f"Initial TFP level '{initial_level_label}' not found in {tfp_levels_path}."
            )
        tfp_initial = float(tfp_matches.iloc[0]["initial_level"])

        tfp_decline_rate = float(tfp_cfg.get("decline_rate", 0.005))
        capital_share = float(cfg.get("capital_share", 0.36507))
        depreciation_rate = float(capital_cfg.get("depreciation_rate", 0.0369791))
        savings_rate = float(capital_cfg.get("savings_rate", 0.223))
        capital_output_ratio = float(capital_cfg.get("capital_output_ratio", 3.41))
        initial_capital = capital_cfg.get("initial_stock_trillions")
        initial_capital_val = None if initial_capital is None else float(initial_capital)

        return cls(
            start_year=start_year,
            scenario=scenario,
            initial_population_million=initial_population,
            logistic_growth=logistic_growth,
            population_asymptote_million=population_asymptote,
            tfp_initial_level=tfp_initial,
            tfp_initial_growth=tfp_growth,
            tfp_decline_rate=tfp_decline_rate,
            capital_share=capital_share,
            depreciation_rate=depreciation_rate,
            savings_rate=savings_rate,
            capital_output_ratio=capital_output_ratio,
            initial_capital_trillions=initial_capital_val,
        )

    def _initial_equilibrium_output(self, population_million: float) -> float:
        labour_billions = max(population_million, 0.0) / 1000.0
        base = (
            self.tfp_initial_level
            * (self.capital_output_ratio**self.capital_share)
            * (labour_billions ** (1.0 - self.capital_share))
        )
        exponent = 1.0 / max(1e-9, 1.0 - self.capital_share)
        return base**exponent

    def project(self, end_year: int, *, currency_conversion: float = 1.0) -> pd.DataFrame:
        if end_year < self.start_year:
            raise ValueError("end_year must be greater than or equal to start_year.")

        years = np.arange(self.start_year, end_year + 1, dtype=int)
        population_series = np.zeros(years.shape[0], dtype=float)
        gdp_series = np.zeros_like(population_series)
        gdp_per_capita = np.zeros_like(population_series)
        gdp_growth_rate = np.full_like(population_series, fill_value=np.nan, dtype=float)
        capital_series = np.zeros_like(population_series)
        consumption_series = np.zeros_like(population_series)
        consumption_per_capita = np.zeros_like(population_series)
        tfp_series = np.zeros_like(population_series)

        population_million = float(self.initial_population_million)
        tfp_level = float(self.tfp_initial_level)
        capital = (
            float(self.initial_capital_trillions)
            if self.initial_capital_trillions is not None
            else self.capital_output_ratio * self._initial_equilibrium_output(population_million)
        )

        for idx, year in enumerate(years):
            labour_input = max(population_million, 1e-6) / 1000.0
            gdp = (
                tfp_level
                * (capital**self.capital_share)
                * (labour_input ** (1.0 - self.capital_share))
            )

            population_series[idx] = population_million
            gdp_series[idx] = gdp
            gdp_per_capita[idx] = (gdp * 1e12) / max(population_million * 1e6, 1.0)
            if idx > 0 and gdp_series[idx - 1] > 0:
                gdp_growth_rate[idx] = (gdp_series[idx] - gdp_series[idx - 1]) / gdp_series[idx - 1]

            tfp_series[idx] = tfp_level
            capital_series[idx] = capital
            consumption = (1.0 - self.savings_rate) * gdp
            consumption_series[idx] = consumption
            consumption_per_capita[idx] = (consumption * 1e12) / max(population_million * 1e6, 1.0)

            capital = (1.0 - self.depreciation_rate) * capital + self.savings_rate * gdp

            growth_rate = float(self.tfp_initial_growth) * np.exp(
                -self.tfp_decline_rate * max(year - self.start_year, 0)
            )
            tfp_level = tfp_level / max(1.0 - growth_rate, 1e-9)

            population_million = population_million * (
                (self.population_asymptote_million / max(population_million, 1e-9))
                ** self.logistic_growth
            )

        conversion = float(currency_conversion)
        if conversion not in (1.0, 1):
            gdp_series = gdp_series * conversion
            gdp_per_capita = gdp_per_capita * conversion
            capital_series = capital_series * conversion
            consumption_series = consumption_series * conversion
            consumption_per_capita = consumption_per_capita * conversion

        frame = pd.DataFrame(
            {
                "year": years,
                "population_million": population_series,
                "population_persons": population_series * 1.0e6,
                "gdp_trillion_usd": gdp_series,
                "gdp_per_capita_usd": gdp_per_capita,
                "gdp_growth_rate": gdp_growth_rate,
                "capital_stock_trillion_usd": capital_series,
                "consumption_trillion_usd": consumption_series,
                "consumption_per_capita_usd": consumption_per_capita,
                "tfp_level": tfp_series,
            }
        )
        return frame

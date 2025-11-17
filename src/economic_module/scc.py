"""Social cost of carbon calculations based on prescribed temperature and emission pathways."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from climate_module.scenario_runner import (
    ScenarioSpec,
    TemperatureResult,
    run_scenarios,
    step_change,
)

DamageFunction = Callable[..., np.ndarray]
SCCMethod = Literal["constant_discount", "ramsey_discount", "pulse"]
SCCAggregation = Literal["per_year", "average"]

KERNEL_REGULARIZATION = 1e-6

_PULSE_RESULT_CACHE: dict[
    tuple[str, tuple[int, ...], float, float, float],
    dict[str, TemperatureResult],
] = {}


def _safe_key(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    return safe.strip("_") or "scenario"


def _column(prefix: str, scenario: str) -> str:
    return f"{prefix}_{_safe_key(scenario)}"


def _extract_climate_scenarios(temperature_frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label, frame in temperature_frames.items():
        if "climate_scenario" not in frame.columns:
            continue
        values = frame["climate_scenario"].dropna().unique()
        if values.size == 0:
            continue
        if len(values) > 1:
            raise ValueError(
                f"Temperature series '{label}' references multiple climate scenarios: {values}."
            )
        mapping[label] = str(values[0])
    return mapping


def _infer_ssp_family(climate_id: str) -> str:
    match = re.match(r"ssp\s*([0-9])", climate_id, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer SSP family from climate scenario '{climate_id}'.")
    return f"SSP{match.group(1)}"


def _load_ssp_table(path: Path, key_column: str, key: str) -> pd.Series:
    try:
        df = pd.read_excel(path)
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "Reading SSP workbooks requires 'openpyxl'. Install it or provide a "
            "CSV override via gdp_series."
        ) from exc
    if key_column not in df.columns:
        raise ValueError(f"Expected column '{key_column}' in {path.name}.")
    df = df.rename(columns={key_column: "key"}).set_index("key")
    key_upper = key.upper()
    if key_upper not in df.index:
        raise ValueError(f"Scenario '{key_upper}' not found in {path.name}.")
    series = df.loc[key_upper].dropna()
    series.index = series.index.astype(int)
    return series.astype(float)


def _load_ssp_economic_data(ssp_family: str, directory: Path) -> tuple[pd.Series, pd.Series | None]:
    """Load SSP GDP and population series for the requested family.

    Preference order:
    1) Long-format CSVs with 2020 PPP GDP and population in millions:
       - ``SSP_gdp_long_2020ppp.csv`` expects columns: scenario, year, gdp_billion_ppp_2020
       - ``SSP_population_long.csv`` expects columns: scenario, year, population_millions
    2) Legacy Excel workbooks ``GDP_SSP1_5.xlsx`` and ``POP_SSP1_5.xlsx``.

    Returns GDP in trillions of USD (PPP-2020) and population in millions.
    """

    family_key = ssp_family.upper().strip()

    # Try CSV sources first
    gdp_csv = directory / "SSP_gdp_long_2020ppp.csv"
    pop_csv = directory / "SSP_population_long.csv"
    if gdp_csv.exists():
        gdp_df = pd.read_csv(gdp_csv, comment="#")
        if not {"scenario", "year"}.issubset(gdp_df.columns):
            raise ValueError(f"{gdp_csv.name} must contain 'scenario' and 'year' columns.")
        # Choose correct GDP column (prefer 2020 PPP). Fall back to 2010 PPP if needed.
        gdp_col = None
        if "gdp_billion_ppp_2020" in gdp_df.columns:
            gdp_col = "gdp_billion_ppp_2020"
        elif "gdp_billion_ppp_2010" in gdp_df.columns:
            gdp_col = "gdp_billion_ppp_2010"
        else:
            raise ValueError(
                f"{gdp_csv.name} missing expected GDP column "
                "('gdp_billion_ppp_2020' or 'gdp_billion_ppp_2010')."
            )
        gdp_sel = gdp_df[gdp_df["scenario"].str.upper() == family_key][["year", gdp_col]].copy()
        if gdp_sel.empty:
            raise ValueError(f"Scenario '{family_key}' not found in {gdp_csv.name}.")
        gdp_sel["year"] = gdp_sel["year"].astype(int)
        gdp_series = (
            gdp_sel.set_index("year")[gdp_col].astype(float) / 1000.0
        )  # billions → trillions

        population_series: pd.Series | None = None
        if pop_csv.exists():
            pop_df = pd.read_csv(pop_csv, comment="#")
            if not {"scenario", "year"}.issubset(pop_df.columns):
                raise ValueError(f"{pop_csv.name} must contain 'scenario' and 'year' columns.")
            # Prefer 'population_millions'; fall back to 'population' (persons)
            pop_col = None
            if "population_millions" in pop_df.columns:
                pop_col = "population_millions"
                scale = 1.0
            elif "population" in pop_df.columns:
                pop_col = "population"
                scale = 1.0 / 1e6  # persons → millions
            else:
                raise ValueError(
                    f"{pop_csv.name} missing expected population column "
                    "('population_millions' or 'population')."
                )
            pop_sel = pop_df[pop_df["scenario"].str.upper() == family_key][["year", pop_col]].copy()
            if not pop_sel.empty:
                pop_sel["year"] = pop_sel["year"].astype(int)
                population_series = pop_sel.set_index("year")[pop_col].astype(float) * scale

        return (
            gdp_series.sort_index(),
            None if population_series is None else population_series.sort_index(),
        )

    # Fallback to legacy Excel workbooks
    gdp_path = directory / "GDP_SSP1_5.xlsx"
    pop_path = directory / "POP_SSP1_5.xlsx"
    if not gdp_path.exists():
        raise FileNotFoundError(f"Missing GDP dataset: {gdp_csv if gdp_csv.exists() else gdp_path}")

    gdp_series = (
        _load_ssp_table(gdp_path, "GDP", ssp_family) / 1000.0
    )  # convert billions to trillions

    population_series: pd.Series | None = None
    if pop_path.exists():
        population_series = _load_ssp_table(pop_path, "Population", ssp_family)

    return gdp_series, population_series


def _align_series(series: pd.Series, years: Iterable[int]) -> pd.Series:
    idx = pd.Index(sorted(set(int(y) for y in years)))
    aligned = series.reindex(idx)
    aligned = aligned.sort_index()
    aligned = aligned.ffill().bfill()
    return aligned


@dataclass(slots=True)
class EconomicInputs:
    """Economic, temperature, and emission series required to compute SCC."""

    years: np.ndarray
    gdp_trillion_usd: np.ndarray
    temperature_scenarios_c: dict[str, np.ndarray]
    emission_scenarios_tco2: dict[str, np.ndarray]
    population_million: np.ndarray | None = None
    climate_scenarios: dict[str, str] | None = None
    ssp_family: str | None = None

    def __post_init__(self) -> None:
        if not self.temperature_scenarios_c:
            raise ValueError("temperature_scenarios_c must contain at least one scenario.")
        if set(self.temperature_scenarios_c) != set(self.emission_scenarios_tco2):
            raise ValueError("Temperature and emission scenario labels must match.")
        length = len(self.years)
        if len(self.gdp_trillion_usd) != length:
            raise ValueError("gdp_trillion_usd must match the length of years.")
        if self.population_million is not None and len(self.population_million) != length:
            raise ValueError("population_million must match the length of years.")
        for name, temps in self.temperature_scenarios_c.items():
            if len(temps) != length:
                raise ValueError(
                    f"Temperature series '{name}' does not match the year vector length."
                )
        for name, emissions in self.emission_scenarios_tco2.items():
            if len(emissions) != length:
                raise ValueError(f"Emission series '{name}' does not match the year vector length.")
        if self.climate_scenarios is None:
            self.climate_scenarios = {}
        else:
            missing = set(self.temperature_scenarios_c) - set(self.climate_scenarios)
            if missing:
                raise ValueError(f"Missing climate scenario metadata for: {sorted(missing)}")

    @property
    def scenario_names(self) -> list[str]:
        return list(self.temperature_scenarios_c.keys())

    def temperature(self, scenario: str) -> np.ndarray:
        try:
            return self.temperature_scenarios_c[scenario]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown temperature scenario '{scenario}'.") from exc

    def emission(self, scenario: str) -> np.ndarray:
        try:
            return self.emission_scenarios_tco2[scenario]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown emission scenario '{scenario}'.") from exc

    @classmethod
    def from_csv(
        cls,
        temperature_paths: Mapping[str, Path | str],
        emission_paths: Mapping[str, Path | str],
        gdp_path: Path | str | None = None,
        *,
        temperature_column: str = "temperature_c",
        emission_column: str = "delta_mtco2",
        emission_to_tonnes: float = 1e6,
        gdp_column: str = "gdp_trillion_usd",
        population_column: str = "population_million",
        gdp_frame: pd.DataFrame | None = None,
        gdp_population_directory: Path | str | None = None,
    ) -> "EconomicInputs":
        """Load inputs from CSV files.

        ``temperature_paths`` and ``emission_paths`` map scenario labels to CSVs
        containing ``year`` and the respective data columns. Emission values are
        scaled by ``emission_to_tonnes`` (default converts Mt CO₂ to t CO₂).
        ``gdp_frame`` can be provided directly to supply custom GDP/population
        trajectories without reading from disk.
        """

        if not temperature_paths or not emission_paths:
            raise ValueError("Provide temperature and emission series for at least one scenario.")

        temperature_labels = set(temperature_paths)
        emission_labels = set(emission_paths)
        if temperature_labels != emission_labels:
            raise ValueError("Temperature and emission scenario labels must match.")

        temp_frames: dict[str, pd.DataFrame] = {}
        for label, path in temperature_paths.items():
            frame = _load_yearly_csv(
                path,
                required_columns={temperature_column},
                optional_columns={"climate_scenario"},
            )
            temp_frames[label] = frame.rename(columns={temperature_column: "temperature_c"})

        emission_frames: dict[str, pd.DataFrame] = {}
        for label, path in emission_paths.items():
            frame = _load_yearly_csv(path, required_columns={emission_column})
            emission_frames[label] = frame.rename(columns={emission_column: "emission_raw"})

        climate_scenarios = _extract_climate_scenarios(temp_frames)
        if not climate_scenarios:
            climate_scenarios = None
        ssp_family: str | None = None

        if gdp_frame is not None:
            gdp_data_frame = gdp_frame.copy()
        elif gdp_population_directory is not None:
            if not climate_scenarios:
                raise ValueError(
                    "Temperature CSVs must include 'climate_scenario' to determine "
                    "SSP-specific GDP."
                )
            families = {_infer_ssp_family(climate_id) for climate_id in climate_scenarios.values()}
            if len(families) != 1:
                raise ValueError(
                    "Temperature CSVs reference multiple SSP families; supply a single SSP "
                    "or provide custom GDP data."
                )
            ssp_family = families.pop()
            gdp_series, population_series = _load_ssp_economic_data(
                ssp_family, Path(gdp_population_directory)
            )
            gdp_series = gdp_series.sort_index()
            if population_series is not None:
                population_series = _align_series(population_series.sort_index(), gdp_series.index)
            gdp_data_frame = pd.DataFrame(
                {
                    "year": gdp_series.index.astype(int),
                    gdp_column: gdp_series.values,
                }
            )
            if population_series is not None:
                gdp_data_frame[population_column] = population_series.loc[
                    gdp_data_frame["year"]
                ].to_numpy()
        else:
            if gdp_path is None:
                raise ValueError("Provide either gdp_path, gdp_frame, or gdp_population_directory.")
            gdp_data_frame = _load_yearly_csv(
                gdp_path,
                required_columns={gdp_column},
                optional_columns={population_column},
            )

        year_bounds: list[tuple[int, int]] = []
        gdp_years = gdp_data_frame["year"].astype(int)
        year_bounds.append((int(gdp_years.min()), int(gdp_years.max())))
        for frame in temp_frames.values():
            series_years = frame["year"].astype(int)
            year_bounds.append((int(series_years.min()), int(series_years.max())))
        for frame in emission_frames.values():
            series_years = frame["year"].astype(int)
            year_bounds.append((int(series_years.min()), int(series_years.max())))
        start_year = max(bound[0] for bound in year_bounds)
        end_year = min(bound[1] for bound in year_bounds)
        if start_year > end_year:
            raise ValueError(
                "Temperature, emission, and GDP datasets do not share overlapping years."
            )

        years = np.arange(start_year, end_year + 1, dtype=int)

        def _reindex_series(frame: pd.DataFrame, column: str, *, scale: float = 1.0) -> np.ndarray:
            series = frame.set_index("year")[column].astype(float).sort_index().reindex(years)
            if series.isna().any():
                series = series.interpolate(method="linear", limit_direction="both")
                series = series.ffill().bfill()
            if series.isna().any():
                missing = series.index[series.isna()].tolist()
                raise ValueError(
                    f"Unable to interpolate column '{column}' for all years. Missing: {missing}"
                )
            if scale != 1.0:
                series = series * scale
            return series.to_numpy(dtype=float)

        gdp_series = _reindex_series(gdp_data_frame, gdp_column)
        population = None
        if population_column in gdp_data_frame:
            population = _reindex_series(gdp_data_frame, population_column)

        temperature_series = {
            label: _reindex_series(frame, "temperature_c") for label, frame in temp_frames.items()
        }
        emission_series = {
            label: _reindex_series(frame, "emission_raw", scale=emission_to_tonnes)
            for label, frame in emission_frames.items()
        }

        return cls(
            years=years,
            gdp_trillion_usd=gdp_series,
            temperature_scenarios_c=temperature_series,
            emission_scenarios_tco2=emission_series,
            population_million=population,
            climate_scenarios=climate_scenarios,
            ssp_family=ssp_family,
        )

    def to_frame(self, *, scenarios: Iterable[str] | None = None) -> pd.DataFrame:
        scenario_list = list(scenarios) if scenarios is not None else self.scenario_names
        data: dict[str, np.ndarray | list[int]] = {
            "year": self.years,
            "gdp_trillion_usd": self.gdp_trillion_usd,
        }
        if self.population_million is not None:
            data["population_million"] = self.population_million
        for scenario in scenario_list:
            data[_column("temperature_c", scenario)] = self.temperature(scenario)
            data[_column("emissions_tco2", scenario)] = self.emission(scenario)
            if self.climate_scenarios and scenario in self.climate_scenarios:
                data[_column("climate_scenario", scenario)] = np.full_like(
                    self.years, self.climate_scenarios[scenario], dtype=object
                )
        return pd.DataFrame(data)


@dataclass(slots=True)
class SCCResult:
    """Container for SCC outputs."""

    method: SCCMethod
    aggregation: SCCAggregation
    scenario: str
    reference: str
    base_year: int
    add_tco2: float
    scc_usd_per_tco2: float
    per_year: pd.DataFrame
    details: pd.DataFrame
    temperature_kernel: np.ndarray | None = None
    run_method: str | None = None


def _load_yearly_csv(
    path: Path | str,
    *,
    required_columns: set[str],
    optional_columns: set[str] | None = None,
) -> pd.DataFrame:
    optional_columns = optional_columns or set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, comment="#")
    if "year" not in df.columns:
        raise ValueError(f"'{path}' must contain a 'year' column.")

    missing = required_columns.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"'{path}' is missing required columns: {missing_list}.")

    available_optionals = [col for col in optional_columns if col in df.columns]
    keep_columns = ["year", *sorted(required_columns), *sorted(available_optionals)]
    df = df[keep_columns].copy()
    df["year"] = df["year"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Damage helpers
# ---------------------------------------------------------------------------


def damage_dice(
    temp: np.ndarray,
    *,
    delta1: float = 0.0,
    delta2: float = 0.002,
    # Threshold amplification
    use_threshold: bool = False,
    threshold_temperature: float = 3.0,
    threshold_scale: float = 0.2,
    threshold_power: float = 2.0,
    # Saturation / cap
    use_saturation: bool = False,
    max_fraction: float = 0.99,
    saturation_mode: str = "rational",
    # Catastrophic add-ons
    use_catastrophic: bool = False,
    catastrophic_temperature: float = 5.0,
    disaster_fraction: float = 0.75,
    disaster_gamma: float = 1.0,
    disaster_mode: str = "prob",
) -> np.ndarray:
    """Return GDP damage fractions using a configurable DICE-style function.

    The baseline follows ``delta1 * T + delta2 * T^2``. Optional extensions:

    - ``use_threshold`` scales damages once temperature exceeds
      ``threshold_temperature`` (power-law amplification).
    - ``use_saturation`` keeps damages below ``max_fraction`` via a rational
      curve (default) or a hard clamp.
    - ``use_catastrophic`` adds disaster losses when temperatures cross
      ``catastrophic_temperature`` either step-wise or probabilistically.
    """

    temperatures = np.asarray(temp, dtype=float)
    damage = delta1 * temperatures + delta2 * temperatures**2

    if use_threshold:
        amplify = (
            1.0
            + threshold_scale
            * np.maximum(0.0, temperatures - threshold_temperature) ** threshold_power
        )
        damage = damage * amplify

    if use_saturation:
        positive = np.maximum(0.0, damage)
        if saturation_mode == "rational":
            with np.errstate(divide="ignore", invalid="ignore"):
                scaled = np.divide(
                    positive,
                    1.0 + positive,
                    out=np.zeros_like(positive),
                    where=positive >= 0.0,
                )
            damage = max_fraction * scaled
        elif saturation_mode == "clamp":
            damage = np.clip(damage, 0.0, max_fraction)
        else:
            raise ValueError("saturation_mode must be 'rational' or 'clamp'")

    if use_catastrophic:
        if disaster_mode == "step":
            extra = np.where(temperatures >= catastrophic_temperature, disaster_fraction, 0.0)
        elif disaster_mode == "prob":
            exceed = np.maximum(0.0, temperatures - catastrophic_temperature)
            extra = (1.0 - np.exp(-disaster_gamma * exceed)) * disaster_fraction
        else:
            raise ValueError("disaster_mode must be 'prob' or 'step'")
        damage = damage + extra

    # Always cap to max_fraction to avoid >100% GDP losses.
    damage = np.minimum(damage, max_fraction)
    return damage


def compute_damages(
    inputs: EconomicInputs,
    *,
    scenarios: Iterable[str] | None = None,
    damage_func: DamageFunction = damage_dice,
    damage_kwargs: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return damages per scenario in absolute USD."""

    scenario_list = list(scenarios) if scenarios is not None else inputs.scenario_names
    if not scenario_list:
        raise ValueError("No scenarios provided for damage calculation.")

    damage_kwargs = dict(damage_kwargs or {})
    # Strip internal tuning flags not part of the damage function signature
    safe_damage_kwargs = {k: v for k, v in damage_kwargs.items() if not str(k).startswith("_")}
    gdp_usd = inputs.gdp_trillion_usd.astype(float) * 1e12

    data: dict[str, np.ndarray | list[int]] = {
        "year": inputs.years,
        "gdp_trillion_usd": inputs.gdp_trillion_usd,
    }
    if inputs.population_million is not None:
        data["population_million"] = inputs.population_million

    for scenario in scenario_list:
        temps = inputs.temperature(scenario)
        emissions = inputs.emission(scenario)
        fractions = damage_func(temps, **safe_damage_kwargs)
        damages = fractions * gdp_usd
        data[_column("temperature_c", scenario)] = temps
        data[_column("damage_fraction", scenario)] = fractions
        data[_column("damage_usd", scenario)] = damages
        data[_column("emissions_tco2", scenario)] = emissions

    return pd.DataFrame(data)


def compute_damage_difference(
    inputs: EconomicInputs,
    scenario: str,
    reference: str,
    *,
    damage_func: DamageFunction = damage_dice,
    damage_kwargs: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return damage table including differences between two scenarios."""

    damage_df = compute_damages(
        inputs,
        scenarios=[reference, scenario],
        damage_func=damage_func,
        damage_kwargs=damage_kwargs,
    )

    scenario_col = _column("damage_usd", scenario)
    reference_col = _column("damage_usd", reference)
    damage_df["delta_damage_usd"] = damage_df[scenario_col] - damage_df[reference_col]

    scenario_em_col = _column("emissions_tco2", scenario)
    reference_em_col = _column("emissions_tco2", reference)
    damage_df["delta_emissions_tco2"] = damage_df[scenario_em_col] - damage_df[reference_em_col]

    return damage_df


# ---------------------------------------------------------------------------
# Temperature kernel and damage attribution helpers
# ---------------------------------------------------------------------------


def _estimate_temperature_kernel(
    emission_delta: np.ndarray,
    temperature_delta: np.ndarray,
    *,
    regularization: float = KERNEL_REGULARIZATION,
    horizon: int | None = None,
    smoothing_lambda: float = 0.0,
    nonnegativity: bool = False,
) -> np.ndarray:
    emission_delta = np.asarray(emission_delta, dtype=float)
    temperature_delta = np.asarray(temperature_delta, dtype=float)
    n = emission_delta.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    m = n if horizon is None else max(1, min(n, int(horizon)))
    toeplitz = np.zeros((n, m), dtype=float)
    for i in range(n):
        length = min(i + 1, m)
        seg = emission_delta[i - length + 1 : i + 1][::-1]
        toeplitz[i, :length] = seg
    gram = toeplitz.T @ toeplitz
    if regularization > 0:
        gram = gram + regularization * np.eye(n, dtype=float)
    # Adjust identity to match m if truncated
    if gram.shape[0] != n and regularization > 0:
        gram = toeplitz.T @ toeplitz + regularization * np.eye(gram.shape[0], dtype=float)
    # Optional smoothing penalty (first-order differences)
    if smoothing_lambda > 0.0 and gram.shape[0] >= 2:
        D = np.eye(gram.shape[0], dtype=float) - np.eye(gram.shape[0], k=1, dtype=float)
        D = D[:-1, :]
        gram = gram + smoothing_lambda * (D.T @ D)
    rhs = toeplitz.T @ temperature_delta
    try:
        kernel_m = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        kernel_m = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    if nonnegativity:
        kernel_m = np.maximum(kernel_m, 0.0)
    # Embed in full-length kernel (pad zeros if truncated)
    if m == n:
        return kernel_m
    kernel = np.zeros(n, dtype=float)
    kernel[:m] = kernel_m
    return kernel


def _temperature_contributions(
    emission_delta: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    emission_delta = np.asarray(emission_delta, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    n = emission_delta.shape[0]
    contributions = np.zeros((n, n), dtype=float)
    for tau in range(n):
        value = emission_delta[tau]
        if np.isclose(value, 0.0):
            continue
        remaining = n - tau
        if remaining <= 0:
            continue
        contributions[tau:, tau] = kernel[:remaining] * value
    return contributions


def _allocate_damages_to_emission_years(
    *,
    base_temperatures: np.ndarray,
    gdp_trillion_usd: np.ndarray,
    damage_func: DamageFunction,
    damage_kwargs: Mapping[str, float] | None,
    temperature_contrib: np.ndarray,
    base_damage_fraction: np.ndarray,
) -> np.ndarray:
    damage_kwargs = dict(damage_kwargs or {})
    gdp_usd = np.asarray(gdp_trillion_usd, dtype=float) * 1e12
    base_temperatures = np.asarray(base_temperatures, dtype=float)
    base_damage_fraction = np.asarray(base_damage_fraction, dtype=float)
    base_damage_usd = base_damage_fraction * gdp_usd
    n_years = temperature_contrib.shape[0]
    damage_contrib = np.zeros_like(temperature_contrib, dtype=float)
    # Optional linearization slope (fraction per °C) for additivity
    linearize = False
    slope_fraction = None
    if isinstance(damage_kwargs, dict) and damage_kwargs.get("_linearized_flag__", False):
        linearize = True
        eps = float(damage_kwargs.get("_linearized_eps__", 1e-4))
        safe_kwargs = {k: v for k, v in damage_kwargs.items() if not str(k).startswith("_")}
        damage_fraction_eps = damage_func(base_temperatures + eps, **safe_kwargs)
        slope_fraction = (damage_fraction_eps - base_damage_fraction) / eps

    for tau in range(n_years):
        temp_delta = temperature_contrib[:, tau]
        if np.allclose(temp_delta, 0.0):
            continue
        if linearize and slope_fraction is not None:
            damage_contrib[:, tau] = slope_fraction * temp_delta * gdp_usd
        else:
            temps_with_tau = base_temperatures + temp_delta
            safe_kwargs = {
                k: v for k, v in (damage_kwargs or {}).items() if not str(k).startswith("_")
            }
            damage_fraction_with_tau = damage_func(temps_with_tau, **safe_kwargs)
            damage_usd_with_tau = damage_fraction_with_tau * gdp_usd
            damage_contrib[:, tau] = damage_usd_with_tau - base_damage_usd
    return damage_contrib


# ---------------------------------------------------------------------------
# Discounting helpers
# ---------------------------------------------------------------------------


def _constant_discount_factors(years: np.ndarray, base_year: int, rate: float) -> np.ndarray:
    if rate < -1.0:
        raise ValueError("Discount rate must be greater than -100%.")
    years = np.asarray(years, dtype=float)
    offsets = years - float(base_year)
    factors = np.zeros_like(offsets, dtype=float)
    future_mask = offsets >= 0
    factors[future_mask] = 1.0 / np.power(1.0 + rate, offsets[future_mask])
    return factors


def _compute_consumption_growth(
    inputs: EconomicInputs,
    reference_damage_usd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if inputs.population_million is None:
        raise ValueError("Population data required for Ramsey discounting.")

    gdp_usd = inputs.gdp_trillion_usd * 1e12
    population = np.asarray(inputs.population_million, dtype=float) * 1e6
    consumption = np.maximum(gdp_usd - reference_damage_usd, 0.0)
    consumption_per_capita = np.divide(
        consumption, population, out=np.zeros_like(consumption), where=population > 0
    )

    growth = np.full_like(consumption_per_capita, fill_value=np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth[1:] = (
            consumption_per_capita[1:] - consumption_per_capita[:-1]
        ) / consumption_per_capita[:-1]
    return consumption_per_capita, growth


def _ramsey_discount_factors(
    years: np.ndarray,
    base_year: int,
    growth: np.ndarray,
    *,
    rho: float,
    eta: float,
) -> np.ndarray:
    if rho < -1.0:
        raise ValueError("rho must be greater than -100%.")
    if eta < 0:
        raise ValueError("eta must be non-negative.")

    years = np.asarray(years, dtype=int)
    growth = np.asarray(growth, dtype=float)
    idx_candidates = np.where(years == int(base_year))[0]
    if len(idx_candidates) == 0:
        raise ValueError(f"base_year {base_year} not present in years array")
    base_idx = int(idx_candidates[0])

    factors = np.zeros(len(years), dtype=float)
    factors[base_idx] = 1.0

    for idx in range(base_idx + 1, len(years)):
        g = np.nan_to_num(growth[idx], nan=0.0)
        r = rho + eta * g
        factors[idx] = factors[idx - 1] / (1.0 + r)

    return factors


# ---------------------------------------------------------------------------
# SCC pipelines
# ---------------------------------------------------------------------------


def _build_per_year_table(
    damage_df: pd.DataFrame,
    *,
    scenario: str,
    reference: str,
    discount_factors: np.ndarray,
    damage_func: DamageFunction,
    damage_kwargs: Mapping[str, float] | None,
    kernel_regularization: float = KERNEL_REGULARIZATION,
    kernel_horizon: int | None = None,
    kernel_nonnegativity: bool = False,
    kernel_smoothing_lambda: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    per_year = damage_df[["year", "delta_damage_usd", "delta_emissions_tco2"]].copy()
    years = per_year["year"].astype(int).to_numpy()
    discount_factors = np.asarray(discount_factors, dtype=float)
    if discount_factors.shape[0] != years.shape[0]:
        raise ValueError("Discount factor length must match the number of years.")

    per_year["discount_factor"] = discount_factors
    per_year["discounted_delta_usd"] = per_year["delta_damage_usd"].astype(float) * discount_factors

    delta_emissions = per_year["delta_emissions_tco2"].astype(float).to_numpy()
    temperature_reference = damage_df[_column("temperature_c", reference)].astype(float).to_numpy()
    temperature_scenario = damage_df[_column("temperature_c", scenario)].astype(float).to_numpy()
    temperature_delta = temperature_scenario - temperature_reference

    kernel = _estimate_temperature_kernel(
        delta_emissions,
        temperature_delta,
        regularization=kernel_regularization,
        horizon=kernel_horizon,
        smoothing_lambda=kernel_smoothing_lambda,
        nonnegativity=kernel_nonnegativity,
    )
    temperature_contrib = _temperature_contributions(delta_emissions, kernel)

    gdp_trillion = damage_df["gdp_trillion_usd"].astype(float).to_numpy()
    base_damage_fraction = damage_df[_column("damage_fraction", reference)].astype(float).to_numpy()

    # Pass a special flag for linearized allocations via damage_kwargs
    damage_contrib = _allocate_damages_to_emission_years(
        base_temperatures=temperature_reference,
        gdp_trillion_usd=gdp_trillion,
        damage_func=damage_func,
        damage_kwargs=damage_kwargs,
        temperature_contrib=temperature_contrib,
        base_damage_fraction=base_damage_fraction,
    )

    damage_attributed = damage_contrib.sum(axis=0)
    discounted_damage_attributed = (discount_factors[:, None] * damage_contrib).sum(axis=0)

    scc_values = np.full(years.shape[0], np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        for idx, emission_delta in enumerate(delta_emissions):
            beta_tau = discount_factors[idx]
            if beta_tau <= 0 or np.isclose(emission_delta, 0.0):
                continue
            pv_base = discounted_damage_attributed[idx]
            scc_values[idx] = pv_base / (beta_tau * emission_delta)

    per_year["damage_attributed_usd"] = damage_attributed
    per_year["discounted_damage_attributed_usd"] = discounted_damage_attributed
    per_year["scc_usd_per_tco2"] = scc_values

    temperature_reconstructed = temperature_contrib.sum(axis=1)
    return (
        per_year,
        kernel,
        temperature_contrib,
        damage_contrib,
        temperature_delta,
        temperature_reconstructed,
    )


def _aggregate_scc(per_year: pd.DataFrame, add_tco2: float) -> float:
    discounted_damage = per_year["discounted_delta_usd"].sum()
    if np.isclose(add_tco2, 0.0):
        return float("nan")
    return discounted_damage / add_tco2


def compute_scc_constant_discount(
    inputs: EconomicInputs,
    *,
    scenario: str,
    reference: str,
    base_year: int,
    discount_rate: float,
    aggregation: SCCAggregation,
    add_tco2: float | None = None,
    damage_func: DamageFunction = damage_dice,
    damage_kwargs: Mapping[str, float] | None = None,
) -> SCCResult:
    """Compute SCC using a constant discount rate."""

    damage_df = compute_damage_difference(
        inputs,
        scenario=scenario,
        reference=reference,
        damage_func=damage_func,
        damage_kwargs=damage_kwargs,
    )

    factors = _constant_discount_factors(damage_df["year"].to_numpy(), base_year, discount_rate)
    (
        per_year,
        kernel,
        temperature_contrib,
        damage_contrib,
        temperature_delta,
        temperature_reconstructed,
    ) = _build_per_year_table(
        damage_df,
        scenario=scenario,
        reference=reference,
        discount_factors=factors,
        damage_func=damage_func,
        damage_kwargs=damage_kwargs,
        kernel_regularization=float(damage_kwargs.get("_kernel_alpha__", KERNEL_REGULARIZATION))
        if isinstance(damage_kwargs, dict)
        else KERNEL_REGULARIZATION,
        kernel_horizon=int(damage_kwargs.get("_kernel_horizon__", 0))
        if isinstance(damage_kwargs, dict) and int(damage_kwargs.get("_kernel_horizon__", 0)) > 0
        else None,
        kernel_nonnegativity=bool(damage_kwargs.get("_kernel_nonneg__", False))
        if isinstance(damage_kwargs, dict)
        else False,
        kernel_smoothing_lambda=float(damage_kwargs.get("_kernel_smooth__", 0.0))
        if isinstance(damage_kwargs, dict)
        else 0.0,
    )

    total_delta_emissions = per_year["delta_emissions_tco2"].sum()
    effective_add_tco2 = add_tco2 if add_tco2 is not None else total_delta_emissions

    scc_value = _aggregate_scc(per_year, effective_add_tco2)

    damage_df = damage_df.assign(
        discount_factor=factors,
        discounted_delta_usd=per_year["discounted_delta_usd"],
        temperature_delta_c=temperature_delta,
        temperature_reconstructed_c=temperature_reconstructed,
        damage_reconstructed_usd=damage_contrib.sum(axis=1),
    )

    return SCCResult(
        method="constant_discount",
        aggregation=aggregation,
        scenario=scenario,
        reference=reference,
        base_year=base_year,
        add_tco2=effective_add_tco2,
        scc_usd_per_tco2=scc_value,
        per_year=per_year,
        details=damage_df,
        temperature_kernel=kernel,
        run_method="kernel",
    )


def compute_scc_ramsey_discount(
    inputs: EconomicInputs,
    *,
    scenario: str,
    reference: str,
    base_year: int,
    rho: float,
    eta: float,
    aggregation: SCCAggregation,
    add_tco2: float | None = None,
    damage_func: DamageFunction = damage_dice,
    damage_kwargs: Mapping[str, float] | None = None,
) -> SCCResult:
    """Compute SCC using Ramsey rule discounting."""

    damage_df = compute_damage_difference(
        inputs,
        scenario=scenario,
        reference=reference,
        damage_func=damage_func,
        damage_kwargs=damage_kwargs,
    )

    reference_col = _column("damage_usd", reference)
    consumption_pc, growth = _compute_consumption_growth(
        inputs, damage_df[reference_col].to_numpy()
    )

    factors = _ramsey_discount_factors(
        damage_df["year"].to_numpy(),
        base_year,
        growth,
        rho=rho,
        eta=eta,
    )

    (
        per_year,
        kernel,
        temperature_contrib,
        damage_contrib,
        temperature_delta,
        temperature_reconstructed,
    ) = _build_per_year_table(
        damage_df,
        scenario=scenario,
        reference=reference,
        discount_factors=factors,
        damage_func=damage_func,
        damage_kwargs=damage_kwargs,
        kernel_regularization=float(damage_kwargs.get("_kernel_alpha__", KERNEL_REGULARIZATION))
        if isinstance(damage_kwargs, dict)
        else KERNEL_REGULARIZATION,
        kernel_horizon=int(damage_kwargs.get("_kernel_horizon__", 0))
        if isinstance(damage_kwargs, dict) and int(damage_kwargs.get("_kernel_horizon__", 0)) > 0
        else None,
        kernel_nonnegativity=bool(damage_kwargs.get("_kernel_nonneg__", False))
        if isinstance(damage_kwargs, dict)
        else False,
        kernel_smoothing_lambda=float(damage_kwargs.get("_kernel_smooth__", 0.0))
        if isinstance(damage_kwargs, dict)
        else 0.0,
    )

    total_delta_emissions = per_year["delta_emissions_tco2"].sum()
    effective_add_tco2 = add_tco2 if add_tco2 is not None else total_delta_emissions
    scc_value = _aggregate_scc(per_year, effective_add_tco2)

    damage_df = damage_df.assign(
        consumption_per_capita_usd=consumption_pc,
        consumption_growth=growth,
        discount_factor=factors,
        discounted_delta_usd=per_year["discounted_delta_usd"],
        temperature_delta_c=temperature_delta,
        temperature_reconstructed_c=temperature_reconstructed,
        damage_reconstructed_usd=damage_contrib.sum(axis=1),
    )

    return SCCResult(
        method="ramsey_discount",
        aggregation=aggregation,
        scenario=scenario,
        reference=reference,
        base_year=base_year,
        add_tco2=effective_add_tco2,
        scc_usd_per_tco2=scc_value,
        per_year=per_year,
        details=damage_df,
        temperature_kernel=kernel,
        run_method="kernel",
    )


def compute_scc_pulse(
    inputs: EconomicInputs,
    *,
    scenario: str,
    reference: str,
    base_year: int,
    discount_method: str,
    discount_rate: float | None = None,
    rho: float | None = None,
    eta: float | None = None,
    aggregation: SCCAggregation,
    add_tco2: float | None = None,
    damage_func: DamageFunction = damage_dice,
    damage_kwargs: Mapping[str, float] | None = None,
    pulse_max_year: int | None = None,
) -> SCCResult:
    """Compute SCC by emission year using FaIR pulse runs per calendar year."""

    climate_map = inputs.climate_scenarios or {}
    climate_id = climate_map.get(reference) or climate_map.get(scenario)
    if not climate_id:
        raise ValueError(
            "Climate scenario metadata (climate_scenario) is required for pulse method."
        )

    years = inputs.years.astype(int)
    start_year = int(years[0])
    end_year = int(years[-1])

    pulse_size_tco2 = 1_000_000.0
    if isinstance(damage_kwargs, Mapping):
        pulse_size_tco2 = float(damage_kwargs.get("_pulse_size_tco2__", pulse_size_tco2))
    pulse_size_mt = pulse_size_tco2 / 1e6

    specie = "CO2 FFI"
    if isinstance(damage_kwargs, Mapping):
        specie = str(damage_kwargs.get("_pulse_specie__", specie))

    safe_damage_kwargs = {
        k: v for k, v in (damage_kwargs or {}).items() if not str(k).startswith("_")
    }

    pulse_years = years
    if pulse_max_year is not None:
        pulse_years = pulse_years[pulse_years <= pulse_max_year]
    if pulse_years.size == 0:
        raise ValueError("No pulse years fall within the requested evaluation window.")

    cache_key = (
        climate_id,
        tuple(int(v) for v in pulse_years.tolist()),
        float(start_year),
        float(end_year),
        pulse_size_mt,
    )
    if cache_key in _PULSE_RESULT_CACHE:
        results = _PULSE_RESULT_CACHE[cache_key]
    else:
        specs: list[ScenarioSpec] = []
        for tau in pulse_years:
            up = step_change(pulse_size_mt, start_year=float(tau))
            down = step_change(pulse_size_mt, start_year=float(tau + 1))

            def pulse_builder_factory(up_fun, down_fun):
                return lambda timepoints, cfg: up_fun(timepoints, cfg) - down_fun(timepoints, cfg)

            pulse_builder = pulse_builder_factory(up, down)
            specs.append(
                ScenarioSpec(
                    label=f"pulse_{tau}",
                    scenario=climate_id,
                    emission_adjustments={specie: pulse_builder},
                    start_year=float(start_year),
                    end_year=float(end_year),
                    timestep=1.0,
                )
            )

        climate_logger = logging.getLogger("climate_module")
        previous_level = climate_logger.level
        climate_logger.setLevel(max(previous_level, logging.WARNING))
        try:
            results = run_scenarios(specs)
        finally:
            climate_logger.setLevel(previous_level)
        _PULSE_RESULT_CACHE[cache_key] = results

    damage_df = compute_damage_difference(
        inputs,
        scenario=scenario,
        reference=reference,
        damage_func=damage_func,
        damage_kwargs=damage_kwargs,
    )

    temperature_ref = damage_df[_column("temperature_c", reference)].astype(float).to_numpy()
    temperature_scenario = damage_df[_column("temperature_c", scenario)].astype(float).to_numpy()
    temperature_delta_scenario = temperature_scenario - temperature_ref

    gdp_usd = inputs.gdp_trillion_usd.astype(float) * 1e12
    base_damage_fraction = damage_df[_column("damage_fraction", reference)].astype(float).to_numpy()
    base_damage_usd = base_damage_fraction * gdp_usd

    years_float = damage_df["year"].to_numpy()
    if discount_method == "constant_discount":
        if discount_rate is None:
            raise ValueError("discount_rate must be provided for pulse constant discounting")
        factors = _constant_discount_factors(years_float, base_year, discount_rate)
        consumption_pc = None
        growth = None
    elif discount_method == "ramsey_discount":
        if rho is None or eta is None:
            raise ValueError("rho and eta must be provided for pulse Ramsey discounting")
        reference_col = _column("damage_usd", reference)
        consumption_pc, growth = _compute_consumption_growth(
            inputs, damage_df[reference_col].to_numpy()
        )
        factors = _ramsey_discount_factors(years_float, base_year, growth, rho=rho, eta=eta)
    else:
        raise ValueError(f"Unsupported discount method for pulse: {discount_method}")

    scc_series = np.full_like(years_float, fill_value=np.nan, dtype=float)
    pv_attributed = np.zeros_like(years_float, dtype=float)
    for tau in pulse_years:
        idx_candidates = np.where(years == tau)[0]
        if idx_candidates.size == 0:
            continue
        idx = int(idx_candidates[0])
        delta_temp = results[f"pulse_{tau}"].delta
        if delta_temp.shape[0] > temperature_ref.shape[0]:
            delta_temp = delta_temp[: temperature_ref.shape[0]]
        elif delta_temp.shape[0] < temperature_ref.shape[0]:
            raise ValueError(
                "Pulse temperature series shorter than baseline; check FaIR configuration."
            )
        damages_with_pulse = (
            damage_func(temperature_ref + delta_temp, **safe_damage_kwargs) * gdp_usd
        )
        delta_damage = damages_with_pulse - base_damage_usd
        pv = float(np.dot(delta_damage, factors))
        pv_attributed[idx] = pv
        beta_tau = float(factors[idx])
        if beta_tau > 0 and pulse_size_tco2 != 0.0:
            scc_series[idx] = pv / (beta_tau * pulse_size_tco2)

    per_year = damage_df[["year"]].copy()
    per_year["delta_emissions_tco2"] = damage_df[_column("emissions_tco2", scenario)].astype(
        float
    ) - damage_df[_column("emissions_tco2", reference)].astype(float)
    per_year["discount_factor"] = factors
    delta_damage_usd = damage_df[_column("damage_usd", scenario)].astype(float) - damage_df[
        _column("damage_usd", reference)
    ].astype(float)
    per_year["delta_damage_usd"] = delta_damage_usd
    per_year["discounted_delta_usd"] = delta_damage_usd * factors
    per_year["discounted_damage_attributed_usd"] = pv_attributed
    per_year["scc_usd_per_tco2"] = scc_series
    per_year["pulse_size_tco2"] = pulse_size_tco2

    total_delta_emissions = float(per_year["delta_emissions_tco2"].sum())
    effective_add_tco2 = add_tco2 if add_tco2 is not None else total_delta_emissions
    scc_value = _aggregate_scc(per_year, effective_add_tco2)

    details = damage_df.assign(
        discount_factor=factors,
        discounted_delta_usd=per_year["discounted_delta_usd"],
        temperature_delta_c=temperature_delta_scenario,
    )
    if discount_method == "ramsey_discount" and consumption_pc is not None and growth is not None:
        details = details.assign(
            consumption_per_capita_usd=consumption_pc,
            consumption_growth=growth,
        )

    return SCCResult(
        method=discount_method,
        aggregation=aggregation,
        scenario=scenario,
        reference=reference,
        base_year=base_year,
        add_tco2=effective_add_tco2,
        scc_usd_per_tco2=scc_value,
        per_year=per_year,
        details=details,
        temperature_kernel=None,
        run_method="pulse",
    )


def compute_scc(
    inputs: EconomicInputs,
    method: SCCMethod,
    *,
    scenario: str,
    reference: str,
    base_year: int,
    aggregation: SCCAggregation = "average",
    add_tco2: float | None = None,
    damage_func: DamageFunction = damage_dice,
    damage_kwargs: Mapping[str, float] | None = None,
    discount_rate: float | None = None,
    rho: float | None = None,
    eta: float | None = None,
    discount_method: str | None = None,
    pulse_max_year: int | None = None,
) -> SCCResult:
    """Dispatch SCC computation for the requested method."""

    if method == "constant_discount":
        if discount_rate is None:
            raise ValueError("discount_rate must be provided for constant discount method")
        return compute_scc_constant_discount(
            inputs,
            scenario=scenario,
            reference=reference,
            base_year=base_year,
            discount_rate=discount_rate,
            aggregation=aggregation,
            add_tco2=add_tco2,
            damage_func=damage_func,
            damage_kwargs=damage_kwargs,
        )
    if method == "ramsey_discount":
        if rho is None or eta is None:
            raise ValueError("rho and eta must be provided for Ramsey discount method")
        return compute_scc_ramsey_discount(
            inputs,
            scenario=scenario,
            reference=reference,
            base_year=base_year,
            rho=rho,
            eta=eta,
            aggregation=aggregation,
            add_tco2=add_tco2,
            damage_func=damage_func,
            damage_kwargs=damage_kwargs,
        )
    if method == "pulse":
        pulse_discount_method = discount_method
        if pulse_discount_method is None:
            if discount_rate is not None:
                pulse_discount_method = "constant_discount"
            elif rho is not None and eta is not None:
                pulse_discount_method = "ramsey_discount"
        if pulse_discount_method is None:
            raise ValueError("discount_method must be provided for pulse method")
        return compute_scc_pulse(
            inputs,
            scenario=scenario,
            reference=reference,
            base_year=base_year,
            discount_method=pulse_discount_method,
            discount_rate=discount_rate,
            rho=rho,
            eta=eta,
            aggregation=aggregation,
            add_tco2=add_tco2,
            damage_func=damage_func,
            damage_kwargs=damage_kwargs,
            pulse_max_year=pulse_max_year,
        )

    raise ValueError(f"Unsupported SCC method: {method}")


__all__ = [
    "EconomicInputs",
    "SCCResult",
    "SCCAggregation",
    "compute_damages",
    "compute_damage_difference",
    "compute_scc",
    "compute_scc_constant_discount",
    "compute_scc_ramsey_discount",
    "compute_scc_pulse",
    "damage_dice",
]

"""Social cost of carbon calculations based on prescribed temperature and emission pathways."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from climate_module.scenario_runner import (
    ScenarioSpec,
    TemperatureResult,
    run_scenarios,
    step_change,
)

LOGGER = logging.getLogger(__name__)

DamageFunction = Callable[..., np.ndarray]
SCCMethod = Literal["pulse"]
SCCAggregation = Literal["per_year", "average"]

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


def _wide_row_to_series(row: pd.Series) -> pd.Series:
    values: dict[int, float] = {}
    for column, raw in row.items():
        column_str = str(column).strip()
        try:
            year = int(column_str)
        except ValueError:
            continue
        if isinstance(raw, str) and not raw.strip():
            continue
        if pd.isna(raw):
            continue
        values[year] = float(raw)
    if not values:
        raise ValueError("No year columns found in the IIASA dataset.")
    series = pd.Series(values).sort_index()
    return series


def _load_iiasa_economic_data(
    ssp_family: str, directory: Path, *, gdp_conversion: float = 1.0
) -> tuple[pd.Series, pd.Series | None]:
    gdp_csv = directory / "GDP.csv"
    if not gdp_csv.exists():
        raise FileNotFoundError(f"Missing IIASA GDP CSV at {gdp_csv}")
    gdp_df = pd.read_csv(gdp_csv)
    gdp_df["Scenario_upper"] = gdp_df["Scenario"].str.upper().str.strip()
    gdp_df["Region_upper"] = gdp_df["Region"].str.upper().str.strip()
    gdp_row = gdp_df[
        (gdp_df["Scenario_upper"] == ssp_family.upper()) & (gdp_df["Region_upper"] == "WORLD")
    ]
    if gdp_row.empty:
        raise ValueError(f"Scenario '{ssp_family}' not found in {gdp_csv.name}.")
    gdp_series = _wide_row_to_series(gdp_row.iloc[0]) / 1000.0  # billions → trillions

    population_csv = directory / "Population.csv"
    population_series: pd.Series | None = None
    if population_csv.exists():
        pop_df = pd.read_csv(population_csv)
        pop_df["Scenario_upper"] = pop_df["Scenario"].str.upper().str.strip()
        pop_df["Region_upper"] = pop_df["Region"].str.upper().str.strip()
        pop_row = pop_df[
            (pop_df["Scenario_upper"] == ssp_family.upper()) & (pop_df["Region_upper"] == "WORLD")
        ]
        if not pop_row.empty:
            population_series = _wide_row_to_series(pop_row.iloc[0])

    gdp_series = gdp_series.sort_index() * float(gdp_conversion)
    population_series = None if population_series is None else population_series.sort_index()
    return gdp_series, population_series


def _extend_gdp_frame(
    frame: pd.DataFrame,
    target_year: int,
    *,
    label: str | None = None,
) -> pd.DataFrame:
    if target_year <= int(frame["year"].max()):
        return frame
    frame = frame.sort_values("year").reset_index(drop=True)
    last_row = frame.iloc[-1].copy()
    start_year = int(last_row["year"])
    new_rows: list[dict[str, float]] = []
    for year in range(start_year + 1, target_year + 1):
        row: dict[str, float] = {
            "year": year,
            "gdp_trillion_usd": float(last_row["gdp_trillion_usd"]),
        }
        if "population_million" in frame.columns:
            row["population_million"] = float(last_row.get("population_million", np.nan))
        new_rows.append(row)
    if new_rows:
        LOGGER.info(
            "Extending GDP/Population series for %s from %s to %s by holding last values.",
            label or "SSP data",
            start_year,
            target_year,
        )
        frame = pd.concat([frame, pd.DataFrame(new_rows)], ignore_index=True)
    return frame


def _load_ssp_economic_data(
    ssp_family: str, directory: Path, *, gdp_conversion: float = 1.0
) -> tuple[pd.Series, pd.Series | None]:
    """Load SSP GDP and population series for the requested family.

    Prefers the IIASA SSP Scenario Explorer extracts (`IIASA/GDP.csv`, `IIASA/Population.csv`).
    """

    base_dir = Path(directory)
    candidates = []
    if (base_dir / "IIASA").is_dir():
        candidates.append(base_dir / "IIASA")
    candidates.append(base_dir)

    last_error: Exception | None = None
    for candidate in candidates:
        gdp_csv = candidate / "GDP.csv"
        if gdp_csv.exists():
            try:
                return _load_iiasa_economic_data(
                    ssp_family,
                    candidate,
                    gdp_conversion=float(gdp_conversion),
                )
            except Exception as exc:  # pragma: no cover - propagate to fallback
                last_error = exc
                continue

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(
        f"Could not locate IIASA GDP/Population CSVs under {base_dir} "
        "(expected GDP.csv and Population.csv)."
    )


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
    consumption_trillion_usd: np.ndarray | None = None
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
        if (
            self.consumption_trillion_usd is not None
            and len(self.consumption_trillion_usd) != length
        ):
            raise ValueError("consumption_trillion_usd must match the length of years.")
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
        consumption_column: str = "consumption_trillion_usd",
        gdp_frame: pd.DataFrame | None = None,
        gdp_population_directory: Path | str | None = None,
        gdp_currency_conversion: float = 1.0,
    ) -> "EconomicInputs":
        """Load inputs from CSV files.

        ``temperature_paths`` and ``emission_paths`` map scenario labels to CSVs
        containing ``year`` and the respective data columns. Emission values are
        scaled by ``emission_to_tonnes`` (default converts Mt CO₂ to t CO₂).
        ``gdp_frame`` can be provided directly to supply custom GDP/population
        trajectories without reading from disk. ``gdp_currency_conversion`` scales
        GDP read from disk (IIASA/CSV sources) to the desired price year.
        """

        if not temperature_paths or not emission_paths:
            raise ValueError("Provide temperature and emission series for at least one scenario.")

        temperature_labels = set(temperature_paths)
        emission_labels = set(emission_paths)
        if temperature_labels != emission_labels:
            raise ValueError("Temperature and emission scenario labels must match.")

        temp_frames: dict[str, pd.DataFrame] = {}
        temp_year_bounds: list[tuple[int, int]] = []
        for label, source in temperature_paths.items():
            if isinstance(source, pd.DataFrame):
                frame = source.copy()
            else:
                frame = _load_yearly_csv(
                    source,
                    required_columns={temperature_column},
                    optional_columns={"climate_scenario"},
                )
            frame = frame.rename(columns={temperature_column: "temperature_c"})
            temp_frames[label] = frame
            years = frame["year"].astype(int)
            temp_year_bounds.append((int(years.min()), int(years.max())))

        emission_frames: dict[str, pd.DataFrame] = {}
        emission_year_bounds: list[tuple[int, int]] = []
        for label, source in emission_paths.items():
            if isinstance(source, pd.DataFrame):
                frame = source.copy()
            else:
                frame = _load_yearly_csv(source, required_columns={emission_column})
            frame = frame.rename(columns={emission_column: "emission_raw"})
            emission_frames[label] = frame
            years = frame["year"].astype(int)
            emission_year_bounds.append((int(years.min()), int(years.max())))

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
            try:
                gdp_series, population_series = _load_ssp_economic_data(
                    ssp_family,
                    Path(gdp_population_directory),
                    gdp_conversion=gdp_currency_conversion,
                )
            except TypeError:
                gdp_series, population_series = _load_ssp_economic_data(
                    ssp_family,
                    Path(gdp_population_directory),
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
            other_bounds = temp_year_bounds + emission_year_bounds
            if other_bounds:
                target_end = min(bound[1] for bound in other_bounds)
                gdp_data_frame = _extend_gdp_frame(
                    gdp_data_frame,
                    target_end,
                    label=ssp_family,
                )
        else:
            if gdp_path is None:
                raise ValueError("Provide either gdp_path, gdp_frame, or gdp_population_directory.")
            gdp_data_frame = _load_yearly_csv(
                gdp_path,
                required_columns={gdp_column},
                optional_columns={population_column, consumption_column},
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
        if (
            gdp_frame is None
            and gdp_population_directory is None
            and gdp_currency_conversion not in (1.0, 1, None)
        ):
            gdp_series = gdp_series * float(gdp_currency_conversion)
        population = None
        consumption = None
        if population_column in gdp_data_frame:
            population = _reindex_series(gdp_data_frame, population_column)
        if consumption_column in gdp_data_frame:
            consumption = _reindex_series(gdp_data_frame, consumption_column)

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
            consumption_trillion_usd=consumption,
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
        if self.consumption_trillion_usd is not None:
            data["consumption_trillion_usd"] = self.consumption_trillion_usd
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
    run_method: str | None = None
    pulse_details: pd.DataFrame | None = None


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
    custom_terms: Sequence[Mapping[str, float]] | None = None,
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

    The baseline follows ``delta1 * T + delta2 * T^2`` unless ``custom_terms`` is
    supplied, in which case damage is computed as ``Σ coeff_i × T^{power_i}``.
    Optional extensions are applied in the following order:

    1. Threshold amplification above ``threshold_temperature``.
    2. Catastrophic add-ons once temperatures cross ``catastrophic_temperature``.
    3. A single saturation step (when enabled) that caps the combined damages
       via either a smooth rational curve or a hard clamp.

    The final result is always clipped to ``[0, max_fraction]`` for numerical safety.
    """

    temperatures = np.asarray(temp, dtype=float)

    # 1. Base polynomial damages (DICE-style or custom coefficients)
    if custom_terms:
        damage = np.zeros_like(temperatures, dtype=float)
        for term in custom_terms:
            if not isinstance(term, Mapping):
                continue
            coeff = float(term.get("coefficient", 0.0))
            power = float(term.get("exponent", term.get("power", 1.0)))
            damage = damage + coeff * np.power(temperatures, power)
    else:
        damage = delta1 * temperatures + delta2 * temperatures**2

    # 2. Threshold amplification
    if use_threshold:
        amplify = (
            1.0
            + threshold_scale
            * np.maximum(0.0, temperatures - threshold_temperature) ** threshold_power
        )
        damage = damage * amplify

    # 3. Catastrophic / disaster component
    if use_catastrophic:
        if disaster_mode == "step":
            extra = np.where(temperatures >= catastrophic_temperature, disaster_fraction, 0.0)
        elif disaster_mode == "prob":
            exceed = np.maximum(0.0, temperatures - catastrophic_temperature)
            extra = (1.0 - np.exp(-disaster_gamma * exceed)) * disaster_fraction
        else:
            raise ValueError("disaster_mode must be 'prob' or 'step'")
        damage = damage + extra

    # Ensure non-negative before saturation
    damage = np.maximum(damage, 0.0)

    # 4. Saturation (applies to base + threshold + catastrophe)
    if use_saturation:
        if saturation_mode == "rational":
            # Smooth cap: ~linear for small damages, approaches max_fraction for large damages.
            x = damage
            with np.errstate(divide="ignore", invalid="ignore"):
                damage = np.divide(
                    max_fraction * x,
                    x + max_fraction,
                    out=np.zeros_like(x),
                    where=x >= 0.0,
                )
        elif saturation_mode == "clamp":
            damage = np.clip(damage, 0.0, max_fraction)
        else:
            raise ValueError("saturation_mode must be 'rational' or 'clamp'")

    # Final numeric safety: keep within [0, max_fraction].
    damage = np.clip(damage, 0.0, max_fraction)
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
    gross_consumption = (
        np.asarray(inputs.consumption_trillion_usd, dtype=float) * 1e12
        if inputs.consumption_trillion_usd is not None
        else gdp_usd
    )
    consumption = np.maximum(gross_consumption - reference_damage_usd, 0.0)
    consumption_per_capita = np.divide(
        consumption, population, out=np.zeros_like(consumption), where=population > 0
    )
    np.divide(
        gross_consumption, population, out=np.zeros_like(gross_consumption), where=population > 0
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


def _aggregate_scc(per_year: pd.DataFrame, add_tco2: float) -> float:
    discounted_damage = per_year["discounted_delta_usd"].sum()
    if np.isclose(add_tco2, 0.0):
        return float("nan")
    return discounted_damage / add_tco2


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
    fair_calibration: object | None = None,
    climate_start_year: int | None = None,
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
    if climate_start_year is not None:
        start_year = min(start_year, int(climate_start_year))
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

    calibration_token = id(fair_calibration) if fair_calibration is not None else None
    cache_key = (
        climate_id,
        tuple(int(v) for v in pulse_years.tolist()),
        float(start_year),
        float(end_year),
        pulse_size_mt,
        calibration_token,
    )
    cached_results = _PULSE_RESULT_CACHE.get(cache_key)
    if cached_results is None:
        cached_results = {}
        _PULSE_RESULT_CACHE[cache_key] = cached_results

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
        consumption_pc_gross = None
    elif discount_method == "ramsey_discount":
        if rho is None or eta is None:
            raise ValueError("rho and eta must be provided for pulse Ramsey discounting")
        reference_col = _column("damage_usd", reference)
        consumption_pc, growth = _compute_consumption_growth(
            inputs, damage_df[reference_col].to_numpy()
        )
        if inputs.population_million is not None:
            population = np.asarray(inputs.population_million, dtype=float) * 1e6
            gross_consumption = (
                np.asarray(inputs.consumption_trillion_usd, dtype=float) * 1e12
                if inputs.consumption_trillion_usd is not None
                else gdp_usd
            )
            consumption_pc_gross = np.divide(
                gross_consumption,
                population,
                out=np.zeros_like(gross_consumption),
                where=population > 0,
            )
        else:
            consumption_pc_gross = None
        factors = _ramsey_discount_factors(years_float, base_year, growth, rho=rho, eta=eta)
    else:
        raise ValueError(f"Unsupported discount method for pulse: {discount_method}")

    scc_series = np.full_like(years_float, fill_value=np.nan, dtype=float)
    pv_attributed = np.zeros_like(years_float, dtype=float)
    pulse_detail_records: list[dict[str, float]] = []
    total_pulses = len(pulse_years)
    years_int = years_float.astype(int)
    climate_logger = logging.getLogger("climate_module")

    for idx_counter, tau in enumerate(pulse_years, start=1):
        idx_candidates = np.where(years == tau)[0]
        if idx_candidates.size == 0:
            continue
        idx = int(idx_candidates[0])

        label = f"pulse_{tau}"
        pulse_result = cached_results.get(label)
        if pulse_result is None:
            up = step_change(pulse_size_mt, start_year=float(tau))
            down = step_change(pulse_size_mt, start_year=float(tau + 1))

            def pulse_builder_factory(up_fun, down_fun):
                return lambda timepoints, cfg: up_fun(timepoints, cfg) - down_fun(timepoints, cfg)

            pulse_builder = pulse_builder_factory(up, down)
            spec = ScenarioSpec(
                label=label,
                scenario=climate_id,
                emission_adjustments={specie: pulse_builder},
                start_year=float(start_year),
                end_year=float(end_year),
                timestep=1.0,
                compute_kwargs={"fair_calibration": fair_calibration} if fair_calibration else None,
            )
            previous_level = climate_logger.level
            climate_logger.setLevel(max(previous_level, logging.WARNING))
            try:
                single_result = run_scenarios([spec])
            finally:
                climate_logger.setLevel(previous_level)
            pulse_result = single_result[label]
            cached_results[label] = pulse_result

        pulse_years_full = pulse_result.years.astype(int)
        delta_full = pulse_result.delta
        delta_series = pd.Series(delta_full, index=pulse_years_full)
        # Align pulse temperature response to SCC evaluation years
        delta_temp = delta_series.reindex(years_int, method=None).fillna(0.0).to_numpy(dtype=float)

        damages_with_pulse_fraction = damage_func(
            temperature_ref + delta_temp, **safe_damage_kwargs
        )
        damages_with_pulse = damages_with_pulse_fraction * gdp_usd
        delta_damage = damages_with_pulse - base_damage_usd
        pv = float(np.dot(delta_damage, factors))
        pv_attributed[idx] = pv
        beta_tau = float(factors[idx])
        if beta_tau > 0 and pulse_size_tco2 != 0.0:
            scc_series[idx] = pv / (beta_tau * pulse_size_tco2)
        delta_fraction = damages_with_pulse_fraction - base_damage_fraction
        for year_idx, year_value in enumerate(years_float):
            pulse_detail_records.append(
                {
                    "pulse_year": int(tau),
                    "year": int(year_value),
                    "delta_temperature_c": float(delta_temp[year_idx]),
                    "pulse_mass_tco2": float(pulse_size_tco2),
                    "baseline_temperature_c": float(temperature_ref[year_idx]),
                    "gdp_trillion_usd": float(inputs.gdp_trillion_usd[year_idx]),
                    "delta_damage_fraction": float(delta_fraction[year_idx]),
                    "delta_damage_usd": float(delta_damage[year_idx]),
                    "discount_factor": float(factors[year_idx]),
                    "pv_damage_usd": float(delta_damage[year_idx] * factors[year_idx]),
                }
            )
        if total_pulses >= 10 and (idx_counter % 10 == 0 or idx_counter == total_pulses):
            pct = 100.0 * idx_counter / float(total_pulses)
            LOGGER.info("Computed pulse %s/%s (%.1f%%)", idx_counter, total_pulses, pct)

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
            consumption_per_capita_gross_usd=consumption_pc_gross,
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
        run_method="pulse",
        pulse_details=pd.DataFrame(pulse_detail_records) if pulse_detail_records else None,
    )


def compute_scc(
    inputs: EconomicInputs,
    method: SCCMethod = "pulse",
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
    fair_calibration: object | None = None,
    climate_start_year: int | None = None,
) -> SCCResult:
    """Dispatch SCC computation (pulse method only)."""

    if method != "pulse":
        raise ValueError(f"Unsupported SCC method '{method}'. Only 'pulse' is available.")

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
        fair_calibration=fair_calibration,
        climate_start_year=climate_start_year,
    )


__all__ = [
    "EconomicInputs",
    "SCCResult",
    "SCCAggregation",
    "compute_damages",
    "compute_damage_difference",
    "compute_scc",
    "compute_scc_pulse",
    "damage_dice",
]

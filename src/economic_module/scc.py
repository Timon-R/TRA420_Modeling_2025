"""Social cost of carbon calculations based on prescribed temperature and emission pathways."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Literal
import re

import numpy as np
import pandas as pd

DamageFunction = Callable[..., np.ndarray]
SCCMethod = Literal["constant_discount", "ramsey_discount"]
SCCAggregation = Literal["per_year", "average"]


def _safe_key(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    return safe.strip("_") or "scenario"


def _column(prefix: str, scenario: str) -> str:
    return f"{prefix}_{_safe_key(scenario)}"


def _extract_climate_scenarios(
    temperature_frames: Mapping[str, pd.DataFrame]
) -> dict[str, str]:
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
            "Reading SSP workbooks requires 'openpyxl'. Install it or provide a CSV override via gdp_series."
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
    gdp_path = directory / "GDP_SSP1_5.xlsx"
    pop_path = directory / "POP_SSP1_5.xlsx"
    if not gdp_path.exists():
        raise FileNotFoundError(f"Missing GDP dataset: {gdp_path}")

    gdp_series = _load_ssp_table(gdp_path, "GDP", ssp_family) / 1000.0  # convert billions to trillions

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
                raise ValueError(f"Temperature series '{name}' does not match the year vector length.")
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
        gdp_population_directory: Path | str | None = None,
    ) -> "EconomicInputs":
        """Load inputs from CSV files.

        ``temperature_paths`` and ``emission_paths`` map scenario labels to CSVs
        containing ``year`` and the respective data columns. Emission values are
        scaled by ``emission_to_tonnes`` (default converts Mt CO₂ to t CO₂).
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
        ssp_family: str | None = None

        if gdp_population_directory is not None:
            if not climate_scenarios:
                raise ValueError(
                    "Temperature CSVs must include 'climate_scenario' to determine SSP-specific GDP."
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
            gdp_frame = pd.DataFrame(
                {
                    "year": gdp_series.index.astype(int),
                    gdp_column: gdp_series.values,
                }
            )
            if population_series is not None:
                gdp_frame[population_column] = population_series.loc[gdp_frame["year"]].to_numpy()
        else:
            if gdp_path is None:
                raise ValueError("Provide either gdp_path or gdp_population_directory.")
            gdp_frame = _load_yearly_csv(
                gdp_path,
                required_columns={gdp_column},
                optional_columns={population_column},
            )

        common_years = set(gdp_frame["year"].astype(int))
        for frame in temp_frames.values():
            common_years &= set(frame["year"].astype(int))
        for frame in emission_frames.values():
            common_years &= set(frame["year"].astype(int))
        if not common_years:
            raise ValueError("Temperature, emission, and GDP datasets do not share common years.")

        years = np.array(sorted(common_years), dtype=int)

        gdp_series = gdp_frame.set_index("year").loc[years, gdp_column].to_numpy(dtype=float)
        population = None
        if population_column in gdp_frame:
            population = gdp_frame.set_index("year").loc[years, population_column].to_numpy(dtype=float)

        temperature_series = {
            label: frame.set_index("year").loc[years, "temperature_c"].to_numpy(dtype=float)
            for label, frame in temp_frames.items()
        }
        emission_series = {
            label: frame.set_index("year").loc[years, "emission_raw"].to_numpy(dtype=float) * emission_to_tonnes
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

    keep_columns = ["year", *sorted(required_columns.union(optional_columns))]
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
        amplify = 1.0 + threshold_scale * np.maximum(
            0.0, temperatures - threshold_temperature
        ) ** threshold_power
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
        fractions = damage_func(temps, **damage_kwargs)
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
    consumption = np.maximum(gdp_usd - reference_damage_usd, 0.0)
    consumption_per_capita = np.divide(consumption, population, out=np.zeros_like(consumption), where=population > 0)

    growth = np.full_like(consumption_per_capita, fill_value=np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth[1:] = (consumption_per_capita[1:] - consumption_per_capita[:-1]) / consumption_per_capita[:-1]
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
    discount_factors: np.ndarray,
) -> pd.DataFrame:
    per_year = damage_df[["year", "delta_damage_usd", "delta_emissions_tco2"]].copy()
    per_year["discount_factor"] = discount_factors
    per_year["discounted_delta_usd"] = per_year["delta_damage_usd"] * discount_factors
    with np.errstate(divide="ignore", invalid="ignore"):
        per_year["scc_usd_per_tco2"] = np.divide(
            per_year["discounted_delta_usd"],
            per_year["delta_emissions_tco2"],
            out=np.full_like(per_year["discounted_delta_usd"], np.nan, dtype=float),
            where=np.abs(per_year["delta_emissions_tco2"]) > 0,
        )
    return per_year


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
    per_year = _build_per_year_table(damage_df, discount_factors=factors)

    total_delta_emissions = per_year["delta_emissions_tco2"].sum()
    effective_add_tco2 = add_tco2 if add_tco2 is not None else total_delta_emissions

    scc_value = _aggregate_scc(per_year, effective_add_tco2)

    damage_df = damage_df.assign(
        discount_factor=factors,
        discounted_delta_usd=per_year["discounted_delta_usd"],
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
    consumption_pc, growth = _compute_consumption_growth(inputs, damage_df[reference_col].to_numpy())

    factors = _ramsey_discount_factors(
        damage_df["year"].to_numpy(),
        base_year,
        growth,
        rho=rho,
        eta=eta,
    )

    per_year = _build_per_year_table(damage_df, discount_factors=factors)

    total_delta_emissions = per_year["delta_emissions_tco2"].sum()
    effective_add_tco2 = add_tco2 if add_tco2 is not None else total_delta_emissions
    scc_value = _aggregate_scc(per_year, effective_add_tco2)

    damage_df = damage_df.assign(
        consumption_per_capita_usd=consumption_pc,
        consumption_growth=growth,
        discount_factor=factors,
        discounted_delta_usd=per_year["discounted_delta_usd"],
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
    "damage_dice",
]

"""FaIR wrappers for quick global-mean temperature projections.

The helpers in this module prepare :class:`fair.FAIR` with sensible defaults and
return easy-to-use data structures that capture how emissions perturbations shift
global mean surface temperature.

Supported SSP pathways
======================
FaIR bundles the RCMIP data set, so the following scenario identifiers are ready to
use:

``ssp245`` (default)
    Middle-of-the-road mitigation and development, ideal as a reference.
``ssp119``
    Strong mitigation pathway consistent with limiting warming well below 2 °C.
``ssp370``
    High-emission pathway with limited mitigation, useful for pessimistic bounds.

Any other scenario present in the RCMIP catalogue (for example ``ssp534-over`` or
``ssp245-baseline``) can be supplied directly; the table above merely highlights a
few common choices.

Emission adjustments in a nutshell
==================================
Emissions in FaIR live on mid-year *timepoints* (e.g. 2025.5 for the 2025–2026
interval). ``compute_temperature_change`` returns those timepoints so you can align
custom perturbations precisely. Scalars apply the same change at every timepoint;
arrays or callables let you vary the intervention over time.

Key species handles
===================
The helper loads a focused subset of FaIR's AR6 catalogue so you can tweak the most
impactful forcings straight away:

``CO2 FFI``
    Carbon dioxide from fossil-fuel and industrial processes—the usual lever for
    mitigation studies.
``CO2 AFOLU``
    Carbon dioxide from agriculture, forestry and other land-use change.
``CH4`` / ``N2O``
    Methane and nitrous oxide emissions.
Other forcing categories such as aerosols (``Aerosol-radiation interactions``),
``Solar`` and ``Volcanic`` are also available for completeness.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from fair import FAIR

from .calibration import FairCalibration, apply_fair_calibration

# Path to the default AR6 species configuration bundled with FaIR.
_SPECIES_CONFIG_PATH = Path(FAIR.fill_species_configs.__defaults__[0]).resolve()
_SPECIES_CONFIG = pd.read_csv(_SPECIES_CONFIG_PATH, index_col=0)

LOGGER = logging.getLogger("climate_module")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

# Species required to reproduce the main AR6 forcing components while keeping
# the configuration compact. Emission adjustments are only supported for the
# species that are driven by emissions in this list.
DEFAULT_SPECIES: Sequence[str] = (
    "CO2 FFI",
    "CO2 AFOLU",
    "CO2",
    "CH4",
    "N2O",
    "Solar",
    "Volcanic",
    "Aerosol-radiation interactions",
    "Aerosol-cloud interactions",
    "Ozone",
    "Stratospheric water vapour",
    "Contrails",
    "Land use",
    "Light absorbing particles on snow and ice",
)

# Scenario aliases exposed for convenience. All names must exist in the RCMIP
# database that ships with FaIR.
# Simple climate parameter presets. Values are broadcast to the number of
# configurations used in the FaIR run (baseline and adjusted by default).
CLIMATE_PRESETS: Mapping[str, Mapping[str, Sequence[float]]] = {
    "ar6": {
        "ocean_heat_capacity": (7.0, 100.0, 1000.0),
        "ocean_heat_transfer": (0.7, 0.3, 0.05),
        "deep_ocean_efficacy": 1.28,
        "forcing_4co2": 7.4,
    },
    "two_box": {
        # Compact two-layer alternative if users prefer to experiment.
        "ocean_heat_capacity": (8.2, 109.0),
        "ocean_heat_transfer": (1.7, 0.18),
        "deep_ocean_efficacy": 1.0,
        "forcing_4co2": 7.2,
    },
}

ArrayLike = Union[float, Sequence[float], np.ndarray]
ClimateSetup = Union[str, Mapping[str, Sequence[float]]]
EmissionAdjustments = Mapping[str, ArrayLike] | None

__all__ = [
    "TemperatureResult",
    "compute_temperature_change",
    "DEFAULT_SPECIES",
    "CLIMATE_PRESETS",
]


@dataclass
class TemperatureResult:
    """Container for FaIR global-mean temperature trajectories."""

    years: np.ndarray
    timepoints: np.ndarray
    baseline: np.ndarray
    adjusted: np.ndarray

    @property
    def delta(self) -> np.ndarray:
        """Adjusted minus baseline temperature change."""
        return self.adjusted - self.baseline

    @property
    def final_delta(self) -> float:
        """Temperature difference in the last simulated year."""
        return float(self.delta[-1])

    def to_frame(self) -> pd.DataFrame:
        """Return the trajectories as a tidy :class:`pandas.DataFrame`."""
        return pd.DataFrame(
            {
                "year": self.years,
                "temperature_baseline": self.baseline,
                "temperature_adjusted": self.adjusted,
                "temperature_delta": self.delta,
            }
        )


def compute_temperature_change(
    emission_adjustments: EmissionAdjustments = None,
    *,
    scenario: str = "ssp245",
    start_year: int = 1750,
    end_year: int = 2100,
    timestep: float = 1.0,
    climate_setup: ClimateSetup = "ar6",
    climate_overrides: Mapping[str, ArrayLike] | None = None,
    fair_calibration: FairCalibration | None = None,
) -> TemperatureResult:
    """Run FaIR for a baseline scenario and an adjusted emissions case.

    Parameters
    ----------
    emission_adjustments:
        Mapping from species name (e.g. ``"CO2 FFI"`` or ``"CH4"``) to either a
        scalar perturbation applied at every timestep, or an array whose length
        matches the number of FaIR timepoints. Adjustments are applied to the
        ``"adjusted"`` configuration only; the baseline remains untouched.
    scenario:
        Scenario key. Must exist in :data:`SUPPORTED_SCENARIOS`.
    start_year, end_year, timestep:
        Temporal resolution arguments passed to :meth:`fair.FAIR.define_time`.
    climate_setup:
        Either one of the keys in :data:`CLIMATE_PRESETS` or an explicit mapping
        with the fields ``ocean_heat_capacity``, ``ocean_heat_transfer``,
        ``deep_ocean_efficacy`` and ``forcing_4co2``. Custom values can be
        provided as scalars (broadcast to all configs) or per-config sequences.
    climate_overrides:
        Optional mapping used to tweak climate parameters after the preset is
        loaded. Keys can include ``ocean_heat_capacity``, ``ocean_heat_transfer``,
        ``deep_ocean_efficacy``, ``forcing_4co2`` and
        ``equilibrium_climate_sensitivity``. Values may be scalars (applied to all
        configs) or per-config sequences.

    Returns
    -------
    TemperatureResult
        FaIR time bounds, mid-point timepoints, baseline temperatures,
        adjusted temperatures, and their difference (adjusted minus baseline).

    Notes
    -----
    To model interventions that switch on at a specific calendar year, align the
    emission adjustments with the returned ``timepoints``. For example, with an
    annual timestep the entry for 2025 corresponds to ``2025.5``. Passing a
    callable (see :func:`climate_module.step_change`) lets you generate the array
    programmatically.
    """
    scenario_name = scenario
    LOGGER.info("Running FaIR for scenario '%s'", scenario_name)
    model = _build_model(
        scenario_name,
        start_year,
        end_year,
        timestep,
        climate_setup,
    )
    if fair_calibration is not None:
        apply_fair_calibration(model, fair_calibration, scenario_name)
    _apply_emission_adjustments(model, scenario_name, emission_adjustments)
    _apply_climate_overrides(model, climate_overrides)
    model.run(progress=False)

    surface_temp = model.temperature.sel(layer=0, scenario=scenario_name)
    baseline = surface_temp.sel(config="baseline").values
    adjusted = surface_temp.sel(config="adjusted").values
    years = model.timebounds.copy()
    timepoints = model.timepoints.copy()

    return TemperatureResult(
        years=years,
        timepoints=timepoints,
        baseline=baseline,
        adjusted=adjusted,
    )


def _build_model(
    scenario: str,
    start_year: int,
    end_year: int,
    timestep: float,
    climate_setup: ClimateSetup,
) -> FAIR:
    """Instantiate and prepare a FaIR model with default species and configs."""
    model = FAIR()
    model.define_time(start_year, end_year, timestep)
    model.define_scenarios([scenario])
    model.define_configs(["baseline", "adjusted"])

    properties = _species_properties(DEFAULT_SPECIES)
    model.define_species(list(DEFAULT_SPECIES), properties)
    model.allocate()
    model.fill_species_configs()

    _apply_climate_setup(model, climate_setup)
    model.fill_from_rcmip()
    _initialise_model_state(model)
    return model


def _species_properties(species: Sequence[str]) -> Mapping[str, Mapping[str, object]]:
    """Extract the FaIR species metadata for the requested subset."""
    try:
        subset = _SPECIES_CONFIG.loc[list(species)]
    except KeyError as exc:
        missing = sorted(set(species) - set(_SPECIES_CONFIG.index))
        raise ValueError(f"Species not recognised by FaIR defaults: {missing}") from exc

    properties: dict[str, dict[str, object]] = {}
    for specie, row in subset.iterrows():
        properties[specie] = {
            "type": row["type"],
            "input_mode": row["input_mode"],
            "greenhouse_gas": bool(row["greenhouse_gas"]),
            "aerosol_chemistry_from_emissions": bool(row["aerosol_chemistry_from_emissions"]),
            "aerosol_chemistry_from_concentration": bool(
                row["aerosol_chemistry_from_concentration"]
            ),
        }
    return properties


def _apply_climate_setup(model: FAIR, setup: ClimateSetup) -> None:
    """Populate :attr:`FAIR.climate_configs` with the requested parameters."""
    if isinstance(setup, str):
        key = setup.lower()
        if key not in CLIMATE_PRESETS:
            raise ValueError(
                f"Unknown climate preset '{setup}'. Available presets: {sorted(CLIMATE_PRESETS)}"
            )
        params = CLIMATE_PRESETS[key]
    else:
        params = setup

    required = {"ocean_heat_capacity", "ocean_heat_transfer", "deep_ocean_efficacy", "forcing_4co2"}
    missing = required - params.keys()
    if missing:
        raise ValueError("Climate setup is missing required keys: " + ", ".join(sorted(missing)))

    n_configs = len(model.configs)

    capacity = _broadcast_matrix(params["ocean_heat_capacity"], n_configs)
    transfer = _broadcast_matrix(params["ocean_heat_transfer"], n_configs)
    efficacy = _broadcast_vector(params["deep_ocean_efficacy"], n_configs)
    forcing = _broadcast_vector(params["forcing_4co2"], n_configs)

    for idx, cfg in enumerate(model.configs):
        model.climate_configs["ocean_heat_capacity"].loc[cfg, :] = capacity[idx]
        model.climate_configs["ocean_heat_transfer"].loc[cfg, :] = transfer[idx]
        model.climate_configs["deep_ocean_efficacy"].loc[cfg] = efficacy[idx]
        model.climate_configs["forcing_4co2"].loc[cfg] = forcing[idx]


def _apply_climate_overrides(model: FAIR, overrides: Mapping[str, ArrayLike] | None) -> None:
    if not overrides:
        return

    n_configs = len(model.configs)

    if "ocean_heat_capacity" in overrides:
        matrix = _broadcast_matrix(overrides["ocean_heat_capacity"], n_configs)
        model.climate_configs["ocean_heat_capacity"].loc[:, :] = matrix

    if "ocean_heat_transfer" in overrides:
        matrix = _broadcast_matrix(overrides["ocean_heat_transfer"], n_configs)
        model.climate_configs["ocean_heat_transfer"].loc[:, :] = matrix

    if "deep_ocean_efficacy" in overrides:
        vector = _broadcast_vector(overrides["deep_ocean_efficacy"], n_configs)
        model.climate_configs["deep_ocean_efficacy"].loc[:] = vector

    if "forcing_4co2" in overrides:
        vector = _broadcast_vector(overrides["forcing_4co2"], n_configs)
        model.climate_configs["forcing_4co2"].loc[:] = vector

    if "equilibrium_climate_sensitivity" in overrides:
        try:
            model.equilibrium_climate_sensitivity = float(
                overrides["equilibrium_climate_sensitivity"]
            )
        except (TypeError, ValueError):
            LOGGER.warning(
                "Could not parse equilibrium climate sensitivity override: %s",
                overrides["equilibrium_climate_sensitivity"],
            )


def _broadcast_matrix(values: Sequence[float] | np.ndarray, n_configs: int) -> np.ndarray:
    """Ensure climate arrays match FaIR's ``(config, layer)`` layout."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.array([arr.item()], dtype=float)
    if arr.ndim == 1:
        return np.tile(arr, (n_configs, 1))
    if arr.shape[0] == n_configs:
        return arr
    if arr.shape[0] == 1:
        return np.tile(arr, (n_configs, 1))
    raise ValueError(
        f"Expected first dimension of climate array to be 1 or {n_configs}; "
        f"received {arr.shape[0]}."
    )


def _broadcast_vector(values: Sequence[float] | float, n_configs: int) -> np.ndarray:
    """Broadcast per-config scalars to a 1D array sized to ``n_configs``."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(n_configs, float(arr.item()))
    if arr.size == 1:
        return np.full(n_configs, float(arr.reshape(-1)[0]))
    if arr.shape[0] != n_configs:
        raise ValueError(
            f"Expected {n_configs} entries, received {arr.shape[0]} in climate vector."
        )
    return arr.astype(float)


def _initialise_model_state(model: FAIR) -> None:
    """Set FaIR state variables to sensible preindustrial baselines."""
    first_timebound = float(model.timebounds[0])
    model.cumulative_emissions.loc[{"timebounds": first_timebound}] = 0.0
    model.airborne_emissions.loc[{"timebounds": first_timebound}] = 0.0
    model.temperature.loc[{"timebounds": first_timebound}] = 0.0

    greenhouse_species = model.properties_df.index[model.properties_df["greenhouse_gas"]]
    baselines = model.species_configs["baseline_concentration"]
    for specie in greenhouse_species:
        baseline = baselines.sel(specie=specie).values
        model.concentration.loc[{"timebounds": first_timebound, "specie": specie}] = baseline


def _apply_emission_adjustments(
    model: FAIR,
    scenario: str,
    emission_adjustments: Mapping[str, ArrayLike] | None,
) -> None:
    """Apply user-supplied emission perturbations to the "adjusted" config."""
    if not emission_adjustments:
        return

    timepoints = model.timepoints
    for specie, delta in emission_adjustments.items():
        if specie not in model.species:
            raise ValueError(f"Species '{specie}' is not included in the climate model setup.")
        if model.properties[specie]["input_mode"] != "emissions":
            raise ValueError(
                f"Species '{specie}' is not emissions-driven in this setup; "
                "adjustments can only be applied to emissions-driven species."
            )
        delta_array = _to_delta_array(delta, timepoints)
        logger = logging.getLogger("climate.run")
        logger.debug(
            "Applying adjustment for specie %s: delta length %d, timepoints length %d",
            specie,
            len(delta_array),
            len(timepoints),
        )
        selection = {
            "specie": specie,
            "scenario": scenario,
            "config": "adjusted",
        }
        baseline = model.emissions.loc[selection].values
        if baseline.shape[0] != delta_array.shape[0]:
            raise ValueError(
                f"Delta for '{specie}' must have {baseline.shape[0]} entries, "
                f"received {delta_array.shape[0]}."
            )
        model.emissions.loc[selection] = baseline + delta_array


def _to_delta_array(delta: ArrayLike, timepoints: np.ndarray) -> np.ndarray:
    """Normalise user-supplied emission adjustments to a NumPy array."""
    if np.isscalar(delta):
        return np.full_like(timepoints, float(delta), dtype=float)

    array = np.asarray(delta, dtype=float)
    if array.shape[0] != timepoints.shape[0]:
        logger = logging.getLogger("climate.run")
        logger.debug(
            "Mismatch in adjustment length: delta=%d, timepoints=%d",
            array.shape[0],
            timepoints.shape[0],
        )
        raise ValueError(
            f"Emission adjustments must have length {timepoints.shape[0]}; "
            f"received {array.shape[0]}."
        )
    return array

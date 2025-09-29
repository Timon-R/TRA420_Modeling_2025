"""Utilities for running sets of FaIR scenarios with emission adjustments."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np

from .FaIR import TemperatureResult, compute_temperature_change

DEFAULT_TIME_CONFIG: Mapping[str, float] = {
    "start_year": 1750,
    "end_year": 2100,
    "timestep": 1.0,
}


AdjustmentCallable = Callable[[np.ndarray, Mapping[str, float]], np.ndarray]
AdjustmentInput = Union[float, Sequence[float], np.ndarray, AdjustmentCallable]
EmissionAdjustments = Mapping[str, AdjustmentInput] | None


@dataclass
class ScenarioSpec:
    """Inputs needed to evaluate a FaIR scenario."""

    label: str
    scenario: str
    emission_adjustments: EmissionAdjustments = None
    start_year: float | None = None
    end_year: float | None = None
    timestep: float | None = None
    compute_kwargs: Mapping[str, object] | None = None


def step_change(value: float, start_year: float) -> AdjustmentCallable:
    """Return an adjustment that steps to ``value`` from ``start_year`` onward."""

    def _builder(timepoints: np.ndarray, cfg: Mapping[str, float]) -> np.ndarray:
        threshold = start_year + cfg["timestep"] / 2
        return np.where(timepoints >= threshold, value, 0.0)

    return _builder


def run_scenarios(specs: Iterable[ScenarioSpec]) -> dict[str, TemperatureResult]:
    """Execute each scenario and return FaIR results keyed by label."""
    results: dict[str, TemperatureResult] = {}
    logger = logging.getLogger("climate_module")
    for spec in specs:
        time_cfg = _time_config(spec)
        timepoints = _build_timepoints(time_cfg)
        adjustments = _prepare_adjustments(spec.emission_adjustments, timepoints, time_cfg)
        logger.info("Performing calculation for scenario '%s'", spec.label)
        result = compute_temperature_change(
            scenario=spec.scenario,
            emission_adjustments=adjustments,
            start_year=time_cfg["start_year"],
            end_year=time_cfg["end_year"],
            timestep=time_cfg["timestep"],
            **(spec.compute_kwargs or {}),
        )
        results[spec.label] = result
    return results


def _time_config(spec: ScenarioSpec) -> Mapping[str, float]:
    cfg = dict(DEFAULT_TIME_CONFIG)
    if spec.start_year is not None:
        cfg["start_year"] = spec.start_year
    if spec.end_year is not None:
        cfg["end_year"] = spec.end_year
    if spec.timestep is not None:
        cfg["timestep"] = spec.timestep
    return cfg


def _build_timepoints(cfg: Mapping[str, float]) -> np.ndarray:
    return np.arange(
        cfg["start_year"] + cfg["timestep"] / 2,
        cfg["end_year"] + cfg["timestep"] / 2,
        cfg["timestep"],
    )


def _prepare_adjustments(
    adjustments: EmissionAdjustments,
    timepoints: np.ndarray,
    cfg: Mapping[str, float],
) -> EmissionAdjustments:
    if not adjustments:
        return None
    prepared: dict[str, AdjustmentInput] = {}
    for specie, value in adjustments.items():
        if callable(value):
            prepared[specie] = value(timepoints, cfg)
        else:
            prepared[specie] = value
    return prepared


__all__ = [
    "ScenarioSpec",
    "DEFAULT_TIME_CONFIG",
    "step_change",
    "run_scenarios",
]

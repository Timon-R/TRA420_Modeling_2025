"""Convenient FaIR wrappers for temperature projections and scenario sweeps."""

from .FaIR import TemperatureResult, compute_temperature_change
from .scenario_runner import DEFAULT_TIME_CONFIG, ScenarioSpec, run_scenarios, step_change

__all__ = [
    "TemperatureResult",
    "compute_temperature_change",
    "DEFAULT_TIME_CONFIG",
    "ScenarioSpec",
    "run_scenarios",
    "step_change",
]

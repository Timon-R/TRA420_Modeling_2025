"""Utilities for translating electricity scenarios into emission differences."""

from .calculator import EmissionScenarioResult, calculate_emissions, run_from_config

__all__ = [
    "EmissionScenarioResult",
    "calculate_emissions",
    "run_from_config",
]

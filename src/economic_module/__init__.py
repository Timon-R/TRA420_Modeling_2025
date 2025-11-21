"""Economic module exposing social cost of carbon utilities."""

from .scc import (
    EconomicInputs,
    SCCAggregation,
    SCCResult,
    compute_damage_difference,
    compute_damages,
    compute_scc,
    compute_scc_pulse,
    damage_dice,
)

__all__ = [
    "EconomicInputs",
    "SCCAggregation",
    "SCCResult",
    "compute_damages",
    "compute_damage_difference",
    "compute_scc",
    "compute_scc_pulse",
    "damage_dice",
]

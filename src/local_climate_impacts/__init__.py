"""Local climate impacts (temperature/precipitation scaling) utilities."""

from .scaling import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_ROOT,
    get_scaling_factors,
    load_config,
    scale_results,
)

__all__ = [
    "load_config",
    "get_scaling_factors",
    "scale_results",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_ROOT",
]

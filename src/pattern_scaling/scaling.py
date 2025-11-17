"""Calculates local temperature changes using pattern scaling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from config_paths import apply_results_run_directory, get_results_run_directory

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = DEFAULT_ROOT / "config.yaml"


def _config_value(config: Dict[str, Any], *keys: str, required: bool = True) -> Any:
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            if required:
                path = ".".join(keys)
                raise KeyError(f"Missing configuration key '{path}'.")
            return None
        current = current[key]
    return current


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load ``config.yaml`` (defaults to the repository root copy)."""

    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = DEFAULT_ROOT / path
    return path


def get_scaling_factors(config: Dict[str, Any]) -> pd.DataFrame:
    """Load and filter pattern-scaling factors based on configuration."""

    ps_cfg = _config_value(config, "pattern_scaling")
    cm_cfg = _config_value(config, "climate_module")

    scaling_file = _resolve_path(ps_cfg.get("scaling_factors_file"))
    if not scaling_file.exists():
        raise FileNotFoundError(f"Scaling factors file not found: {scaling_file}")

    weighting_key = ps_cfg.get("scaling_weighting", "area")
    weighting_column = f"patterns.{weighting_key}"

    ps_df = pd.read_csv(scaling_file)
    if weighting_column not in ps_df.columns:
        raise ValueError(
            f"Column '{weighting_column}' not found in scaling factors file '{scaling_file}'."
        )

    countries = ps_cfg.get("countries", [])
    if not countries:
        raise ValueError("'pattern_scaling.countries' must list at least one ISO3 code.")

    definitions = _config_value(cm_cfg, "climate_scenarios", "definitions")
    scenario_prefixes = {str(entry["id"]).lower()[:4] for entry in definitions if "id" in entry}
    if not scenario_prefixes:
        raise ValueError("No climate scenarios found in configuration definitions.")

    filtered = ps_df[
        (ps_df["iso3"].isin(countries)) & (ps_df["scenario"].str.lower().isin(scenario_prefixes))
    ][["name", "iso3", "scenario", weighting_column]].copy()
    filtered.rename(columns={weighting_column: "scaling_factor"}, inplace=True)
    return filtered


def _detect_scenario_prefix(
    frame: pd.DataFrame, scenario_prefixes: set[str], filename: str
) -> str | None:
    if "climate_scenario" in frame.columns:
        values = frame["climate_scenario"].dropna().astype(str).str.lower().unique()
        if values.size:
            return values[0][:4]
    filename_lower = filename.lower()
    for prefix in scenario_prefixes:
        if prefix in filename_lower:
            return prefix
    return None


def scale_results(config: Dict[str, Any], scaling_factors: pd.DataFrame) -> None:
    """Apply pattern-scaling factors to climate-module temperature results."""

    ps_cfg = _config_value(config, "pattern_scaling")
    cm_cfg = _config_value(config, "climate_module")
    run_directory = get_results_run_directory(config)

    climate_dir = _resolve_path(cm_cfg.get("output_directory", "results/climate"))
    climate_dir = apply_results_run_directory(climate_dir, run_directory, repo_root=DEFAULT_ROOT)
    if not climate_dir.exists():
        raise FileNotFoundError(
            f"Climate output directory '{climate_dir}' not found. Run the climate module first."
        )

    output_dir = _resolve_path(ps_cfg.get("output_directory", "results/climate_scaled"))
    output_dir = apply_results_run_directory(output_dir, run_directory, repo_root=DEFAULT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    definitions = _config_value(cm_cfg, "climate_scenarios", "definitions")
    scenario_prefixes = {str(entry["id"]).lower()[:4] for entry in definitions if "id" in entry}

    countries = set(ps_cfg.get("countries", []))
    if not countries:
        raise ValueError("'pattern_scaling.countries' must list at least one ISO3 code.")

    for climate_file in sorted(climate_dir.glob("*.csv")):
        frame = pd.read_csv(climate_file)
        if not {"temperature_baseline", "temperature_adjusted"}.issubset(frame.columns):
            continue
        scenario_prefix = _detect_scenario_prefix(frame, scenario_prefixes, climate_file.name)
        if scenario_prefix is None:
            continue

        for country in countries:
            row = scaling_factors[
                (scaling_factors["scenario"].str.lower() == scenario_prefix)
                & (scaling_factors["iso3"] == country)
            ]
            if row.empty:
                continue

            factor = float(row.iloc[0]["scaling_factor"])
            scaled = pd.DataFrame(
                {
                    "year": frame["year"],
                    "temperature_baseline": frame["temperature_baseline"] * factor,
                    "temperature_adjusted": frame["temperature_adjusted"] * factor,
                }
            )
            scaled["temperature_delta"] = (
                scaled["temperature_adjusted"] - scaled["temperature_baseline"]
            )
            if "climate_scenario" in frame.columns:
                scaled["climate_scenario"] = frame["climate_scenario"]
            scaled["iso3"] = country
            scaled["scaling_factor"] = factor

            output_path = output_dir / f"{country}_{climate_file.name}"
            scaled.to_csv(output_path, index=False)


if __name__ == "__main__":
    config = load_config(DEFAULT_CONFIG_PATH)
    sf = get_scaling_factors(config)
    scale_results(config, sf)

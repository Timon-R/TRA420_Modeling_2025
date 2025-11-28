"""Calculates local temperature and precipitation changes using pattern scaling."""

from __future__ import annotations

from os import listdir
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

DEFAULT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = DEFAULT_ROOT / "config.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML config into a dict."""
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_scaling_factors(config: Dict[str, Any]) -> pd.DataFrame:
    ps_df = pd.read_csv(DEFAULT_ROOT / config["pattern_scaling"].get("scaling_factors_file"))
    ps_df_precip = pd.read_csv(DEFAULT_ROOT / config["pattern_scaling"].get("scaling_factors_file_precipitation"))
    """Perform pattern scaling based on the provided configuration."""
    countries = config["pattern_scaling"].get("countries")
    # In the pattern scaling csv, scenarios are defined without rc
    scenarios = [
        s["id"][0:4] for s in config["climate_module"]["climate_scenarios"].get("definitions")
    ]
    weighting = f"patterns.{config["pattern_scaling"].get("scaling_weighting")}"
    sf = ps_df[(ps_df["iso3"].isin(countries)) & (ps_df["scenario"].isin(scenarios))]
    sf = sf[["name", "iso3", "scenario", weighting]]
    sf_precip = ps_df_precip[(ps_df_precip["iso3"].isin(countries)) & (ps_df_precip["scenario"].isin(scenarios))]
    sf_precip = (sf_precip[["name", "iso3", "scenario", weighting]]
                        .rename({weighting: f"precipitation.{weighting}"}, axis=1))[[f"precipitation.{weighting}"]]
    sf = pd.concat([sf, sf_precip], axis=1, sort=False)
    return sf

def scale_results(config: Dict[str, Any], scaling_factors: pd.DataFrame) -> None:
    """Apply scaling factors to climate results."""
    results = listdir(DEFAULT_ROOT / config["climate_module"].get("output_directory"))
    scenarios = [
        s["id"][0:4] for s in config["climate_module"]["climate_scenarios"].get("definitions")
    ]
    weighting = f"patterns.{config["pattern_scaling"].get("scaling_weighting")}"
    output_dir = DEFAULT_ROOT / config["pattern_scaling"].get("output_directory")
    countries = config["pattern_scaling"].get("countries")
    for result in results:
        if result.endswith(".csv"):
            df = pd.read_csv(
                DEFAULT_ROOT / config["climate_module"].get("output_directory") / result
            )
            for scenario in scenarios:
                for country in countries:
                    if scenario in result:
                        sf = scaling_factors[
                            (scaling_factors["scenario"] == scenario)
                            & (scaling_factors["iso3"] == country)
                        ][weighting].values[0]
                        sf_precip = scaling_factors[
                            (scaling_factors["scenario"] == scenario)
                            & (scaling_factors["iso3"] == country)
                            ][f"precipitation.{weighting}"].values[0]
                        sdf = pd.concat(
                            [
                                df["year"],
                                df["temperature_baseline"] * sf,
                                df["temperature_adjusted"] * sf,
                                df["temperature_adjusted"] * sf - df["temperature_baseline"] * sf,
                                df["temperature_baseline"] * sf_precip,
                                df["temperature_adjusted"] * sf_precip,
                            ],
                            axis=1,
                        )
                        sdf.columns.values[3:6] = ["temperature_delta", "precipitation_baseline", "precipitation_adjusted"]
                        name = country + "_" + result
                        sdf.to_csv(output_dir / name, index=False)


if __name__ == "__main__":
    config = load_config(DEFAULT_CONFIG_PATH)
    sf = get_scaling_factors(config)
    scale_results(config, sf)

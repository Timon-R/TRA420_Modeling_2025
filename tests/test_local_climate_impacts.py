from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_climate_impacts import scaling as ps


@pytest.fixture
def temp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(ps, "DEFAULT_ROOT", tmp_path)

    climate_dir = tmp_path / "results" / "climate"
    climate_dir.mkdir(parents=True)
    scaling_dir = tmp_path / "results" / "climate_scaled"
    scaling_dir.mkdir(parents=True)

    scaling_file = tmp_path / "data" / "scaling.csv"
    scaling_file.parent.mkdir(parents=True)
    scaling_df = pd.DataFrame(
        {
            "name": ["United States"],
            "iso3": ["USA"],
            "continent": ["North America"],
            "scenario": ["ssp2"],
            "patterns.area": [1.5],
        }
    )
    scaling_df.to_csv(scaling_file, index=False)

    precip_file = tmp_path / "data" / "precip_scaling.csv"
    precip_df = pd.DataFrame(
        {
            "name": ["United States"],
            "iso3": ["USA"],
            "continent": ["North America"],
            "scenario": ["ssp2"],
            "patterns.area": [2.5],
        }
    )
    precip_df.to_csv(precip_file, index=False)

    config = {
        "local_climate_impacts": {
            "output_directory": "results/climate_scaled",
            "scaling_factors_file": "data/scaling.csv",
            "scaling_factors_file_precipitation": "data/precip_scaling.csv",
            "scaling_weighting": "area",
            "countries": ["USA"],
        },
        "climate_module": {
            "output_directory": "results/climate",
            "climate_scenarios": {
                "definitions": [{"id": "ssp245"}],
            },
        },
    }

    return config


def test_get_scaling_factors_filters_to_countries_and_scenarios(temp_config: dict):
    factors = ps.get_scaling_factors(temp_config)
    assert len(factors) == 1
    assert factors.iloc[0]["iso3"] == "USA"
    assert np.isclose(factors.iloc[0]["scaling_factor"], 1.5)
    assert np.isclose(factors.iloc[0]["precipitation_scaling_factor"], 2.5)


def test_scale_results_writes_scaled_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ps, "DEFAULT_ROOT", tmp_path)
    climate_dir = tmp_path / "results" / "climate"
    climate_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "climate_scaled").mkdir(parents=True, exist_ok=True)

    climate_df = pd.DataFrame(
        {
            "year": [2030, 2040],
            "temperature_baseline": [1.0, 1.5],
            "temperature_adjusted": [1.2, 1.8],
            "temperature_delta": [0.2, 0.3],
            "climate_scenario": ["ssp245", "ssp245"],
        }
    )
    climate_file = climate_dir / "policy_ssp245.csv"
    climate_df.to_csv(climate_file, index=False)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    scaling_file = data_dir / "scaling.csv"
    pd.DataFrame(
        {
            "name": ["United States"],
            "iso3": ["USA"],
            "continent": ["North America"],
            "scenario": ["ssp2"],
            "patterns.area": [2.0],
        }
    ).to_csv(scaling_file, index=False)
    precip_file = data_dir / "precip_scaling.csv"
    pd.DataFrame(
        {
            "name": ["United States"],
            "iso3": ["USA"],
            "continent": ["North America"],
            "scenario": ["ssp2"],
            "patterns.area": [4.0],
        }
    ).to_csv(precip_file, index=False)

    config = {
        "local_climate_impacts": {
            "output_directory": "results/climate_scaled",
            "scaling_factors_file": "data/scaling.csv",
            "scaling_factors_file_precipitation": "data/precip_scaling.csv",
            "scaling_weighting": "area",
            "countries": ["USA"],
        },
        "climate_module": {
            "output_directory": "results/climate",
            "climate_scenarios": {
                "definitions": [{"id": "ssp245"}],
            },
        },
    }

    factors = ps.get_scaling_factors(config)
    ps.scale_results(config, factors)

    output_path = tmp_path / "results" / "climate_scaled" / "USA_policy_ssp245.csv"
    assert output_path.exists()
    out = pd.read_csv(output_path)
    np.testing.assert_allclose(out["temperature_baseline"], [2.0, 3.0])
    np.testing.assert_allclose(out["temperature_delta"], [0.4, 0.6])
    np.testing.assert_allclose(out["precipitation_baseline_mm_per_day"], [4.0, 6.0])
    np.testing.assert_allclose(out["precipitation_adjusted_mm_per_day"], [4.8, 7.2])
    np.testing.assert_allclose(out["precipitation_delta_mm_per_day"], [0.8, 1.2])
    assert (out["iso3"] == "USA").all()

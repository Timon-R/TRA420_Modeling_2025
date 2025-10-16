import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calc_emissions.calculator import (
    EmissionScenarioResult,
    _generate_years,
    _load_emission_factors,
    _resolve_demand_series,
    _resolve_mix_shares,
    _values_to_series,
    calculate_emissions,
    run_from_config,
)


def test_generate_years_produces_expected_sequence():
    years = _generate_years({"start": 2020, "end": 2030, "step": 5})
    assert years == [2020, 2025, 2030]


def test_values_to_series_interpolates_and_aligns_years():
    series = _values_to_series({2020: 100, 2030: 200}, [2020, 2025, 2030], "demand")
    assert series.loc[2025] == pytest.approx(150.0)
    assert series.index.tolist() == [2020, 2025, 2030]


def test_calculate_emissions_combines_demand_and_mix():
    years = [2020, 2025]
    demand = pd.Series([100.0, 110.0], index=years, name="demand_twh")
    mix = pd.DataFrame({"coal": [0.6, 0.5], "solar": [0.4, 0.5]}, index=years)
    factors = pd.DataFrame(
        {
            "co2_mt_per_twh": [0.9, 0.0],
            "so2_kt_per_twh": [5.0, 0.0],
            "nox_kt_per_twh": [2.0, 0.0],
            "pm25_kt_per_twh": [1.0, 0.0],
        },
        index=["coal", "solar"],
    )

    result = calculate_emissions("scenario", demand, mix, factors)

    assert isinstance(result, EmissionScenarioResult)
    np.testing.assert_allclose(result.generation_twh.loc[2020, "coal"], 60.0)
    np.testing.assert_allclose(result.total_emissions_mt["co2"].iloc[0], 54.0)
    np.testing.assert_allclose(result.total_emissions_mt["so2"].iloc[0], 0.3)


def test_run_from_config_creates_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path
    (root / "resources").mkdir()
    (root / "results").mkdir()

    emission_factors = pd.DataFrame(
        {
            "technology": ["coal", "solar"],
            "co2_mt_per_twh": [1.0, 0.0],
            "so2_kt_per_twh": [2.0, 0.0],
            "nox_kt_per_twh": [1.0, 0.0],
            "pm25_kt_per_twh": [0.5, 0.0],
        }
    )
    factors_path = root / "factors.csv"
    emission_factors.to_csv(factors_path, index=False)

    config = {
        "calc_emissions": {
            "emission_factors_file": str(factors_path),
            "output_directory": str(root / "resources"),
            "results_directory": str(root / "results"),
            "years": {"start": 2020, "end": 2025, "step": 5},
            "demand_scenarios": {},
            "mix_scenarios": {},
            "baseline": {
                "demand_custom": {2020: 100.0, 2025: 110.0},
                "mix_custom": {"shares": {"coal": 0.7, "solar": 0.3}},
            },
            "scenarios": [
                {
                    "name": "policy",
                    "demand_custom": {2020: 80.0, 2025: 90.0},
                    "mix_custom": {"shares": {"coal": 0.2, "solar": 0.8}},
                }
            ],
        }
    }

    config_path = root / "config.yaml"
    config_path.write_text(json.dumps(config))

    # run_from_config expects YAML; using json-compatible structure is valid for yaml.safe_load.
    results = run_from_config(config_path)

    assert "policy" in results
    output_file = root / "resources" / "policy" / "co2.csv"
    assert output_file.exists()
    csv = pd.read_csv(output_file)
    assert list(csv.columns) == ["year", "delta"]
    assert csv["delta"].dtype == float

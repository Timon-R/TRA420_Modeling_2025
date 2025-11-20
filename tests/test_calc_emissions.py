import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calc_emissions.calculator import (
    EmissionScenarioResult,
    _generate_years,
    _values_to_series,
    calculate_emissions,
    run_from_config,
)
from calc_emissions.writers import write_per_country_results


def test_generate_years_produces_expected_sequence():
    years = _generate_years({"start": 2020, "end": 2030, "step": 5})
    assert years == [2020, 2025, 2030]


def test_generate_years_uses_fallback_when_missing_config():
    years = _generate_years(None, {"start": 2030, "end": 2040, "step": 5})
    assert years == [2030, 2035, 2040]


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
            "co2": [0.9, 0.0],
            "sox": [0.005, 0.0],
            "nox": [0.002, 0.0],
            "pm25": [0.001, 0.0],
        },
        index=["coal", "solar"],
    )

    result = calculate_emissions(
        "scenario",
        demand,
        mix,
        factors,
        demand_case="base_demand",
        mix_case="base_mix",
    )

    assert isinstance(result, EmissionScenarioResult)
    np.testing.assert_allclose(result.generation_twh.loc[2020, "coal"], 60.0)
    np.testing.assert_allclose(result.total_emissions_mt["co2"].iloc[0], 54.0)
    np.testing.assert_allclose(result.total_emissions_mt["sox"].iloc[0], 0.3)


def test_run_from_config_creates_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path
    (root / "resources").mkdir()
    (root / "results").mkdir()

    emission_factors = pd.DataFrame(
        {
            "technology": ["coal", "solar"],
            "co2_mt_per_twh": [1.0, 0.0],
            "sox_kg_per_kwh": [0.002, 0.0],
            "nox_kg_per_kwh": [0.001, 0.0],
            "pm25_kg_per_kwh": [0.0005, 0.0],
        }
    )
    factors_path = root / "factors.csv"
    emission_factors.to_csv(factors_path, index=False)

    config = {
        "calc_emissions": {
            "emission_factors_file": str(factors_path),
            "output_directory": str(root / "resources"),
            "results_directory": str(root / "results"),
            "demand_scenarios": {
                "base_demand": {
                    "values": {2020: 100.0, 2025: 110.0},
                },
                "scen1_lower": {
                    "values": {2020: 80.0, 2025: 90.0},
                },
                "scen1_upper": {
                    "values": {2020: 120.0, 2025: 130.0},
                },
            },
            "mix_scenarios": {
                "base_mix": {
                    "shares": {"coal": 0.7, "solar": 0.3},
                }
            },
        }
    }

    config_path = root / "config.yaml"
    config_path.write_text(json.dumps(config))

    # run_from_config expects YAML; using json-compatible structure is valid for yaml.safe_load.
    results = run_from_config(
        config_path,
        default_years={"start": 2020, "end": 2025, "step": 5},
    )

    assert {
        "base_mix__base_demand",
        "base_mix__scen1_lower",
        "base_mix__scen1_upper",
        "base_mix__scen1_mean",
    } <= set(results.keys())
    per_country_root = root / "results" / "emissions"
    write_per_country_results({"Testland": results}, per_country_root, "base_demand")
    output_file = per_country_root / "base_mix" / "Testland" / "co2.csv"
    assert output_file.exists()
    csv = pd.read_csv(output_file, comment="#")
    assert "absolute_base_demand" in csv.columns
    assert "absolute_scen1_lower" in csv.columns
    assert "absolute_scen1_mean" in csv.columns
    assert "delta_scen1_lower" in csv.columns
    assert csv["delta_scen1_lower"].dtype == float
    mean_series = csv.loc[csv["year"] == 2025, "absolute_scen1_mean"].iloc[0]
    lower_series = csv.loc[csv["year"] == 2025, "absolute_scen1_lower"].iloc[0]
    upper_series = csv.loc[csv["year"] == 2025, "absolute_scen1_upper"].iloc[0]
    assert mean_series == pytest.approx((lower_series + upper_series) / 2.0)


def test_run_from_config_enforces_allowed_cases(tmp_path: Path):
    root = tmp_path
    (root / "resources").mkdir()
    (root / "results").mkdir()

    factors = pd.DataFrame(
        {
            "technology": ["coal"],
            "co2_mt_per_twh": [1.0],
        }
    )
    factors_path = root / "factors.csv"
    factors.to_csv(factors_path, index=False)

    config = {
        "calc_emissions": {
            "emission_factors_file": str(factors_path),
            "output_directory": str(root / "resources"),
            "results_directory": str(root / "results"),
            "demand_scenarios": {
                "base_demand": {"values": {2020: 10, 2025: 11}},
                "scen1_lower": {"values": {2020: 8, 2025: 9}},
                "custom_extra": {"values": {2020: 12, 2025: 13}},
            },
            "mix_scenarios": {
                "base_mix": {"shares": {"coal": 1.0}},
                "custom_mix": {"shares": {"coal": 1.0}},
            },
        }
    }
    config_path = root / "config.yaml"
    config_path.write_text(json.dumps(config))

    results = run_from_config(
        config_path,
        default_years={"start": 2020, "end": 2025, "step": 5},
        allowed_demand_cases=["base_demand", "scen1_lower"],
        allowed_mix_cases=["base_mix"],
    )

    assert set(results) == {"base_mix__base_demand", "base_mix__scen1_lower"}

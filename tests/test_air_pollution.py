import json
import math
from pathlib import Path

import pandas as pd
import pytest

from air_pollution import run_from_config as run_air_pollution


def test_air_pollution_run_from_config_creates_outputs(tmp_path: Path):
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

    pollution_stats = pd.DataFrame(
        {
            "country": ["Testland_A", "Testland_B"],
            "count": [1, 1],
            "mean": [10.0, 30.0],
            "median": [10.0, 30.0],
            "std": [None, None],
            "min": [10.0, 30.0],
            "max": [10.0, 30.0],
        }
    )
    stats_path = root / "pm25.csv"
    pollution_stats.to_csv(stats_path, index=False)

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
        },
        "air_pollution": {
            "output_directory": str(root / "results" / "air_pollution"),
            "electricity_share": 1.0,
            "country_weights": {"Testland_A": 3.0, "Testland_B": 1.0},
            "pollutants": {
                "pm25": {
                    "stats_file": str(stats_path),
                    "relative_risk": 1.08,
                    "reference_delta": 10.0,
                },
                "nox": {
                    "stats_file": str(stats_path),
                    "relative_risk": 1.03,
                    "reference_delta": 10.0,
                },
            },
            "baseline_deaths": {"per_year": 7000.0},
        },
    }

    config_path = root / "config.yaml"
    config_path.write_text(json.dumps(config))

    results = run_air_pollution(config_path=config_path)

    assert "policy" in results
    output_file = (
        Path(config["air_pollution"]["output_directory"]) / "policy" / "pm25_health_impact.csv"
    )
    assert output_file.exists()

    df = pd.read_csv(output_file)
    expected_columns = {
        "country",
        "year",
        "baseline_concentration",
        "emission_ratio",
        "new_concentration",
        "delta_concentration",
        "percent_change_mortality",
    }
    assert set(df.columns) == expected_columns

    beta = math.log(1.08) / 10.0
    ratio_2020 = 0.008 / 0.035
    ratio_2025 = 0.009 / 0.0385
    baseline_concentrations = {"Testland_A": 10.0, "Testland_B": 30.0}
    country_weights = {"Testland_A": 3.0 / 4.0, "Testland_B": 1.0 / 4.0}

    def expected_pct_change(concentration: float, ratio: float, beta_value: float) -> float:
        delta_conc = concentration * (ratio - 1.0)
        return math.exp(beta_value * delta_conc) - 1.0

    expected_pm25_2020_per_country = {
        country: expected_pct_change(conc, ratio_2020, beta)
        for country, conc in baseline_concentrations.items()
    }
    expected_pm25_2025_per_country = {
        country: expected_pct_change(conc, ratio_2025, beta)
        for country, conc in baseline_concentrations.items()
    }

    for country, _conc in baseline_concentrations.items():
        country_series = (
            df.loc[df["country"] == country].set_index("year")["percent_change_mortality"].to_dict()
        )
        assert math.isclose(
            country_series[2020], expected_pm25_2020_per_country[country], rel_tol=1e-6
        )
        assert math.isclose(
            country_series[2025], expected_pm25_2025_per_country[country], rel_tol=1e-6
        )

    expected_weighted_pm25_2020 = sum(
        country_weights[country] * value
        for country, value in expected_pm25_2020_per_country.items()
    )
    expected_weighted_pm25_2025 = sum(
        country_weights[country] * value
        for country, value in expected_pm25_2025_per_country.items()
    )

    summary_file = (
        Path(config["air_pollution"]["output_directory"]) / "policy" / "pm25_mortality_summary.csv"
    )
    assert not summary_file.exists()

    impact_pm25 = results["policy"].pollutant_results["pm25"]
    pm25_weights_dict = impact_pm25.country_weights.to_dict()
    for country, weight in country_weights.items():
        assert math.isclose(pm25_weights_dict[country], weight, rel_tol=1e-6)
    assert math.isclose(
        impact_pm25.weighted_percent_change.loc[2020], expected_weighted_pm25_2020, rel_tol=1e-6
    )
    assert math.isclose(
        impact_pm25.weighted_percent_change.loc[2025], expected_weighted_pm25_2025, rel_tol=1e-6
    )

    beta_nox = math.log(1.03) / 10.0
    expected_nox_2020_per_country = {
        country: expected_pct_change(conc, ratio_2020, beta_nox)
        for country, conc in baseline_concentrations.items()
    }
    expected_nox_2025_per_country = {
        country: expected_pct_change(conc, ratio_2025, beta_nox)
        for country, conc in baseline_concentrations.items()
    }
    expected_weighted_nox_2020 = sum(
        country_weights[country] * value for country, value in expected_nox_2020_per_country.items()
    )
    expected_weighted_nox_2025 = sum(
        country_weights[country] * value for country, value in expected_nox_2025_per_country.items()
    )

    impact_nox = results["policy"].pollutant_results["nox"]
    assert math.isclose(
        impact_nox.weighted_percent_change.loc[2020], expected_weighted_nox_2020, rel_tol=1e-6
    )
    assert math.isclose(
        impact_nox.weighted_percent_change.loc[2025], expected_weighted_nox_2025, rel_tol=1e-6
    )

    total_file = (
        Path(config["air_pollution"]["output_directory"]) / "policy" / "total_mortality_summary.csv"
    )
    assert total_file.exists()
    total_df = pd.read_csv(total_file)
    expected_percent_change_2020 = (expected_weighted_pm25_2020 + expected_weighted_nox_2020) / 2.0
    expected_percent_change_2025 = (expected_weighted_pm25_2025 + expected_weighted_nox_2025) / 2.0
    expected_total_2020 = 7000.0 * expected_percent_change_2020
    expected_total_2025 = 7000.0 * expected_percent_change_2025
    total_delta_by_year = dict(
        zip(total_df["year"], total_df["delta_deaths_per_year"], strict=False)
    )
    assert math.isclose(total_delta_by_year[2020], expected_total_2020, rel_tol=1e-6)
    assert math.isclose(total_delta_by_year[2025], expected_total_2025, rel_tol=1e-6)
    baseline_by_year = dict(
        zip(total_df["year"], total_df["baseline_deaths_per_year"], strict=False)
    )
    assert baseline_by_year[2020] == pytest.approx(7000.0)


def test_air_pollution_with_baseline_deaths_and_vsl(tmp_path: Path):
    root = tmp_path
    (root / "resources").mkdir()
    (root / "results").mkdir()

    emission_factors = pd.DataFrame(
        {
            "technology": ["coal"],
            "co2_mt_per_twh": [1.0],
            "so2_kt_per_twh": [1.0],
            "nox_kt_per_twh": [1.0],
            "pm25_kt_per_twh": [1.0],
        }
    )
    factors_path = root / "factors.csv"
    emission_factors.to_csv(factors_path, index=False)

    stats = pd.DataFrame(
        {
            "country": ["A", "B"],
            "median": [1.0, 2.0],
            "baseline_deaths_per_year": [100.0, 300.0],
        }
    )
    stats_path = root / "stats.csv"
    stats.to_csv(stats_path, index=False)

    config = {
        "calc_emissions": {
            "emission_factors_file": str(factors_path),
            "output_directory": str(root / "resources"),
            "results_directory": str(root / "results"),
            "years": {"start": 2020, "end": 2025, "step": 5},
            "demand_scenarios": {},
            "mix_scenarios": {},
            "baseline": {
                "demand_custom": {2020: 100.0, 2025: 100.0},
                "mix_custom": {"shares": {"coal": 1.0}},
            },
            "scenarios": [
                {
                    "name": "policy",
                    "demand_custom": {2020: 50.0, 2025: 50.0},
                    "mix_custom": {"shares": {"coal": 1.0}},
                }
            ],
        },
        "air_pollution": {
            "output_directory": str(root / "results" / "air_pollution"),
            "electricity_share": 1.0,
            "value_of_statistical_life_usd": 1_000_000.0,
            "pollutants": {
                "pm25": {"stats_file": str(stats_path)},
                "nox": {
                    "stats_file": str(stats_path),
                    "relative_risk": 1.03,
                    "reference_delta": 10,
                },
            },
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(json.dumps(config))

    results = run_air_pollution(config_path=config_path)
    pm25_summary_path = (
        Path(config["air_pollution"]["output_directory"]) / "policy" / "pm25_mortality_summary.csv"
    )
    assert pm25_summary_path.exists()
    pm25_summary = pd.read_csv(pm25_summary_path)
    assert "delta_value_usd" in pm25_summary.columns
    assert math.isclose(
        pm25_summary["delta_value_usd"].iloc[0],
        pm25_summary["delta_deaths_per_year"].iloc[0] * 1_000_000.0,
    )
    impact_pm25 = results["policy"].pollutant_results["pm25"]
    expected_weights = {"A": 0.25, "B": 0.75}
    for country, weight in expected_weights.items():
        assert math.isclose(impact_pm25.country_weights[country], weight, rel_tol=1e-6)

    total_path = (
        Path(config["air_pollution"]["output_directory"]) / "policy" / "total_mortality_summary.csv"
    )
    assert total_path.exists()
    total_summary = pd.read_csv(total_path)
    assert "delta_value_usd" in total_summary.columns
    assert math.isclose(
        total_summary["delta_value_usd"].iloc[0],
        total_summary["delta_deaths_per_year"].iloc[0] * 1_000_000.0,
    )
    assert math.isclose(total_summary["baseline_deaths_per_year"].iloc[0], 800.0)

    assert results["policy"].total_mortality_summary is not None

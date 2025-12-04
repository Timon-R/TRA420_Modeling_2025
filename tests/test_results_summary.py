from pathlib import Path

import pandas as pd
import pytest

from results_summary import (
    SummarySettings,
    _include_background_plots,
    _lookup_extreme_weather,
    _plot_socioeconomic_timeseries,
    build_summary,
    write_summary_csv,
)


def _write_csv(path: Path, data: list[dict]) -> None:
    frame = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_build_summary_collects_metrics(tmp_path: Path):
    root = tmp_path

    gdp_path = root / "gdp.csv"
    _write_csv(
        gdp_path,
        [
            {"year": 2025, "gdp_trillion_usd": 1.0, "population_million": 5.0},
            {"year": 2030, "gdp_trillion_usd": 1.2, "population_million": 5.1},
            {"year": 2050, "gdp_trillion_usd": 1.5, "population_million": 5.2},
        ],
    )

    emission_root = root / "results" / "emissions" / "All_countries"
    temperature_root = root / "results" / "climate"

    config = {
        "calc_emissions": {
            "countries": {
                "baseline_demand_case": "base_demand",
                "demand_scenarios": ["base_demand", "policy"],
                "mix_scenarios": ["base_mix"],
                "aggregate_output_directory": "results/emissions/All_countries",
                "resources_root": "results/emissions",
            }
        },
        "economic_module": {
            "reference_scenario": "base_mix__base_demand",
            "evaluation_scenarios": ["base_mix__policy"],
            "gdp_series": str(gdp_path),
            "methods": {
                "run": "ramsey_discount",
            },
            "output_directory": "results/economic",
            "base_year": 2025,
            "aggregation": "average",
            "aggregation_horizon": {"start": 2025, "end": 2050},
            "data_sources": {
                "emission_root": str(emission_root),
                "temperature_root": str(temperature_root),
                "climate_scenarios": ["ssp245"],
            },
        },
        "results": {
            "summary": {
                "years": [2030, 2050],
                "output_directory": "results/summary",
                "include_plots": False,
                "plot_format": "png",
            }
        },
        "local_climate_impacts": {
            "output_directory": "results/climate_scaled",
            "countries": ["SRB"],
            "extreme_weather_costs_file": str(root / "extreme_weather.csv"),
        },
    }

    # Emission deltas
    _write_csv(
        emission_root / "base_mix" / "co2.csv",
        [
            {
                "year": 2025,
                "absolute_base_demand": 0.0,
                "absolute_policy": 0.0,
                "delta_base_demand": 0.0,
                "delta_policy": 0.0,
            },
            {
                "year": 2030,
                "absolute_base_demand": 0.0,
                "absolute_policy": -10.0,
                "delta_base_demand": 0.0,
                "delta_policy": -10.0,
            },
            {
                "year": 2050,
                "absolute_base_demand": 0.0,
                "absolute_policy": -15.0,
                "delta_base_demand": 0.0,
                "delta_policy": -15.0,
            },
        ],
    )

    # Per-country pollutant deltas
    _write_csv(
        root / "results" / "emissions" / "base_mix" / "Serbia" / "co2.csv",
        [
            {
                "year": 2025,
                "absolute_base_demand": 0.0,
                "absolute_policy": 0.0,
                "delta_base_demand": 0.0,
                "delta_policy": 0.0,
            },
            {
                "year": 2030,
                "absolute_base_demand": 0.0,
                "absolute_policy": -5.0,
                "delta_base_demand": 0.0,
                "delta_policy": -5.0,
            },
        ],
    )

    _write_csv(
        root / "results" / "emissions" / "base_mix" / "Serbia" / "nox.csv",
        [
            {
                "year": 2030,
                "absolute_base_demand": 0.0,
                "absolute_policy": -1.0,
                "delta_base_demand": 0.0,
                "delta_policy": -1.0,
            }
        ],
    )

    # Temperature deltas
    _write_csv(
        temperature_root / "base_mix__policy_ssp245.csv",
        [
            {"year": 2030, "temperature_baseline": 3.0, "temperature_adjusted": 2.8},
            {"year": 2050, "temperature_baseline": 3.5, "temperature_adjusted": 3.2},
        ],
    )
    _write_csv(
        temperature_root / "base_mix__base_demand_ssp245.csv",
        [
            {"year": 2030, "temperature_baseline": 3.0, "temperature_adjusted": 3.0},
            {"year": 2050, "temperature_baseline": 3.5, "temperature_adjusted": 3.5},
        ],
    )

    # SCC timeseries
    _write_csv(
        root / "results" / "economic" / "pulse_scc_timeseries_ramsey_discount_ssp245.csv",
        [
            {
                "year": 2030,
                "discount_factor": 1.0,
                "pv_damage_per_pulse_usd": -5.0e9,
                "scc_usd_per_tco2": 50.0,
                "pulse_size_tco2": 1_000_000,
                "delta_emissions_tco2": -100_000_000.0,
                "discounted_delta_usd": -5.0e9,
            },
            {
                "year": 2050,
                "discount_factor": 0.9,
                "pv_damage_per_pulse_usd": -7.5e9,
                "scc_usd_per_tco2": 60.0,
                "pulse_size_tco2": 1_000_000,
                "delta_emissions_tco2": -150_000_000.0,
                "discounted_delta_usd": -7.5e9,
            },
        ],
    )

    # Mortality summary
    _write_csv(
        root / "results" / "air_pollution" / "base_mix__policy" / "total_mortality_summary.csv",
        [
            {
                "year": 2030,
                "percent_change_mortality": -0.02,
                "baseline_deaths_per_year": 1000,
                "delta_deaths_per_year": -20,
                "new_deaths_per_year": 980,
                "value_of_statistical_life_usd": 1_000_000,
                "baseline_value_usd": 1_000_000_000,
                "delta_value_usd": -20_000_000,
                "new_value_usd": 980_000_000,
            },
            {
                "year": 2050,
                "percent_change_mortality": -0.03,
                "baseline_deaths_per_year": 1000,
                "delta_deaths_per_year": -30,
                "new_deaths_per_year": 970,
                "value_of_statistical_life_usd": 1_000_000,
                "baseline_value_usd": 1_000_000_000,
                "delta_value_usd": -30_000_000,
                "new_value_usd": 970_000_000,
            },
        ],
    )

    # Local climate impacts (pattern-scaled outputs)
    climate_scaled_dir = root / "results" / "climate_scaled"
    _write_csv(
        climate_scaled_dir / "SRB_base_mix__policy_ssp245.csv",
        [
            {
                "year": 2030,
                "temperature_delta": 0.2,
                "precipitation_delta_mm_per_day": 0.005,
            },
            {
                "year": 2050,
                "temperature_delta": 0.3,
                "precipitation_delta_mm_per_day": 0.007,
            },
        ],
    )
    _write_csv(
        climate_scaled_dir / "SRB_base_mix__base_demand_ssp245.csv",
        [
            {
                "year": 2030,
                "temperature_delta": 0.0,
                "precipitation_delta_mm_per_day": 0.0,
            },
            {
                "year": 2050,
                "temperature_delta": 0.0,
                "precipitation_delta_mm_per_day": 0.0,
            },
        ],
    )

    _write_csv(
        root / "extreme_weather.csv",
        [
            {
                "Country": "SRB",
                "Scenario": "ssp2",
                "2030_pct_gdp": 6.0,
                "2050_pct_gdp": 7.0,
            }
        ],
    )

    settings, methods, metrics_map = build_summary(root, config)
    summary_csv = write_summary_csv(settings, methods, metrics_map)
    assert summary_csv.exists()
    summary_df = pd.read_csv(summary_csv)
    assert "energy_mix" in summary_df.columns
    assert "delta_co2_Mt_all_countries_2030" in summary_df.columns
    assert "delta_co2_Serbia_2030" in summary_df.columns
    assert summary_df.iloc[0]["delta_co2_Serbia_2030"] == pytest.approx(-5.0)

    assert settings.years == [2030, 2050]
    assert settings.aggregation_mode == "average"
    assert settings.aggregation_horizon == (2025, 2050)
    assert methods == ["ramsey_discount"]

    policy_key = "base_mix__policy_ssp245"

    assert policy_key in metrics_map
    policy_metrics = metrics_map[policy_key]
    assert policy_metrics.emission_delta_mt[2030] == -10.0
    assert policy_metrics.emission_timeseries == {2025: 0.0, 2030: -10.0, 2050: -15.0}
    assert policy_metrics.temperature_timeseries is not None
    assert policy_metrics.temperature_timeseries[2030] == pytest.approx(-0.2)
    assert policy_metrics.mortality_delta[2050] == -30
    assert policy_metrics.mortality_value_delta is not None
    assert policy_metrics.mortality_value_delta[2030] == -20_000_000
    assert policy_metrics.scc_average["ramsey_discount"] == 50.0
    assert "baseline_ssp245" not in metrics_map

    # Re-run with per-year aggregation to ensure SCC series are reported for configured years.
    config["economic_module"]["aggregation"] = "per_year"
    settings_py, methods_py, metrics_map_py = build_summary(root, config)
    assert settings_py.aggregation_mode == "per_year"

    policy_row = summary_df.loc[summary_df["demand_case"] == "policy"].iloc[0]
    expected_precip = 0.005 * 365.0
    assert policy_row["local_climate_precipitation_delta_mm_per_year_SRB_2030"] == pytest.approx(
        expected_precip
    )
    assert policy_row[
        "local_climate_precipitation_delta_mm_per_year_all_countries_2030"
    ] == pytest.approx(expected_precip)
    assert policy_row["extreme_weather_pct_gdp_SRB_2030"] == pytest.approx(6.0)
    assert policy_row["extreme_weather_pct_gdp_all_countries_2050"] == pytest.approx(7.0)
    assert policy_row["delta_T_local_climate_C_all_countries_2030"] == pytest.approx(0.2)


def test_include_background_plots_copies_files(tmp_path: Path):
    climate_dir = tmp_path / "resources" / "climate"
    plots_dir = tmp_path / "summary" / "plots"
    climate_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    for stem in ("background_climate_full", "background_climate_horizon"):
        (climate_dir / f"{stem}.png").write_bytes(b"fake")

    settings = SummarySettings(
        years=[2030],
        output_directory=tmp_path / "summary",
        climate_output_directory=climate_dir,
    )

    _include_background_plots(settings, plots_dir)

    for stem in ("background_climate_full", "background_climate_horizon"):
        assert (plots_dir / f"{stem}.png").exists()


def test_lookup_extreme_weather_uses_closest_available_scenario():
    costs = {
        ("ALB", "ssp2"): {2030: 2.7},
        ("ALB", "ssp5"): {2030: 3.0},
    }
    assert _lookup_extreme_weather(costs, "ALB", "ssp370", 2030) == pytest.approx(2.7)
    assert _lookup_extreme_weather(costs, "ALB", "ssp585", 2030) == pytest.approx(3.0)
    assert _lookup_extreme_weather(costs, "ALB", "ssp119", 2030) == pytest.approx(2.7)


def test_plot_socioeconomics(tmp_path: Path):
    plots_dir = tmp_path / "plots"
    settings = SummarySettings(
        years=[2030, 2040],
        output_directory=tmp_path,
        gdp_series={
            "policy_ssp245": {2030: 100.0, 2040: 120.0},
            "policy_ssp370": {2030: 105.0, 2040: 125.0},
        },
        population_series={
            "policy_ssp245": {2030: 7000.0, 2040: 7100.0},
            "policy_ssp370": {2030: 7100.0, 2040: 7200.0},
        },
        plot_start=2030,
        plot_end=2040,
        plot_format="png",
        climate_labels=["ssp245", "ssp370"],
    )
    _plot_socioeconomic_timeseries(settings)
    assert (plots_dir / "socioeconomics.png").exists()

from pathlib import Path

import pandas as pd
import pytest

from results_summary import (
    build_summary,
    write_summary_json,
    write_summary_text,
)


def _write_csv(path: Path, data: list[dict]) -> None:
    frame = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_build_summary_collects_metrics(tmp_path: Path):
    root = tmp_path

    config = {
        "economic_module": {
            "reference_scenario": "baseline",
            "evaluation_scenarios": ["policy"],
            "methods": {
                "run": "ramsey_discount",
            },
            "output_directory": "results/economic",
            "base_year": 2025,
            "aggregation": "average",
            "aggregation_horizon": {"start": 2025, "end": 2050},
            "data_sources": {
                "emission_root": str(root / "resources"),
                "temperature_root": str(root / "resources" / "climate"),
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
    }

    # Emission deltas
    _write_csv(
        root / "resources" / "policy" / "co2.csv",
        [
            {"year": 2025, "delta": 0.0},
            {"year": 2030, "delta": -10.0},
            {"year": 2050, "delta": -15.0},
        ],
    )
    _write_csv(
        root / "resources" / "baseline" / "co2.csv",
        [
            {"year": 2025, "delta": 0.0},
            {"year": 2030, "delta": 0.0},
            {"year": 2050, "delta": 0.0},
        ],
    )

    # Temperature deltas
    _write_csv(
        root / "resources" / "climate" / "policy_ssp245.csv",
        [
            {"year": 2030, "temperature_baseline": 3.0, "temperature_adjusted": 2.8},
            {"year": 2050, "temperature_baseline": 3.5, "temperature_adjusted": 3.2},
        ],
    )
    _write_csv(
        root / "resources" / "climate" / "baseline_ssp245.csv",
        [
            {"year": 2030, "temperature_baseline": 3.0, "temperature_adjusted": 3.0},
            {"year": 2050, "temperature_baseline": 3.5, "temperature_adjusted": 3.5},
        ],
    )

    # SCC timeseries
    _write_csv(
        root / "results" / "economic" / "scc_timeseries_ramsey_discount_policy_ssp245.csv",
        [
            {
                "year": 2030,
                "delta_damage_usd": -5.0e9,
                "delta_emissions_tco2": -1.0,
                "scc_usd_per_tco2": 50.0,
                "discounted_delta_usd": -5.0e9,
            },
            {
                "year": 2050,
                "delta_damage_usd": -7.5e9,
                "delta_emissions_tco2": -0.8,
                "scc_usd_per_tco2": 60.0,
                "discounted_delta_usd": -7.5e9,
            },
        ],
    )

    _write_csv(
        root / "results" / "economic" / "scc_summary.csv",
        [
            {
                "scenario": "policy_ssp245",
                "reference": "baseline_ssp245",
                "method": "ramsey_discount",
                "aggregation": "average",
                "scc_usd_per_tco2": 50.0,
                "base_year": 2025,
            }
        ],
    )

    # Mortality summary
    _write_csv(
        root / "results" / "air_pollution" / "policy" / "total_mortality_summary.csv",
        [
            {
                "year": 2030,
                "percent_change_mortality": -0.02,
                "baseline_deaths_per_year": 1000,
                "delta_deaths_per_year": -20,
                "new_deaths_per_year": 980,
            },
            {
                "year": 2050,
                "percent_change_mortality": -0.03,
                "baseline_deaths_per_year": 1000,
                "delta_deaths_per_year": -30,
                "new_deaths_per_year": 970,
            },
        ],
    )

    settings, methods, metrics_map = build_summary(root, config)

    assert settings.years == [2030, 2050]
    assert settings.aggregation_mode == "average"
    assert settings.aggregation_horizon == (2025, 2050)
    assert methods == ["ramsey_discount"]

    policy_key = "policy_ssp245"

    assert policy_key in metrics_map
    policy_metrics = metrics_map[policy_key]
    assert policy_metrics.emission_delta_mt[2030] == -10.0
    assert policy_metrics.emission_timeseries == {2025: 0.0, 2030: -10.0, 2050: -15.0}
    assert policy_metrics.temperature_timeseries is not None
    assert policy_metrics.temperature_timeseries[2030] == pytest.approx(-0.2)
    assert policy_metrics.scc_usd_per_tco2["ramsey_discount"][2030] == 50.0
    assert policy_metrics.damages_usd["ramsey_discount"][2050] == -7.5e9
    assert policy_metrics.mortality_delta[2050] == -30
    assert policy_metrics.scc_average["ramsey_discount"] == 50.0
    assert policy_metrics.damage_total_usd["ramsey_discount"] == pytest.approx(-12.5e9)
    assert "baseline_ssp245" not in metrics_map

    summary_txt = write_summary_text(
        settings, methods, metrics_map, output_path=root / "results" / "summary"
    )
    assert summary_txt.exists()
    content = summary_txt.read_text()
    assert "Scenario: policy_ssp245" in content
    assert "SCC average (2025-2050):" in content

    summary_json = write_summary_json(
        settings, methods, metrics_map, output_path=root / "results" / "summary"
    )
    assert summary_json.exists()
    data = summary_json.read_text()
    assert '"policy_ssp245"' in data

    # Re-run with per-year aggregation to ensure SCC series are reported for configured years.
    config["economic_module"]["aggregation"] = "per_year"
    settings_py, methods_py, metrics_map_py = build_summary(root, config)
    assert settings_py.aggregation_mode == "per_year"

    summary_txt_py = write_summary_text(
        settings_py, methods_py, metrics_map_py, output_path=root / "results" / "summary_per_year"
    )
    text_py = summary_txt_py.read_text()
    assert "SCC (USD/tCO₂):" in text_py
    assert "    ramsey_discount:" in text_py
    assert "      2030: 50.00" in text_py

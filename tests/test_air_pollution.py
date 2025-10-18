import json
import math
from pathlib import Path

import pandas as pd

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
            "country": ["Testland"],
            "count": [1],
            "mean": [20.0],
            "median": [20.0],
            "std": [None],
            "min": [20.0],
            "max": [20.0],
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
            "pollutants": {
                "pm25": {
                    "stats_file": str(stats_path),
                    "relative_risk": 1.08,
                    "reference_delta": 10.0,
                }
            },
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
    pct_change_2020 = math.exp(beta * (20.0 * ratio_2020 - 20.0)) - 1.0
    pct_change_2025 = math.exp(beta * (20.0 * ratio_2025 - 20.0)) - 1.0

    pct_by_year = dict(zip(df["year"], df["percent_change_mortality"], strict=False))
    assert math.isclose(pct_by_year[2020], pct_change_2020, rel_tol=1e-6)
    assert math.isclose(pct_by_year[2025], pct_change_2025, rel_tol=1e-6)

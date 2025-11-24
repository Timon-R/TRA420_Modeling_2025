from pathlib import Path

import numpy as np

from economic_module.socioeconomics import DiceSocioeconomics


def _dice_config() -> dict:
    base_path = Path.cwd()
    return {
        "start_year": 2019,
        "scenario": "SSP2",
        "population": {
            "initial_population_million": 7713.47,
            "logistic_growth": 0.028,
            "asymptote_table_path": "data/GDP_and_Population_data/DICE/dice_scenarios.csv",
        },
        "tfp": {
            "initial_level_label": "nordhaus",
            "decline_rate": 0.005,
            "scenario_growth_table_path": ("data/GDP_and_Population_data/DICE/dice_scenarios.csv"),
            "initial_levels_table_path": (
                "data/GDP_and_Population_data/DICE/dice_tfp_initial_levels.csv"
            ),
        },
        "capital": {
            "depreciation_rate": 0.0369791,
            "savings_rate": 0.223,
            "capital_output_ratio": 3.41,
            "initial_stock_trillions": None,
        },
        "capital_share": 0.36507,
        "__base_path__": base_path,
    }


def test_dice_projection_increases_gdp():
    cfg = _dice_config()
    base_path = cfg.pop("__base_path__")
    model = DiceSocioeconomics.from_config(cfg, base_path=base_path)
    frame = model.project(2030)

    assert {"year", "gdp_trillion_usd", "population_persons"}.issubset(frame.columns)
    assert np.all(frame["population_persons"].values >= 0)
    assert frame.iloc[-1]["gdp_trillion_usd"] > frame.iloc[0]["gdp_trillion_usd"]
    np.testing.assert_allclose(
        frame["population_persons"].to_numpy(),
        frame["population_million"].to_numpy() * 1e6,
    )

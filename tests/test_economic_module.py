from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from economic_module import (
    EconomicInputs,
    compute_damage_difference,
    compute_damages,
    compute_scc_constant_discount,
    compute_scc_ramsey_discount,
    damage_dice,
)
from economic_module.scc import (
    _compute_consumption_growth,
    _constant_discount_factors,
    _ramsey_discount_factors,
)


@pytest.fixture
def sample_inputs() -> EconomicInputs:
    years = np.array([2025, 2030])
    gdp = np.array([100.0, 110.0])
    temps = {
        "baseline": np.array([1.0, 1.2]),
        "policy": np.array([1.0, 1.1]),
    }
    emissions = {
        "baseline": np.array([0.0, 0.0]),
        "policy": np.array([-1.0, -1.0]),
    }
    population = np.array([50.0, 51.0])
    return EconomicInputs(
        years=years,
        gdp_trillion_usd=gdp,
        temperature_scenarios_c=temps,
        emission_scenarios_tco2=emissions,
        population_million=population,
    )


def test_compute_damages_returns_expected_columns(sample_inputs: EconomicInputs):
    df = compute_damages(
        sample_inputs, scenarios=["policy"], damage_func=damage_dice, damage_kwargs={"delta2": 0.0}
    )
    assert "damage_usd_policy" in df.columns
    damages = df["damage_usd_policy"].to_numpy()
    expected = np.zeros_like(damages)
    np.testing.assert_allclose(damages, expected)


def test_compute_damage_difference(sample_inputs: EconomicInputs):
    df = compute_damage_difference(sample_inputs, scenario="policy", reference="baseline")
    assert "delta_damage_usd" in df.columns
    np.testing.assert_allclose(df["delta_emissions_tco2"], [-1.0, -1.0])


def test_constant_discount_factors(sample_inputs: EconomicInputs):
    factors = _constant_discount_factors(sample_inputs.years, base_year=2025, rate=0.02)
    np.testing.assert_allclose(factors[0], 1.0)
    np.testing.assert_allclose(factors[1], 1.0 / 1.02**5)


def test_ramsey_discount_pipeline(sample_inputs: EconomicInputs):
    damages = np.zeros(sample_inputs.years.shape[0])
    consumption_pc, growth = _compute_consumption_growth(sample_inputs, damages)
    assert consumption_pc.shape == growth.shape
    factors = _ramsey_discount_factors(sample_inputs.years, 2025, growth, rho=0.01, eta=1.0)
    assert factors[0] == pytest.approx(1.0)


def test_compute_scc_constant_discount(sample_inputs: EconomicInputs):
    result = compute_scc_constant_discount(
        sample_inputs,
        scenario="policy",
        reference="baseline",
        base_year=2025,
        discount_rate=0.03,
        aggregation="average",
        add_tco2=-2.0,
        damage_kwargs={"delta2": 0.0},
    )
    assert not np.isnan(result.scc_usd_per_tco2)
    per_year = result.per_year
    assert "damage_attributed_usd" in per_year.columns
    assert "discounted_damage_attributed_usd" in per_year.columns
    np.testing.assert_allclose(
        per_year["discounted_damage_attributed_usd"].sum(),
        per_year["discounted_delta_usd"].sum(),
    )
    assert result.temperature_kernel is not None
    assert result.temperature_kernel.shape[0] == len(per_year)
    assert result.run_method == "kernel"


def test_compute_scc_ramsey_discount(sample_inputs: EconomicInputs):
    result = compute_scc_ramsey_discount(
        sample_inputs,
        scenario="policy",
        reference="baseline",
        base_year=2025,
        rho=0.01,
        eta=1.0,
        aggregation="average",
        add_tco2=-2.0,
        damage_kwargs={"delta2": 0.0},
    )
    assert result.per_year["discount_factor"].iloc[0] == pytest.approx(1.0)
    assert result.run_method == "kernel"


def test_from_csv_automatically_loads_ssp_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    temp_path = tmp_path / "temp.csv"
    emission_path = tmp_path / "emission.csv"

    temp_df = pd.DataFrame(
        {
            "year": [2025, 2030],
            "temperature_adjusted": [1.2, 1.3],
            "temperature_baseline": [1.1, 1.2],
            "temperature_delta": [0.1, 0.1],
            "climate_scenario": ["ssp245", "ssp245"],
        }
    )
    temp_df.to_csv(temp_path, index=False)

    emission_df = pd.DataFrame({"year": [2025, 2030], "delta": [-1.0, -1.0]})
    emission_df.to_csv(emission_path, index=False)

    gdp_series = pd.Series([80_000, 90_000], index=[2025, 2030])
    pop_series = pd.Series([7000, 7100], index=[2025, 2030])

    def fake_loader(ssp_family: str, directory: Path):
        assert ssp_family == "SSP2"
        return gdp_series / 1000.0, pop_series

    monkeypatch.setattr("economic_module.scc._load_ssp_economic_data", fake_loader)

    inputs = EconomicInputs.from_csv(
        {"policy": temp_path},
        {"policy": emission_path},
        gdp_path=None,
        temperature_column="temperature_adjusted",
        emission_column="delta",
        emission_to_tonnes=1e6,
        gdp_population_directory=tmp_path,
    )

    assert inputs.ssp_family == "SSP2"
    np.testing.assert_array_equal(inputs.years, np.arange(2025, 2031))
    np.testing.assert_allclose(inputs.gdp_trillion_usd[[0, -1]], np.array([80.0, 90.0]))
    assert inputs.population_million is not None
    np.testing.assert_allclose(inputs.population_million[[0, -1]], np.array([7000.0, 7100.0]))

from pathlib import Path

import pandas as pd
import pytest
from scripts import run_calc_emissions_all

from calc_emissions.calculator import EmissionScenarioResult
from calc_emissions.writers import write_per_country_results


def test_write_per_country_results_writes_csvs(tmp_path: Path):
    # Build two minimal EmissionScenarioResult objects for a country
    years = [2025, 2030]
    baseline_totals = pd.Series([10.0, 12.0], index=years)
    scenario_totals = pd.Series([8.0, 11.0], index=years)

    baseline = EmissionScenarioResult(
        name="mixA__base_demand",
        demand_case="base_demand",
        mix_case="mixA",
        years=years,
        demand_twh=pd.Series([5.0, 6.0], index=years),
        generation_twh=pd.DataFrame({"coal": [3.0, 3.5]}, index=years),
        technology_emissions_mt={"co2": pd.DataFrame({"coal": [3.0, 3.5]}, index=years)},
        total_emissions_mt={"co2": baseline_totals},
        delta_mtco2=pd.Series([0.0, 0.0], index=years),
    )

    scenario = EmissionScenarioResult(
        name="mixA__scen1",
        demand_case="scen1",
        mix_case="mixA",
        years=years,
        demand_twh=pd.Series([5.0, 6.0], index=years),
        generation_twh=pd.DataFrame({"coal": [2.5, 3.0]}, index=years),
        technology_emissions_mt={"co2": pd.DataFrame({"coal": [2.5, 3.0]}, index=years)},
        total_emissions_mt={"co2": scenario_totals},
        delta_mtco2=scenario_totals - baseline_totals,
    )

    per_country_map = {"Testland": {baseline.name: baseline, scenario.name: scenario}}

    # Call the writer imported from the src package
    resources_root = tmp_path / "results" / "emissions"
    write_per_country_results(per_country_map, resources_root)

    co2_csv = resources_root / "mixA" / "Testland" / "co2.csv"
    assert co2_csv.exists()
    # Read while skipping commented header line
    df = pd.read_csv(co2_csv, comment="#")
    assert list(df["year"]) == years
    assert "absolute_base_demand" in df.columns
    assert "absolute_scen1" in df.columns
    assert "delta_scen1" in df.columns
    assert df["absolute_scen1"].iloc[0] == 8.0
    assert pytest.approx(df["delta_scen1"].iloc[0]) == pytest.approx(8.0 - 10.0)


def test_build_aggregated_results_creates_scenario_names():
    years = [2025]
    base = EmissionScenarioResult(
        name="mixA__base_demand",
        demand_case="base_demand",
        mix_case="mixA",
        years=years,
        demand_twh=pd.Series([5.0], index=years),
        generation_twh=pd.DataFrame({"coal": [3.0]}, index=years),
        technology_emissions_mt={"co2": pd.DataFrame({"coal": [3.0]}, index=years)},
        total_emissions_mt={"co2": pd.Series([10.0], index=years)},
        delta_mtco2=pd.Series([0.0], index=years),
    )
    scen = EmissionScenarioResult(
        name="mixA__scen1",
        demand_case="scen1",
        mix_case="mixA",
        years=years,
        demand_twh=pd.Series([6.0], index=years),
        generation_twh=pd.DataFrame({"coal": [2.5]}, index=years),
        technology_emissions_mt={"co2": pd.DataFrame({"coal": [2.5]}, index=years)},
        total_emissions_mt={"co2": pd.Series([8.0], index=years)},
        delta_mtco2=pd.Series([-2.0], index=years),
    )
    aggregated = run_calc_emissions_all._build_aggregated_results(
        [{"mixA__base_demand": base, "mixA__scen1": scen}],
        baseline_case="base_demand",
    )
    assert {"mixA__base_demand", "mixA__scen1"} <= set(aggregated.keys())
    delta = aggregated["mixA__scen1"].delta_mtco2.iloc[0]
    assert pytest.approx(delta) == pytest.approx(-2.0)

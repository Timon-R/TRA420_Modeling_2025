from pathlib import Path

import pandas as pd

from calc_emissions.calculator import EmissionScenarioResult
from calc_emissions.writers import write_per_country_results


def test_write_per_country_results_writes_csvs(tmp_path: Path):
    # Build two minimal EmissionScenarioResult objects for a country
    years = [2025, 2030]
    baseline_totals = pd.Series([10.0, 12.0], index=years)
    scenario_totals = pd.Series([8.0, 11.0], index=years)

    baseline = EmissionScenarioResult(
        name="baseline",
        years=years,
        demand_twh=pd.Series([5.0, 6.0], index=years),
        generation_twh=pd.DataFrame({"coal": [3.0, 3.5]}, index=years),
        technology_emissions_mt={"co2": pd.DataFrame({"coal": [3.0, 3.5]}, index=years)},
        total_emissions_mt={"co2": baseline_totals},
        delta_mtco2=pd.Series([0.0, 0.0], index=years),
    )

    scenario = EmissionScenarioResult(
        name="scenario_1",
        years=years,
        demand_twh=pd.Series([5.0, 6.0], index=years),
        generation_twh=pd.DataFrame({"coal": [2.5, 3.0]}, index=years),
        technology_emissions_mt={"co2": pd.DataFrame({"coal": [2.5, 3.0]}, index=years)},
        total_emissions_mt={"co2": scenario_totals},
        delta_mtco2=scenario_totals - baseline_totals,
    )

    per_country_map = {"Testland": {"baseline": baseline, "scenario_1": scenario}}

    # Call the writer imported from the src package
    resources_root = tmp_path / "resources"
    write_per_country_results(per_country_map, resources_root)

    # Assert files exist and contain expected deltas (scenario - baseline)
    co2_csv = resources_root / "Testland" / "scenario_1" / "co2.csv"
    assert co2_csv.exists()
    df = pd.read_csv(co2_csv)
    assert list(df["year"]) == years
    # Expected delta: [-2.0, -1.0]
    assert df["delta"].iloc[0] == 8.0 - 10.0
    assert df["delta"].iloc[1] == 11.0 - 12.0

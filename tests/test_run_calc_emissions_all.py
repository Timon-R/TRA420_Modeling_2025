from pathlib import Path

import pandas as pd
from scripts import run_calc_emissions_all

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

    # New format: per-country map keys are '<mix>/<demand>' and writer produces
    # a single per-mix pollutant CSV under resources/<Country>/<mix>/<pollutant>.csv
    per_country_map = {"Testland": {"mixA/baseline": baseline, "mixA/scenario_1": scenario}}

    # Call the writer imported from the src package
    resources_root = tmp_path / "resources"
    write_per_country_results(per_country_map, resources_root)

    # Assert files exist and contain expected absolute and delta columns
    co2_csv = resources_root / "Testland" / "mixA" / "co2.csv"
    assert co2_csv.exists()
    # Read while skipping commented header line
    df = pd.read_csv(co2_csv, comment="#")
    assert list(df["year"]) == years
    # Expected absolute and delta values
    assert df["absolute_baseline"].iloc[0] == 10.0
    assert df["absolute_baseline"].iloc[1] == 12.0
    assert df["absolute_scenario_1"].iloc[0] == 8.0
    assert df["absolute_scenario_1"].iloc[1] == 11.0
    assert df["delta_scenario_1"].iloc[0] == 8.0 - 10.0
    assert df["delta_scenario_1"].iloc[1] == 11.0 - 12.0


def test_canonicalize_baseline_demands_handles_alias():
    years = [2025]
    totals = pd.Series([10.0], index=years)
    baseline = EmissionScenarioResult(
        name="reference_case",
        years=years,
        demand_twh=pd.Series([5.0], index=years),
        generation_twh=pd.DataFrame({"coal": [3.0]}, index=years),
        technology_emissions_mt={"co2": pd.DataFrame({"coal": [3.0]}, index=years)},
        total_emissions_mt={"co2": totals},
        delta_mtco2=pd.Series([0.0], index=years),
    )
    per_mix = {"mixA/reference": baseline}
    renamed = run_calc_emissions_all._canonicalize_baseline_demands(per_mix, "reference")
    assert "mixA/baseline" in renamed
    assert "mixA/reference" not in renamed


def test_apply_scenario_aliases_maps_named_cases():
    years = [2025]
    totals = pd.Series([12.0], index=years)
    scenario = EmissionScenarioResult(
        name="reference/lower_bound",
        years=years,
        demand_twh=pd.Series([6.0], index=years),
        generation_twh=pd.DataFrame({"coal": [3.0]}, index=years),
        technology_emissions_mt={"co2": pd.DataFrame({"coal": [3.0]}, index=years)},
        total_emissions_mt={"co2": totals},
        delta_mtco2=pd.Series([-1.0], index=years),
    )
    aggregated = {"reference/lower_bound": scenario}
    aliases = {"scenario_1_lower_bound": ("reference", "lower_bound")}
    run_calc_emissions_all._apply_scenario_aliases(aggregated, aliases)
    assert aggregated["scenario_1_lower_bound"] is scenario

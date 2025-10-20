import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCC_PATH = REPO_ROOT / "scripts" / "run_scc.py"

spec = importlib.util.spec_from_file_location("run_scc_module", RUN_SCC_PATH)
run_scc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_scc)  # type: ignore[attr-defined]


def _write_csv(path: Path, data: list[dict]) -> None:
    frame = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_config(tmp_path: Path, include_horizon: bool) -> dict:
    years = [2025, 2030]

    gdp_path = tmp_path / "gdp.csv"
    _write_csv(
        gdp_path,
        [
            {"year": year, "gdp_trillion_usd": 1.0 + 0.1 * idx, "population_million": 5.0}
            for idx, year in enumerate(years)
        ],
    )

    emission_root = tmp_path / "emissions"
    temperature_root = tmp_path / "climate"
    for scenario, deltas in {
        "baseline": [0.0, 0.0],
        "policy": [0.0, -5.0],
    }.items():
        _write_csv(
            emission_root / scenario / "co2.csv",
            [
                {"year": year, "delta": value if idx else 0.0}
                for idx, (year, value) in enumerate(zip(years, deltas, strict=False))
            ],
        )

    _write_csv(
        temperature_root / "baseline_ssp245.csv",
        [
            {
                "year": year,
                "temperature_adjusted": 3.0 + 0.1 * idx,
                "temperature_baseline": 3.0 + 0.1 * idx,
                "climate_scenario": "ssp245",
            }
            for idx, year in enumerate(years)
        ],
    )
    _write_csv(
        temperature_root / "policy_ssp245.csv",
        [
            {
                "year": year,
                "temperature_adjusted": 2.8 + 0.1 * idx,
                "temperature_baseline": 3.0 + 0.1 * idx,
                "climate_scenario": "ssp245",
            }
            for idx, year in enumerate(years)
        ],
    )

    output_dir = tmp_path / "results" / "economic"

    economic_cfg = {
        "gdp_series": str(gdp_path),
        "reference_scenario": "baseline",
        "evaluation_scenarios": ["policy"],
        "base_year": 2025,
        "aggregation": "average",
        "methods": {"run": "ramsey_discount"},
        "output_directory": str(output_dir),
        "data_sources": {
            "emission_root": str(emission_root),
            "temperature_root": str(temperature_root),
            "climate_scenarios": ["ssp245"],
        },
    }

    if include_horizon:
        economic_cfg["aggregation_horizon"] = {"start": 2025, "end": 2030}

    config = {
        "time_horizon": {"start": 2025, "end": 2030, "step": 5},
        "economic_module": economic_cfg,
    }
    return config


@pytest.mark.parametrize("include_horizon", [False, True])
def test_run_scc_aggregation_horizon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, include_horizon: bool, capsys
):
    config = _build_config(tmp_path, include_horizon=include_horizon)

    monkeypatch.setattr(run_scc, "_load_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["run_scc.py"])

    if not include_horizon:
        with pytest.raises(SystemExit):
            run_scc.main()
        captured = capsys.readouterr()
        assert "aggregation" in captured.err.lower()
    else:
        run_scc.main()
        output_dir = Path(config["economic_module"]["output_directory"])
        timeseries = output_dir / "scc_timeseries_ramsey_discount_policy_ssp245.csv"
        assert timeseries.exists()
        df = pd.read_csv(timeseries)
        assert set(df["year"].astype(int)) == {2025, 2030}

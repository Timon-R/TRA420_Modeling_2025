import numpy as np
import pytest

try:  # pragma: no cover - handled via pytest skip
    from climate_module import TemperatureResult
    from climate_module.scenario_runner import (
        DEFAULT_TIME_CONFIG,
        ScenarioSpec,
        _build_timepoints,
        _prepare_adjustments,
        run_scenarios,
        step_change,
    )
except ImportError:
    pytest.skip("FaIR dependency not available.", allow_module_level=True)


def test_step_change_returns_callable_with_expected_shape():
    builder = step_change(-2.0, start_year=2025.0)
    timepoints = np.array([2024.5, 2025.5, 2026.5])
    cfg = DEFAULT_TIME_CONFIG
    delta = builder(timepoints, cfg)
    np.testing.assert_allclose(delta, np.array([0.0, -0.002, -0.002]))


def test_build_timepoints_matches_config():
    cfg = {"start_year": 2020.0, "end_year": 2022.0, "timestep": 1.0}
    timepoints = _build_timepoints(cfg)
    np.testing.assert_allclose(timepoints, np.array([2020.5, 2021.5]))


def test_prepare_adjustments_handles_callable_and_sequence():
    timepoints = np.array([2020.5, 2021.5])
    cfg = DEFAULT_TIME_CONFIG
    adjustments = _prepare_adjustments({"CO2 FFI": [1000.0, 2000.0]}, timepoints, cfg)
    np.testing.assert_allclose(adjustments["CO2 FFI"], np.array([1.0, 2.0]))

    builder = step_change(-2.0, start_year=2025.0)
    adjustments = _prepare_adjustments({"CO2 FFI": builder}, timepoints, cfg)
    assert isinstance(adjustments["CO2 FFI"], np.ndarray)


def test_run_scenarios_invokes_compute_temperature_change(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_compute_temperature_change(**kwargs):
        calls.append(kwargs)
        return TemperatureResult(
            years=np.array([2020, 2025]),
            timepoints=np.array([2020.5, 2025.5]),
            baseline=np.zeros(2),
            adjusted=np.ones(2),
        )

    monkeypatch.setattr(
        "climate_module.scenario_runner.compute_temperature_change", fake_compute_temperature_change
    )

    spec = ScenarioSpec(
        label="case",
        scenario="ssp245",
        emission_adjustments={"CO2 FFI": [0.0, 0.0]},
        start_year=2020.0,
        end_year=2025.0,
        timestep=5.0,
        compute_kwargs={},
    )

    results = run_scenarios([spec])
    assert "case" in results
    assert len(calls) == 1
    assert calls[0]["scenario"] == "ssp245"

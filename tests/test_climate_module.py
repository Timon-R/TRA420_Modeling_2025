import numpy as np
import pandas as pd
import pytest

fair = pytest.importorskip("fair")

from climate_module.FaIR import (
    TemperatureResult,
    _broadcast_matrix,
    _broadcast_vector,
    _to_delta_array,
)


def test_temperature_result_properties():
    result = TemperatureResult(
        years=np.array([2020, 2025]),
        timepoints=np.array([2020.5, 2025.5]),
        baseline=np.array([1.0, 1.2]),
        adjusted=np.array([1.1, 1.5]),
    )
    np.testing.assert_allclose(result.delta, np.array([0.1, 0.3]))
    assert result.final_delta == pytest.approx(0.3)
    frame = result.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == [
        "year",
        "temperature_baseline",
        "temperature_adjusted",
        "temperature_delta",
    ]


def test_broadcast_matrix_handles_scalars_vectors_and_matrices():
    matrix = _broadcast_matrix(5.0, 2)
    np.testing.assert_allclose(matrix, np.array([[5.0], [5.0]]))

    matrix = _broadcast_matrix([5.0, 3.0], 2)
    np.testing.assert_allclose(matrix, np.array([[5.0, 3.0], [5.0, 3.0]]))

    preset = np.array([[1.0, 2.0], [3.0, 4.0]])
    matrix = _broadcast_matrix(preset, 2)
    np.testing.assert_allclose(matrix, preset)


def test_broadcast_vector_expands_scalars_to_configs():
    vec = _broadcast_vector(2.0, 3)
    np.testing.assert_allclose(vec, np.array([2.0, 2.0, 2.0]))

    vec = _broadcast_vector([1.0, 2.0, 3.0], 3)
    np.testing.assert_allclose(vec, np.array([1.0, 2.0, 3.0]))


def test_to_delta_array_accepts_scalar_and_array():
    timepoints = np.array([2020.5, 2021.5])
    scalar = _to_delta_array(1.5, timepoints)
    np.testing.assert_allclose(scalar, np.array([1.5, 1.5]))

    arr = _to_delta_array([0.1, 0.2], timepoints)
    np.testing.assert_allclose(arr, np.array([0.1, 0.2]))

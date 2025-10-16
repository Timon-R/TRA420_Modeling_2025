import numpy as np

from economic_module import damage_dice


def test_damage_matches_quadratic_without_extensions():
    temps = np.array([0.0, 1.0, 2.0, 3.0])
    baseline = damage_dice(
        temps,
        use_threshold=False,
        use_saturation=False,
        use_catastrophic=False,
        max_fraction=1.0,
    )
    expected = 0.002 * temps**2
    np.testing.assert_allclose(baseline, expected)


def test_threshold_amplification_increases_high_temperature_damages():
    temps = np.array([0.0, 1.0, 2.0, 3.0])
    result = damage_dice(
        temps,
        delta2=0.01,
        use_threshold=True,
        threshold_temperature=1.0,
        threshold_scale=1.0,
        threshold_power=1.0,
        use_saturation=False,
        use_catastrophic=False,
        max_fraction=5.0,
    )
    baseline = 0.01 * temps**2
    expected = baseline * (1.0 + np.maximum(0.0, temps - 1.0))
    np.testing.assert_allclose(result, expected)


def test_saturation_keeps_damage_below_max_fraction():
    temps = np.array([0.0, 5.0, 10.0, 20.0])
    result = damage_dice(
        temps,
        delta2=0.1,
        use_saturation=True,
        max_fraction=0.4,
    )
    assert np.all(result <= 0.4 + 1e-12)
    assert np.all(result >= 0.0)


def test_catastrophic_step_adds_fraction_above_threshold():
    temps = np.array([4.5, 5.0, 6.0])
    disaster_fraction = 0.3
    result = damage_dice(
        temps,
        delta2=0.0,
        use_catastrophic=True,
        catastrophic_temperature=5.0,
        disaster_fraction=disaster_fraction,
        disaster_mode="step",
        use_threshold=False,
        use_saturation=False,
        max_fraction=1.0,
    )
    expected = np.array([0.0, disaster_fraction, disaster_fraction])
    np.testing.assert_allclose(result, expected)

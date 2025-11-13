"""Test profile visualization calculations independently from HCA."""
import numpy as np
import pytest
from trace_calc.services.profile_data_calculator import ProfileDataCalculator


def test_curved_profile_applies_curvature_in_meters():
    """Verify curvature correction is properly converted from km to meters."""
    # Flat terrain profile
    distances_km = np.array([0, 50, 100])
    elevations_m = np.array([100.0, 100.0, 100.0])

    calc = ProfileDataCalculator(distances_km, elevations_m)
    curved, baseline = calc.curved_profile()

    # At 50km from the center, the drop is d^2 / (2*R) = 50^2 / (2 * 6371) = 0.196 km = 196 m
    expected_drop_m = (50**2 / (2 * 6371)) * 1000

    # Midpoint should be relatively unchanged (small distance from itself)
    assert abs(curved[1] - elevations_m[1]) < 0.1, "Midpoint should be near original"

    # Edges should drop significantly due to Earth curvature
    actual_drop_left = elevations_m[0] - curved[0]
    actual_drop_right = elevations_m[2] - curved[2]

    assert abs(actual_drop_left - expected_drop_m) < 1, \
        f"Left edge drop should be ~{expected_drop_m:.2f}m, got {actual_drop_left:.2f}m"
    assert abs(actual_drop_right - expected_drop_m) < 1, \
        f"Right edge drop should be ~{expected_drop_m:.2f}m, got {actual_drop_right:.2f}m"


def test_curved_profile_symmetric_around_midpoint():
    """Curvature should be symmetric around the midpoint."""
    distances_km = np.linspace(0, 100, 256)
    elevations_m = np.ones(256) * 150.0

    calc = ProfileDataCalculator(distances_km, elevations_m)
    curved, _ = calc.curved_profile()

    mid_idx = len(curved) // 2

    # Points equidistant from midpoint should have same curvature
    for offset in [10, 50, 100]:
        if mid_idx - offset >= 0 and mid_idx + offset < len(curved):
            left_drop = elevations_m[mid_idx - offset] - curved[mid_idx - offset]
            right_drop = elevations_m[mid_idx + offset] - curved[mid_idx + offset]
            assert abs(left_drop - right_drop) < 1.0, \
                f"Curvature should be symmetric at offset {offset}"


def test_profile_calculator_independent_from_hca():
    """
    Verify profile calculation doesn't affect HCA values.
    This test documents architectural separation.
    """
    from trace_calc.models.input_data import InputData
    from trace_calc.models.path import PathData
    from trace_calc.services.analyzers import GrozaAnalyzer

    # Use same test data as HCA tests
    np.random.seed(42)
    random_array = np.random.rand(256)
    scaled_array = random_array * 40 + 135
    test_elevations = np.convolve(scaled_array, np.ones(5)/5, mode="same")
    test_elevations[0] += 70
    test_elevations[-1] += 70

    test_profile = PathData(
        coordinates=np.linspace(123.10, 234.50, 256),
        distances=np.linspace(0, 100, 256),
        elevations=test_elevations,
    )

    analyzer = GrozaAnalyzer(test_profile, InputData("test_file"))

    # HCA should still be correct (independent of profile bug fix)
    assert f"{analyzer.hca_data.b_sum:.3f}" == "0.552", \
        "HCA calculation must remain unchanged"

    # Profile should now have correct units
    assert analyzer.profile_data is not None, "Profile should be calculated"

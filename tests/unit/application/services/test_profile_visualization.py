"""Test profile visualization calculations independently from HCA."""

import numpy as np
from trace_calc.application.services.profile_data_calculator import (
    ProfileDataCalculator,
)


def test_curved_profile_applies_curvature_in_meters():
    """Verify curvature correction is properly converted from km to meters."""
    # Flat terrain profile
    distances_km = np.array([0, 50, 100])
    elevations_m = np.array([100.0, 100.0, 100.0])

    calc = ProfileDataCalculator(distances_km, elevations_m)
    curved, baseline = calc.curved_profile()

    # At 50km from the center, the bulge is x*(d-x)/(2*R) = 50*(100-50)/(2*6371) = 0.196 km = 196 m
    expected_bulge_m = (50.0 * (100.0 - 50.0) / (2 * 6371.0)) * 1000

    # Midpoint should now be higher due to the bulge
    assert np.isclose(curved[1], elevations_m[1] + expected_bulge_m, atol=0.1), "Midpoint should show expected bulge"

    # Edges should be at original elevation (0 bulge)
    assert np.isclose(curved[0], elevations_m[0], atol=0.1), "Left edge should have zero bulge"
    assert np.isclose(curved[2], elevations_m[2], atol=0.1), "Right edge should have zero bulge"


def test_curved_profile_symmetric_around_midpoint():
    """Curvature should be symmetric around the midpoint."""
    distances_km = np.linspace(0, 100, 256)
    elevations_m = np.ones(256) * 150.0

    calc = ProfileDataCalculator(distances_km, elevations_m)
    curved, _ = calc.curved_profile()

    # Check symmetry by comparing points equidistant from the start/end
    for i in range(len(curved) // 2):
        left_bulge = curved[i] - elevations_m[i]
        right_bulge = curved[len(curved) - 1 - i] - elevations_m[len(curved) - 1 - i]
        assert np.isclose(left_bulge, right_bulge, atol=0.01), ( # Use a tighter atol
            f"Curvature bulge should be symmetric at index {i}"
        )

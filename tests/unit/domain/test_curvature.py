# tests/unit/domain/test_curvature.py
import numpy as np
import pytest

from trace_calc.domain.curvature import apply_geometric_curvature, calculate_earth_drop
from trace_calc.domain.constants import EARTH_RADIUS_KM

def test_apply_geometric_curvature_upward_bulge():
    """
    Test that apply_geometric_curvature produces an upward bulge relative to endpoints.
    """
    # Arrange: A simple 100km path
    distances_km = np.array([0.0, 25.0, 50.0, 75.0, 100.0])

    # Act
    curvature = apply_geometric_curvature(distances_km)

    # Assert
    # The curve should be 0 at the start and end
    assert np.isclose(curvature[0], 0.0)
    assert np.isclose(curvature[-1], 0.0)

    # The curve should be positive in the middle and symmetrical
    assert curvature[2] > 0  # Max height at the midpoint
    assert np.isclose(curvature[1], curvature[3])
    assert curvature[1] > 0
    assert curvature[2] > curvature[1]

    # Check the midpoint height with the known formula h = x*(d-x)/(2R)
    # x=50, d=100, R=6371.0
    expected_midpoint_height = (50.0 * (100.0 - 50.0)) / (2 * EARTH_RADIUS_KM) * 1000
    assert np.isclose(curvature[2], expected_midpoint_height, atol=0.1)

def test_apply_geometric_curvature_empty_input():
    """
    Test that apply_geometric_curvature handles empty input gracefully.
    """
    # Arrange
    distances_km = np.array([])

    # Act
    curvature = apply_geometric_curvature(distances_km)

    # Assert
    assert curvature.shape == (0,)

def test_apply_geometric_curvature_single_point():
    """
    Test that apply_geometric_curvature handles a single point path.
    """
    # Arrange
    distances_km = np.array([50.0])

    # Act
    curvature = apply_geometric_curvature(distances_km)

    assert np.isclose(curvature[0], 0.0)

def test_calculate_earth_drop():
    """
    Test that calculate_earth_drop correctly computes the Earth's drop from a tangent.
    """
    # Arrange
    distances_km = np.array([0.0, 10.0, 50.0, 100.0])
    R = EARTH_RADIUS_KM

    # Act
    drop = calculate_earth_drop(distances_km)

    # Assert
    # Expected drop = x^2 / (2 * R) * 1000 (to meters)
    expected_drop_0km = (0.0**2 / (2 * R)) * 1000
    expected_drop_10km = (10.0**2 / (2 * R)) * 1000
    expected_drop_50km = (50.0**2 / (2 * R)) * 1000
    expected_drop_100km = (100.0**2 / (2 * R)) * 1000

    assert np.isclose(drop[0], expected_drop_0km)
    assert np.isclose(drop[1], expected_drop_10km)
    assert np.isclose(drop[2], expected_drop_50km)
    assert np.isclose(drop[3], expected_drop_100km)



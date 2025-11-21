# trace_calc/domain/curvature.py
import numpy as np
from numpy.typing import NDArray

from trace_calc.domain.constants import EARTH_RADIUS_KM, CURVATURE_SCALE
from trace_calc.domain.models.units import Kilometers


def apply_geometric_curvature(distances_km: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply geometric Earth curvature correction, showing an upward bulge.

    Calculates curvature relative to a straight line between the path's endpoints.

    Args:
        distances_km: 1D array of distances along the path in kilometers.

    Returns:
        1D array of curvature corrections in meters (positive values).
    """
    if distances_km.size < 2:
        return np.zeros_like(distances_km, dtype=float)

    # Earth radius in km
    R = EARTH_RADIUS_KM

    # Calculate curvature relative to straight line between endpoints
    # Formula: h = x * (d - x) / (2 * R)
    # where x is distance from start, d is total distance
    d = distances_km[-1] - distances_km[0]  # Total path length
    x = distances_km - distances_km[0]  # Distance from start point

    # Curvature correction (positive = terrain bulges above straight line)
    # The result of the division is in km, so multiply by 1000 to get meters.
    curvature = x * (d - x) / (2 * R) * 1000

    return curvature


def get_empirical_curvature_correction(distance_km: Kilometers) -> float:
    """
    Calculates the empirical curvature correction.

    Args:
        distance_km: Distance from the site in kilometers.

    Returns:
        A correction factor.
    """
    return (distance_km**2) / CURVATURE_SCALE


def calculate_earth_drop(distance_km: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate the drop of the Earth's surface from a tangent line at the start.

    Args:
        distance_km: 1D array of distances from the starting point in kilometers.

    Returns:
        1D array of vertical drops in meters.
    """
    R = EARTH_RADIUS_KM  # Earth radius in km
    # Formula: drop = x^2 / (2 * R)
    # The result of the division is in km, so multiply by 1000 to get meters.
    drop = (distance_km**2) / (2 * R) * 1000
    return drop

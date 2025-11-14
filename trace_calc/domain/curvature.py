# trace_calc/domain/curvature.py
import numpy as np
from numpy.typing import NDArray

from trace_calc.domain.models.units import Kilometers
from trace_calc.domain.constants import CURVATURE_SCALE, GEOMETRIC_CURVATURE_SCALE


def apply_geometric_curvature(distances_km: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculates geometric Earth curvature drop in meters from a tangent at the path's midpoint.

    The result of the division is treated as being in kilometers, and then converted to meters.

    Args:
        distances_km: 1D array of distances along the path in kilometers.

    Returns:
        1D array of curvature corrections in meters.
    """
    if distances_km.size == 0:
        return np.array([], dtype=np.float64)

    mid_idx = distances_km.size // 2

    # The formula `d**2 / SCALE` is assumed to result in kilometers.
    curve_km: NDArray[np.float64] = (
        -((distances_km - distances_km[mid_idx]) ** 2) / GEOMETRIC_CURVATURE_SCALE
    )

    # Shift the curve so the peak (at midpoint) is at zero, representing the tangent line.
    # The other points will have negative values, representing the drop.
    curve_km -= curve_km.max()

    return curve_km * 1000


def get_empirical_curvature_correction(distance_km: Kilometers) -> float:
    """
    Calculates the empirical curvature correction.

    Args:
        distance_km: Distance from the site in kilometers.

    Returns:
        A correction factor.
    """
    return (distance_km**2) / CURVATURE_SCALE

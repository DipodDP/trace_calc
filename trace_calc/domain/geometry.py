import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


def rotate_line_by_angle(
    line_coeffs: NDArray[np.float64],
    pivot_point: tuple[float, float],
    angle_degrees: float,
) -> NDArray[np.float64]:
    """Rotate a line equation by a given angle around a pivot point.

    Args:
        line_coeffs: A numpy array [k, b] for the line y = kx + b.
        pivot_point: A tuple (x0, y0) representing the pivot for rotation.
        angle_degrees: The rotation angle in degrees (positive is counter-clockwise).

    Returns:
        A numpy array [k_new, b_new] for the rotated line.

    Raises:
        ValueError: If the input line is too steep, the resulting line is too
                    steep, or the angle is out of the valid range [-90, 90].
    """
    if not -90 <= angle_degrees <= 90:
        raise ValueError("Angle must be between -90 and 90 degrees")

    k, b = line_coeffs
    x0, y0 = pivot_point

    if abs(k) > 1e6:
        raise ValueError("Cannot rotate vertical line")

    # CRITICAL: Convert slope from m/km to m/m for geometric calculations
    k_actual = k / 1000.0

    alpha = np.arctan(k_actual)
    theta = np.deg2rad(angle_degrees)

    # Determine rotation direction based on slope sign
    # For descending lines (negative slope), subtract offset to keep line above
    # For ascending lines (positive slope), add offset to make steeper
    if k < 0:  # Descending line
        new_angle_rad = alpha - theta
    else:  # Ascending or horizontal line
        new_angle_rad = alpha + theta

    k_new_actual = np.tan(new_angle_rad)

    # Convert slope back to m/km
    k_new = k_new_actual * 1000.0

    if abs(k_new) > 1e6:
        raise ValueError("Near-vertical result")

    b_new = y0 - k_new * x0

    return np.array([k_new, b_new])


def find_line_intersection(
    line1_coeffs: NDArray[np.float64], line2_coeffs: NDArray[np.float64]
) -> tuple[float, float]:
    """
    Find the intersection point of two lines.

    Args:
        line1_coeffs: Coefficients [k, b] for line 1.
        line2_coeffs: Coefficients [k, b] for line 2.

    Returns:
        A tuple (x, y) for the intersection point.

    Raises:
        ValueError: If lines are parallel or coincident.
    """
    k1, b1 = line1_coeffs
    k2, b2 = line2_coeffs

    if np.isclose(k1, k2):
        if np.isclose(b1, b2):
            raise ValueError("Lines are Coincident")
        raise ValueError("Lines are Parallel")

    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1

    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError("Intersection calculation resulted in non-finite values.")

    return x, y


def calculate_height_above_terrain(
    distance_km: float,
    elevation_sea_level: float,
    distances: NDArray[np.float64],
    elevations: NDArray[np.float64],
) -> float:
    """
    Interpolate terrain elevation and compute relative height.

    Args:
        distance_km: The distance along the path to calculate the height at.
        elevation_sea_level: The absolute elevation of the point in question.
        distances: An array of distances along the path.
        elevations: An array of terrain elevations corresponding to the distances.

    Returns:
        The height of the point above the terrain.

    Raises:
        ValueError: If the distance is outside the bounds of the path.
    """
    if not (distances[0] <= distance_km <= distances[-1]):
        raise ValueError("Distance is outside the bounds of the path")

    terrain_elev = np.interp(distance_km, distances, elevations)
    height = elevation_sea_level - terrain_elev

    if not np.isfinite(height):
        raise ValueError("Height calculation resulted in a non-finite value.")

    return height


def calculate_cone_intersection_volume(
    lower_a: NDArray[np.float64],
    lower_b: NDArray[np.float64],
    upper_a: NDArray[np.float64],
    upper_b: NDArray[np.float64],
    distances: NDArray[np.float64],
    lower_intersection_x: float,
    upper_intersection_x: float,
    cross_ab_x: float,
    cross_ba_x: float,
) -> float:
    """
    Numerically integrate the volume of the 3D region bounded by four sight lines.

    Args:
        lower_a, lower_b, upper_a, upper_b: Coefficients for the four lines.
        distances: The array of distances along the path.
        lower_intersection_x: x-coordinate of the lower intersection.
        upper_intersection_x: x-coordinate of the upper intersection.
        cross_ab_x: x-coordinate of the intersection of upper_a and lower_b.
        cross_ba_x: x-coordinate of the intersection of upper_b and lower_a.

    Returns:
        The calculated volume in cubic meters.

    Raises:
        ValueError: If the geometry is invalid (e.g., negative height).
    """
    x_min = min(cross_ab_x, cross_ba_x)
    x_max = max(cross_ab_x, cross_ba_x)

    if np.isclose(x_min, x_max):
        return 0.0

    n_samples = max(100, int((x_max - x_min) * 10))
    x_samples = np.linspace(x_min, x_max, n_samples)

    y_lower_a = np.polyval(lower_a, x_samples)
    y_lower_b = np.polyval(lower_b, x_samples)
    y_upper_a = np.polyval(upper_a, x_samples)
    y_upper_b = np.polyval(upper_b, x_samples)

    y_bottom = np.minimum(y_lower_a, y_lower_b)
    y_top = np.maximum(y_upper_a, y_upper_b)

    heights = y_top - y_bottom

    if np.any(heights < 0):
        raise ValueError("Invalid geometry: Negative height encountered.")

    # Simplified circular cross-section approximation
    areas = np.pi * (heights**2)
    
    # Integrate over distance in meters (x_samples is in km)
    volume = np.trapz(areas, x=x_samples * 1000)

    if volume < 0:
        # This shouldn't happen with non-negative areas, but as a safeguard
        raise ValueError("Volume calculation resulted in a negative value.")

    return volume


def calculate_distance_between_points(
    point_a: tuple[float, float], point_b: tuple[float, float]
) -> float:
    """
    Calculate the Euclidean distance between two points in mixed units.

    Args:
        point_a: A tuple (x, y) where x is in km and y is in meters.
        point_b: A tuple (x, y) where x is in km and y is in meters.

    Returns:
        The distance in kilometers.
    """
    x1, y1 = point_a
    x2, y2 = point_b

    dx = x2 - x1  # in km
    dy = (y2 - y1) / 1000  # convert meters to km

    distance = np.sqrt(dx**2 + dy**2)

    return distance

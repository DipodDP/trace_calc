"""Input validation utilities for troposcatter calculations."""

import numpy as np
from numpy.typing import NDArray

from trace_calc.domain.models.units import Elevation, Kilometers, Meters
from trace_calc.domain.models.coordinates import Coordinates


class ValidationError(ValueError):
    """Raised when validation fails."""

    pass


def validate_coordinates(coord: Coordinates) -> None:
    """Validate geographic coordinates.

    Args:
        coord: Coordinates to validate

    Raises:
        ValidationError: If coordinates are out of valid range
    """
    if not isinstance(coord, Coordinates):
        raise ValidationError(f"Expected Coordinates, got {type(coord)}")

    if not -90 <= coord.lat <= 90:
        raise ValidationError(
            f"Invalid latitude {coord.lat}°. Must be in range [-90, 90]"
        )

    if not -180 <= coord.lon <= 180:
        raise ValidationError(
            f"Invalid longitude {coord.lon}°. Must be in range [-180, 180]"
        )


def validate_distance(distance_km: Kilometers) -> None:
    """Validate distance parameter.

    Args:
        distance_km: Distance in kilometers

    Raises:
        ValidationError: If distance is invalid
    """
    if not isinstance(distance_km, (int, float)):
        raise ValidationError(f"Distance must be numeric, got {type(distance_km)}")

    if distance_km <= 0:
        raise ValidationError(f"Distance must be positive, got {distance_km} km")

    if distance_km > 20000:
        raise ValidationError(
            f"Distance {distance_km} km exceeds maximum Earth surface distance (~20,000 km)"
        )


def validate_wavelength(wavelength_m: Meters) -> None:
    """Validate wavelength parameter.

    Args:
        wavelength_m: Wavelength in meters

    Raises:
        ValidationError: If wavelength is invalid
    """
    if not isinstance(wavelength_m, (int, float)):
        raise ValidationError(f"Wavelength must be numeric, got {type(wavelength_m)}")

    if wavelength_m <= 0:
        raise ValidationError(f"Wavelength must be positive, got {wavelength_m} m")

    # Typical radio wavelengths: 1mm to 100m
    if not 0.001 <= wavelength_m <= 100:
        raise ValidationError(
            f"Wavelength {wavelength_m} m outside typical radio range [0.001, 100] m"
        )


def validate_antenna_height(height_m: Meters, name: str = "antenna") -> None:
    """Validate antenna height parameter.

    Args:
        height_m: Antenna height in meters
        name: Name for error messages

    Raises:
        ValidationError: If height is invalid
    """
    if not isinstance(height_m, (int, float)):
        raise ValidationError(f"{name} height must be numeric, got {type(height_m)}")

    if height_m < 0:
        raise ValidationError(f"{name} height must be non-negative, got {height_m} m")

    if height_m > 1000:
        raise ValidationError(
            f"{name} height {height_m} m is unrealistically high (max reasonable: 1000 m)"
        )


def validate_elevation(elevation_m: Elevation) -> bool:
    """Check if elevation is plausible for Earth's surface.

    Args:
        elevation_m: Elevation in meters

    Returns:
        True if elevation is plausible, False otherwise

    Note:
        Valid range: -500m (below Dead Sea) to 9000m (above Everest)
    """
    return -500 <= elevation_m <= 9000


def validate_elevation_array(
    elevations: NDArray[np.float64], max_jump: Meters = Meters(1000)
) -> None:
    """Validate an array of elevations for data quality.

    Args:
        elevations: Array of elevation values in meters
        max_jump: Maximum allowed elevation change between adjacent points (default: 1000m)

    Raises:
        ValidationError: If elevation data is invalid

    Checks:
        - No NaN or Inf values
        - Values within plausible Earth elevation range
        - No suspicious jumps between adjacent points
    """
    if not isinstance(elevations, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(elevations)}")

    if elevations.size == 0:
        raise ValidationError("Elevation array is empty")

    # Check for NaN/Inf
    if np.any(np.isnan(elevations)):
        nan_indices = np.where(np.isnan(elevations))[0]
        raise ValidationError(
            f"Elevation array contains NaN values at indices: {nan_indices[:10].tolist()}"
        )

    if np.any(np.isinf(elevations)):
        inf_indices = np.where(np.isinf(elevations))[0]
        raise ValidationError(
            f"Elevation array contains Inf values at indices: {inf_indices[:10].tolist()}"
        )

    # Check plausible bounds
    invalid_mask = (elevations < Elevation(Meters(-500))) | (
        elevations > Elevation(Meters(9000))
    )
    if np.any(invalid_mask):
        invalid_indices = np.where(invalid_mask)[0]
        invalid_values = elevations[invalid_indices]
        raise ValidationError(
            f"Implausible elevations (range: -500 to 9000 m). "
            f"Found {len(invalid_indices)} invalid values. "
            f"First 5 indices: {invalid_indices[:5].tolist()}, "
            f"values: {invalid_values[:5].tolist()}"
        )

    # Check for suspicious jumps (potential data errors)
    if elevations.size > 1:
        diffs = np.abs(np.diff(elevations))
        jump_mask = diffs > max_jump
        if np.any(jump_mask):
            jump_indices = np.where(jump_mask)[0]
            jump_values = diffs[jump_indices]
            raise ValidationError(
                f"Suspicious elevation jumps (>{max_jump}m between adjacent points). "
                f"Found {len(jump_indices)} jumps. "
                f"First 5 at indices: {jump_indices[:5].tolist()}, "
                f"magnitudes: {jump_values[:5].tolist()} m"
            )


def validate_arccos_domain(value: float) -> float:
    """Clip value to valid arccos domain [-1, 1] with warning.

    Args:
        value: Input to arccos

    Returns:
        Clipped value within [-1, 1]

    Note:
        Numerical precision errors can push cos() results slightly outside [-1, 1].
        This function safely clips with a small epsilon tolerance.
    """
    epsilon = 1e-10

    if value < -1 - epsilon or value > 1 + epsilon:
        raise ValidationError(
            f"Value {value} is far outside arccos domain [-1, 1]. "
            "This indicates a serious calculation error, not just floating-point precision."
        )

    # Clip to valid range
    return float(np.clip(value, -1.0, 1.0))

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class GeoData:
    """
    Model that holds geo data of the sites.
    """

    distance: float
    mag_declination_a: float
    mag_declination_b: float
    true_azimuth_a_b: float
    true_azimuth_b_a: float
    mag_azimuth_a_b: float
    mag_azimuth_b_a: float

    def __repr__(self):
        return (
            f"GeoData(distance={self.distance}, "
            f"true_azimuth_a_b={self.true_azimuth_a_b}, true_azimuth_b_a={self.true_azimuth_b_a},"
            f"mag_declination_a={self.mag_declination_a}, mag_declination_b={self.mag_declination_b},"
            " ... )"
        )


@dataclass(slots=True)
class PathData:
    """
    Model that holds path-related data.

    :param coordinates: An array of coordinates (e.g., [[lat, lon], ...])
    :param distances: An array of distances
    :param elevations: An array of elevations
    """

    coordinates: NDArray[np.floating[Any]]
    distances: NDArray[np.float64]
    elevations: NDArray[np.float64]

    def __repr__(self):
        return (
            f"PathData(coordinates={self.coordinates}, "
            f"distances={self.distances}, elevations={self.elevations})"
        )


class HCAData(NamedTuple):
    """
    Model that holds horizon close angle, calculated for a path between two sites.
    """

    b1_max: float
    b2_max: float
    b_sum: float
    b1_idx: int
    b2_idx: int


class ProfileViewData(NamedTuple):
    """
    Container for elevation profile and its corresponding baseline values.
    """

    elevations: NDArray[np.float64]  # Elevation values along the path
    baseline: NDArray[np.float64]  # Corresponding baseline values


@dataclass(slots=True)
class ProfileData:
    """
    Aggregated profile outputs: flat, curved, and sight-line data.

    :param plain: Plain profile ProfileViewData
    :param curved: Curved profile ProfileViewData
    :param elevations: A tuple of line A, line B coefficients and intersection point
    """

    plain: ProfileViewData
    curved: ProfileViewData
    lines_of_sight: tuple[np.ndarray, np.ndarray, tuple[float, float]]

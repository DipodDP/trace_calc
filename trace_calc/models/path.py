from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from trace_calc.domain.units import Angle, Kilometers


@dataclass(slots=True)
class GeoData:
    """
    Model that holds geo data of the sites.
    """

    distance: Kilometers
    mag_declination_a: Angle
    mag_declination_b: Angle
    true_azimuth_a_b: Angle
    true_azimuth_b_a: Angle
    mag_azimuth_a_b: Angle
    mag_azimuth_b_a: Angle

    def __repr__(self) -> str:
        return (
            f"GeoData(distance={self.distance:.2f}, "
            f"true_azimuth_a_b={self.true_azimuth_a_b:.2f}, "
            f"true_azimuth_b_a={self.true_azimuth_b_a:.2f}, "
            f"mag_declination_a={self.mag_declination_a:.2f}, "
            f"mag_declination_b={self.mag_declination_b:.2f}, "
            "... )"
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

    def __repr__(self) -> str:
        return (
            f"PathData(coordinates={self.coordinates}, "
            f"distances={self.distances}, elevations={self.elevations})"
        )


class HCAData(NamedTuple):
    """
    Model that holds horizon close angle, calculated for a path between two sites.
    """

    b1_max: Angle
    b2_max: Angle
    b_sum: Angle
    b1_idx: int
    b2_idx: int


class ProfileViewData(NamedTuple):
    """
    Container for elevation profile and its corresponding baseline values.
    """

    elevations: NDArray[np.float64]
    baseline: NDArray[np.float64]


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
    lines_of_sight: tuple[NDArray[np.float64], NDArray[np.float64], tuple[float, float]]

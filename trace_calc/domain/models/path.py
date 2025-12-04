from dataclasses import dataclass, fields, is_dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .units import Angle, Kilometers


class BaseModel:
    def to_dict(self):
        """Converts a dataclass instance to a dictionary, handling nested dataclasses,
        NamedTuples, and numpy arrays.
        """
        result = {}
        for f in fields(self):
            value = self._convert_value(getattr(self, f.name))
            result[f.name] = value
        return result

    def _convert_value(self, value: Any) -> Any:
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        if type(value).__name__ in ('Angle', 'Kilometers'):
            return float(value)
        if isinstance(value, tuple) and hasattr(value, '_asdict'):  # Handle NamedTuple
            return {k: self._convert_value(v) for k, v in value._asdict().items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [self._convert_value(v) for v in value]
        return value


@dataclass(slots=True)
class IntersectionPoint(BaseModel):
    """Represents a point where two sight lines intersect."""
    distance_km: float
    elevation_sea_level: float
    elevation_terrain: float
    angle: Angle | None = None


@dataclass(slots=True)
class SightLinesData(BaseModel):
    """Container for all four sight line equations."""
    lower_a: NDArray[np.float64]
    lower_b: NDArray[np.float64]
    upper_a: NDArray[np.float64]
    upper_b: NDArray[np.float64]
    antenna_elevation_angle_a: NDArray[np.float64]
    antenna_elevation_angle_b: NDArray[np.float64]


@dataclass(slots=True)
class IntersectionsData(BaseModel):
    """All intersection points between sight lines."""
    lower: IntersectionPoint
    upper: IntersectionPoint
    cross_ab: IntersectionPoint
    cross_ba: IntersectionPoint
    beam_intersection_point: IntersectionPoint | None


@dataclass(slots=True)
class VolumeData(BaseModel):
    """Volumetric and distance metrics for analysis region."""
    cone_intersection_volume_m3: float
    distance_a_to_cross_ab: float
    distance_b_to_cross_ba: float
    distance_between_crosses: float
    distance_a_to_lower_intersection: float
    distance_b_to_lower_intersection: float
    distance_a_to_upper_intersection: float
    distance_b_to_upper_intersection: float
    distance_between_lower_upper_intersections: float
    antenna_elevation_angle_a: Angle
    antenna_elevation_angle_b: Angle


@dataclass(slots=True)
class GeoData(BaseModel):
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

    def to_dict(self) -> dict[str, Any]:
        data = BaseModel.to_dict(self)
        data["distance_km"] = data.pop("distance")
        return data

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
class PathData(BaseModel):
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


@dataclass(slots=True)
class ProfileViewData(BaseModel):
    """
    Container for elevation profile and its corresponding baseline values.
    """

    elevations: NDArray[np.float64]
    baseline: NDArray[np.float64]


@dataclass(slots=True)
class ProfileData(BaseModel):
    """
    Aggregated profile outputs: flat, curved, and sight-line data.

    :param plain: Plain profile ProfileViewData
    :param curved: Curved profile ProfileViewData
    :param lines_of_sight: SightLinesData object
    :param intersections: IntersectionsData object
    :param volume: VolumeData object
    """

    plain: ProfileViewData
    curved: ProfileViewData
    lines_of_sight: SightLinesData
    intersections: IntersectionsData
    volume: VolumeData
import numpy as np
from numpy.typing import NDArray
import dataclasses

from trace_calc.domain import geometry
from trace_calc.domain.curvature import apply_geometric_curvature, calculate_earth_drop
from trace_calc.domain.models.units import Angle, Meters
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import (
    IntersectionsData,
    IntersectionPoint,
    PathData,
    ProfileData,
    ProfileViewData,
    SightLinesData,
    VolumeData,
)
from trace_calc.application.services.hca import HCACalculatorCC


class ProfileDataCalculator:
    """
    Calculates elevation profiles: flat, curved, and lines of sight.
    """

    def __init__(self, distances: NDArray[np.float64], elevations: NDArray[np.float64]):
        """
        Initialize with distances and elevations arrays.

        Args:
            distances: 1D array of distances along the path.
            elevations: 1D array of corresponding elevations.

        Raises:
            ValueError: if inputs are not 1D arrays of equal length >= 2.
        """
        self.distances: NDArray[np.float64] = np.atleast_1d(distances)
        self.elevations: NDArray[np.float64] = np.atleast_1d(elevations)

        if self.distances.ndim != 1 or self.elevations.ndim != 1:
            raise ValueError("`distances` and `elevations` must be 1D arrays.")
        if self.distances.size < 2:
            raise ValueError(
                "`distances` and `elevations` must contain at least two elements."
            )
        if self.distances.size != self.elevations.size:
            raise ValueError("`distances` and `elevations` must have the same length.")

    @staticmethod
    def _line_coeff(
        p0: tuple[float, float], p1: tuple[float, float]
    ) -> NDArray[np.float64]:
        """
        Compute linear coefficients (k, b) for the line y = k*x + b between two points.
        """
        k = (p1[1] - p0[1]) / (p1[0] - p0[0])
        b = p0[1] - k * p0[0]
        return np.array([k, b])

    def _calculate_upper_lines(
        self,
        lower_line_coeffs: NDArray[np.float64],
        pivot_point: tuple[float, float],
        angle_offset: Angle,
    ) -> NDArray[np.float64]:
        """
        Calculate upper sight line by rotating lower line.

        Args:
            lower_line_coeffs: [k, b] for lower sight line
            pivot_point: (distance, elevation) of site
            angle_offset: Angular offset in degrees

        Returns:
            [k_upper, b_upper] for upper sight line

        Raises:
            ValueError: If rotation produces invalid line
        """
        try:
            upper_coeffs = geometry.rotate_line_by_angle(
                lower_line_coeffs, pivot_point, float(angle_offset)
            )
            return upper_coeffs
        except ValueError as e:
            raise ValueError(f"Failed to calculate upper line: {e}") from e

    def _calculate_antenna_elevation_angle_lines(
        self,
        sight_lines: SightLinesData,
        distances: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Calculate the antenna elevation angle line of the angle between upper and lower sight lines.
        The antenna elevation angle line is calculated as the average of the upper and lower lines.
        """
        y_lower_a = np.polyval(sight_lines.lower_a, distances)
        y_upper_a = np.polyval(sight_lines.upper_a, distances)
        antenna_elevation_angle_a = (y_lower_a + y_upper_a) / 2

        y_lower_b = np.polyval(sight_lines.lower_b, distances)
        y_upper_b = np.polyval(sight_lines.upper_b, distances)
        antenna_elevation_angle_b = (y_lower_b + y_upper_b) / 2

        return antenna_elevation_angle_a, antenna_elevation_angle_b

    def _calculate_antenna_elevation_angle_intersection(
        self,
        antenna_elevation_angle_a: NDArray[np.float64],
        antenna_elevation_angle_b: NDArray[np.float64],
        distances: NDArray[np.float64],
        elevations: NDArray[np.float64],
    ) -> IntersectionPoint | None:
        """
        Calculate the intersection point of the two antenna elevation angle lines.
        """
        diff = antenna_elevation_angle_a - antenna_elevation_angle_b
        # Find where the difference changes sign
        sign_changes = np.where(np.diff(np.sign(diff)))[0]



        if not sign_changes.size:

            return None  # No intersection found

        # Take the first intersection for simplicity
        idx = sign_changes[0]

        # Use linear interpolation to find the exact intersection point
        x1, x2 = distances[idx], distances[idx + 1]
        y1_a, y2_a = antenna_elevation_angle_a[idx], antenna_elevation_angle_a[idx + 1]
        y1_b, y2_b = antenna_elevation_angle_b[idx], antenna_elevation_angle_b[idx + 1]

        # Line A: y = m_a * x + c_a
        m_a = (y2_a - y1_a) / (x2 - x1)
        c_a = y1_a - m_a * x1

        # Line B: y = m_b * x + c_b
        m_b = (y2_b - y1_b) / (x2 - x1)
        c_b = y1_b - m_b * x1

        # Intersection: m_a * x + c_a = m_b * x + c_b
        # (m_a - m_b) * x = c_b - c_a
        # x = (c_b - c_a) / (m_a - m_b)
        if np.isclose(m_a, m_b):  # Parallel lines, no unique intersection
            return None

        x_intersectionersect = (c_b - c_a) / (m_a - m_b)
        y_intersectionersect = m_a * x_intersectionersect + c_a

        # Calculate angle between the two lines at intersection
        # Convert slopes from m/km to unitless ratio
        m1 = m_a / 1000
        m2 = m_b / 1000
        angle_rad = np.arctan(np.abs((m1 - m2) / (1 + m1 * m2)))
        angle_deg = Angle(np.degrees(angle_rad))

        # Correct for Earth drop
        drop = calculate_earth_drop(np.array([x_intersectionersect]))[0]
        y_corrected = y_intersectionersect - drop

        h_terrain = geometry.calculate_height_above_terrain(
            x_intersectionersect, y_corrected, distances, elevations
        )
        return IntersectionPoint(
            x_intersectionersect, y_corrected, h_terrain, angle=angle_deg
        )

    def _calculate_all_intersections(
        self,
        sight_lines: SightLinesData,
        distances: NDArray[np.float64],
        elevations: NDArray[np.float64],
    ) -> IntersectionsData:
        """
        Calculate all intersection points between sight lines.

        Args:
            sight_lines: All four line coefficients
            distances: Path distances (km)
            elevations: Terrain elevations (meters)

        Returns:
            IntersectionsData with all 4 intersections

        Raises:
            ValueError: If any intersection is invalid or outside path
        """
        # Lower intersection
        x, y = geometry.find_line_intersection(sight_lines.lower_a, sight_lines.lower_b)
        if not (distances[0] <= x <= distances[-1]):
            raise ValueError(f"Lower intersection at {x} km is outside path bounds")

        drop = calculate_earth_drop(np.array([x]))[0]
        y_corrected = y - drop

        h_terrain = geometry.calculate_height_above_terrain(
            x, y_corrected, distances, elevations
        )
        lower_intersection = IntersectionPoint(x, y_corrected, h_terrain)

        # Upper intersection
        x, y = geometry.find_line_intersection(sight_lines.upper_a, sight_lines.upper_b)
        if not (distances[0] <= x <= distances[-1]):
            raise ValueError(f"Upper intersection at {x} km is outside path bounds")

        drop = calculate_earth_drop(np.array([x]))[0]
        y_corrected = y - drop

        h_terrain = geometry.calculate_height_above_terrain(
            x, y_corrected, distances, elevations
        )
        upper_intersection = IntersectionPoint(x, y_corrected, h_terrain)

        # Cross AB intersection
        x, y = geometry.find_line_intersection(sight_lines.upper_a, sight_lines.lower_b)
        if not (distances[0] <= x <= distances[-1]):
            raise ValueError(f"Cross AB intersection at {x} km is outside path bounds")

        drop = calculate_earth_drop(np.array([x]))[0]
        y_corrected = y - drop

        h_terrain = geometry.calculate_height_above_terrain(
            x, y_corrected, distances, elevations
        )
        cross_ab = IntersectionPoint(x, y_corrected, h_terrain)

        # Cross BA intersection
        x, y = geometry.find_line_intersection(sight_lines.upper_b, sight_lines.lower_a)
        if not (distances[0] <= x <= distances[-1]):
            raise ValueError(f"Cross BA intersection at {x} km is outside path bounds")

        drop = calculate_earth_drop(np.array([x]))[0]
        y_corrected = y - drop

        h_terrain = geometry.calculate_height_above_terrain(
            x, y_corrected, distances, elevations
        )
        cross_ba = IntersectionPoint(x, y_corrected, h_terrain)

        return IntersectionsData(
            lower_intersection,
            upper_intersection,
            cross_ab,
            cross_ba,
            beam_intersection_point=None,
        )

    def _calculate_volume_metrics(
        self,
        sight_lines: SightLinesData,
        distances: NDArray[np.float64],
        intersections: IntersectionsData,
    ) -> VolumeData:
        """
        Calculate volume and distance metrics.

        Args:
            sight_lines: All four lines
            distances: Path distances
            intersections: All intersection points

        Returns:
            VolumeData with volume and distance metrics

        Raises:
            ValueError: If volume calculation fails
        """
        # Calculate volume
        volume = geometry.calculate_cone_intersection_volume(
            sight_lines.lower_a,
            sight_lines.lower_b,
            sight_lines.upper_a,
            sight_lines.upper_b,
            distances,
            intersections.lower.distance_km,
            intersections.upper.distance_km,
            intersections.cross_ab.distance_km,
            intersections.cross_ba.distance_km,
        )

        # Calculate distances
        distance_a_to_cross_ab = intersections.cross_ab.distance_km
        total_distance = distances[-1]
        distance_b_to_cross_ba = total_distance - intersections.cross_ba.distance_km
        distance_between = abs(
            intersections.cross_ab.distance_km - intersections.cross_ba.distance_km
        )

        # Calculate distances to lower/upper intersections
        distance_a_to_lower = intersections.lower.distance_km
        distance_b_to_lower = total_distance - intersections.lower.distance_km
        distance_a_to_upper = intersections.upper.distance_km
        distance_b_to_upper = total_distance - intersections.upper.distance_km
        distance_between_lower_upper = abs(
            intersections.upper.distance_km - intersections.lower.distance_km
        )

        # Validate
        if distance_a_to_cross_ab < 0 or distance_b_to_cross_ba < 0:
            raise ValueError("Distance metrics produced negative values")
        if distance_a_to_lower < 0 or distance_b_to_lower < 0:
            raise ValueError(
                "Lower intersection distance metrics produced negative values"
            )
        if distance_a_to_upper < 0 or distance_b_to_upper < 0:
            raise ValueError(
                "Upper intersection distance metrics produced negative values"
            )

        return VolumeData(
            cone_intersection_volume_m3=volume,
            distance_a_to_cross_ab=distance_a_to_cross_ab,
            distance_b_to_cross_ba=distance_b_to_cross_ba,
            distance_between_crosses=distance_between,
            distance_a_to_lower_intersection=distance_a_to_lower,
            distance_b_to_lower_intersection=distance_b_to_lower,
            distance_a_to_upper_intersection=distance_a_to_upper,
            distance_b_to_upper_intersection=distance_b_to_upper,
            distance_between_lower_upper_intersections=distance_between_lower_upper,
            antenna_elevation_angle_a=Angle(0.0),
            antenna_elevation_angle_b=Angle(0.0),
        )

    def plain_profile(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate the flat profile and its baseline.

        Returns:
            Tuple of (elevations, baseline), where baseline is adjusted if there are negative elevations.
        """
        baseline = np.zeros_like(self.distances)
        min_elev = self.elevations.min()
        if min_elev < 0:
            baseline += min_elev
        return self.elevations, baseline

    def curved_profile(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Apply a parabolic curve to simulate Earth curvature using the geometric model.

        Returns:
            Tuple of (curved_elevations, curved_baseline).
        """
        curve = apply_geometric_curvature(self.distances)

        curved_elevations = self.elevations + curve
        min_elev = self.elevations.min()
        curved_baseline = curve + (min_elev if min_elev < 0 else 0)
        return curved_elevations, curved_baseline

    def lines_of_sight(
        self, hca_indices: tuple[int, int], height_offsets: tuple[Meters, Meters]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[float, float]]:
        """
        Calculate lines of sight between two points with height offsets.

        Args:
            hca_indices: indices for horizon clearance analysis points (start_idx, end_idx).
            height_offsets: height offsets at path start and end.

        Returns:
            Coefficients for start and end sight lines and their intersection point (x, y).
        """
        i1, i2 = hca_indices
        if not (0 <= i1 < self.distances.size) or not (0 <= i2 < self.distances.size):
            raise IndexError(
                f"HCA indices {hca_indices} out of range for size {self.distances.size}."
            )

        offset_start, offset_end = height_offsets
        p0 = (
            float(self.distances[0]),
            float(self.elevations_curved[0]) + float(offset_start),
        )
        p1 = (float(self.distances[i1]), float(self.elevations_curved[i1]))
        coeff1 = self._line_coeff(p0, p1)

        p2 = (
            float(self.distances[-1]),
            float(self.elevations_curved[-1]) + float(offset_end),
        )
        p3 = (float(self.distances[i2]), float(self.elevations_curved[i2]))
        coeff2 = self._line_coeff(p2, p3)

        x_cross = (coeff2[1] - coeff1[1]) / (coeff1[0] - coeff2[0])
        y_cross = float(np.polyval(coeff1, x_cross))
        return coeff1, coeff2, (x_cross, y_cross)

    def calculate_all(
        self,
        hca_indices: tuple[int, int],
        height_offsets: tuple[Meters, Meters],
        hpbw: Angle = Angle(0.0),
    ) -> ProfileData:
        """
        Compute all profile variants and lines of sight.

        Returns:
            ProfileData containing:
                plain: flat profile and baseline
                curved: curved profile and baseline
                lines_of_sight: sight lines and intersection
        """
        plain, baseline_plain = self.plain_profile()
        curved, baseline_curved = self.curved_profile()
        self.elevations_curved = curved

        offset_start, offset_end = height_offsets
        coeff1, coeff2, _ = self.lines_of_sight(hca_indices, height_offsets)

        # Calculate upper sight lines
        pivot_a = (self.distances[0], self.elevations_curved[0] + offset_start)
        pivot_b = (self.distances[-1], self.elevations_curved[-1] + offset_end)

        coeff1_upper = self._calculate_upper_lines(coeff1, pivot_a, hpbw)
        coeff2_upper = self._calculate_upper_lines(coeff2, pivot_b, hpbw)

        # Assemble sight lines data
        sight_lines_no_antenna_elevation_angle = SightLinesData(
            lower_a=coeff1,
            lower_b=coeff2,
            upper_a=coeff1_upper,
            upper_b=coeff2_upper,
            antenna_elevation_angle_a=np.array([]),  # Placeholder
            antenna_elevation_angle_b=np.array([]),  # Placeholder
        )

        antenna_elevation_angle_a, antenna_elevation_angle_b = (
            self._calculate_antenna_elevation_angle_lines(
                sight_lines_no_antenna_elevation_angle, self.distances
            )
        )

        sight_lines = SightLinesData(
            lower_a=coeff1,
            lower_b=coeff2,
            upper_a=coeff1_upper,
            upper_b=coeff2_upper,
            antenna_elevation_angle_a=antenna_elevation_angle_a,
            antenna_elevation_angle_b=antenna_elevation_angle_b,
        )

        # Calculate all intersections
        beam_intersection_point = (
            self._calculate_antenna_elevation_angle_intersection(
                antenna_elevation_angle_a,
                antenna_elevation_angle_b,
                self.distances,
                self.elevations,
            )
        )

        intersections = self._calculate_all_intersections(
            sight_lines, self.distances, self.elevations
        )
        intersections = dataclasses.replace(
            intersections,
            beam_intersection_point=beam_intersection_point,
        )

        # Calculate volume metrics
        volume = self._calculate_volume_metrics(
            sight_lines, self.distances, intersections
        )
        volume = VolumeData(
            cone_intersection_volume_m3=volume.cone_intersection_volume_m3,
            distance_a_to_cross_ab=volume.distance_a_to_cross_ab,
            distance_b_to_cross_ba=volume.distance_b_to_cross_ba,
            distance_between_crosses=volume.distance_between_crosses,
            distance_a_to_lower_intersection=volume.distance_a_to_lower_intersection,
            distance_b_to_lower_intersection=volume.distance_b_to_lower_intersection,
            distance_a_to_upper_intersection=volume.distance_a_to_upper_intersection,
            distance_b_to_upper_intersection=volume.distance_b_to_upper_intersection,
            distance_between_lower_upper_intersections=volume.distance_between_lower_upper_intersections,
            antenna_elevation_angle_a=Angle(0.0),
            antenna_elevation_angle_b=Angle(0.0),
        )

        return ProfileData(
            plain=ProfileViewData(plain, baseline_plain),
            curved=ProfileViewData(curved, baseline_curved),
            lines_of_sight=sight_lines,
            intersections=intersections,
            volume=volume,
        )


class DefaultProfileDataCalculator(object):
    """
    Integrates HCA data with profile calculations for a given path.
    """

    def __init__(self, profile: PathData, input_data: InputData):
        hca_calculator = HCACalculatorCC(profile, input_data)
        self.hca_calculator = hca_calculator
        self.hca_data = hca_calculator.hca_data

        offsets = (hca_calculator.antenna_a_height, hca_calculator.antenna_b_height)

        calculator = ProfileDataCalculator(
            hca_calculator.distances, hca_calculator.elevations
        )
        self.profile_data = calculator.calculate_all(
            (self.hca_data.b1_idx, self.hca_data.b2_idx),
            offsets,
            hpbw=input_data.hpbw,
        )

        # Per user request, override antenna elevation angle calculation to be based on Lower Sight Line.
        # Antenna Elevation Angle = Lower Sight Line Angle + 1.25 * (Beamwidth / 2)
        slope_a = self.profile_data.lines_of_sight.lower_a[0]
        slope_b = self.profile_data.lines_of_sight.lower_b[0] * (-1)

        # Slopes are in m/km, convert to unitless ratio for arctan
        angle_a_rad = np.arctan(slope_a / 1000)
        angle_b_rad = np.arctan(slope_b / 1000)

        angle_a_deg = np.rad2deg(angle_a_rad)
        angle_b_deg = np.rad2deg(angle_b_rad)

        antenna_elevation_angle_a = Angle(angle_a_deg) + (input_data.hpbw / 2)
        antenna_elevation_angle_b = Angle(angle_b_deg) + (input_data.hpbw / 2)

        # Create a new VolumeData with the correct angles
        new_volume_data = dataclasses.replace(
            self.profile_data.volume,
            antenna_elevation_angle_a=antenna_elevation_angle_a,
            antenna_elevation_angle_b=antenna_elevation_angle_b,
        )

        # Update the profile_data with the new volume data
        self.profile_data = dataclasses.replace(self.profile_data, volume=new_volume_data)

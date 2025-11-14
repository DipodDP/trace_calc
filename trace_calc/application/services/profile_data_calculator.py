import numpy as np
from numpy.typing import NDArray

from trace_calc.domain.curvature import apply_geometric_curvature
from trace_calc.domain.models.units import Meters
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import PathData, ProfileData, ProfileViewData
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
        self, hca_indices: tuple[int, int], height_offsets: tuple[Meters, Meters]
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

        sight1, sight2, intersection = self.lines_of_sight(hca_indices, height_offsets)
        return ProfileData(
            plain=ProfileViewData(plain, baseline_plain),
            curved=ProfileViewData(curved, baseline_curved),
            lines_of_sight=(sight1, sight2, intersection),
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
            (self.hca_data.b1_idx, self.hca_data.b2_idx), offsets
        )

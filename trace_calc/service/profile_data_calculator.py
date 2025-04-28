import numpy as np
from numpy.typing import NDArray

from trace_calc.models.input_data import InputData
from trace_calc.models.path import PathData, ProfileData, ProfileViewData
from trace_calc.service.hca_calculator import HCACalculatorCC


class ProfileDataCalculator:
    """
    Calculates elevation profiles: flat, curved, and lines of sight.
    """

    def __init__(self, distances: np.ndarray, elevations: np.ndarray):
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
            raise ValueError("`distances` and `elevations` must contain at least two elements.")
        if self.distances.size != self.elevations.size:
            raise ValueError("`distances` and `elevations` must have the same length.")

    @staticmethod
    def _line_coeff(p0: tuple[float, float], p1: tuple[float, float]) -> np.ndarray:
        """
        Compute linear coefficients (k, b) for the line y = k*x + b between two points.
        """
        k = (p1[1] - p0[1]) / (p1[0] - p0[0])
        b = p0[1] - k * p0[0]
        return np.array([k, b])

    def plain_profile(self) -> tuple[np.ndarray, np.ndarray]:
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

    def curved_profile(self, curvature_scale: float = 12.742) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply a parabolic curve to simulate Earth curvature.

        Args:
            curvature_scale: factor controlling the curvature magnitude.

        Returns:
            Tuple of (curved_elevations, curved_baseline).
        """
        mid_idx = self.distances.size // 2
        curve = -((self.distances - self.distances[mid_idx]) ** 2) / curvature_scale
        curve -= curve[0]

        curved_elevations = self.elevations + curve
        min_elev = self.elevations.min()
        curved_baseline = curve + (min_elev if min_elev < 0 else 0)
        return curved_elevations, curved_baseline

    def lines_of_sight(
        self,
        hca_indices: tuple[int, int],
        height_offsets: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
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
            raise IndexError(f"HCA indices {hca_indices} out of range for size {self.distances.size}.")

        offset_start, offset_end = height_offsets
        p0 = (self.distances[0], self.elevations_curved[0] + offset_start)
        p1 = (self.distances[i1], self.elevations_curved[i1])
        coeff1 = self._line_coeff(p0, p1)

        p2 = (self.distances[-1], self.elevations_curved[-1] + offset_end)
        p3 = (self.distances[i2], self.elevations_curved[i2])
        coeff2 = self._line_coeff(p2, p3)

        x_cross = (coeff2[1] - coeff1[1]) / (coeff1[0] - coeff2[0])
        y_cross = np.polyval(coeff1, x_cross)
        return coeff1, coeff2, (x_cross, y_cross)

    def calculate_all(
        self,
        hca_indices: tuple[int, int],
        height_offsets: tuple[float, float],
        curvature_scale: float = 12.742
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
        curved, baseline_curved = self.curved_profile(curvature_scale)
        self.elevations_curved = curved

        sight1, sight2, intersection = self.lines_of_sight(hca_indices, height_offsets)
        return ProfileData(
            plain=ProfileViewData(plain, baseline_plain),
            curved=ProfileViewData(curved, baseline_curved),
            lines_of_sight=(sight1, sight2, intersection)
        )


class DefaultProfileDataCalculator(HCACalculatorCC):
    """
    Integrates HCA data with profile calculations for a given path.
    """

    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)
        offsets = (self.antenna_a_height, self.antenna_b_height)

        calculator = ProfileDataCalculator(self.distances, self.elevations)
        self.profile_data = calculator.calculate_all(
            (self.hca_data.b1_idx, self.hca_data.b2_idx),
            offsets
        )

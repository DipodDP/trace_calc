"""
Custom HCA Calculator Example
===============================

This example demonstrates how to create a custom Horizon Clearance Angle (HCA)
calculator with alternative elevation filtering approaches.

HCA calculators are used to:
- Filter elevation profiles to remove noise
- Calculate maximum elevation angles at each site
- Determine obstacles affecting the radio path

This example shows:
1. Creating a custom elevation filter
2. Extending the base HCA calculator
3. Using the custom calculator in analysis

Requirements:
- Python 3.10+
- trace_calc package installed
- numpy
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter  # Optional: for Savitzky-Golay filtering

from trace_calc.domain.models.path import HCAData, PathData
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.units import Angle, Degrees, Elevation, Kilometers, Meters
from trace_calc.application.services.hca import HCACalculator


# ==============================================================================
# Custom Elevation Filter - Savitzky-Golay
# ==============================================================================

class ElevationsFilterSavGol:
    """
    Mixin providing Savitzky-Golay filtering for elevation profiles.

    Savitzky-Golay filter is useful for smoothing noisy data while
    preserving peak features better than simple moving averages.
    """

    elevations: NDArray[np.float64]

    def filter_elevation_profile(
        self, window_length: int = 11, polyorder: int = 3
    ) -> NDArray[np.float64]:
        """
        Apply Savitzky-Golay filter to elevation profile.

        Args:
            window_length: Length of the filter window (must be odd, >= polyorder+1)
            polyorder: Order of the polynomial used to fit the samples

        Returns:
            Filtered elevation array
        """
        # Ensure window_length is odd and valid
        if window_length % 2 == 0:
            window_length += 1
        if window_length > len(self.elevations):
            window_length = len(self.elevations) if len(self.elevations) % 2 == 1 else len(self.elevations) - 1

        try:
            filtered_elevations = savgol_filter(
                self.elevations,
                window_length=window_length,
                polyorder=polyorder,
                mode='interp'
            )
            return filtered_elevations
        except Exception:
            # Fallback to original elevations if filtering fails
            return self.elevations.copy()


# ==============================================================================
# Custom Elevation Filter - Simple Moving Average
# ==============================================================================

class ElevationsFilterMovingAverage:
    """
    Mixin providing simple moving average filtering.

    This is a straightforward smoothing approach that works well
    for moderately noisy data.
    """

    elevations: NDArray[np.float64]

    def filter_elevation_profile(self, window_size: int = 5) -> NDArray[np.float64]:
        """
        Apply moving average filter to elevation profile.

        Args:
            window_size: Number of points to average

        Returns:
            Filtered elevation array
        """
        # Use numpy convolve for moving average
        weights = np.ones(window_size) / window_size
        filtered_elevations = np.convolve(self.elevations, weights, mode='same')
        return filtered_elevations


# ==============================================================================
# Custom Elevation Filter - Median Filter
# ==============================================================================

class ElevationsFilterMedian:
    """
    Mixin providing median filtering for elevation profiles.

    Median filters are excellent at removing outliers and spike noise
    while preserving sharp edges in the terrain profile.
    """

    elevations: NDArray[np.float64]

    def filter_elevation_profile(self, kernel_size: int = 5) -> NDArray[np.float64]:
        """
        Apply median filter to elevation profile.

        Args:
            kernel_size: Size of the median filter kernel

        Returns:
            Filtered elevation array
        """
        filtered_elevations = np.zeros_like(self.elevations)
        half_kernel = kernel_size // 2

        for i in range(len(self.elevations)):
            # Get window bounds
            start = max(0, i - half_kernel)
            end = min(len(self.elevations), i + half_kernel + 1)

            # Apply median
            filtered_elevations[i] = np.median(self.elevations[start:end])

        return filtered_elevations


# ==============================================================================
# Custom HCA Calculators
# ==============================================================================

class HCACalculatorSavGol(HCACalculator, ElevationsFilterSavGol):
    """
    HCA calculator using Savitzky-Golay filtering.

    Combines the base HCA calculation algorithm with Savitzky-Golay
    elevation filtering for noise reduction.
    """

    def __init__(
        self,
        profile: PathData,
        input_data: InputData,
        window_length: int = 11,
        polyorder: int = 3,
    ):
        super().__init__(profile, input_data)
        self.elevations = self.filter_elevation_profile(window_length, polyorder)


class HCACalculatorMovingAverage(HCACalculator, ElevationsFilterMovingAverage):
    """
    HCA calculator using moving average filtering.

    Simple and fast smoothing approach suitable for moderately noisy data.
    """

    def __init__(self, profile: PathData, input_data: InputData, window_size: int = 5):
        super().__init__(profile, input_data)
        self.elevations = self.filter_elevation_profile(window_size)


class HCACalculatorMedian(HCACalculator, ElevationsFilterMedian):
    """
    HCA calculator using median filtering.

    Excellent at removing spike noise and outliers.
    """

    def __init__(self, profile: PathData, input_data: InputData, kernel_size: int = 5):
        super().__init__(profile, input_data)
        self.elevations = self.filter_elevation_profile(kernel_size)


# ==============================================================================
# Custom HCA Calculator with Adaptive Filtering
# ==============================================================================

class HCACalculatorAdaptive(HCACalculator):
    """
    HCA calculator with adaptive filtering based on terrain characteristics.

    This calculator analyzes the elevation profile and chooses an appropriate
    filtering strategy based on the terrain roughness.
    """

    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)

        # Analyze terrain roughness
        elevation_std = np.std(self.elevations)
        elevation_range = np.ptp(self.elevations)  # Peak-to-peak

        # Choose filtering strategy based on roughness
        roughness_ratio = elevation_std / max(elevation_range, 1.0)

        if roughness_ratio > 0.15:
            # Very rough terrain - use median filter to remove spikes
            self.elevations = self._apply_median_filter(kernel_size=7)
        elif roughness_ratio > 0.08:
            # Moderately rough - use Savitzky-Golay for feature preservation
            self.elevations = self._apply_savgol_filter(window_length=11, polyorder=3)
        else:
            # Smooth terrain - light smoothing with moving average
            self.elevations = self._apply_moving_average(window_size=3)

    def _apply_median_filter(self, kernel_size: int) -> NDArray[np.float64]:
        """Apply median filter."""
        filtered = np.zeros_like(self.elevations)
        half_kernel = kernel_size // 2

        for i in range(len(self.elevations)):
            start = max(0, i - half_kernel)
            end = min(len(self.elevations), i + half_kernel + 1)
            filtered[i] = np.median(self.elevations[start:end])

        return filtered

    def _apply_savgol_filter(
        self, window_length: int, polyorder: int
    ) -> NDArray[np.float64]:
        """Apply Savitzky-Golay filter."""
        try:
            return savgol_filter(
                self.elevations, window_length=window_length, polyorder=polyorder, mode='interp'
            )
        except Exception:
            return self.elevations.copy()

    def _apply_moving_average(self, window_size: int) -> NDArray[np.float64]:
        """Apply moving average filter."""
        weights = np.ones(window_size) / window_size
        return np.convolve(self.elevations, weights, mode='same')


# ==============================================================================
# Usage Example
# ==============================================================================

def example_usage():
    """
    Example showing how to use custom HCA calculators.

    Note: In practice, you would integrate these into your analyzer by
    creating a custom ProfileDataCalculator that uses your HCA calculator.
    """
    from trace_calc.domain.models.coordinates import Coordinates
    from trace_calc.domain.models.units import Meters
    import numpy as np

    # Create sample data
    input_data = InputData(
        path_name="test_path",
        site_a_coordinates=Coordinates(55.7558, 37.6173),
        site_b_coordinates=Coordinates(59.9343, 30.3351),
        frequency_mhz=5000.0,
        antenna_a_height=Meters(30.0),
        antenna_b_height=Meters(30.0),
    )

    # Create sample path data (in practice, this comes from elevation API)
    sample_distances = np.linspace(0, 100, 1000)  # 100 km path
    sample_elevations = 200 + 50 * np.sin(sample_distances / 10) + 10 * np.random.randn(1000)

    sample_path = PathData(
        coordinates=np.array([
            [input_data.site_a_coordinates.lat, input_data.site_a_coordinates.lon],
            [input_data.site_b_coordinates.lat, input_data.site_b_coordinates.lon],
        ]),
        distances=sample_distances,
        elevations=sample_elevations,
    )

    # Compare different HCA calculators
    print("Comparing HCA Calculators")
    print("=" * 50)

    # Standard cross-correlation filter
    from trace_calc.application.services.hca import HCACalculatorCC
    hca_cc = HCACalculatorCC(sample_path, input_data)
    result_cc = hca_cc.calculate_hca()
    print(f"\nCross-Correlation Filter:")
    print(f"  b1_max: {result_cc.b1_max:.4f}°")
    print(f"  b2_max: {result_cc.b2_max:.4f}°")
    print(f"  b_sum:  {result_cc.b_sum:.4f}°")

    # Savitzky-Golay filter
    hca_savgol = HCACalculatorSavGol(sample_path, input_data)
    result_savgol = hca_savgol.calculate_hca()
    print(f"\nSavitzky-Golay Filter:")
    print(f"  b1_max: {result_savgol.b1_max:.4f}°")
    print(f"  b2_max: {result_savgol.b2_max:.4f}°")
    print(f"  b_sum:  {result_savgol.b_sum:.4f}°")

    # Moving average filter
    hca_ma = HCACalculatorMovingAverage(sample_path, input_data)
    result_ma = hca_ma.calculate_hca()
    print(f"\nMoving Average Filter:")
    print(f"  b1_max: {result_ma.b1_max:.4f}°")
    print(f"  b2_max: {result_ma.b2_max:.4f}°")
    print(f"  b_sum:  {result_ma.b_sum:.4f}°")

    # Adaptive filter
    hca_adaptive = HCACalculatorAdaptive(sample_path, input_data)
    result_adaptive = hca_adaptive.calculate_hca()
    print(f"\nAdaptive Filter:")
    print(f"  b1_max: {result_adaptive.b1_max:.4f}°")
    print(f"  b2_max: {result_adaptive.b2_max:.4f}°")
    print(f"  b_sum:  {result_adaptive.b_sum:.4f}°")


if __name__ == "__main__":
    print("Custom HCA Calculator Example")
    print("=" * 50)
    print()
    print("This example demonstrates custom elevation filtering strategies:")
    print("1. Savitzky-Golay filter - Preserves peaks while smoothing")
    print("2. Moving average filter - Simple and fast smoothing")
    print("3. Median filter - Excellent for removing spike noise")
    print("4. Adaptive filter - Automatically chooses best strategy")
    print()

    example_usage()

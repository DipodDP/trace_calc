import math

from numpy import correlate, float64, pad, ones
from numpy.fft import fft, ifft
from numpy.typing import NDArray

from trace_calc.models.input_data import InputData
from trace_calc.models.path import HCAData, PathData
from trace_calc.service.base import BaseHCACalculator


class ElevationsFilterFFT:
    elevations: NDArray[float64]

    def filter_elevation_profile(self, aa_level):
        """Filter profile via FFT"""

        elevations_fft = fft(pad(self.elevations, (10, 10), "edge"))
        z0 = round(elevations_fft.size / (1.4 * aa_level))
        z1 = round(elevations_fft.size - elevations_fft.size / (1.4 * aa_level))
        elevations_fft[z0:z1] = 0
        filtered_elevations = ifft(elevations_fft)[10 : elevations_fft.size - 10].real

        return filtered_elevations


class ElevationsFilterCC:
    elevations: NDArray[float64]

    def filter_elevation_profile(self):
        """Filter profile via cross correlation"""

        window = ones(2) / 2
        filtered_elevations = correlate(
            pad(self.elevations, (10, 10), "edge"), window, "same"
        )[10 : self.elevations.size + 10]

        return filtered_elevations


class HCACalculator(BaseHCACalculator):
    def calculate_hca(self) -> HCAData:
        # find left hca (horizon close angle)
        start_point = self.elevations[0]
        b1_max = -360
        b1_idx = 0
        for i in range(self.distances.size):
            b1 = self.betta_calc(
                start_point,
                self.elevations[i],
                self.distances[i],
                self.antenna_a_height,
            )
            if b1 > b1_max:
                b1_max = b1
                b1_idx = i

        # find right hca
        end_point = self.elevations[-1]
        b2_max = -360
        b2_idx = 0
        for i in range(self.distances.size - 1, -1, -1):
            b2 = self.betta_calc(
                end_point,
                self.elevations[i],
                self.distances[-1] - self.distances[i],
                self.antenna_b_height,
            )
            if b2 > b2_max:
                b2_max = b2
                b2_idx = i

        b_sum = b1_max + b2_max

        return HCAData(b1_max, b2_max, b_sum, b1_idx, b2_idx)

    @staticmethod
    def betta_calc(site_height, obstacle_height, R, antenna_height=2):
        return (
            math.atan2(
                (obstacle_height - (R**2 / 12.742) - site_height - antenna_height),
                (R * 1000),
            )
            * 180
            / math.pi
        )


class HCACalculatorCC(HCACalculator, ElevationsFilterCC):
    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)

        self.elevations = self.filter_elevation_profile()
        self.hca_data = self.calculate_hca()


class HCACalculatorFFT(HCACalculator, ElevationsFilterFFT):
    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)

        self.elevations = self.filter_elevation_profile(3.6)
        self.hca_data = self.calculate_hca()

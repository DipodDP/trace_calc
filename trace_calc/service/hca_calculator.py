import math

import numpy as np
from numpy import correlate, float64, pad, ones
from numpy.fft import fft, ifft
from numpy.typing import NDArray

from trace_calc.models.input_data import InputData
from trace_calc.models.path import HCAData, PathData
from trace_calc.service.base import BaseHCACalulator


class ElevationsFilterFFT:
    elevations: NDArray[float64]

    def filter_elevation_profile(self, aa_level):
        # filter profile via FFT
        elevations_fft = fft(np.pad(self.elevations, (10, 10), "edge"))
        z0 = round(elevations_fft.size / (2 * aa_level))
        z1 = round(elevations_fft.size - elevations_fft.size / (2 * aa_level))
        elevations_fft[z0:z1] = 0
        filtered_elevations = ifft(elevations_fft)[10 : elevations_fft.size - 10].real

        return filtered_elevations


class ElevationsFilterDefault:
    elevations: NDArray[float64]

    def filter_elevation_profile(self):
        window = ones(5) / 5
        filtered_elevations = correlate(
            pad(self.elevations, (10, 10), "edge"), window, "same"
        )[10 : self.elevations.size + 10]

        return filtered_elevations


class HCACalulator(BaseHCACalulator):
    def calculate_hca(self):
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
    def betta_calc(h1, h2, R, ha=2):
        return math.atan2((h2 - (R**2 / 12.742) - h1 - ha), (R * 1000)) * 180 / math.pi


class HCACalulatorDefault(HCACalulator, ElevationsFilterDefault):
    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)

        self.elevations = self.filter_elevation_profile()
        self.hca_data = self.calculate_hca()


class HCACalulatorFFT(HCACalulator, ElevationsFilterFFT):
    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)

        self.elevations = self.filter_elevation_profile(2.5)
        self.hca_data = self.calculate_hca()

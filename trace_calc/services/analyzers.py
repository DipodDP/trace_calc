import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trace_calc.domain.units import Angle, Kilometers, Loss, Meters, Speed, Degrees
from trace_calc.models.input_data import InputData
from trace_calc.models.path import PathData, ProfileData
from trace_calc.services.base import BaseAnalyzer
from trace_calc.services.plotter import ProfilePlotter
from trace_calc.services.profile_data_calculator import DefaultProfileDataCalculator
from trace_calc.services.speed_calculators import (
    GrozaSpeedCalculator,
    SosnikSpeedCalculator,
)
from trace_calc.services.validators import validate_distance, validate_wavelength


class PlotterMixin:
    distances: NDArray[np.float64]
    profile_data: ProfileData
    input_data: InputData

    def draw_plot(self) -> None:
        plotter = ProfilePlotter(self.profile_data)
        plotter.plot(self.distances, self.input_data.path_name)


class PrinterMixin:
    def print_results(self, **values: Any) -> None:
        speed_prefix = values.get("speed_prefix")
        extra_dist = values.get("extra_dist")
        speed = values.get("speed")

        print(
            f"Extra distance = {extra_dist:.1f} km"
        ) if extra_dist is not None else ...
        print(
            f"Estimated median speed = {speed:.1f} {speed_prefix}bits/s"
        ) if speed is not None else ...

        for key, value in values.items():
            if key not in ("extra_dist", "speed"):
                print(f"{key}: {value}")


class GrozaAnalyzer(
    BaseAnalyzer,
    GrozaSpeedCalculator,
    PrinterMixin,
    PlotterMixin,
):
    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)

        profile_data_calculator = DefaultProfileDataCalculator(profile, input_data)
        self.profile_data = profile_data_calculator.profile_data
        self.hca_data = profile_data_calculator.hca_data

        hca_calc = profile_data_calculator.hca_calculator
        self.distances = hca_calc.distances
        self.elevations = hca_calc.elevations
        self.antenna_a_height = hca_calc.antenna_a_height
        self.antenna_b_height = hca_calc.antenna_b_height
        self.input_data = input_data

    @staticmethod
    def _l0_calc(R: Kilometers, lam: Meters = Meters(0.06)) -> Loss:
        """Calculate free space path loss.

        Args:
            R: Distance in kilometers
            lam: Wavelength in meters (default: 0.06m = 5 GHz)

        Returns:
            Free space loss in dB
        """
        validate_distance(R)
        validate_wavelength(lam)
        return Loss(20 * math.log10(4 * math.pi * R * 1000 / lam))

    @staticmethod
    def _lmed_calc(R: Kilometers, lam: Meters = Meters(0.06)) -> Loss:
        """Calculate median propagation loss (empirical Groza model).

        Args:
            R: Distance in kilometers
            lam: Wavelength in meters (default: 0.06m = 5 GHz)

        Returns:
            Median loss in dB
        """
        validate_distance(R)
        validate_wavelength(lam)
        l = 0.3  # Reference wavelength: 0.3m (1 GHz)
        k = (70 - 85) / (146 - 345)  # Empirical slope
        b = 70 - k * 146  # Empirical intercept
        return Loss((k * R + b) - 10 * math.log10(lam / l))

    @staticmethod
    def _lr_calc(R: Kilometers, delta: Angle) -> Loss:
        """Calculate terrain roughness loss.

        Args:
            R: Distance in kilometers
            delta: Terrain roughness parameter in degrees

        Returns:
            Roughness loss in dB
        """
        validate_distance(R)
        a = 183.6242531493953  # Empirical coefficient [km]
        b = 0.30840274015885827  # Empirical coefficient
        k = a / R + b
        c = k * delta + 1
        if c > 0:
            return Loss(20 / 3 * math.log2(c))
        else:
            return Loss(-20)

    def _delta_calc(self) -> Angle:
        if self.hca_data.b_sum < Degrees(-0.6):
            b_sum = Angle(Degrees(-0.6))
        else:
            b_sum = self.hca_data.b_sum

        return Angle(
            Degrees(
                b_sum
                + 0.056 * math.sqrt((self.antenna_a_height + self.antenna_b_height) / 2)
            )
        )

    def analyze(self, *, Lk: Loss = Loss(0.0), **kwargs: Any) -> dict[str, Any]:
        trace_dist = self.distances[-1]

        # calc losses
        L0 = self._l0_calc(trace_dist, Meters(0.06))
        Lmed = self._lmed_calc(trace_dist, Meters(0.06))
        Lr = self._lr_calc(trace_dist, self._delta_calc())

        Ltot, dL, speed = self.calculate_speed(L0, Lmed, Lr, Lk, 2)

        print(f"Total losses = {Ltot:.1f} dB")
        print(f"Delta to reference trace = {dL:.1f} dB")
        speed_prefix = "M"
        if speed < 1:
            speed = Speed(speed * 1024)
            speed_prefix = "k"

        data = {
            "L0": L0,
            "Lmed": Lmed,
            "Lr": Lr,
            "trace_dist": trace_dist,
            "b1_max": self.hca_data.b1_max,
            "b2_max": self.hca_data.b2_max,
            "b_sum": self.hca_data.b_sum,
            "Ltot": Ltot,
            "dL": dL,
            "speed": speed,
            "speed_prefix": speed_prefix,
        }
        self.print_results(**data)
        self.draw_plot()

        return data


class SosnikAnalyzer(
    BaseAnalyzer,
    SosnikSpeedCalculator,
    PrinterMixin,
    PlotterMixin,
):
    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)

        profile_data_calculator = DefaultProfileDataCalculator(profile, input_data)
        self.profile_data = profile_data_calculator.profile_data
        self.hca_data = profile_data_calculator.hca_data

        hca_calc = profile_data_calculator.hca_calculator
        self.distances = hca_calc.distances
        self.elevations = hca_calc.elevations
        self.antenna_a_height = hca_calc.antenna_a_height
        self.antenna_b_height = hca_calc.antenna_b_height
        self.input_data = input_data

    def analyze(self, **kwargs: Any) -> dict[str, Any]:
        trace_dist: Kilometers = self.distances[-1]
        b_sum: Angle = self.hca_data.b_sum

        arg = 1 + (
            b_sum
            * 60
            / (0.4 * trace_dist + b_sum * 60)
            * (1 + (b_sum * 60 / (0.2 * trace_dist)))
        )
        print(f"Argument {arg}")
        if arg > 0:
            Lr = Loss(-40 * math.log10(arg))
        else:
            Lr = Loss(0)
        speed, extra_dist = self.calculate_speed(trace_dist, Lr, b_sum)
        data = {
            "Lr": Lr,
            "trace_dist": trace_dist,
            "extra_dist": extra_dist,
            "b1_max": self.hca_data.b1_max,
            "b2_max": self.hca_data.b2_max,
            "b_sum": self.hca_data.b_sum,
            "speed": speed,
            "speed_prefix": "k",
        }
        self.print_results(**data)
        self.draw_plot()

        return data

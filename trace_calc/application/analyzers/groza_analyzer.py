import math
from typing import Any

from trace_calc.domain.models.units import (
    Angle,
    Kilometers,
    Loss,
    Meters,
    Speed,
    Degrees,
)
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import PathData
from trace_calc.application.analyzers.base import BaseServiceAnalyzer
from trace_calc.domain.speed_calculators import GrozaSpeedCalculator
from trace_calc.domain.validators import validate_distance, validate_wavelength
from trace_calc.logging_config import get_logger

logger = get_logger(__name__)


class GrozaAnalyzer(
    BaseServiceAnalyzer,
    GrozaSpeedCalculator,
):
    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)

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

        logger.debug(f"Total losses = {Ltot:.1f} dB")
        logger.debug(f"Delta to reference trace = {dL:.1f} dB")
        speed_prefix = "M"
        if speed < 1:
            speed = Speed(speed * 1024)
            speed_prefix = "k"

        data = {
            "method": "groza",
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

        return data
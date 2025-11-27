import math
from typing import Any

from trace_calc.domain.models.units import (
    Angle,
    Kilometers,
    Loss,
)
from trace_calc.domain.models.analysis import AnalyzerResult
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import PathData
from trace_calc.application.analyzers.base import BaseServiceAnalyzer
from trace_calc.domain.speed_calculators import SosnikSpeedCalculator
from trace_calc.logging_config import get_logger

logger = get_logger(__name__)


class SosnikAnalyzer(
    BaseServiceAnalyzer,
    SosnikSpeedCalculator,
):
    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)

    def analyze(self, **kwargs: Any) -> AnalyzerResult:
        trace_dist: Kilometers = self.distances[-1]
        b_sum: Angle = self.hca_data.b_sum

        c = 299792458
        frequency_hz = self.input_data.frequency_mhz * 1e6
        wavelength_m = c / frequency_hz

        arg = 1 + (
            b_sum
            * 60
            / (0.4 * trace_dist + b_sum * 60)
            * (1 + (b_sum * 60 / (0.2 * trace_dist)))
        )
        logger.debug(f"Argument {arg}")
        if arg > 0:
            L_correction = Loss(-40 * math.log10(arg))
        else:
            L_correction = Loss(0)
        speed, extra_dist, equal_dist = self.calculate_speed(
            trace_dist, L_correction, b_sum
        )

        model_parameters = {
            "L_correction": L_correction,
            "extra_dist": extra_dist,
            "equal_dist": equal_dist,
            "method": "sosnik",
        }

        return AnalyzerResult(
            model_parameters=model_parameters,
            link_speed=speed,
            wavelength=wavelength_m,
            hca=self.hca_data,
            profile_data=self.profile_data,
            speed_prefix="k",
        )

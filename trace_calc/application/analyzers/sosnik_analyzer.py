import math
from typing import Any

from trace_calc.domain.models.units import (
    Angle,
    Kilometers,
    Loss,
)
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

    def analyze(self, **kwargs: Any) -> dict[str, Any]:
        trace_dist: Kilometers = self.distances[-1]
        b_sum: Angle = self.hca_data.b_sum

        arg = 1 + (
            b_sum
            * 60
            / (0.4 * trace_dist + b_sum * 60)
            * (1 + (b_sum * 60 / (0.2 * trace_dist)))
        )
        logger.debug(f"Argument {arg}")
        if arg > 0:
            Lr = Loss(-40 * math.log10(arg))
        else:
            Lr = Loss(0)
        speed, extra_dist = self.calculate_speed(trace_dist, Lr, b_sum)
        data = {
            "method": "sosnik",
            "Lr": Lr,
            "trace_dist": trace_dist,
            "extra_dist": extra_dist,
            "b1_max": self.hca_data.b1_max,
            "b2_max": self.hca_data.b2_max,
            "b_sum": self.hca_data.b_sum,
            "speed": speed,
            "speed_prefix": "k",
        }

        return data
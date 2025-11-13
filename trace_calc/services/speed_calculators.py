from typing import Tuple

from trace_calc.domain.units import Angle, Kilometers, Loss, Speed
from trace_calc.services.base import BaseSpeedCalculator


class GrozaSpeedCalculator(BaseSpeedCalculator):
    def calculate_speed(
        self, L0: Loss, Lmed: Loss, Lr: Loss, Lk: Loss, ld: float
    ) -> Tuple[Loss, Loss, Speed]:
        L: Loss = Loss(L0 + Lmed + Lr + Lk + ld)
        dL: Loss = Loss(L - 233.8)
        if dL > -1.66:
            speed: Speed = Speed(15.2 * 10 ** (-dL / 10))
        elif dL < -6.66:  # -1.66 - 5
            speed = Speed(44.6)
        else:
            speed = Speed(22.3)
        return L, dL, speed


class SosnikSpeedCalculator(BaseSpeedCalculator):
    def calculate_speed(
        self, trace_dist: Kilometers, Lr: Loss, b_sum: Angle
    ) -> Tuple[Speed, Kilometers]:
        extra_dist: Kilometers = Kilometers(148 * b_sum) if b_sum > 0 else Kilometers(0)
        equal_dist: Kilometers = Kilometers(trace_dist + extra_dist)

        if trace_dist < 40 and equal_dist < 90 and Lr >= -35:
            speed: Speed = Speed(2048)
        elif Lr < -45:
            speed = Speed(0)
        elif equal_dist < 120:
            speed = Speed(512)
        elif equal_dist < 140:
            speed = Speed(256)
        elif equal_dist < 210:
            speed = Speed(64)
        else:
            speed = Speed(0)

        return speed, extra_dist

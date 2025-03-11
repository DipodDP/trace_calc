from typing import Tuple

from trace_calc.service.base import BaseCalculator


class GrozaCalculator(BaseCalculator):
    def calculate(self, L0, Lmed, Lr, Lk, ld) -> Tuple[float, float, float]:
        L = L0 + Lmed + Lr + Lk + ld
        dL = L - 233.8
        if dL > -1.66:
            speed = 15.2 * 10 ** (-dL / 10)
        elif dL < -6.66:  # -1.66 - 5
            speed = 44.6
        else:
            speed = 22.3
        return L, dL, speed


class SosnikCalculator(BaseCalculator):
    def calculate(self, trace_dist, Lr, b_sum) -> Tuple[float, float]:
        extra_dist = 148 * b_sum if b_sum > 0 else 0
        equal_dist = trace_dist + extra_dist

        if trace_dist < 40 and equal_dist < 90 and Lr >= -35:
            speed = 2048
        elif Lr < -45:
            speed = 0
        elif equal_dist < 120:
            speed = 512
        elif equal_dist < 140:
            speed = 256
        elif equal_dist < 210:
            speed = 64
        else:
            speed = 0

        return speed, extra_dist

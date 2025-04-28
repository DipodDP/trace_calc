import math

from trace_calc.service.base import BaseAnalyzer
from trace_calc.service.profile_data_calculator import DefaultProfileDataCalculator
from trace_calc.service.speed_calculators import (
    GrozaSpeedCalculator,
    SosnikSpeedCalculator,
)


class PrinterMixin:
    def print_results(self, **values) -> None:
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
    DefaultProfileDataCalculator,
    GrozaSpeedCalculator,
    PrinterMixin,
):
    @staticmethod
    def _l0_calc(R, lam=0.06):
        return 20 * math.log10(4 * math.pi * R * 1000 / lam)

    @staticmethod
    def _lmed_calc(R, lam=0.06):
        l = 0.3
        k = (70 - 85) / (146 - 345)
        b = 70 - k * 146
        return (k * R + b) - 10 * math.log10(lam / l)

    @staticmethod
    def _lr_calc(R, delta):
        a = 183.6242531493953
        b = 0.30840274015885827
        k = a / R + b
        c = k * delta + 1
        if c > 0:
            return 20 / 3 * math.log2(c)
        else:
            return -20

    def _delta_calc(self):
        if self.hca_data.b_sum < -0.6:
            b_sum = -0.6
        else:
            b_sum = self.hca_data.b_sum

        return b_sum + 0.056 * math.sqrt(
            (self.antenna_a_height + self.antenna_b_height) / 2
        )

    def analyze(self, *, Lk=0.0, **kwargs) -> dict:
        trace_dist = self.distances[-1]

        # calc losses
        L0 = self._l0_calc(trace_dist)
        Lmed = self._lmed_calc(trace_dist)
        Lr = self._lr_calc(trace_dist, self._delta_calc())

        Ltot, dL, speed = self.calculate_speed(L0, Lmed, Lr, Lk, 2)

        print(f"Total losses = {Ltot:.1f} dB")
        print(f"Delta to reference trace = {dL:.1f} dB")
        speed_prefix = "M"
        if speed < 1:
            speed *= 1024
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

        return data


class SosnikAnalyzer(
    BaseAnalyzer,
    DefaultProfileDataCalculator,
    SosnikSpeedCalculator,
    PrinterMixin,
):
    def analyze(self, **kwargs) -> dict:
        trace_dist = self.distances[-1]
        b_sum = self.hca_data.b_sum

        arg = 1 + (
            self.hca_data.b_sum
            * 60
            / (0.4 * trace_dist + b_sum * 60)
            * (1 + (b_sum * 60 / (0.2 * trace_dist)))
        )
        print(f"Argument {arg}")
        if arg > 0:
            Lr = -40 * math.log10(arg)
        else:
            Lr = 0
        speed, extra_dist = self.calculate_speed(trace_dist, Lr, self.hca_data.b_sum)
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

        return data

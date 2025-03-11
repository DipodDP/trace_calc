from trace_calc.service.base import BaseAnalyzer
from trace_calc.service.path_calculator import GrozaCalculator, SosnikCalculator


class GrozaAnalyzer(BaseAnalyzer):

    calculator = GrozaCalculator()
    async def analyze(self) -> dict:
        # print(f"!--- Groza -------- Bot mode is: {self.bot_mode} -----------!")
        # path = await self.path_loader.load(self.path_filename)

        # Assume profile_an_legacy is defined elsewhere and returns the following tuple:
        # (L0, Lmed, Lr, trace_dist, b1_max, b2_max, b_sum)
        L0, Lmed, Lr, trace_dist, b1_max, b2_max, b_sum = profile_an_legacy(
            path, self.path_filename, ha1=self.ha1, ha2=self.ha2
        )

        Ltot, dL, speed = self.calculator.calculate(L0, Lmed, Lr, self.Lk, 2)
        print(f"Total losses = {Ltot:.1f} dB")
        print(f"Delta to reference trace = {dL:.1f} dB")

        sp_pref = "M"
        if speed < 1:
            speed *= 1024
            sp_pref = "k"
        print(f"Estimated median speed = {speed:.1f} {sp_pref}bits/s")

        return {
            "L0": L0,
            "Lmed": Lmed,
            "Lr": Lr,
            "trace_dist": trace_dist,
            "b1_max": b1_max,
            "b2_max": b2_max,
            "b_sum": b_sum,
            "Ltot": Ltot,
            "dL": dL,
            "speed": speed,
            "sp_pref": sp_pref,
        }


class SosnikAnalyzer(BaseAnalyzer):
    calculator = SosnikCalculator()
    async def analyze(self) -> dict:
        # print(f"!--- Sosnik -------- Bot mode is: {self.bot_mode} -----------!")
        # path = await self.path_loader.load(self.path_filename)

        # Assume profile_analyzer is defined elsewhere and returns:
        # (trace_dist, b1_max, b2_max, b_sum, Lr)
        trace_dist, b1_max, b2_max, b_sum, Lr = profile_analyzer(
            path, self.path_filename, ha1=self.ha1, ha2=self.ha2
        )

        speed, extra_dist = self.calculator.calculate(trace_dist, Lr, b_sum)
        sp_pref = "k"
        print(f"Extra distance = {extra_dist:.1f} km")
        print(f"Estimated median speed = {speed:.1f} {sp_pref}bits/s")

        return {
            "trace_dist": trace_dist,
            "extra_dist": extra_dist,
            "b1_max": b1_max,
            "b2_max": b2_max,
            "b_sum": b_sum,
            "Lr": Lr,
            "speed": speed,
            "sp_pref": sp_pref,
        }

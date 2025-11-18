import numpy as np
from trace_calc.domain.models.coordinates import InputData
from trace_calc.application.services.hca import HCACalculatorCC, HCACalculatorFFT
from tests.fixtures import test_profile


def test_hca_calculator_cross_correlation():
    result = HCACalculatorCC(test_profile, InputData("test_file", frequency_mhz=1000.0))
    assert f"{result.hca_data.b_sum:.3f}" == "0.552"


def test_hca_calculator_fft():
    result = HCACalculatorFFT(
        test_profile, InputData("test_file", frequency_mhz=1000.0)
    )
    assert f"{result.hca_data.b_sum:.3f}" == "0.352"

from trace_calc.domain.models.coordinates import InputData
from trace_calc.application.analysis import GrozaAnalyzer, SosnikAnalyzer
from .test_hca_calculator import test_profile


def test_sosnic_analyzer():
    analyzer = SosnikAnalyzer(test_profile, InputData('test_file', frequency_mhz=1000.0))
    result = analyzer.analyze()
    assert f"{result.get('b_sum'):.3f}" == "0.552"
    assert result.get("speed") == 64
    assert result.get("speed_prefix") == "k"

def test_groza_analyzer():
    analyzer = GrozaAnalyzer(test_profile, InputData('test_file', frequency_mhz=1000.0))
    result = analyzer.analyze()
    assert f"{result.get('b_sum'):.3f}" == "0.552"
    assert result.get("speed") == 22.3
    assert result.get("speed_prefix") == "M"

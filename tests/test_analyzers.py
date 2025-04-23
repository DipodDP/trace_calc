from trace_calc.models.input_data import InputData
from trace_calc.service.analyzers import GrozaAnalyzer, SosnikAnalyzer
from .test_hca_calculator import test_profile


def test_sosnic_analyzer():
    analyzer = SosnikAnalyzer(test_profile, InputData('test_file'))
    result = analyzer.analyze()
    assert f"{result.get('b_sum'):.3f}" == "0.163"
    assert result.get("speed") == 256
    assert result.get("speed_prefix") == "k"

def test_groza_analyzer():
    analyzer = GrozaAnalyzer(test_profile, InputData('test_file'))
    result = analyzer.analyze()
    assert f"{result.get('b_sum'):.3f}" == "0.163"
    assert result.get("speed") == 44.6
    assert result.get("speed_prefix") == "M"

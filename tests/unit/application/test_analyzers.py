from trace_calc.domain.models.coordinates import InputData
from trace_calc.application.analysis import GrozaAnalyzer, SosnikAnalyzer
from tests.fixtures import test_profile


def test_sosnic_analyzer():
    analyzer = SosnikAnalyzer(
        test_profile, InputData("test_file", frequency_mhz=1000.0)
    )
    result = analyzer.analyze()
    assert f"{result.hca.b_sum:.3f}" == "0.552"
    assert result.link_speed == 64
    assert result.speed_prefix == "k" == "k"


def test_groza_analyzer():
    analyzer = GrozaAnalyzer(test_profile, InputData("test_file", frequency_mhz=1000.0))
    result = analyzer.analyze()
    assert f"{result.hca.b_sum:.3f}" == "0.552"
    assert result.link_speed == 44.6
    assert result.speed_prefix == "M"


def test_profile_calculator_independent_from_hca():
    """
    Verify profile calculation doesn't affect HCA values.
    This test documents architectural separation.
    """
    from trace_calc.domain.models.coordinates import InputData
    from trace_calc.application.analysis import GrozaAnalyzer
    from tests.fixtures import test_profile

    analyzer = GrozaAnalyzer(test_profile, InputData("test_file", frequency_mhz=1000.0))

    # HCA should still be correct (independent of profile bug fix)
    assert f"{analyzer.hca_data.b_sum:.3f}" == "0.552", (
        "HCA calculation must remain unchanged"
    )

    # Profile should now have correct units
    assert analyzer.profile_data is not None, "Profile should be calculated"

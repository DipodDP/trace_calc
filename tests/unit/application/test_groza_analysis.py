"""Unit tests for Groza propagation analysis (no I/O)"""

import pytest
import numpy as np
from trace_calc.application.analysis import GrozaAnalysisService
from trace_calc.domain.models.path import PathData
from trace_calc.domain.models.coordinates import InputData, Coordinates
from trace_calc.domain.models.units import Meters
from trace_calc.domain.models.analysis import AnalysisResult


@pytest.fixture
def sample_path_data():
    """Create sample path data for testing"""
    return PathData(
        coordinates=np.array([[50.0, 14.0], [50.1, 14.1]]),
        distances=np.linspace(0, 100, 101),  # 100 km, 101 points
        elevations=np.random.uniform(200, 500, 101),  # Random elevations
    )


@pytest.fixture
def sample_input_data():
    """Create sample input data"""
    from trace_calc.domain.models.units import Angle

    return InputData(
        path_name="test_path",
        frequency_mhz=1000.0,
        site_a_coordinates=Coordinates(50.0, 14.0),
        site_b_coordinates=Coordinates(50.1, 14.1),
        antenna_a_height=Meters(300.0),
        antenna_b_height=Meters(350.0),
        hpbw=Angle(0.0),  # Use zero offset for basic tests
    )


@pytest.mark.asyncio
async def test_groza_analysis_returns_result(sample_path_data, sample_input_data):
    service = GrozaAnalysisService()

    result = await service.analyze(
        path=sample_path_data,
        input_data=sample_input_data,
        antenna_a_height=10.0,
        antenna_b_height=10.0,
    )

    # Verify it returns correct type
    assert isinstance(result, AnalysisResult)

    # Verify all fields are populated
    assert result.basic_transmission_loss > 0
    assert result.total_path_loss > 0
    assert result.link_speed > 0
    assert result.wavelength > 0
    assert result.metadata["method"] == "groza"


@pytest.mark.asyncio
async def test_groza_analysis_no_side_effects(sample_path_data, sample_input_data, capsys):
    service = GrozaAnalysisService()

    result = await service.analyze(
        path=sample_path_data,
        input_data=sample_input_data,
        antenna_a_height=10.0,
        antenna_b_height=10.0,
    )

    # Capture stdout/stderr
    captured = capsys.readouterr()

    # Should have NO output (no printing!)
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.asyncio
async def test_groza_analysis_deterministic(sample_path_data, sample_input_data):
    """Test that analysis is deterministic (same inputs = same outputs)"""
    service1 = GrozaAnalysisService()
    service2 = GrozaAnalysisService()

    result1 = await service1.analyze(
        path=sample_path_data,
        input_data=sample_input_data,
        antenna_a_height=10.0,
        antenna_b_height=10.0,
    )
    result2 = await service2.analyze(
        path=sample_path_data,
        input_data=sample_input_data,
        antenna_a_height=10.0,
        antenna_b_height=10.0,
    )

    # Results should be identical
    assert result1.basic_transmission_loss == result2.basic_transmission_loss
    assert result1.total_path_loss == result2.total_path_loss
    assert result1.link_speed == result2.link_speed


@pytest.mark.asyncio
async def test_groza_wavelength_calculation(sample_path_data, sample_input_data):
    """Test wavelength calculation: λ = c / f"""
    service = GrozaAnalysisService()

    result = await service.analyze(
        path=sample_path_data,
        input_data=sample_input_data,
        antenna_a_height=10.0,
        antenna_b_height=10.0,
    )

    # Frequency = 1000 MHz = 1e9 Hz
    # Speed of light = 3e8 m/s
    # Expected wavelength = 3e8 / 1e9 = 0.3 m
    expected_wavelength = 3e8 / (sample_input_data.frequency_mhz * 1e6)

    assert abs(result.wavelength - expected_wavelength) < 0.01

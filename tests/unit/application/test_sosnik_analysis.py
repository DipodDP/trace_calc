"""Unit tests for Sosnik propagation analysis (no I/O)"""

import pytest
import numpy as np
import math
from unittest.mock import MagicMock

from trace_calc.application.analysis import SosnikAnalysisService
from trace_calc.application.analyzers.sosnik_analyzer import SosnikAnalyzer
from trace_calc.domain.models.path import (
    PathData, HCAData, ProfileData,
    IntersectionsData, IntersectionPoint, SightLinesData, VolumeData, ProfileViewData
)
from trace_calc.domain.models.coordinates import InputData, Coordinates
from trace_calc.domain.models.units import Meters, Angle, Kilometers, Loss, Speed
from trace_calc.domain.models.analysis import AnalysisResult
from trace_calc.domain.speed_calculators import SosnikSpeedCalculator

@pytest.fixture
def sample_path_data():
    """Create sample path data for testing"""
    distances = np.linspace(0, 100, 101)
    # Create a simple concave (valley) profile
    elevations = 300 + ((distances - 50) ** 2) / 25
    return PathData(
        coordinates=np.array([[50.0, 14.0], [50.1, 14.1]]),
        distances=distances,
        elevations=elevations,
    )

@pytest.fixture
def sample_input_data():
    """Create sample input data"""
    return InputData(
        path_name="test_path",
        frequency_mhz=1000.0,
        site_a_coordinates=Coordinates(50.0, 14.0),
        site_b_coordinates=Coordinates(50.1, 14.1),
        antenna_a_height=Meters(300.0),
        antenna_b_height=Meters(350.0),
        hpbw=Angle(0.1),  # Use a small non-zero angle
    )

@pytest.fixture
def simple_profile_data():
    """A simple, valid ProfileData object to avoid geometry calculation errors."""
    return ProfileData(
        plain=ProfileViewData(elevations=np.array([1, 2]), baseline=np.array([1, 2])),
        curved=ProfileViewData(elevations=np.array([1, 2]), baseline=np.array([1, 2])),
        lines_of_sight=SightLinesData(
            lower_a=np.array([0, 0]), lower_b=np.array([0, 0]),
            upper_a=np.array([0, 0]), upper_b=np.array([0, 0]),
            antenna_elevation_angle_a=np.array([0, 0]),
            antenna_elevation_angle_b=np.array([0, 0]),
        ),
        intersections=IntersectionsData(
            lower=IntersectionPoint(0, 0, 0, Angle(0)),
            upper=IntersectionPoint(0, 0, 0, Angle(0)),
            cross_ab=IntersectionPoint(0, 0, 0, Angle(0)),
            cross_ba=IntersectionPoint(0, 0, 0, Angle(0)),
            beam_intersection_point=IntersectionPoint(0, 0, 0, Angle(0)),
        ),
        volume=VolumeData(
            cone_intersection_volume_m3=0,
            distance_a_to_cross_ab=0, distance_b_to_cross_ba=0,
            distance_between_crosses=0, distance_a_to_lower_intersection=0,
            distance_b_to_lower_intersection=0, distance_a_to_upper_intersection=0,
            distance_b_to_upper_intersection=0,
            distance_between_lower_upper_intersections=0,
            antenna_elevation_angle_a=Angle(0), antenna_elevation_angle_b=Angle(0),
        )
    )

@pytest.mark.asyncio
async def test_sosnik_analysis_service_returns_result(sample_path_data, sample_input_data, simple_profile_data, monkeypatch):
    """Test that the SosnikAnalysisService returns a valid AnalysisResult."""
    service = SosnikAnalysisService()
    
    monkeypatch.setattr(
        "trace_calc.application.analyzers.base.DefaultProfileDataCalculator",
        lambda *args, **kwargs: MagicMock(
            profile_data=simple_profile_data,
            hca_data=HCAData(b1_max=0, b2_max=0, b_sum=Angle(0.1), b1_idx=0, b2_idx=0),
            hca_calculator=MagicMock(distances=np.linspace(0, 100, 101)),
        ),
    )

    result = await service.analyze(
        path=sample_path_data,
        input_data=sample_input_data,
    )
    
    assert isinstance(result, AnalysisResult)
    assert result.result["method"] == "sosnik"
    assert "L_correction" in result.model_propagation_loss_parameters
    assert "extra_distance_km" in result.model_propagation_loss_parameters
    assert "equal_dist" in result.model_propagation_loss_parameters
    assert result.link_speed is not None
    assert result.wavelength > 0
    assert result.result["speed_prefix"] == "k"

# Test cases for SosnikSpeedCalculator
# (trace_dist, L_correction, b_sum, expected_speed)
speed_test_cases = [
    (Kilometers(30), Loss(-10), Angle(0.1), Speed(2048)),  # trace_dist < 40, equal_dist < 90, L_correction >= -35
    (Kilometers(100), Loss(-50), Angle(0.1), Speed(0)),    # L_correction < -45
    (Kilometers(100), Loss(-40), Angle(0.1), Speed(512)),  # equal_dist < 120
    (Kilometers(120), Loss(-40), Angle(0.1), Speed(256)),  # equal_dist < 140
    (Kilometers(150), Loss(-40), Angle(0.1), Speed(64)),   # equal_dist < 210
    (Kilometers(220), Loss(-40), Angle(0.1), Speed(0)),    # equal_dist >= 210
]

@pytest.mark.parametrize("trace_dist, L_correction, b_sum, expected_speed", speed_test_cases)
def test_sosnik_speed_calculator_branches(trace_dist, L_correction, b_sum, expected_speed):
    """Test all branches of the SosnikSpeedCalculator logic."""
    calculator = SosnikSpeedCalculator()
    speed, extra_dist, equal_dist = calculator.calculate_speed(trace_dist, L_correction, b_sum)
    assert speed == expected_speed

def test_sosnik_analyzer_model_parameters(sample_path_data, sample_input_data, simple_profile_data, monkeypatch):
    """Test that the SosnikAnalyzer correctly calculates and returns model parameters."""
    
    monkeypatch.setattr(
        "trace_calc.application.analyzers.base.DefaultProfileDataCalculator",
        lambda *args, **kwargs: MagicMock(
            profile_data=simple_profile_data,
            hca_data=HCAData(b1_max=0, b2_max=0, b_sum=Angle(0.1), b1_idx=0, b2_idx=0),
            hca_calculator=MagicMock(distances=np.linspace(0, 100, 101)),
        ),
    )
    
    analyzer = SosnikAnalyzer(sample_path_data, sample_input_data)
    result = analyzer.analyze()

    model_params = result.model_parameters
    assert "L_correction" in model_params
    assert "extra_distance_km" in model_params
    assert "equal_dist" in model_params

    # Recalculate expected values for verification
    trace_dist = analyzer.distances[-1]
    b_sum = analyzer.hca_data.b_sum
    
    arg = 1 + (
        b_sum
        * 60
        / (0.4 * trace_dist + b_sum * 60)
        * (1 + (b_sum * 60 / (0.2 * trace_dist)))
    )
    expected_L_correction = Loss(-40 * math.log10(arg)) if arg > 0 else Loss(0)
    
    expected_extra_dist = Kilometers(148 * b_sum) if b_sum > 0 else Kilometers(0)
    expected_equal_dist = Kilometers(trace_dist + expected_extra_dist)
    
    assert model_params["L_correction"] == pytest.approx(expected_L_correction)
    assert model_params["extra_distance_km"] == pytest.approx(expected_extra_dist)
    assert model_params["equal_dist"] == pytest.approx(expected_equal_dist)
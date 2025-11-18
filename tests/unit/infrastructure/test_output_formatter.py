"""Test output formatting (console display)"""

import pytest
from io import StringIO
import sys
import numpy as np

from trace_calc.infrastructure.output.formatters import ConsoleOutputFormatter
from trace_calc.domain.models.analysis import AnalysisResult, PropagationLoss
from trace_calc.domain.models.path import (
    ProfileData,
    IntersectionsData,
    IntersectionPoint,
    SightLinesData,
    VolumeData,
)


@pytest.fixture
def sample_analysis_result():
    """Create sample analysis result for testing"""
    # Dummy SightLinesData
    dummy_sight_lines = SightLinesData(
        lower_a=np.array([0.1, 100.0]),
        lower_b=np.array([-0.1, 200.0]),
        upper_a=np.array([0.2, 100.0]),
        upper_b=np.array([-0.2, 220.0]),
        bisector_a=np.array([150.0, 160.0]),
        bisector_b=np.array([210.0, 200.0]),
    )

    # Dummy IntersectionPoint (values in meters)
    dummy_lower_int = IntersectionPoint(
        distance_km=50.0, elevation_sea_level=100.0, elevation_terrain=10.0
    )
    dummy_upper_int = IntersectionPoint(
        distance_km=60.0, elevation_sea_level=120.0, elevation_terrain=20.0
    )
    dummy_cross_ab = IntersectionPoint(
        distance_km=70.0, elevation_sea_level=130.0, elevation_terrain=30.0
    )
    dummy_cross_ba = IntersectionPoint(
        distance_km=80.0, elevation_sea_level=140.0, elevation_terrain=40.0
    )

    # Dummy IntersectionsData
    dummy_intersections = IntersectionsData(
        lower_intersection=dummy_lower_int,
        upper_intersection=dummy_upper_int,
        cross_ab=dummy_cross_ab,
        cross_ba=dummy_cross_ba,
    )

    # Dummy VolumeData (cone_intersection_volume_m3 is in m^3, others in km)
    dummy_volume = VolumeData(
        cone_intersection_volume_m3=1000000000,  # 1 km^3
        distance_a_to_cross_ab=10.0,
        distance_b_to_cross_ba=20.0,
        distance_between_crosses=30.0,
        distance_a_to_lower_intersection=5.0,
        distance_b_to_lower_intersection=15.0,
        distance_a_to_upper_intersection=25.0,
        distance_b_to_upper_intersection=35.0,
        distance_between_lower_upper_intersections=45.0,
    )

    dummy_profile_data = ProfileData(
        plain=None,  # Not relevant for this test
        curved=None,  # Not relevant for this test
        lines_of_sight=dummy_sight_lines,
        intersections=dummy_intersections,
        volume=dummy_volume,
    )

    return AnalysisResult(
        basic_transmission_loss=120.5,
        total_path_loss=145.2,
        link_speed=87.3,
        wavelength=0.3,
        propagation_loss=PropagationLoss(
            free_space_loss=92.4,
            atmospheric_loss=0.5,
            diffraction_loss=12.3,
            total_loss=105.2,
        ),
        metadata={
            "method": "groza",
            "distance_km": 100.5,
            "frequency_mhz": 1000.0,
            "profile_data": dummy_profile_data,
        },
    )


def test_console_formatter_prints_summary(sample_analysis_result):
    """Test that formatter prints readable summary"""
    formatter = ConsoleOutputFormatter()

    # Capture output
    captured_output = StringIO()
    sys.stdout = captured_output

    formatter.format_result(sample_analysis_result)

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    # Verify key information is printed
    assert "GROZA Analysis Result" in output
    assert "120.5" in output  # Basic loss
    assert "145.2" in output  # Total loss
    assert "87.3" in output  # Link speed

    # Verify kilometer units and values for profile data
    assert "intercept=0.1000km" in output
    assert "intercept=0.2000km" in output
    assert "0.1300km ASL" in output # cross_ab.elevation_sea_level / 1000
    assert "0.0300km above terrain" in output # cross_ab.elevation_terrain / 1000
    assert "Common scatter volume: 1.000 km³" in output # volume.cone_intersection_volume_m3 / 1e9
    assert "Common volume bottom (lower intersection): 50.000 km, 0.0100km above terrain, 0.1000km ASL" in output
    assert "Common volume top (upper intersection): 60.000 km, 0.0200km above terrain, 0.1200km ASL" in output


def test_console_formatter_handles_missing_loss_breakdown(sample_analysis_result):
    """Test formatter handles missing propagation_loss gracefully"""
    result_without_loss = AnalysisResult(
        basic_transmission_loss=120.5,
        total_path_loss=145.2,
        link_speed=87.3,
        wavelength=0.3,
        propagation_loss=None,  # Missing!
        metadata={"method": "groza"},
    )

    formatter = ConsoleOutputFormatter()

    # Should not raise exception
    captured_output = StringIO()
    sys.stdout = captured_output
    formatter.format_result(result_without_loss)
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    assert "120.5" in output  # Still prints basic info

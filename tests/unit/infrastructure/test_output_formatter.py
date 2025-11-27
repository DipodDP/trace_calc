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
    ProfileViewData,
)
from trace_calc.domain.models.units import Angle
from trace_calc.domain.models.coordinates import InputData, Coordinates
from trace_calc.domain.models.path import GeoData


@pytest.fixture
def sample_geo_data():
    """Create sample geo data for testing"""
    return GeoData(
        distance=100.0,
        mag_declination_a=Angle(5.0),
        mag_declination_b=Angle(-3.0),
        true_azimuth_a_b=Angle(90.0),
        true_azimuth_b_a=Angle(270.0),
        mag_azimuth_a_b=Angle(85.0),
        mag_azimuth_b_a=Angle(273.0),
    )


@pytest.fixture
def sample_input_data():
    """Create sample input data for testing"""
    return InputData(
        path_name="test_path",
        site_a_coordinates=Coordinates(lat=50.0, lon=14.0),
        site_b_coordinates=Coordinates(lat=50.1, lon=14.1),
        frequency_mhz=1000.0,
        hpbw=Angle(2.5),
    )


@pytest.fixture
def sample_analysis_result(sample_input_data, sample_geo_data):
    """Create sample analysis result for testing"""
    # Dummy SightLinesData
    dummy_sight_lines = SightLinesData(
        lower_a=np.array([0.1, 100.0]),
        lower_b=np.array([-0.1, 200.0]),
        upper_a=np.array([0.2, 100.0]),
        upper_b=np.array([-0.2, 220.0]),
        antenna_elevation_angle_a=np.array([150.0, 160.0]),
        antenna_elevation_angle_b=np.array([210.0, 200.0]),
    )

    # Dummy IntersectionPoint (values in meters)
    dummy_lower_intersection = IntersectionPoint(
        distance_km=50.0, elevation_sea_level=100.0, elevation_terrain=10.0
    )
    dummy_upper_intersection = IntersectionPoint(
        distance_km=60.0, elevation_sea_level=120.0, elevation_terrain=20.0
    )
    dummy_cross_ab = IntersectionPoint(
        distance_km=70.0, elevation_sea_level=130.0, elevation_terrain=30.0
    )
    dummy_cross_ba = IntersectionPoint(
        distance_km=80.0, elevation_sea_level=140.0, elevation_terrain=40.0
    )
    dummy_beam_intersection_point = IntersectionPoint(
        distance_km=90.0,
        elevation_sea_level=150.0,
        elevation_terrain=50.0,
        angle=Angle(12.34),
    )

    # Dummy IntersectionsData
    dummy_intersections = IntersectionsData(
        lower=dummy_lower_intersection,
        upper=dummy_upper_intersection,
        cross_ab=dummy_cross_ab,
        cross_ba=dummy_cross_ba,
        beam_intersection_point=dummy_beam_intersection_point,
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
        antenna_elevation_angle_a=Angle(1.23),
        antenna_elevation_angle_b=Angle(-4.56),
    )

    dummy_profile_data = ProfileData(
        plain=ProfileViewData(elevations=np.array([]), baseline=np.array([])),
        curved=ProfileViewData(elevations=np.array([]), baseline=np.array([])),
        lines_of_sight=dummy_sight_lines,
        intersections=dummy_intersections,
        volume=dummy_volume,
    )

    return AnalysisResult(
        link_speed=87.3,
        wavelength=0.3,
        model_propagation_loss_parameters={
            "basic_transmission_loss": 120.5,
            "total_loss": 145.2,
            "propagation_loss": PropagationLoss(
                free_space_loss=132.4,
                atmospheric_loss=0.5,
                diffraction_loss=12.3,
                refraction_loss=0.0,
                total_loss=145.2,
            ),
        },
        result={
            "method": "groza",
            "distance_km": 100.5,
            "frequency_mhz": 1000.0,
            "hpbw": sample_input_data.hpbw,
            "profile_data": dummy_profile_data,
            "geo_data": sample_geo_data,  # Add geo_data to result
        },
    )


def test_console_formatter_prints_summary(
    sample_analysis_result, sample_input_data, sample_geo_data
):
    """Test that formatter prints readable summary with geo data"""
    formatter = ConsoleOutputFormatter()

    # Capture output
    captured_output = StringIO()
    sys.stdout = captured_output

    profile_data = sample_analysis_result.result.get("profile_data")
    formatter.format_result(
        sample_analysis_result,
        input_data=sample_input_data,
        geo_data=sample_geo_data,
        profile_data=profile_data,
    )

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    # Verify key information is printed
    assert "GROZA Analysis Result" in output
    assert "Total Path Loss (Ltot):  145.20 dB" in output  # Total loss
    assert "Estimated Speed:         87.3 Mbps" in output  # Link speed

    # Verify link parameters are printed and formatted
    assert "Wavelength:              0.30 m" in output
    assert "Frequency:               1000.00 MHz" in output
    assert "HPBW:                    2.50°" in output  # From sample_input_data

    # Verify site coordinates are printed
    assert "Site Coordinates:" in output
    assert "Site A:                  50.000000°, 14.000000°" in output
    assert "Site B:                  50.100000°, 14.100000°" in output

    # Verify geo data is printed
    assert "Geographic Data:" in output
    assert "Distance:                100.00 km" in output  # From sample_geo_data

    # Verify geo data is printed
    assert "Geographic Data:" in output
    assert "Distance:                100.00 km" in output
    assert "True Azimuth A→B:        90.00°" in output
    assert "True Azimuth B→A:        270.00°" in output
    assert "Mag Declination A:       5.00°" in output
    assert "Mag Declination B:       -3.00°" in output
    assert "Mag Azimuth A→B:         85.00°" in output
    assert "Mag Azimuth B→A:         273.00°" in output

    # Verify antennas elevations intersection sections
    assert "Beam Intersection Point:" in output
    assert "Distance: 90.00 km" in output
    assert "Elevation ASL: 0.15 km" in output
    assert "Elevation above terrain: 0.05 km" in output
    assert "Intersection angle: 12.34°" in output
    assert "Antenna Elevation Angles:" in output
    assert "Antenna Elevation Angle A: 1.23°" in output
    assert "Antenna Elevation Angle B: -4.56°" in output

    # Verify kilometer units and values for profile data
    # Note: The cross_ab line in the formatter is missing elevation details, this is expected
    assert "Upper A x Lower B: 70.00 km" in output
    assert "Upper B x Lower A: 80.00 km" in output
    assert "0.14 km ASL" in output  # cross_ba.elevation_sea_level / 1000
    assert "0.04 km above terrain" in output  # cross_ba.elevation_terrain / 1000
    assert (
        "Common scatter volume: 1.00 km³" in output
    )  # volume.cone_intersection_volume_m3 / 1e9
    assert "Common volume bottom (lower intersection): 50.00 km" in output
    assert "0.01 km above terrain" in output
    assert "0.10 km ASL" in output
    assert "Common volume top (upper intersection): 60.00 km" in output
    assert "0.02 km above terrain" in output
    assert "0.12 km ASL" in output


def test_console_formatter_handles_missing_loss_breakdown(
    sample_analysis_result, sample_input_data
):
    """Test formatter handles missing propagation_loss gracefully"""
    result_without_loss = AnalysisResult(
        link_speed=87.3,
        wavelength=0.3,
        model_propagation_loss_parameters={
            "basic_transmission_loss": 120.5,
            "total_loss": 145.2,
            "propagation_loss": None,  # Missing!
        },
        result={
            "method": "groza",
            "geo_data": sample_analysis_result.result["geo_data"],
        },
    )

    formatter = ConsoleOutputFormatter()

    # Should not raise exception
    captured_output = StringIO()
    sys.stdout = captured_output
    formatter.format_result(
        result_without_loss,
        input_data=sample_input_data,
        geo_data=sample_analysis_result.result["geo_data"],
    )
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    assert "Total Path Loss (Ltot):  145.20 dB" in output  # Still prints total loss

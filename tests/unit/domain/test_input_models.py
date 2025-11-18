import pytest
from trace_calc.domain.models.coordinates import Coordinates, InputData
from trace_calc.domain.models.units import Angle

# Dummy Coordinates for instantiation
coords = Coordinates(lat=0.0, lon=0.0)


def test_calculation_input_default_angle_offset():
    """Test default angle_offset is 2.5"""
    inp = InputData(
        path_name="test",
        site_a_coordinates=coords,
        site_b_coordinates=coords,
    )
    assert inp.elevation_angle_offset == Angle(2.5)


def test_calculation_input_negative_angle_offset_raises():
    """Test validation rejects negative angle_offset"""
    with pytest.raises(ValueError, match="elevation_angle_offset must be non-negative"):
        InputData(
            path_name="test",
            site_a_coordinates=coords,
            site_b_coordinates=coords,
            elevation_angle_offset=Angle(-0.1),
        )


def test_calculation_input_too_large_angle_offset_raises():
    """Test validation rejects angle_offset > 45"""
    with pytest.raises(
        ValueError, match="elevation_angle_offset must be <= 45 degrees"
    ):
        InputData(
            path_name="test",
            site_a_coordinates=coords,
            site_b_coordinates=coords,
            elevation_angle_offset=Angle(45.1),
        )


def test_calculation_input_valid_angle_offset():
    """Test valid angle_offset accepted"""
    inp = InputData(
        path_name="test",
        site_a_coordinates=coords,
        site_b_coordinates=coords,
        elevation_angle_offset=Angle(15.0),
    )
    assert inp.elevation_angle_offset == Angle(15.0)

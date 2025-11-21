import pytest
from trace_calc.domain.models.coordinates import Coordinates, InputData
from trace_calc.domain.models.units import Angle

# Dummy Coordinates for instantiation
coords = Coordinates(lat=0.0, lon=0.0)


def test_input_data_default_hpbw():
    """Test default hpbw is 2.5"""
    inp = InputData(
        path_name="test",
        site_a_coordinates=coords,
        site_b_coordinates=coords,
    )
    assert inp.hpbw == Angle(2.5)


def test_input_data_negative_hpbw_raises():
    """Test validation rejects negative hpbw"""
    with pytest.raises(ValueError, match="HPBW must be non-negative"):
        InputData(
            path_name="test",
            site_a_coordinates=coords,
            site_b_coordinates=coords,
            hpbw=Angle(-0.1),
        )


def test_input_data_too_large_hpbw_raises():
    """Test validation rejects hpbw > 45"""
    with pytest.raises(
        ValueError, match="HPBW must be <= 45 degrees"
    ):
        InputData(
            path_name="test",
            site_a_coordinates=coords,
            site_b_coordinates=coords,
            hpbw=Angle(45.1),
        )


def test_input_data_valid_hpbw():
    """Test valid hpbw accepted"""
    inp = InputData(
        path_name="test",
        site_a_coordinates=coords,
        site_b_coordinates=coords,
        hpbw=Angle(15.0),
    )
    assert inp.hpbw == Angle(15.0)

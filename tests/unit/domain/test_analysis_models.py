"""Test domain models for analysis results"""

import pytest
from trace_calc.domain.models.analysis import AnalysisResult, PropagationLoss


def test_analysis_result_creation():
    """Test creating analysis result with valid data"""
    result = AnalysisResult(
        basic_transmission_loss=120.5,
        total_path_loss=145.2,
        link_speed=100.0,
        wavelength=0.03,
        propagation_loss=None,
        metadata={"model": "groza", "version": "1.0"},
    )

    assert result.basic_transmission_loss == 120.5
    assert result.total_path_loss == 145.2
    assert result.link_speed == 100.0
    assert result.wavelength == 0.03
    assert result.metadata["model"] == "groza"


def test_analysis_result_immutable():
    """Test that AnalysisResult is immutable (frozen dataclass)"""
    result = AnalysisResult(
        basic_transmission_loss=120.5,
        total_path_loss=145.2,
        link_speed=100.0,
        wavelength=0.03,
        propagation_loss=None,
        metadata={},
    )

    with pytest.raises(AttributeError):
        result.link_speed = 200.0  # Should raise FrozenInstanceError


def test_propagation_loss_validation():
    """Test PropagationLoss dataclass with validation"""
    loss = PropagationLoss(
        free_space_loss=92.4,
        atmospheric_loss=0.5,
        diffraction_loss=12.3,
        total_loss=105.2,
    )

    assert loss.free_space_loss == 92.4
    # Validate total matches sum
    expected_total = 92.4 + 0.5 + 12.3
    assert abs(loss.total_loss - expected_total) < 0.1


import numpy as np
from trace_calc.domain.models.path import (
    IntersectionPoint,
    SightLinesData,
    IntersectionsData,
    VolumeData,
    ProfileData,
    ProfileViewData,
)


def test_intersection_point_creation():
    """Test IntersectionPoint instantiation"""
    p = IntersectionPoint(
        distance_km=10.0, elevation_sea_level=200.0, elevation_terrain=50.0
    )
    assert p.distance_km == 10.0
    assert p.elevation_sea_level == 200.0
    assert p.elevation_terrain == 50.0


def test_sight_lines_data_creation():
    """Test SightLinesData instantiation"""
    s = SightLinesData(
        lower_a=np.array([1, 1]),
        lower_b=np.array([2, 2]),
        upper_a=np.array([3, 3]),
        upper_b=np.array([4, 4]),
        bisector_a=np.array([]),
        bisector_b=np.array([]),
    )
    np.testing.assert_array_equal(s.lower_a, np.array([1, 1]))
    np.testing.assert_array_equal(s.lower_b, np.array([2, 2]))
    np.testing.assert_array_equal(s.upper_a, np.array([3, 3]))
    np.testing.assert_array_equal(s.upper_b, np.array([4, 4]))


def test_intersections_data_creation():
    """Test IntersectionsData instantiation"""
    p1 = IntersectionPoint(1, 2, 3)
    p2 = IntersectionPoint(4, 5, 6)
    p3 = IntersectionPoint(7, 8, 9)
    p4 = IntersectionPoint(10, 11, 12)
    i = IntersectionsData(
        lower_intersection=p1, upper_intersection=p2, cross_ab=p3, cross_ba=p4
    )
    assert i.lower_intersection == p1
    assert i.upper_intersection == p2
    assert i.cross_ab == p3
    assert i.cross_ba == p4


def test_volume_data_creation():
    """Test VolumeData instantiation with non-negative values"""
    v = VolumeData(
        cone_intersection_volume_m3=1000.0,
        distance_a_to_cross_ab=10.0,
        distance_b_to_cross_ba=20.0,
        distance_between_crosses=30.0,
        distance_a_to_lower_intersection=5.0,
        distance_b_to_lower_intersection=15.0,
        distance_a_to_upper_intersection=7.0,
        distance_b_to_upper_intersection=13.0,
        distance_between_lower_upper_intersections=2.0,
    )
    assert v.cone_intersection_volume_m3 == 1000.0
    assert v.distance_a_to_cross_ab == 10.0
    assert v.distance_b_to_cross_ba == 20.0
    assert v.distance_between_crosses == 30.0
    assert v.distance_a_to_lower_intersection == 5.0
    assert v.distance_b_to_lower_intersection == 15.0
    assert v.distance_a_to_upper_intersection == 7.0
    assert v.distance_b_to_upper_intersection == 13.0
    assert v.distance_between_lower_upper_intersections == 2.0


def test_profile_data_with_new_fields():
    """Test ProfileData accepts new sight_lines, intersections, volume fields"""
    pvd = ProfileViewData(elevations=np.array([1, 2]), baseline=np.array([0, 0]))
    sl_data = SightLinesData(
        lower_a=np.array([1, 1]),
        lower_b=np.array([2, 2]),
        upper_a=np.array([3, 3]),
        upper_b=np.array([4, 4]),
        bisector_a=np.array([]),
        bisector_b=np.array([]),
    )
    int_p = IntersectionPoint(1, 2, 3)
    int_data = IntersectionsData(int_p, int_p, int_p, int_p)
    vol_data = VolumeData(1, 2, 3, 4, 5, 6, 7, 8, 9)

    profile_data = ProfileData(
        plain=pvd,
        curved=pvd,
        lines_of_sight=sl_data,
        intersections=int_data,
        volume=vol_data,
    )

    assert profile_data.lines_of_sight == sl_data
    assert profile_data.intersections == int_data
    assert profile_data.volume == vol_data

import pytest
import numpy as np

from trace_calc.application.services.profile_data_calculator import (
    ProfileDataCalculator,
)
from trace_calc.domain.models.units import Angle
from trace_calc.domain.models.path import (
    SightLinesData,
    IntersectionPoint,
    IntersectionsData,
    VolumeData,
    ProfileData,
)
from trace_calc.domain.models.units import Meters


class TestProfileDataCalculatorExtensions:
    def setup_method(self):
        self.distances = np.linspace(0, 10, 100)
        # V-shaped terrain, highest in the middle
        self.elevations = 100 - np.abs(self.distances - 5)
        self.calculator = ProfileDataCalculator(self.distances, self.elevations)
        self.calculator.elevations_curved, _ = self.calculator.curved_profile()

    def test_calculate_upper_lines_zero_offset(self):
        """Test upper equals lower with zero offset"""
        lower_line = np.array([0.1, 50])  # y = 0.1x + 50
        pivot = (0.0, 50.0)
        angle = Angle(0.0)

        upper_line = self.calculator._calculate_upper_lines(lower_line, pivot, angle)

        np.testing.assert_allclose(upper_line, lower_line)

    def test_calculate_upper_lines_positive_offset(self):
        """Test upper differs from lower with positive offset"""
        lower_line = np.array([0.0, 100])  # y = 100
        pivot = (0.0, 100.0)
        angle = Angle(1.0)

        upper_line = self.calculator._calculate_upper_lines(lower_line, pivot, angle)

        # Rotated 1 degree around (0, 100)
        expected_k = np.tan(np.deg2rad(1.0)) * 1000
        np.testing.assert_allclose(upper_line, [expected_k, 100.0], atol=1e-6)

    def test_calculate_all_intersections_returns_four_points(self):
        """Test all 4 intersections calculated"""
        sight_lines = SightLinesData(
            lower_a=np.array([100.0, 100]),  # 100 m/km
            lower_b=np.array([-100.0, 11000]),
            upper_a=np.array([200.0, 100]),
            upper_b=np.array([-200.0, 11200]),
            antenna_elevation_angle_a=np.array([]),
            antenna_elevation_angle_b=np.array([]),
        )
        # lower: 100x + 100 = -100x + 11000 => 200x = 10900 => x = 54.5
        # upper: 200x + 100 = -200x + 11200 => 400x = 11100 => x = 27.75
        # cross_ab: 200x + 100 = -100x + 11000 => 300x = 10900 => x = 36.333
        # cross_ba: 100x + 100 = -200x + 11200 => 300x = 11100 => x = 37

        # Mock distances and elevations to be within bounds
        distances = np.linspace(0, 100, 100)
        elevations = np.zeros_like(distances)
        calculator = ProfileDataCalculator(distances, elevations)

        intersections = calculator._calculate_all_intersections(
            sight_lines, distances, elevations
        )

        assert isinstance(intersections, IntersectionsData)
        assert isinstance(intersections.lower, IntersectionPoint)
        assert isinstance(intersections.upper, IntersectionPoint)
        assert isinstance(intersections.cross_ab, IntersectionPoint)
        assert isinstance(intersections.cross_ba, IntersectionPoint)

        assert np.isclose(intersections.lower.distance_km, 54.5)
        assert np.isclose(intersections.upper.distance_km, 27.75)
        assert np.isclose(intersections.cross_ab.distance_km, 109.0 / 3.0)
        assert np.isclose(intersections.cross_ba.distance_km, 37.0)

    def test_calculate_all_intersections_outside_bounds_raises(self):
        """Test error when intersection outside path"""
        sight_lines = SightLinesData(
            lower_a=np.array([0.1, 100]),
            lower_b=np.array([-0.1, 102]),  # Intersection at x=10
            upper_a=np.array([0.2, 100]),
            upper_b=np.array([-0.2, 108]),  # Intersection at x=20
            antenna_elevation_angle_a=np.array([]),
            antenna_elevation_angle_b=np.array([]),
        )

        # Path is only 5 km long
        distances = np.linspace(0, 5, 10)
        elevations = np.zeros_like(distances)
        calculator = ProfileDataCalculator(distances, elevations)

        with pytest.raises(
            ValueError, match="Lower intersection at 10.0 km is outside path bounds"
        ):
            calculator._calculate_all_intersections(sight_lines, distances, elevations)

    def setup_volume_test(self):
        self.sight_lines = SightLinesData(
            lower_a=np.array([0.1, 100]),
            lower_b=np.array([-0.1, 200]),
            upper_a=np.array([0.2, 100]),
            upper_b=np.array([-0.2, 220]),
            antenna_elevation_angle_a=np.array([]),
            antenna_elevation_angle_b=np.array([]),
        )
        # lower: x=250, y=125
        # upper: x=300, y=160
        # cross_ab: x=333.33, y=166.67
        # cross_ba: x=200, y=120
        self.intersections = IntersectionsData(
            lower=IntersectionPoint(250, 125, 25),
            upper=IntersectionPoint(300, 160, 60),
            cross_ab=IntersectionPoint(1000 / 3, 500 / 3, 100),
            cross_ba=IntersectionPoint(200, 120, 20),
            beam_intersection_point=None,
        )
        self.distances_vol = np.linspace(0, 500, 500)
        self.calculator_vol = ProfileDataCalculator(
            self.distances_vol, np.zeros_like(self.distances_vol)
        )

    def test_calculate_volume_metrics_non_negative(self):
        """Test volume is non-negative"""
        self.setup_volume_test()
        volume_data = self.calculator_vol._calculate_volume_metrics(
            self.sight_lines, self.distances_vol, self.intersections
        )
        assert isinstance(volume_data, VolumeData)
        assert volume_data.cone_intersection_volume_m3 >= 0

    def test_calculate_volume_metrics_distances_valid(self):
        """Test distance metrics are valid"""
        self.setup_volume_test()
        total_distance = self.distances_vol[-1]
        volume_data = self.calculator_vol._calculate_volume_metrics(
            self.sight_lines, self.distances_vol, self.intersections
        )

        assert 0 <= volume_data.distance_a_to_cross_ab <= total_distance
        assert 0 <= volume_data.distance_b_to_cross_ba <= total_distance
        assert (
            volume_data.distance_a_to_cross_ab
            == self.intersections.cross_ab.distance_km
        )
        expected_dist_b = total_distance - self.intersections.cross_ba.distance_km
        assert np.isclose(volume_data.distance_b_to_cross_ba, expected_dist_b)

    def test_calculate_volume_metrics_consistency(self):
        """Test distance_between = |cross_ab - cross_ba|"""
        self.setup_volume_test()
        volume_data = self.calculator_vol._calculate_volume_metrics(
            self.sight_lines, self.distances_vol, self.intersections
        )

        expected_dist = abs(
            self.intersections.cross_ab.distance_km
            - self.intersections.cross_ba.distance_km
        )
        assert np.isclose(volume_data.distance_between_crosses, expected_dist)

    def test_calculate_all_profile_data_structure(self):
        """Test full calculation returns ProfileData with correct structure"""
        # Use a simple setup where HCA indices are in the middle
        hca_indices = (50, 50)
        height_offsets = (Meters(10), Meters(10))

        profile_data = self.calculator.calculate_all(hca_indices, height_offsets)

        assert isinstance(profile_data, ProfileData)
        assert isinstance(profile_data.lines_of_sight, SightLinesData)
        assert isinstance(profile_data.intersections, IntersectionsData)
        assert isinstance(profile_data.volume, VolumeData)
        assert isinstance(
            profile_data.lines_of_sight.antenna_elevation_angle_a, np.ndarray
        )
        assert isinstance(
            profile_data.lines_of_sight.antenna_elevation_angle_b, np.ndarray
        )

    def test_calculate_antenna_elevation_angle_lines(self):
        """Test that antenna elevation angle lines are the average of upper and lower lines."""
        sight_lines = SightLinesData(
            lower_a=np.array([0.1, 100]),
            lower_b=np.array([-0.1, 200]),
            upper_a=np.array([0.2, 100]),
            upper_b=np.array([-0.2, 220]),
            antenna_elevation_angle_a=np.array([]),
            antenna_elevation_angle_b=np.array([]),
        )
        distances = np.linspace(0, 100, 101)
        calculator = ProfileDataCalculator(distances, np.zeros_like(distances))

        antenna_elevation_angle_a, antenna_elevation_angle_b = (
            calculator._calculate_antenna_elevation_angle_lines(sight_lines, distances)
        )

        y_lower_a = np.polyval(sight_lines.lower_a, distances)
        y_upper_a = np.polyval(sight_lines.upper_a, distances)
        expected_antenna_elevation_angle_a = (y_lower_a + y_upper_a) / 2

        y_lower_b = np.polyval(sight_lines.lower_b, distances)
        y_upper_b = np.polyval(sight_lines.upper_b, distances)
        expected_antenna_elevation_angle_b = (y_lower_b + y_upper_b) / 2

        np.testing.assert_allclose(
            antenna_elevation_angle_a, expected_antenna_elevation_angle_a
        )
        np.testing.assert_allclose(
            antenna_elevation_angle_b, expected_antenna_elevation_angle_b
        )

    def test_calculate_all_with_angle_offset(self):
        """Test full calculation with HPBW"""
        hca_indices = (50, 50)
        height_offsets = (Meters(2), Meters(40))  # User requested antenna heights
        angle_offset = Angle(2.5)  # User requested offset

        with pytest.raises(
            ValueError, match="Upper intersection .* is outside path bounds"
        ):
            self.calculator.calculate_all(hca_indices, height_offsets, angle_offset)

    def test_calculate_all_zero_angle_backward_compat(self):
        """Test backward compatibility with zero offset"""
        hca_indices = (50, 50)
        height_offsets = (Meters(10), Meters(10))
        angle_offset = Angle(0.0)

        profile_data = self.calculator.calculate_all(
            hca_indices, height_offsets, angle_offset
        )

        # With zero angle, upper and lower lines should be identical
        assert np.allclose(
            profile_data.lines_of_sight.lower_a, profile_data.lines_of_sight.upper_a
        )
        assert np.allclose(
            profile_data.lines_of_sight.lower_b, profile_data.lines_of_sight.upper_b
        )
        assert np.isclose(profile_data.volume.cone_intersection_volume_m3, 0)

    def test_calculate_all_default_parameter(self):
        """Test default angle parameter works"""
        hca_indices = (50, 50)
        height_offsets = (Meters(10), Meters(10))

        # Call without angle_offset parameter
        profile_data = self.calculator.calculate_all(hca_indices, height_offsets)

        # Default should be 0 for now to maintain backward compatibility of the method's result
        # The plan says Angle(0.0) is the default.
        assert np.allclose(
            profile_data.lines_of_sight.lower_a, profile_data.lines_of_sight.upper_a
        )
        assert np.allclose(
            profile_data.lines_of_sight.lower_b, profile_data.lines_of_sight.upper_b
        )

    def test_intersection_elevation_is_corrected_for_curvature(self):
        """Test that intersection elevations are corrected for Earth's drop."""
        # Path is 100km long, terrain is flat at 0m ASL.
        distances = np.linspace(0.0, 100.0, 101)
        elevations = np.zeros(101)
        calculator = ProfileDataCalculator(distances, elevations)

        # Manually get the curved elevations for the call
        elevations_curved, _ = calculator.curved_profile()

        # Sight lines that intersect at midpoint (x=50)
        # y = x  and y = -x + 100. Intersection at (50, 50) on a flat plane.
        sight_lines = SightLinesData(
            lower_a=np.array([1.0, 0.0]),
            lower_b=np.array([-1.0, 100.0]),
            # Dummy upper lines, not used in this part of the test
            upper_a=np.array([1.0, 0.0]),
            upper_b=np.array([-1.0, 100.0]),
            antenna_elevation_angle_a=np.array([]),
            antenna_elevation_angle_b=np.array([]),
        )

        # Expected values
        x_intersectionersect = 50.0
        y_flat = 50.0

        # Earth drop correction: drop = x^2 / (2R)
        # R = 6371 km. drop is in km, convert to meters.
        earth_drop_m = (x_intersectionersect**2 / (2 * 6371)) * 1000
        expected_elevation_sea_level = y_flat - earth_drop_m

        # Terrain height correction: bulge = x(d-x) / (2R)
        # For a flat terrain, the curved profile is just the bulge.
        d = distances[-1]
        terrain_bulge_m = (
            x_intersectionersect * (d - x_intersectionersect) / (2 * 6371)
        ) * 1000

        # For a flat terrain at 0m, the height above terrain should equal the height
        # above sea level.
        expected_elevation_terrain = expected_elevation_sea_level

        # Perform calculation
        # Note: We now pass the raw elevations to this method.
        intersections = calculator._calculate_all_intersections(
            sight_lines, distances, elevations
        )

        # We test the 'lower' intersection, but all are the same here.
        lower_intersection = intersections.lower

        assert lower_intersection.distance_km == pytest.approx(x_intersectionersect)
        assert lower_intersection.elevation_sea_level == pytest.approx(
            expected_elevation_sea_level
        )
        assert lower_intersection.elevation_terrain == pytest.approx(
            expected_elevation_terrain
        )

    def test_beam_intersection_point_and_angles(self):
        """Test calculation of antenna elevation angle intersection and angles with correct units."""
        # 1. Setup
        distances = np.linspace(0, 100, 101)
        elevations = np.zeros_like(distances)  # Flat terrain at 0m
        calculator = ProfileDataCalculator(distances, elevations)
        elevations_curved, _ = calculator.curved_profile()

        # Define antenna elevation angle lines based on y = 1.5x + 100 and y = -1.5x + 250
        # Note: Slopes here are m/km, which is what the old code assumed for angle calcs
        antenna_elevation_angle_a_coeffs = np.array([1.5, 100])
        antenna_elevation_angle_b_coeffs = np.array([-1.5, 250])
        antenna_elevation_angle_a_vals = np.polyval(
            antenna_elevation_angle_a_coeffs, distances
        )
        antenna_elevation_angle_b_vals = np.polyval(
            antenna_elevation_angle_b_coeffs, distances
        )

        # 2. Test antenna elevation angle intersection
        beam_intersection_point = (
            calculator._calculate_antenna_elevation_angle_intersection(
                antenna_elevation_angle_a_vals,
                antenna_elevation_angle_b_vals,
                distances,
                elevations,
            )
        )

        # Expected intersection on flat plane:
        # 1.5x + 100 = -1.5x + 250  => 3x = 150 => x = 50
        # y = 1.5 * 50 + 100 = 175
        x_intersectionersect_flat = 50.0
        y_intersectionersect_flat = 175.0

        # Expected corrections
        earth_drop_m = (x_intersectionersect_flat**2 / (2 * 6371)) * 1000  # ~196.2m
        expected_elev_asl = y_intersectionersect_flat - earth_drop_m
        expected_elev_terrain = expected_elev_asl - 0  # Terrain is at 0m

        assert beam_intersection_point is not None
        assert beam_intersection_point.distance_km == pytest.approx(
            x_intersectionersect_flat
        )
        assert beam_intersection_point.elevation_sea_level == pytest.approx(
            expected_elev_asl
        )
        assert beam_intersection_point.elevation_terrain == pytest.approx(
            expected_elev_terrain
        )

import pytest
import numpy as np
from numpy.testing import assert_allclose

# This import will fail until the file and function are created.
from trace_calc.domain import geometry


class TestRotateLineByAngle:
    def test_rotate_horizontal_line(self):
        """Test rotating a horizontal line by a small angle."""
        line_coeffs = np.array([0.0, 100.0])  # y = 100
        pivot_point = (0.0, 100.0)
        angle_degrees = 1.0

        new_coeffs = geometry.rotate_line_by_angle(
            line_coeffs, pivot_point, angle_degrees
        )

        # Expected slope: tan(1.0 deg) * 1000
        expected_k = np.tan(np.deg2rad(1.0)) * 1000
        assert_allclose(new_coeffs, [expected_k, 100.0], atol=1e-6)

    def test_rotate_with_non_origin_pivot(self):
        """Test rotation maintains the pivot point on the line."""
        # A line with slope 1 m/km is a very small angle
        line_coeffs = np.array([1.0, 0.0])  # y = 1*x
        pivot_point = (10.0, 10.0)
        # Angle of line is arctan(1/1000) = 0.057 degrees.
        # Rotating by -0.057 degrees should make it horizontal.
        angle_of_line_degrees = np.rad2deg(np.arctan(1.0 / 1000.0))

        new_coeffs = geometry.rotate_line_by_angle(
            line_coeffs, pivot_point, -angle_of_line_degrees
        )

        # The new line should be horizontal: y = 10
        assert_allclose(new_coeffs, [0.0, 10.0], atol=1e-6)

        # Verify pivot point is on the new line
        k_new, b_new = new_coeffs
        assert_allclose(k_new * pivot_point[0] + b_new, pivot_point[1])

    def test_rotate_descending_line(self):
        """Test rotating a descending line.

        For descending lines (k < 0), SUBTRACTING the angle offset makes the angle
        more negative, which produces a steeper descent that geometrically moves the line UP.
        This is the correct behavior verified by the prototype implementation and the plan.
        See EXTENDED_VISIBILITY_IMPLEMENTATION_PLAN.md section "CRITICAL IMPLEMENTATION CORRECTIONS".
        """
        line_coeffs = np.array([-10.0, 100.0])  # y = -10x + 100
        pivot_point = (0.0, 100.0)
        angle_degrees = 1.0

        new_coeffs = geometry.rotate_line_by_angle(
            line_coeffs, pivot_point, angle_degrees
        )

        # Original angle is arctan(-10/1000) ≈ -0.573°
        # SUBTRACTING 1.0° gives ≈ -1.573° (more negative → steeper but geometrically higher)
        # This keeps the upper line above the lower line for descending slopes
        original_angle_rad = np.arctan(line_coeffs[0] / 1000.0)
        new_angle_rad = original_angle_rad - np.deg2rad(
            angle_degrees
        )  # SUBTRACT for descending lines
        expected_k = np.tan(new_angle_rad) * 1000.0

        assert_allclose(new_coeffs, [expected_k, 100.0], atol=1e-6)
        # The new slope should be MORE negative (steeper descent)
        assert new_coeffs[0] < line_coeffs[0]

    def test_rotate_vertical_line_raises(self):
        """Test that rotating a near-vertical line raises an error."""
        line_coeffs = np.array([1e7, 0.0])  # Very steep line
        pivot_point = (0.0, 0.0)
        angle_degrees = 10.0

        with pytest.raises(ValueError, match="Cannot rotate vertical line"):
            geometry.rotate_line_by_angle(line_coeffs, pivot_point, angle_degrees)

    def test_rotate_to_near_vertical_raises(self):
        """Test that rotating to a near-vertical line raises an error."""
        line_coeffs = np.array([0.0, 0.0])  # y = 0
        pivot_point = (0.0, 0.0)

        with pytest.raises(ValueError, match="Near-vertical result"):
            # Use an angle that will definitely make it near-vertical after conversion
            geometry.rotate_line_by_angle(line_coeffs, pivot_point, 89.99)

    def test_invalid_angle_raises(self):
        """Test that an angle outside the valid range raises an error."""
        line_coeffs = np.array([0.0, 0.0])
        pivot_point = (0.0, 0.0)

        with pytest.raises(
            ValueError, match="Angle must be between -90 and 90 degrees"
        ):
            geometry.rotate_line_by_angle(line_coeffs, pivot_point, 91.0)

        with pytest.raises(
            ValueError, match="Angle must be between -90 and 90 degrees"
        ):
            geometry.rotate_line_by_angle(line_coeffs, pivot_point, -91.0)


class TestFindLineIntersection:
    def test_simple_intersection(self):
        """Test intersection of two simple lines."""
        line1 = np.array([1.0, 0.0])  # y = x
        line2 = np.array([-1.0, 2.0])  # y = -x + 2
        intersection = geometry.find_line_intersection(line1, line2)
        assert_allclose(intersection, (1.0, 1.0))

    def test_perpendicular_lines(self):
        """Test intersection of perpendicular lines not at the origin."""
        line1 = np.array([2.0, 1.0])  # y = 2x + 1
        line2 = np.array([-0.5, 3.0])  # y = -0.5x + 3
        intersection = geometry.find_line_intersection(line1, line2)
        # 2x + 1 = -0.5x + 3  => 2.5x = 2 => x = 0.8
        # y = 2 * 0.8 + 1 = 1.6 + 1 = 2.6
        assert_allclose(intersection, (0.8, 2.6))

    def test_parallel_lines_raise(self):
        """Test that parallel lines raise a ValueError."""
        line1 = np.array([2.0, 1.0])
        line2 = np.array([2.0, 5.0])
        with pytest.raises(ValueError, match="Parallel"):
            geometry.find_line_intersection(line1, line2)

    def test_coincident_lines_raise(self):
        """Test that coincident lines raise a ValueError."""
        line1 = np.array([2.0, 1.0])
        line2 = np.array([2.0, 1.000000001])
        with pytest.raises(ValueError, match="Coincident"):
            geometry.find_line_intersection(line1, line2)


class TestCalculateHeightAboveTerrain:
    def setup_method(self):
        self.distances = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.elevations = np.array([100.0, 110.0, 105.0, 115.0, 120.0])

    def test_exact_match(self):
        """Test height calculation when distance is an exact point in the array."""
        height = geometry.calculate_height_above_terrain(
            2.0, 150.0, self.distances, self.elevations
        )
        assert np.isclose(height, 45.0)  # 150.0 - 105.0

    def test_interpolation(self):
        """Test height calculation that requires linear interpolation."""
        height = geometry.calculate_height_above_terrain(
            1.5, 120.0, self.distances, self.elevations
        )
        # Terrain at 1.5 is halfway between 110.0 and 105.0, which is 107.5
        # Height is 120.0 - 107.5 = 12.5
        assert np.isclose(height, 12.5)

    def test_out_of_bounds_raises(self):
        """Test that a distance outside the path bounds raises a ValueError."""
        with pytest.raises(
            ValueError, match="Distance is outside the bounds of the path"
        ):
            geometry.calculate_height_above_terrain(
                -0.1, 100.0, self.distances, self.elevations
            )
        with pytest.raises(
            ValueError, match="Distance is outside the bounds of the path"
        ):
            geometry.calculate_height_above_terrain(
                4.1, 100.0, self.distances, self.elevations
            )


class TestCalculateConeIntersectionVolume:
    def setup_method(self):
        # Simple linear lines for predictable volume
        self.lower_a = np.array([0.1, 100])  # y = 0.1x + 100
        self.lower_b = np.array([-0.1, 300])  # y = -0.1x + 300
        self.upper_a = np.array([0.2, 100])  # y = 0.2x + 100
        self.upper_b = np.array([-0.2, 320])  # y = -0.2x + 320
        self.distances = np.linspace(0, 2000, 200)

        # cross_ab: 0.2x + 100 = -0.1x + 300 => 0.3x = 200 => x = 666.67
        self.cross_ab_x = 200 / 0.3
        # cross_ba: 0.1x + 100 = -0.2x + 320 => 0.3x = 220 => x = 733.33
        self.cross_ba_x = 220 / 0.3
        # lower_int: 0.1x + 100 = -0.1x + 300 => 0.2x = 200 => x = 1000
        self.lower_intersection_x = 1000
        # upper_int: 0.2x + 100 = -0.2x + 320 => 0.4x = 220 => x = 550
        self.upper_intersection_x = 550

    def test_simple_volume(self):
        """Test a basic volume calculation."""
        volume = geometry.calculate_cone_intersection_volume(
            self.lower_a,
            self.lower_b,
            self.upper_a,
            self.upper_b,
            self.distances,
            self.lower_intersection_x,
            self.upper_intersection_x,
            self.cross_ab_x,
            self.cross_ba_x,
        )
        # The exact value is complex to calculate here, but it must be positive
        assert volume > 0

    def test_degenerate_case_zero_volume(self):
        """Test case where integration bounds are the same, should be zero volume."""
        volume = geometry.calculate_cone_intersection_volume(
            self.lower_a,
            self.lower_b,
            self.upper_a,
            self.upper_b,
            self.distances,
            self.lower_intersection_x,
            self.upper_intersection_x,
            self.cross_ab_x,
            self.cross_ab_x,  # Same x for both cross intersections
        )
        assert np.isclose(volume, 0.0)

    def test_negative_height_raises(self):
        """Test that invalid geometry (e.g., upper lines cross below lower lines) raises an error."""
        # Swap upper and lower 'a' lines to create an invalid region
        with pytest.raises(ValueError, match="Invalid geometry"):
            geometry.calculate_cone_intersection_volume(
                self.upper_a,
                self.lower_b,
                self.lower_a,
                self.upper_b,
                self.distances,
                self.lower_intersection_x,
                self.upper_intersection_x,
                self.cross_ab_x,
                self.cross_ba_x,
            )


class TestCalculateDistanceBetweenPoints:
    def test_simple_distance(self):
        """Test a simple distance calculation (3-4-5 triangle)."""
        # 3 km horizontal, 4000m (4km) vertical
        point_a = (1.0, 1000.0)
        point_b = (4.0, 5000.0)
        distance = geometry.calculate_distance_between_points(point_a, point_b)
        assert np.isclose(distance, 5.0)

    def test_same_point(self):
        """Test that the distance between the same point is zero."""
        point_a = (10.0, 200.0)
        distance = geometry.calculate_distance_between_points(point_a, point_a)
        assert np.isclose(distance, 0.0)

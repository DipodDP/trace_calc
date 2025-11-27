import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from trace_calc.domain.models.path import PathData, ProfileData
from trace_calc.domain.models.analysis import AnalysisResult
from trace_calc.infrastructure.visualization.plotter import ProfileVisualizer


class TestProfileVisualizer(unittest.TestCase):
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.subplots")
    def test_plot_profile_with_analysis_result(
        self, mock_subplots, mock_savefig, mock_show
    ):
        # Create mock data for PathData, ProfileData, and AnalysisResult
        path_data = MagicMock(spec=PathData)
        path_data.distances = np.array([0, 10, 20])

        profile_data = MagicMock(spec=ProfileData)
        profile_data.plain.elevations = np.array([100, 150, 120])
        profile_data.plain.baseline = 0
        profile_data.curved.elevations = np.array([100, 140, 110])
        profile_data.curved.baseline = 0
        profile_data.lines_of_sight.lower_a = np.array([1, 1])
        profile_data.lines_of_sight.lower_b = np.array([1, 1])
        profile_data.lines_of_sight.upper_a = np.array([1, 1])
        profile_data.lines_of_sight.upper_b = np.array([1, 1])
        profile_data.lines_of_sight.antenna_elevation_angle_a = np.array([1, 1])
        profile_data.lines_of_sight.antenna_elevation_angle_b = np.array([1, 1])
        profile_data.intersections.lower.elevation_sea_level = 1
        profile_data.intersections.lower.distance_km = 1
        profile_data.intersections.upper.elevation_sea_level = 1
        profile_data.intersections.upper.distance_km = 1
        profile_data.intersections.cross_ab.elevation_sea_level = 1
        profile_data.intersections.cross_ab.distance_km = 1
        profile_data.intersections.cross_ba.elevation_sea_level = 1
        profile_data.intersections.cross_ba.distance_km = 1
        profile_data.intersections.beam_intersection_point.elevation_sea_level = 1
        profile_data.intersections.beam_intersection_point.distance_km = 1
        profile_data.intersections.beam_intersection_point.angle = 3.50

        analysis_result = MagicMock(spec=AnalysisResult)
        analysis_result.result = {
            "b1_max": 0.5,
            "b2_max": 0.6,
            "b_sum": 1.1,
            "hpbw": 1.2,
        }
        profile_data.volume.cone_intersection_volume_m3 = 1
        profile_data.volume.distance_a_to_cross_ab = 1
        profile_data.volume.distance_b_to_cross_ba = 1
        profile_data.volume.distance_between_crosses = 1
        profile_data.intersections.upper.elevation_terrain = 1
        profile_data.intersections.upper.elevation_sea_level = 1
        profile_data.intersections.lower.elevation_terrain = 1
        profile_data.intersections.lower.elevation_sea_level = 1
        profile_data.volume.antenna_elevation_angle_a = 1.23
        profile_data.volume.antenna_elevation_angle_b = 4.56

        visualizer = ProfileVisualizer()

        # Call the method to be tested
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        type(mock_ax1).transAxes = MagicMock()
        type(mock_ax2).transAxes = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        visualizer.plot_profile(path_data, profile_data, analysis_result)

        # Assert that the text box contains the correct information
        call_args, call_kwargs = mock_ax2.text.call_args_list[0]

        self.assertIn("HCA=0.50°", call_args[2])
        self.assertIn("Elev=1.23°", call_args[2])
        self.assertIn("Θ=1.20°", call_args[2])
        self.assertIn("HCA sum: 1.10°", call_args[2])
        self.assertIn("Elev=4.56°", call_args[2])
        self.assertIn("BIA: 3.50°", call_args[2]) # BIA stands for Beam Intersection Angle

    @patch("matplotlib.pyplot.subplots")
    def test_plain_profile_y_axis_label(self, mock_subplots):
        # Arrange
        path_data = MagicMock(spec=PathData)
        path_data.distances = np.array([0, 10, 20])

        profile_data = MagicMock(spec=ProfileData)
        profile_data.plain.elevations = np.array([100, 150, 120])
        profile_data.plain.baseline = 0
        profile_data.curved.elevations = np.array([100, 140, 110])
        profile_data.curved.baseline = 0
        profile_data.lines_of_sight.lower_a = np.array([1, 1])
        profile_data.lines_of_sight.lower_b = np.array([1, 1])
        profile_data.lines_of_sight.upper_a = np.array([1, 1])
        profile_data.lines_of_sight.upper_b = np.array([1, 1])
        profile_data.lines_of_sight.antenna_elevation_angle_a = np.array([1, 1])
        profile_data.lines_of_sight.antenna_elevation_angle_b = np.array([1, 1])
        profile_data.intersections.lower.elevation_sea_level = 1
        profile_data.intersections.lower.distance_km = 1
        profile_data.intersections.upper.elevation_sea_level = 1
        profile_data.intersections.upper.distance_km = 1
        profile_data.intersections.cross_ab.elevation_sea_level = 1
        profile_data.intersections.cross_ab.distance_km = 1
        profile_data.intersections.cross_ba.elevation_sea_level = 1
        profile_data.intersections.cross_ba.distance_km = 1
        profile_data.intersections.beam_intersection_point.elevation_sea_level = 1
        profile_data.intersections.beam_intersection_point.distance_km = 1
        profile_data.intersections.beam_intersection_point.angle = 3.50
        profile_data.volume.cone_intersection_volume_m3 = 1
        profile_data.volume.distance_a_to_cross_ab = 1
        profile_data.volume.distance_b_to_cross_ba = 1
        profile_data.volume.distance_between_crosses = 1
        profile_data.intersections.upper.elevation_terrain = 1
        profile_data.intersections.upper.elevation_sea_level = 1
        profile_data.intersections.lower.elevation_terrain = 1
        profile_data.intersections.lower.elevation_sea_level = 1

        visualizer = ProfileVisualizer()
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])

        # Act
        visualizer.plot_profile(path_data, profile_data)

        # Assert
        mock_ax1.set_ylabel.assert_called_with("Elevation (m)", fontsize=10)


if __name__ == "__main__":
    unittest.main()

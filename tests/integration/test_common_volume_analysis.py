import json
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from trace_calc.domain.models.coordinates import InputData, Coordinates
from trace_calc.domain.models.units import Angle, Meters
from trace_calc.application.analysis import GrozaAnalyzer
from trace_calc.application.services.profile_data_calculator import (
    ProfileDataCalculator,
)
from trace_calc.infrastructure.visualization.plotter import ProfileVisualizer
from trace_calc.infrastructure.output.formatters import (
    ConsoleOutputFormatter,
    JSONOutputFormatter,
)
from trace_calc.domain.models.analysis import AnalysisResult
from trace_calc.domain.models.path import (
    PathData,
    ProfileData,
    SightLinesData,
    IntersectionsData,
    VolumeData,
)


# Mock for PathData and HCAData for consistent testing
@pytest.fixture
def mock_path_data():
    distances = np.linspace(0, 100, 101)  # 0 to 100 km
    elevations = np.sin(distances / 10) * 50 + 100  # undulating terrain
    coordinates = np.array([[0, 0]] * 101)  # dummy coordinates
    return PathData(coordinates=coordinates, distances=distances, elevations=elevations)


@pytest.fixture
def mock_input_data():
    return InputData(
        path_name="Test Path",
        site_a_coordinates=Coordinates(lat=0, lon=0),
        site_b_coordinates=Coordinates(lat=1, lon=1),
        frequency_mhz=1000.0,
        antenna_a_height=Meters(30),
        antenna_b_height=Meters(30),
        hpbw=Angle(2.5),
    )


@pytest.fixture
def mock_profile_data(mock_path_data, mock_input_data):
    calculator = ProfileDataCalculator(
        mock_path_data.distances, mock_path_data.elevations
    )
    # Mock HCA indices for simplicity, assuming they are found in the middle
    hca_indices = (
        len(mock_path_data.distances) // 4,
        len(mock_path_data.distances) * 3 // 4,
    )
    height_offsets = (
        mock_input_data.antenna_a_height,
        mock_input_data.antenna_b_height,
    )
    return calculator.calculate_all(hca_indices, height_offsets, mock_input_data.hpbw)


class TestCommonVolumeAnalysis:
    def test_end_to_end_with_real_coordinates(self, mock_path_data, mock_input_data):
        """
        Test the complete calculation pipeline with mock data representing real coordinates.
        Verifies all outputs are present and valid.
        """
        calculator = ProfileDataCalculator(
            mock_path_data.distances, mock_path_data.elevations
        )
        hca_indices = (
            len(mock_path_data.distances) // 4,
            len(mock_path_data.distances) * 3 // 4,
        )
        height_offsets = (
            mock_input_data.antenna_a_height,
            mock_input_data.antenna_b_height,
        )

        profile_data = calculator.calculate_all(
            hca_indices, height_offsets, mock_input_data.hpbw
        )

        assert isinstance(profile_data, ProfileData)
        assert isinstance(profile_data.lines_of_sight, SightLinesData)
        assert isinstance(profile_data.intersections, IntersectionsData)
        assert isinstance(profile_data.volume, VolumeData)

        # Basic checks for values
        assert profile_data.volume.cone_intersection_volume_m3 >= 0
        assert profile_data.intersections.lower.distance_km >= 0
        assert profile_data.intersections.upper.distance_km >= 0

        # Verify new distance metrics exist and are valid
        assert profile_data.volume.distance_a_to_lower_intersection >= 0
        assert profile_data.volume.distance_b_to_lower_intersection >= 0
        assert profile_data.volume.distance_a_to_upper_intersection >= 0
        assert profile_data.volume.distance_b_to_upper_intersection >= 0
        assert profile_data.volume.distance_between_lower_upper_intersections >= 0

    @patch("matplotlib.pyplot.Figure.savefig")
    @patch("matplotlib.pyplot.show")
    def test_visualization_generation(
        self, mock_show, mock_savefig, mock_path_data, mock_profile_data
    ):
        """
        Create profile with new data and verify plot generation.
        """
        visualizer = ProfileVisualizer()
        save_path = "test_plot.png"
        visualizer.plot_profile(
            mock_path_data, mock_profile_data, save_path=save_path, show=False
        )

        mock_savefig.assert_called_once_with(save_path, dpi=300, facecolor="mintcream")
        mock_show.assert_not_called()
        # Clean up created file if any
        if os.path.exists(save_path):
            os.remove(save_path)

    def test_console_output_formatting(self, mock_profile_data, mock_input_data):
        """
        Generate profile and format console output.
        Verify all sections are present.
        """
        formatter = ConsoleOutputFormatter()
        analysis_result = AnalysisResult(
            link_speed=50,
            wavelength=0.3,
            model_propagation_loss_parameters={
                "basic_transmission_loss": 100,
                "total_path_loss": 120,
                "propagation_loss": {
                    "free_space_loss": 10.0,
                    "atmospheric_loss": 5.0,
                    "diffraction_loss": 2.0,
                    "refraction_loss": 0.0,
                    "total_loss": 17.0,
                },
            },
            result={"profile_data": mock_profile_data, "method": "test"},
        )

        with patch("builtins.print") as mock_print:
            formatter.format_result(analysis_result, mock_input_data, None, mock_profile_data)
            printed_output = "".join(call.args[0] for call in mock_print.call_args_list)

            assert "Common Scatter Volume Analysis" in printed_output
            assert "Lower Sight Lines:" in printed_output
            assert "Upper Sight Lines:" in printed_output
            assert "Cross Intersections:" in printed_output
            assert "Volume Metrics:" in printed_output
            assert "Distance Metrics:" in printed_output
            assert "Distance from A to lower intersection:" in printed_output
            assert "Distance between lower and upper intersections:" in printed_output

            # Check for new antennas elevations intersection sections
            assert "Beam Intersection Point:" in printed_output
            assert "Antenna Elevation Angles:" in printed_output
            # A less brittle check for a key metric
            assert "Common scatter volume:" in printed_output

    def test_json_output_schema(self, mock_profile_data, mock_input_data):
        """
        Generate profile, serialize to JSON, and validate against schema.
        """
        from trace_calc.domain.models.analysis import AnalysisResult, PropagationLoss

        formatter = JSONOutputFormatter()
        # Create a real AnalysisResult object
        analysis_result = AnalysisResult(
            link_speed=50,
            wavelength=0.3,
            model_propagation_loss_parameters={
                "basic_transmission_loss": 100,
                "total_path_loss": 120,
                "propagation_loss": {
                    "free_space_loss": 10.0,
                    "atmospheric_loss": 5.0,
                    "diffraction_loss": 2.0,
                    "refraction_loss": 0.0,
                    "total_loss": 17.0,
                },
            },
            result={"profile_data": mock_profile_data, "method": "test"},
        )

        json_output = formatter.format_result(analysis_result, mock_input_data, None, mock_profile_data)
        output_dict = json.loads(json_output)

        assert "profile_data" in output_dict
        profile_json = output_dict["profile_data"]

        assert "sight_lines" in profile_json
        assert "lower_a_slope" in profile_json["sight_lines"]
        assert "upper_a_slope" in profile_json["sight_lines"]
        assert "lower_b_slope" in profile_json["sight_lines"]
        assert "upper_b_slope" in profile_json["sight_lines"]

        assert "intersections" in profile_json
        assert "lower" in profile_json["intersections"]
        assert "upper" in profile_json["intersections"]
        assert "cross_ab" in profile_json["intersections"]
        assert "cross_ba" in profile_json["intersections"]

        assert "common_volume" in profile_json
        assert "cone_intersection_volume_m3" in profile_json["common_volume"]
        assert "distance_a_to_cross_ab" in profile_json["common_volume"]

        # Verify new distance metrics are present in JSON
        assert "distance_a_to_lower_intersection" in profile_json["common_volume"]
        assert "distance_b_to_lower_intersection" in profile_json["common_volume"]
        assert "distance_a_to_upper_intersection" in profile_json["common_volume"]
        assert "distance_b_to_upper_intersection" in profile_json["common_volume"]
        assert "distance_between_lower_upper_intersections" in profile_json["common_volume"]

        # Verify round-trip (serialize/deserialize)
        # Compare formatted values to avoid precision issues with np.isclose
        assert profile_json["common_volume"]["cone_intersection_volume_m3"] == float(f"{mock_profile_data.volume.cone_intersection_volume_m3:.2f}")
        assert profile_json["intersections"]["lower"]["distance_km"] == float(f"{mock_profile_data.intersections.lower.distance_km:.2f}")
        assert np.isclose(
            profile_json["sight_lines"]["lower_a_slope"],
            mock_profile_data.lines_of_sight.lower_a[0], # Keep as is, not formatted to .2f in output
        )
        assert profile_json["common_volume"]["distance_a_to_lower_intersection"] == float(f"{mock_profile_data.volume.distance_a_to_lower_intersection:.2f}")
        assert profile_json["common_volume"]["distance_between_lower_upper_intersections"] == float(f"{mock_profile_data.volume.distance_between_lower_upper_intersections:.2f}")

    def test_different_angle_offsets(self, mock_path_data, mock_input_data):
        """
        Test with 0°, 1°, 2.5°, 5° offsets.
        Verify monotonic relationships and boundary cases.
        """
        calculator = ProfileDataCalculator(
            mock_path_data.distances, mock_path_data.elevations
        )
        hca_indices = (
            len(mock_path_data.distances) // 4,
            len(mock_path_data.distances) * 3 // 4,
        )
        height_offsets = (
            mock_input_data.antenna_a_height,
            mock_input_data.antenna_b_height,
        )

        offsets = [0.0, 1.0, 2.5, 5.0]
        volumes = []
        upper_intersection_elevations = []

        for offset_deg in offsets:
            input_data_with_offset = InputData(
                path_name="Test Path",
                site_a_coordinates=Coordinates(lat=0, lon=0),
                site_b_coordinates=Coordinates(lat=1, lon=1),
                frequency_mhz=1000.0,
                antenna_a_height=Meters(30),
                antenna_b_height=Meters(30),
                hpbw=Angle(offset_deg),
            )
            profile_data = calculator.calculate_all(
                hca_indices, height_offsets, input_data_with_offset.hpbw
            )
            volumes.append(profile_data.volume.cone_intersection_volume_m3)
            upper_intersection_elevations.append(
                profile_data.intersections.upper.elevation_sea_level
            )

        # Verify that volume and upper intersection elevation increase with HPBW
        assert all(volumes[i] <= volumes[i + 1] for i in range(len(volumes) - 1))
        assert all(
            upper_intersection_elevations[i] <= upper_intersection_elevations[i + 1]
            for i in range(len(upper_intersection_elevations) - 1)
        )

        # For 0 offset, volume should be 0 or very close to 0
        assert np.isclose(volumes[0], 0.0, atol=1e-3)
        # For 0 offset, upper intersection should be same as lower intersection
        profile_data_0_offset = calculator.calculate_all(
            hca_indices, height_offsets, Angle(0.0)
        )
        assert np.isclose(
            profile_data_0_offset.intersections.upper.elevation_sea_level,
            profile_data_0_offset.intersections.lower.elevation_sea_level,
        )

    def test_antenna_elevation_angle_calculation(self, mock_path_data, mock_input_data):
        """
        Verify that the antenna elevation angles are correctly calculated based on
        the lower sight line angles and the HPBW.
        Formula: Antenna Elevation Angle = Lower Sight Line Angle + 1.25 * (Θ / 2)
        """
        # 1. Instantiate analyzer to get access to all calculated data
        # The DefaultProfileDataCalculator is called within GrozaAnalyzer's init
        analyzer = GrozaAnalyzer(mock_path_data, mock_input_data)
        profile_data = analyzer.profile_data
        hpbw = mock_input_data.hpbw

        # 2. Calculate the expected angles based on the new formula
        slope_a = profile_data.lines_of_sight.lower_a[0]
        slope_b = profile_data.lines_of_sight.lower_b[0]

        angle_a_rad = np.arctan(slope_a / 1000)
        angle_b_rad = np.arctan(slope_b / 1000)

        angle_a_deg = np.rad2deg(angle_a_rad)
        angle_b_deg = np.rad2deg(angle_b_rad)

        expected_angle_a = abs(angle_a_deg) + (hpbw / 2)
        expected_angle_b = abs(angle_b_deg) + (hpbw / 2)

        # 3. Assert that the angles in the final profile data match the expectation
        assert profile_data.volume.antenna_elevation_angle_a == pytest.approx(
            expected_angle_a
        )
        assert profile_data.volume.antenna_elevation_angle_b == pytest.approx(
            expected_angle_b
        )

    @patch("matplotlib.pyplot.subplots")
    def test_visualization_with_new_metrics_text(
        self, mock_subplots, mock_path_data, mock_input_data
    ):
        """
        Verify that the plot includes the updated metrics text with new labels.
        """
        # Arrange
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax, mock_ax])

        visualizer = ProfileVisualizer()

        # Run the analyzer to get the real result and profile_data
        analyzer = GrozaAnalyzer(mock_path_data, mock_input_data)
        analyzer_result = analyzer.analyze()
        profile_data = analyzer_result.profile_data

        # Create a full AnalysisResult object
        from trace_calc.domain.models.analysis import AnalysisResult

        result = AnalysisResult(
            link_speed=analyzer_result.link_speed,
            wavelength=analyzer_result.wavelength,
            model_propagation_loss_parameters=analyzer_result.model_parameters,
            result={
                "b1_max": analyzer_result.hca.b1_max,
                "b2_max": analyzer_result.hca.b2_max,
                "b_sum": analyzer_result.hca.b_sum,
            },
        )
        result.result["hpbw"] = mock_input_data.hpbw
        # Set the angle attribute on the actual object if it exists.
        # Check if beam_intersection_point is not None first.
        if profile_data.intersections.beam_intersection_point:
            profile_data.intersections.beam_intersection_point.angle = 3.50

        # Act
        visualizer.plot_profile(
            mock_path_data,
            profile_data,
            result=result,
            show=False,
            save_path="test.png",  # Must provide a save path to trigger savefig logic
        )

        # Assert
        # Find the call to `text()` on the mock axes object
        text_call_args = None
        for call in mock_ax.method_calls:
            if call[0] == "text":
                text_call_args = call[1]
                break

        assert text_call_args is not None, "ax.text() was not called"

        metrics_text = text_call_args[2]  # The text string is the 3rd argument

        assert "Site A: HCA=" in metrics_text
        assert "Elev=" in metrics_text
        assert "Θ=" in metrics_text
        assert "Site B: HCA=" in metrics_text
        assert "HCA sum:" in metrics_text
        assert "BIA: 3.50°" in metrics_text # BIA stands for Beam Intersection Angle

        # Clean up dummy file
        if os.path.exists("test.png"):
            os.remove("test.png")

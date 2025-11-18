import json
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from trace_calc.domain.models.coordinates import InputData, Coordinates
from trace_calc.domain.models.units import Angle, Meters
from trace_calc.application.services.profile_data_calculator import ProfileDataCalculator
from trace_calc.infrastructure.visualization.plotter import ProfileVisualizer
from trace_calc.infrastructure.output.formatters import ConsoleOutputFormatter, JSONOutputFormatter
from trace_calc.domain.models.path import PathData, ProfileData, SightLinesData, IntersectionsData, VolumeData


# Mock for PathData and HCAData for consistent testing
@pytest.fixture
def mock_path_data():
    distances = np.linspace(0, 100, 101)  # 0 to 100 km
    elevations = np.sin(distances / 10) * 50 + 100 # undulating terrain
    coordinates = np.array([[0,0]] * 101) # dummy coordinates
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
        elevation_angle_offset=Angle(2.5)
    )

@pytest.fixture
def mock_profile_data(mock_path_data, mock_input_data):
    calculator = ProfileDataCalculator(mock_path_data.distances, mock_path_data.elevations)
    # Mock HCA indices for simplicity, assuming they are found in the middle
    hca_indices = (len(mock_path_data.distances) // 4, len(mock_path_data.distances) * 3 // 4)
    height_offsets = (mock_input_data.antenna_a_height, mock_input_data.antenna_b_height)
    return calculator.calculate_all(hca_indices, height_offsets, mock_input_data.elevation_angle_offset)


class TestCommonVolumeIntegration:

    def test_end_to_end_with_real_coordinates(self, mock_path_data, mock_input_data):
        """
        Test the complete calculation pipeline with mock data representing real coordinates.
        Verifies all outputs are present and valid.
        """
        calculator = ProfileDataCalculator(mock_path_data.distances, mock_path_data.elevations)
        hca_indices = (len(mock_path_data.distances) // 4, len(mock_path_data.distances) * 3 // 4)
        height_offsets = (mock_input_data.antenna_a_height, mock_input_data.antenna_b_height)

        profile_data = calculator.calculate_all(hca_indices, height_offsets, mock_input_data.elevation_angle_offset)

        assert isinstance(profile_data, ProfileData)
        assert isinstance(profile_data.lines_of_sight, SightLinesData)
        assert isinstance(profile_data.intersections, IntersectionsData)
        assert isinstance(profile_data.volume, VolumeData)

        # Basic checks for values
        assert profile_data.volume.cone_intersection_volume_m3 >= 0
        assert profile_data.intersections.lower_intersection.distance_km >= 0
        assert profile_data.intersections.upper_intersection.distance_km >= 0

        # Verify new distance metrics exist and are valid
        assert profile_data.volume.distance_a_to_lower_intersection >= 0
        assert profile_data.volume.distance_b_to_lower_intersection >= 0
        assert profile_data.volume.distance_a_to_upper_intersection >= 0
        assert profile_data.volume.distance_b_to_upper_intersection >= 0
        assert profile_data.volume.distance_between_lower_upper_intersections >= 0

    @patch('matplotlib.pyplot.Figure.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualization_generation(self, mock_show, mock_savefig, mock_path_data, mock_profile_data):
        """
        Create profile with new data and verify plot generation.
        """
        visualizer = ProfileVisualizer()
        save_path = "test_plot.png"
        visualizer.plot_profile(mock_path_data, mock_profile_data, save_path=save_path, show=False)

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
        # Mock AnalysisResult to contain profile_data in metadata
        mock_result = MagicMock()
        mock_result.metadata = {"profile_data": mock_profile_data, "method": "test"}
        mock_result.wavelength = 0.3 # dummy
        mock_result.basic_transmission_loss = 100 # dummy
        mock_result.total_path_loss = 120 # dummy
        mock_result.link_speed = 50 # dummy
        mock_result.propagation_loss = MagicMock(
            free_space_loss=10.0,
            atmospheric_loss=5.0,
            diffraction_loss=2.0,
            total_loss=17.0
        )

        with patch('builtins.print') as mock_print:
            formatter.format_result(mock_result, mock_input_data)
            printed_output = "".join(call.args[0] for call in mock_print.call_args_list)

            assert "Common Scatter Volume Analysis" in printed_output
            assert "Lower Sight Lines:" in printed_output
            assert "Upper Sight Lines:" in printed_output
            assert "Cross Intersections:" in printed_output
            assert "Volume Metrics:" in printed_output
            assert f"Common scatter volume: {mock_profile_data.volume.cone_intersection_volume_m3 / 1e9:.3f} km³" in printed_output
            assert "Distance Metrics:" in printed_output
            assert "Distance from A to lower intersection:" in printed_output
            assert "Distance between lower and upper intersections:" in printed_output

    def test_json_output_schema(self, mock_profile_data, mock_input_data):
        """
        Generate profile, serialize to JSON, and validate against schema.
        """
        from trace_calc.domain.models.analysis import AnalysisResult, PropagationLoss

        formatter = JSONOutputFormatter()
        # Create a real AnalysisResult object
        analysis_result = AnalysisResult(
            basic_transmission_loss=100,
            total_path_loss=120,
            link_speed=50,
            wavelength=0.3,
            propagation_loss=PropagationLoss(
                free_space_loss=10.0,
                atmospheric_loss=5.0,
                diffraction_loss=2.0,
                total_loss=17.0,
            ),
            metadata={"profile_data": mock_profile_data, "method": "test"},
        )

        json_output = formatter.format_result(analysis_result, mock_input_data)
        output_dict = json.loads(json_output)

        assert "profile_data" in output_dict
        profile_json = output_dict["profile_data"]

        assert "sight_lines" in profile_json
        assert "lower_a" in profile_json["sight_lines"]
        assert "upper_a" in profile_json["sight_lines"]

        assert "intersections" in profile_json
        assert "lower" in profile_json["intersections"]
        assert "upper" in profile_json["intersections"]
        assert "cross_ab" in profile_json["intersections"]
        assert "cross_ba" in profile_json["intersections"]

        assert "volume" in profile_json
        assert "cone_intersection_volume_m3" in profile_json["volume"]
        assert "distance_a_to_cross_ab" in profile_json["volume"]

        # Verify new distance metrics are present in JSON
        assert "distance_a_to_lower_intersection" in profile_json["volume"]
        assert "distance_b_to_lower_intersection" in profile_json["volume"]
        assert "distance_a_to_upper_intersection" in profile_json["volume"]
        assert "distance_b_to_upper_intersection" in profile_json["volume"]
        assert "distance_between_lower_upper_intersections" in profile_json["volume"]

        # Verify round-trip (serialize/deserialize)
        assert np.isclose(
            profile_json["volume"]["cone_intersection_volume_m3"],
            mock_profile_data.volume.cone_intersection_volume_m3,
        )
        assert np.isclose(
            profile_json["intersections"]["lower"]["distance_km"],
            mock_profile_data.intersections.lower_intersection.distance_km,
        )
        assert np.isclose(
            profile_json["volume"]["distance_a_to_lower_intersection"],
            mock_profile_data.volume.distance_a_to_lower_intersection,
        )
        assert np.isclose(
            profile_json["volume"]["distance_between_lower_upper_intersections"],
            mock_profile_data.volume.distance_between_lower_upper_intersections,
        )

    def test_different_angle_offsets(self, mock_path_data, mock_input_data):
        """
        Test with 0°, 1°, 2.5°, 5° offsets.
        Verify monotonic relationships and boundary cases.
        """
        calculator = ProfileDataCalculator(mock_path_data.distances, mock_path_data.elevations)
        hca_indices = (len(mock_path_data.distances) // 4, len(mock_path_data.distances) * 3 // 4)
        height_offsets = (mock_input_data.antenna_a_height, mock_input_data.antenna_b_height)

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
                elevation_angle_offset=Angle(offset_deg)
            )
            profile_data = calculator.calculate_all(hca_indices, height_offsets, input_data_with_offset.elevation_angle_offset)
            volumes.append(profile_data.volume.cone_intersection_volume_m3)
            upper_intersection_elevations.append(profile_data.intersections.upper_intersection.elevation_sea_level)

        # Verify that volume and upper intersection elevation increase with angle offset
        assert all(volumes[i] <= volumes[i+1] for i in range(len(volumes)-1))
        assert all(upper_intersection_elevations[i] <= upper_intersection_elevations[i+1] for i in range(len(upper_intersection_elevations)-1))

        # For 0 offset, volume should be 0 or very close to 0
        assert np.isclose(volumes[0], 0.0, atol=1e-3)
        # For 0 offset, upper intersection should be same as lower intersection
        profile_data_0_offset = calculator.calculate_all(hca_indices, height_offsets, Angle(0.0))
        assert np.isclose(profile_data_0_offset.intersections.upper_intersection.elevation_sea_level,
                          profile_data_0_offset.intersections.lower_intersection.elevation_sea_level)

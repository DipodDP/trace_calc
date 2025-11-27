import json
import pytest
import numpy as np
from unittest.mock import MagicMock

from trace_calc.domain.models.coordinates import InputData, Coordinates
from trace_calc.domain.models.units import Angle, Meters
from trace_calc.infrastructure.output.formatters import JSONOutputFormatter
from trace_calc.domain.models.analysis import AnalysisResult, PropagationLoss
from trace_calc.domain.models.path import (
    PathData,
    ProfileData,
    SightLinesData,
    IntersectionsData,
    VolumeData,
    IntersectionPoint,
)
from trace_calc.application.services.profile_data_calculator import (
    ProfileDataCalculator,
)


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
    hca_indices = (
        len(mock_path_data.distances) // 4,
        len(mock_path_data.distances) * 3 // 4,
    )
    height_offsets = (
        mock_input_data.antenna_a_height,
        mock_input_data.antenna_b_height,
    )
    return calculator.calculate_all(hca_indices, height_offsets, mock_input_data.hpbw)


@pytest.fixture
def mock_analysis_result(mock_profile_data):
    return AnalysisResult(
        link_speed=50.0,
        wavelength=0.3,
        model_propagation_loss_parameters={
            "basic_transmission_loss": 100,
            "total_loss": 120,
            "propagation_loss": {
                "free_space_loss": 10.0,
                "atmospheric_loss": 5.0,
                "diffraction_loss": 2.0,
                "total_loss": 17.0,
            },
        },
        result={"profile_data": mock_profile_data, "method": "test"},
    )


class TestJSONOutput:
    def test_json_output_generation(
        self, mock_analysis_result, mock_input_data, mock_profile_data
    ):
        """
        Test that the JSON output is correctly generated and has the correct structure.
        """
        formatter = JSONOutputFormatter()
        json_output = formatter.format_result(
            mock_analysis_result, mock_input_data, None, mock_profile_data
        )
        output_dict = json.loads(json_output)

        assert "analysis_result" in output_dict
        ar = output_dict["analysis_result"]
        assert "link_speed" in ar
        assert ar["link_speed"] == 50.0  # Changed to .1f

        assert "calculated_distance_km" in output_dict
        assert output_dict["calculated_distance_km"] == 157.29 # .2f for other floats

        assert "profile_data" in output_dict
        profile_data_output = output_dict["profile_data"]
        intersections_output = profile_data_output["intersections"]
        
        # Assertions for dummy_lower_intersection
        assert intersections_output["lower"]["distance_km"] == 54.52
        assert intersections_output["lower"]["elevation_sea_level_km"] == 0.22
        assert intersections_output["lower"]["elevation_terrain_km"] == 0.15

        # Assertions for dummy_beam_intersection_point angle
        assert intersections_output["beam_intersection_point"]["angle_deg"] == 3.28
        
        assert "model_propagation_loss_parameters" in ar
        assert "metadata" in ar

        loss_params = ar["model_propagation_loss_parameters"]
        assert "total_loss" in loss_params

        assert ar["metadata"]["method"] == "test"

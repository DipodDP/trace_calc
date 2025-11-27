import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from environs import Env

from trace_calc.main import main
from trace_calc.application.orchestration import OrchestrationService


@pytest.fixture
def mock_dependencies():
    """Mocks all external dependencies for main function."""
    with patch("trace_calc.main.AsyncElevationsApiClient") as MockElevationsApiClient, \
         patch("trace_calc.main.AsyncMagDeclinationApiClient") as MockMagDeclinationApiClient, \
         patch("trace_calc.main.ConsoleOutputFormatter") as MockOutputFormatter, \
         patch("trace_calc.main.ProfileVisualizer") as MockVisualizer, \
         patch("trace_calc.main.PathProfileService") as MockProfileService, \
         patch("trace_calc.main.GrozaAnalysisService") as MockGrozaAnalysisService, \
         patch("trace_calc.main.SosnikAnalysisService") as MockSosnikAnalysisService, \
         patch("trace_calc.main.FilePathStorage") as MockFilePathStorage, \
         patch("trace_calc.main.OrchestrationService") as MockOrchestrationService, \
         patch("trace_calc.main.Env") as MockEnv, \
         patch("builtins.input") as mock_input:

        # Configure mocks
        MockEnv.return_value.str.side_effect = lambda x: f"mock_{x}"
        mock_input.side_effect = ["test_path", "", "", "50.0 14.0", "50.1 14.1"] # Simulate user input for file name, default antenna heights, and coordinates

        # Mock the async load/store methods
        mock_storage_instance = MockFilePathStorage.return_value
        mock_storage_instance.load = AsyncMock(side_effect=FileNotFoundError) # Simulate no stored path
        mock_storage_instance.store = AsyncMock()

        # Mock PathProfileService.get_profile to return a dummy PathData
        mock_path_data = MagicMock()
        mock_path_data.coordinates = [[0.0, 0.0], [1.0, 1.0]] # Dummy coordinates
        MockProfileService.return_value.get_profile = AsyncMock(return_value=mock_path_data)

        # Mock analysis services
        MockGrozaAnalysisService.return_value = MagicMock()
        MockSosnikAnalysisService.return_value = MagicMock()


        # Mock OrchestrationService.process to be an AsyncMock
        mock_analysis_result = MagicMock()
        mock_analysis_result.link_speed = 100.0  # Top-level attribute
        mock_analysis_result.result = {"speed_prefix": "M", "wavelength": 0.3}
        mock_orchestration_service_instance = MockOrchestrationService.return_value
        mock_orchestration_service_instance.process = AsyncMock(return_value=mock_analysis_result)

        yield {
            "MockElevationsApiClient": MockElevationsApiClient,
            "MockMagDeclinationApiClient": MockMagDeclinationApiClient,
            "MockOutputFormatter": MockOutputFormatter,
            "MockVisualizer": MockVisualizer,
            "MockProfileService": MockProfileService,
            "MockGrozaAnalysisService": MockGrozaAnalysisService,
            "MockSosnikAnalysisService": MockSosnikAnalysisService,
            "MockFilePathStorage": MockFilePathStorage,
            "MockOrchestrationService": MockOrchestrationService,
            "MockEnv": MockEnv,
            "mock_input": mock_input,
        }


@pytest.mark.asyncio
async def test_main_instantiates_orchestration_service_correctly(mock_dependencies):
    """
    Test that main function correctly instantiates OrchestrationService
    with all required dependencies.
    """
    # Simulate command line arguments
    with patch("sys.argv", ["main.py", "--method", "groza"]):
        # Run the main function
        await main()

        # Assert that OrchestrationService was called with the correct dependencies
        MockOrchestrationService = mock_dependencies["MockOrchestrationService"]
        MockOrchestrationService.assert_called_once()
        call_args = MockOrchestrationService.call_args[1] # Keyword arguments

        # Assert correct service types are passed
        assert isinstance(call_args["analysis_service"], MagicMock)  # GrozaAnalysisService mock
        assert isinstance(call_args["profile_service"], MagicMock)  # PathProfileService mock
        assert isinstance(call_args["declinations_api_client"], MagicMock)  # AsyncMagDeclinationApiClient mock
        assert isinstance(call_args["visualizer"], MagicMock)  # ProfileVisualizer mock

        # Assert specific mocks are used
        mock_dependencies["MockGrozaAnalysisService"].assert_called_once()
        mock_dependencies["MockProfileService"].assert_called_once()
        mock_dependencies["MockElevationsApiClient"].assert_called_once_with(
            "mock_ELEVATION_API_URL", "mock_ELEVATION_API_KEY"
        )
        mock_dependencies["MockMagDeclinationApiClient"].assert_called_once_with(
            "mock_DECLINATION_API_URL", "mock_DECLINATION_API_KEY"
        )
        mock_dependencies["MockVisualizer"].assert_called_once()
import asyncio
import numpy as np

from trace_calc.domain.models.path import PathData
from trace_calc.domain.models.units import Meters, Angle
from trace_calc.domain.models.coordinates import InputData, Coordinates
from trace_calc.infrastructure.visualization.plotter import ProfileVisualizer
from trace_calc.infrastructure.output.formatters import ConsoleOutputFormatter
from trace_calc.application.analysis import GrozaAnalysisService
from trace_calc.application.orchestration import OrchestrationService


class MockElevationsApiClient:
    """A mock API client that returns a V-shaped terrain profile."""

    async def get_elevations(self, coordinates: list[Coordinates]) -> list[float]:
        num_points = len(coordinates)
        distances = np.linspace(0, 100, num_points)
        elevations = 200 - np.abs(distances - 50) * 5
        return elevations.tolist()


async def main():
    """
    Generates a plot and console output to verify the fix for elevation_terrain.
    """
    print("=" * 70)
    print("RUNNING VERIFICATION SCRIPT")
    print("=" * 70)

    # 1. Create dummy input data
    input_data = InputData(
        path_name="verification_run",
        site_a_coordinates=Coordinates(lat=0, lon=0),
        site_b_coordinates=Coordinates(lat=1, lon=1),
        antenna_a_height=Meters(20),
        antenna_b_height=Meters(20),
        hpbw=Angle(0.5),
    )

    # 2. Set up services with mock dependencies
    mock_api_client = MockElevationsApiClient()

    lats = np.linspace(0, 1, 101)
    lons = np.linspace(0, 1, 101)
    dummy_coords = [Coordinates(lat=lat, lon=lon) for lat, lon in zip(lats, lons)]

    distances = np.linspace(0, 100, 101)
    elevations = await mock_api_client.get_elevations(dummy_coords)
    path = PathData(
        distances=distances,
        elevations=np.array(elevations),
        coordinates=np.array([[c.lat, c.lon] for c in dummy_coords]),
    )

    # 3. Set up the orchestrator
    orchestrator = OrchestrationService(
        analysis_service=GrozaAnalysisService(),
        profile_service=None,
        output_formatter=ConsoleOutputFormatter(),
        visualizer=ProfileVisualizer(),
    )

    # 4. Run the orchestrator to generate output and plot
    print("\nRunning analysis and generating output...")
    await orchestrator.process(
        input_data,
        antenna_a_height=input_data.antenna_a_height,
        antenna_b_height=input_data.antenna_b_height,
        display_output=True,
        generate_plot=True,
        path=path,
        save_plot_path="output_data/verification_plot_km.png",
    )

    print("\n" + "=" * 70)
    print("✅ VERIFICATION COMPLETE")
    print(
        "Please check the console output above for the corrected 'above terrain' height."
    )
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

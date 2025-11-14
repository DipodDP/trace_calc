import asyncio
import argparse
from environs import Env

from trace_calc.application.analysis import (
    BaseAnalysisService,
    GrozaAnalysisService,
    SosnikAnalysisService,
)
from trace_calc.domain.models.coordinates import Coordinates, InputData
from trace_calc.domain.models.path import PathData
from trace_calc.domain.models.units import Meters
from trace_calc.domain.constants import OUTPUT_DATA_DIR
from trace_calc.infrastructure.api.clients import (
    AsyncElevationsApiClient,
    AsyncMagDeclinationApiClient,
)
from trace_calc.infrastructure.storage import FilePathStorage
from trace_calc.application.orchestration import OrchestrationService
from trace_calc.application.services.profile import PathProfileService
from trace_calc.infrastructure.output.formatters import ConsoleOutputFormatter
from trace_calc.infrastructure.visualization.plotter import ProfileVisualizer


class AppDependencies:
    """Container for application dependencies."""

    def __init__(self, env: Env):
        self.storage = FilePathStorage(output_dir=OUTPUT_DATA_DIR)
        self.elevations_api_client = AsyncElevationsApiClient(
            env.str("ELEVATION_API_URL"), env.str("ELEVATION_API_KEY")
        )
        self.declinations_api_client = AsyncMagDeclinationApiClient(
            env.str("DECLINATION_API_URL"), env.str("DECLINATION_API_KEY")
        )
        self.output_formatter = ConsoleOutputFormatter()
        self.visualizer = ProfileVisualizer(style="default")


def get_analysis_service(method: str) -> BaseAnalysisService:
    """Factory for creating analysis services."""
    if method == "groza":
        return GrozaAnalysisService()
    elif method == "sosnik":
        return SosnikAnalysisService()
    else:
        raise ValueError(f"Unknown analysis method: {method}")


class UserInputHandler:
    def get_coordinates(self, prompt: str) -> Coordinates:
        try:
            lat, lon = map(float, input(prompt).split()[:2])
            return Coordinates(lat=lat, lon=lon)
        except ValueError:
            raise ValueError(
                "Coordinates must be in a valid numeric format (e.g., -123.456 12.345)"
            )

    def get_antenna_height(self, prompt: str) -> Meters | None:
        try:
            height_str = input(prompt)
            return Meters(int(height_str)) if height_str else None
        except ValueError:
            raise ValueError("Please enter a valid integer.")

    def get_user_input(self) -> InputData:
        stored_filename = input("Enter stored file name (without .path): ")
        antenna_a_height = self.get_antenna_height(
            "Enter antenna 1 height or skip to use default: "
        )
        antenna_b_height = self.get_antenna_height(
            "Enter antenna 2 height or skip to use default: "
        )

        input_data = InputData(path_name=stored_filename)
        if antenna_a_height is not None:
            input_data.antenna_a_height = antenna_a_height
        if antenna_b_height is not None:
            input_data.antenna_b_height = antenna_b_height
        return input_data


async def run_analysis(
    orchestrator: OrchestrationService, input_data: InputData, path: PathData
):
    """Runs the analysis and prints the result."""
    result = await orchestrator.process(
        input_data,
        antenna_a_height=input_data.antenna_a_height,
        antenna_b_height=input_data.antenna_b_height,
        display_output=True,
        generate_plot=True,
        path=path,
        save_plot_path=f"{OUTPUT_DATA_DIR}/{input_data.path_name}_profile.png",
    )
    print(f"\n✅ Analysis complete! Link speed: {result.link_speed:.1f} Mbps")


async def main():
    parser = argparse.ArgumentParser(description="Trace Calculator")
    parser.add_argument(
        "--method",
        type=str,
        choices=["groza", "sosnik"],
        default="groza",
        help="Analysis method to use (groza or sosnik)",
    )
    args = parser.parse_args()

    env = Env()
    env.read_env(".env")

    deps = AppDependencies(env)
    input_handler = UserInputHandler()

    try:
        input_data = input_handler.get_user_input()

        path: PathData | None = None
        try:
            path = await deps.storage.load(input_data.path_name)
            input_data.site_a_coordinates = Coordinates(*path.coordinates[0])
            input_data.site_b_coordinates = Coordinates(*path.coordinates[-1])
        except (FileNotFoundError, IndexError, ValueError):
            print("Coordinates are required for calculation. Enter them manually.")
            input_data.site_a_coordinates = input_handler.get_coordinates(
                'Input site "A" coordinates (format: -123.456 12.345): '
            )
            input_data.site_b_coordinates = input_handler.get_coordinates(
                'Input site "B" coordinates (format: -123.456 12.345): '
            )

        profile_service = PathProfileService(
            input_data=input_data,
            elevations_api_client=deps.elevations_api_client,
        )

        if path is None:
            path = await profile_service.get_profile()
            await deps.storage.store(input_data.path_name, path)

        analysis_service = get_analysis_service(args.method)

        orchestrator = OrchestrationService(
            analysis_service=analysis_service,
            profile_service=profile_service,
            output_formatter=deps.output_formatter,
            visualizer=deps.visualizer,
        )

        await run_analysis(orchestrator, input_data, path)

    except (ValueError, EOFError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

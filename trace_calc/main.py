import asyncio
import argparse
from environs import Env
import os

from trace_calc.logging_config import setup_logging
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
from trace_calc.infrastructure.output.formatters import (
    ConsoleOutputFormatter,
    JSONOutputFormatter,
)
from trace_calc.infrastructure.visualization.plotter import ProfileVisualizer
from trace_calc.domain.exceptions import APIException


class AppDependencies:
    """Container for application dependencies."""

    def __init__(self, env: Env, args: argparse.Namespace):
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
    orchestrator: OrchestrationService,
    input_data: InputData,
    path: PathData,
    args: argparse.Namespace,
    deps: AppDependencies,
):
    """Runs the analysis and handles output formatting."""
    result = await orchestrator.process(
        input_data,
        antenna_a_height=input_data.antenna_a_height,
        antenna_b_height=input_data.antenna_b_height,
        display_output=False,  # We handle output here
        generate_plot=True,
        path=path,
        save_plot_path=f"{OUTPUT_DATA_DIR}/{input_data.path_name}.png",
    )

    geo_data = result.result.get("geo_data")
    profile_data = result.result.get("profile_data")

    speed_prefix = result.result.get("speed_prefix", "M")
    speed_unit = "Mbps" if speed_prefix == "M" else "kbps"

    if args.save_json:
        formatter = JSONOutputFormatter()
        json_output = formatter.format_result(
            result, input_data, geo_data, profile_data
        )
        file_path = os.path.join(OUTPUT_DATA_DIR, f"{input_data.path_name}.json")
        with open(file_path, "w") as f:
            f.write(json_output)
        print(f"✅ JSON output saved to {file_path}")
        print(f"   Link speed: {result.link_speed:.1f} {speed_unit}")
    else:
        deps.output_formatter.format_result(result, input_data, geo_data, profile_data)
        print(f"✅ Analysis complete! Link speed: {result.link_speed:.1f} {speed_unit}")


async def main():
    parser = argparse.ArgumentParser(description="Trace Calculator")
    parser.add_argument(
        "--method",
        type=str,
        choices=["groza", "sosnik"],
        default="groza",
        help="Analysis method to use (groza or sosnik)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save JSON output to a file in the output_data directory",
    )
    args = parser.parse_args()

    # Load environment variables as early as possible within main()
    env = Env()
    env.read_env(".env")

    # Setup logging ONLY AFTER environment variables are loaded, passing env
    setup_logging(env)  # Pass the env object

    deps = AppDependencies(env, args)
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
            declinations_api_client=deps.declinations_api_client,
            visualizer=deps.visualizer,
        )

        await run_analysis(orchestrator, input_data, path, args, deps)

    except (ValueError, EOFError) as e:
        print(f"Error: {e}")
    except APIException as e:
        print(
            f"API Error: {e}\nPlease ensure your API keys in the .env file are correct and have access to the Geomagnetic Declination API."
        )


if __name__ == "__main__":
    asyncio.run(main())

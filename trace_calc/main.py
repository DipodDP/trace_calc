import asyncio
from pathlib import Path
from trace_calc.models.input_data import Coordinates, InputData
from environs import Env

from trace_calc.service.api_clients import AsyncElevationsApiClient
from trace_calc.service.path_calculator import GrozaCalculator
from trace_calc.service.path_storage import FilePathStorage
from trace_calc.service.process import AnalyzerService


# --- Input Handling ---
class UserInputHandler:
    def get_coordinates(self, prompt: str):
        try:
            coordinates = (float(coord) for coord in input(prompt).split()[:2])
        except ValueError:
            raise ValueError(
                "Coordinates must be in a valid numeric format (e.g., -123.456 12.345)"
            )
        return Coordinates(*coordinates)

    def get_antenna_heights(self, prompt: str) -> int | None:
        try:
            height = input(prompt)
            return int(height) if height else None
        except ValueError:
            raise ValueError("Please enter a valid integer.")


# --- Application Orchestration ---
class Application:
    def __init__(self):
        self.input_handler = UserInputHandler()

    async def run(self, service: AnalyzerService):
        # analyzer_type = input("Choose analyzer (groza/sosnik): ").lower().strip()
        stored_filename = input("Enter stored file name (without .path): ")
        file_path = Path(stored_filename + ".path")

        input_data = InputData(stored_filename)
        antennas_heights = {
            k: v
            for k, v in {
                "antenna_a_height": self.input_handler.get_antenna_heights(
                    "Enter antenna 1 height or skip to use default: "
                ),
                "antenna_b_height": self.input_handler.get_antenna_heights(
                    "Enter antenna 2 height or skip to use default: "
                ),
            }.items()
            if v is not None
        }

        # Process the data: fetch additional info, calculate, and store the result.
        try:
            result = await service.process(input_data)
        except ValueError:
            print(
                f"Coordinates are required for calculation. Enter them manually. The file {file_path} has no stored data."
            )
            input_data.site_a_coordinates = self.input_handler.get_coordinates(
                'Input site "A" coordinates (format: -123.456 12.345): ',
            )
            input_data.site_b_coordinates = self.input_handler.get_coordinates(
                'Input site "B" coordinates (format: -123.456 12.345): ',
            )
            result = await service.process(input_data, **antennas_heights)

        # if analyzer_type == "groza":
        #     analyzer = GrozaAnalyzer(
        #         coord_a, coord_b, Lk=0, path_filename=stored_filename, ha1=ha1, ha2=ha2
        #     )
        # elif analyzer_type == "sosnik":
        #     analyzer = SosnikAnalyzer(
        #         coord_a, coord_b, Lk=0, path_filename=stored_filename, ha1=ha1, ha2=ha2
        #     )
        # else:
        #     print("Unknown analyzer type. Exiting.")
        #     return
        # analyzer = SosnikAnalyzer(
        #     coord_a, coord_b, Lk=0, path_filename=stored_filename, ha1=ha1, ha2=ha2
        # )

        print("Analysis Result:", result)


# --- Run the Application ---
if __name__ == "__main__":
    env = Env()
    env.read_env(".env")

    elevations_api_key = env.str("ELEVATION_API_KEY")
    elevations_api_url = env.str("ELEVATION_API_URL")
    path_fetcher = AsyncElevationsApiClient(elevations_api_url, elevations_api_key)

    calculator = GrozaCalculator()
    storage = FilePathStorage()
    groza_service = AnalyzerService(calculator, storage, path_fetcher)
    asyncio.run(Application().run(groza_service))

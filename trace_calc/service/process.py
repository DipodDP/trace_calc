from trace_calc.models.input_data import Coordinates, InputData
from trace_calc.service.base import (
    BaseCalculator,
    BaseElevationsApiClient,
    BasePathStorage,
)
from trace_calc.service.profile_service import PathProfileService


class AnalyzerService:
    def __init__(
        self,
        calculator: BaseCalculator,
        storage: BasePathStorage,
        elevations_api_client: BaseElevationsApiClient,
    ):
        self.storage = storage
        self.calculator = calculator
        self.elevations_api_client = elevations_api_client

    async def process(self, input_data: InputData) -> tuple[float, ...]:
        # 1. Fetch path information
        try:
            # Attempt to load path data from file
            path_data = await self.storage.load(input_data.path_filename)
            if input_data.site_a_coordinates not in (
                None,
                Coordinates(*path_data.coordinates[0]),
            ) or input_data.site_b_coordinates not in (
                None,
                Coordinates(*path_data.coordinates[-1]),
            ):
                raise ValueError("Stored coordinates don't match!")
        except (FileNotFoundError, IndexError, ValueError):
            # Get the new one if there are problems with file
            # or path coordinates didn't match.
            profile_service = PathProfileService(input_data, self.elevations_api_client)
            path_data = await profile_service.get_profile()
            # 2. Store path information
            await self.storage.store(input_data.path_filename, path_data)
        # 3. Perform calculation using both input and fetched data
        data = self.calculator.calculate(input_data, path_data)
        return data

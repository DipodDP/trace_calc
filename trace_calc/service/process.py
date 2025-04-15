from typing import Any, Type
from trace_calc.models.input_data import Coordinates, InputData
from trace_calc.service.base import (
    BaseAnalyzer,
    BaseElevationsApiClient,
    BasePathStorage,
)
from trace_calc.service.profile_service import PathProfileService


class AnalyzerService:
    def __init__(
        self,
        profile_analyzer_cls: Type[BaseAnalyzer],
        storage: BasePathStorage,
        elevations_api_client: BaseElevationsApiClient | None,
    ):
        self.storage = storage
        self.profile_analyzer = profile_analyzer_cls
        self.elevations_api_client = elevations_api_client

    async def process(self, input_data: InputData) -> dict[str, Any]:
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
            # Get the new one if there are problems with stored path
            # or path coordinates didn't match.
            if not self.elevations_api_client:
                raise RuntimeError("Elevations API client is missing!")
            profile_service = PathProfileService(input_data, self.elevations_api_client)
            path_data = await profile_service.get_profile()
            # 2. Store path information
            await self.storage.store(input_data.path_filename, path_data)
        # 3. Perform calculation using both input and fetched data
        analyzer = self.profile_analyzer(path_data, input_data)
        data = analyzer.analyze(Lk=input_data.climate_losses)
        return data

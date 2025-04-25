from typing import Any, Type

from trace_calc.models.input_data import Coordinates, InputData
from trace_calc.models.path import GeoData, PathData
from trace_calc.service.base import (
    BaseAnalyzer,
    BaseDeclinationsApiClient,
    BaseElevationsApiClient,
    BasePathStorage,
)
from trace_calc.service.coordinates_service import CoordinatesService
from trace_calc.service.profile_service import PathProfileService


class AnalyzerService:
    path_data: PathData

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
            self.path_data = await self.storage.load(input_data.path_filename)
            if input_data.site_a_coordinates not in (
                None,
                Coordinates(*self.path_data.coordinates[0]),
            ) or input_data.site_b_coordinates not in (
                None,
                Coordinates(*self.path_data.coordinates[-1]),
            ):
                raise ValueError("Stored coordinates don't match!")
        except (FileNotFoundError, IndexError, ValueError):
            # Get the new one if there are problems with stored path
            # or path coordinates didn't match.
            if not self.elevations_api_client:
                raise RuntimeError("Elevations API client is missing!")
            profile_service = PathProfileService(input_data, self.elevations_api_client)
            self.path_data = await profile_service.get_profile()
            # 2. Store path information
            await self.storage.store(input_data.path_filename, self.path_data)
        # 3. Perform calculation using both input and fetched data
        analyzer = self.profile_analyzer(self.path_data, input_data)
        data = analyzer.analyze(Lk=input_data.climate_losses)
        return data


class GeoDataService:
    def __init__(
        self,
        declinations_api_client: BaseDeclinationsApiClient,
    ):
        self.declinations_api_client = declinations_api_client

    async def process(
        self,
        coord_a: Coordinates,
        coord_b: Coordinates,
    ) -> GeoData:
        if not self.declinations_api_client:
            raise RuntimeError("Elevations API client is missing!")

        mag_declinations = await self.declinations_api_client.fetch_declinations((
            coord_a,
            coord_b,
        ))
        distance = CoordinatesService(coord_a, coord_b).get_distance()
        azimuth_a_b = CoordinatesService(coord_a, coord_b).get_azimuth()
        azimuth_b_a = CoordinatesService(coord_b, coord_a).get_azimuth()

        return GeoData(
            distance=distance,
            mag_declination_a=mag_declinations[0],
            mag_declination_b=mag_declinations[1],
            true_azimuth_a_b=azimuth_a_b,
            true_azimuth_b_a=azimuth_b_a,
            mag_azimuth_a_b=azimuth_a_b - mag_declinations[0],
            mag_azimuth_b_a=azimuth_b_a - mag_declinations[0],
        )

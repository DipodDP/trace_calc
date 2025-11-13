from typing import Any, Type

from trace_calc.domain.units import Angle, Kilometers, Degrees
from trace_calc.models.input_data import Coordinates, InputData
from trace_calc.models.path import GeoData, PathData
from trace_calc.services.base import (
    BaseAnalyzer,
    BaseDeclinationsApiClient,
    BaseElevationsApiClient,
    BasePathStorage,
)
from trace_calc.services.coordinates_service import CoordinatesService
from trace_calc.services.exceptions import CoordinatesRequiredException
from trace_calc.services.profile_service import PathProfileService


class AnalyzerService:
    """
    Orchestrates path data retrieval and analysis.
    """

    path_data: PathData

    def __init__(
        self,
        profile_analyzer_cls: Type[BaseAnalyzer],
        storage: BasePathStorage,
        elevations_api_client: BaseElevationsApiClient | None,
    ):
        """
        Initialize service with storage and analyzer.

        Args:
            profile_analyzer_cls: Analyzer class for profile computation.
            storage: Persistent storage for path data.
            elevations_api_client: Optional API client for elevation data fetching.
        """
        self.storage = storage
        self.profile_analyzer = profile_analyzer_cls
        self.elevations_api_client = elevations_api_client

    async def process(self, input_data: InputData) -> dict[str, Any]:
        """
        Retrieve or fetch path data, then run analysis.

        Workflow:
          1. Attempt load from storage
          2. On failure, fetch from API and store
          3. Analyze with the provided analyzer

        Returns:
            Analysis results as a dict.
        """
        try:
            await self.get_path_data_from_storage(input_data)
        except (FileNotFoundError, IndexError, ValueError):
            await self.get_path_data_from_api(input_data)
            await self.storage.store(input_data.path_name, self.path_data)

        analyzer = self.profile_analyzer(self.path_data, input_data)
        data = analyzer.analyze(Lk=input_data.climate_losses)
        return data

    async def get_path_data_from_storage(self, input_data: InputData) -> None:
        """
        Load path data from storage, verifying site coordinates.
        """
        self.path_data = await self.storage.load(input_data.path_name)
        if input_data.site_a_coordinates not in (
            None,
            Coordinates(*self.path_data.coordinates[0]),
        ) or input_data.site_b_coordinates not in (
            None,
            Coordinates(*self.path_data.coordinates[-1]),
        ):
            raise CoordinatesRequiredException("Stored coordinates don't match!")

    async def get_path_data_from_api(self, input_data: InputData) -> None:
        """
        Fetch fresh path profile via elevations API.
        """

        if not self.elevations_api_client:
            raise RuntimeError("Elevations API client is missing!")
        profile_service = PathProfileService(input_data, self.elevations_api_client)
        self.path_data = await profile_service.get_profile()


class GeoDataService:
    """
    Computes geographic metadata between two coordinates.
    """

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
        """
        Fetch declinations, calculate distance and azimuths between two coordinates.

        Returns:
            GeoData with distance, true/magnetic azimuths, and declinations.

        Raises:
            RuntimeError: If declination client is missing.
        """

        if not self.declinations_api_client:
            raise RuntimeError("Elevations API client is missing!")

        mag_declinations: list[
            Angle
        ] = await self.declinations_api_client.fetch_declinations(
            (
                coord_a,
                coord_b,
            )
        )
        distance: Kilometers = CoordinatesService(coord_a, coord_b).get_distance()
        azimuth_a_b: Angle = CoordinatesService(coord_a, coord_b).get_azimuth()
        azimuth_b_a: Angle = CoordinatesService(coord_b, coord_a).get_azimuth()

        return GeoData(
            distance=distance,
            mag_declination_a=Angle(Degrees(round(mag_declinations[0], 2))),
            mag_declination_b=Angle(Degrees(round(mag_declinations[1], 2))),
            true_azimuth_a_b=Angle(Degrees(round(azimuth_a_b, 2))),
            true_azimuth_b_a=Angle(Degrees(round(azimuth_b_a, 2))),
            mag_azimuth_a_b=Angle(Degrees(round(azimuth_a_b - mag_declinations[0], 2))),
            mag_azimuth_b_a=Angle(Degrees(round(azimuth_b_a - mag_declinations[1], 2))),
        )

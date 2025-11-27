from typing import Any

import numpy as np
from numpy.typing import NDArray

from trace_calc.domain.models.units import Kilometers
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import PathData
from trace_calc.domain.interfaces import BaseElevationsApiClient
from .coordinates import CoordinatesService
from trace_calc.domain.exceptions import CoordinatesRequiredException

from trace_calc.logging_config import get_logger

logger = get_logger(__name__)


class PathProfileService:
    def __init__(
        self,
        input_data: InputData,
        elevations_api_client: BaseElevationsApiClient,
        block_size: int = 128,
        resolution: float = 0.5,
    ):
        self.elevations_api_client = elevations_api_client
        self.block_size = block_size
        self.resolution = resolution
        if (
            input_data.site_a_coordinates is None
            or input_data.site_b_coordinates is None
        ):
            raise CoordinatesRequiredException(
                "Cannot fetch elevations without coordinates"
            )
        self.coord_a = input_data.site_a_coordinates
        self.coord_b = input_data.site_b_coordinates

    def linspace_coord(self) -> NDArray[np.floating[Any]]:
        """
        Generates a coordinate vector between two points with a given resolution.
        """

        coordinates_service = CoordinatesService(self.coord_a, self.coord_b)
        full_distance: Kilometers = coordinates_service.get_distance()

        # Extend coordinates longitude if crossing 180 degree to create linear space vector
        coord_a, coord_b = coordinates_service.get_extended_coordinates()

        points_num = (np.ceil(full_distance / self.resolution)).astype(int)

        max_points = 4 * self.block_size
        if points_num > max_points:
            points_num = max_points

        if points_num < self.block_size:
            points_num = self.block_size

        points_num = (np.ceil(points_num / self.block_size) * self.block_size).astype(
            int
        )

        lat_vector = np.linspace(coord_a[0], coord_b[0], points_num)
        lon_vector = np.linspace(coord_a[1], coord_b[1], points_num)

        # Getting back normal longitude
        for i in range(lon_vector.size):
            if lon_vector[i] > 180:
                lon_vector[i] = CoordinatesService.normalize_longitude_180(
                    lon_vector[i]
                )

        return np.column_stack((lat_vector, lon_vector))

    async def get_profile(self) -> PathData:
        """
        Entry point method that creates a profile containing distance and elevation arrays.
        """
        self.coord_vect = self.linspace_coord()
        points_num = self.coord_vect.shape[0]
        logger.debug(f"Coordinates: {self.coord_a} {self.coord_b}")

        coordinates_service = CoordinatesService(self.coord_a, self.coord_b)
        full_distance: Kilometers = coordinates_service.get_distance()

        path_profile = PathData(
            coordinates=self.coord_vect,
            distances=np.linspace(0, full_distance, points_num, dtype=np.float64),
            elevations=np.zeros(points_num, dtype=np.float64),
        )

        path_profile.elevations = await self.elevations_api_client.fetch_elevations(
            self.coord_vect, self.block_size
        )
        return path_profile

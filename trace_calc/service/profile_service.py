from typing import Any

import numpy as np
from numpy.typing import NDArray

from trace_calc.models.input_data import Coordinates, InputData
from trace_calc.models.path import PathData
from trace_calc.service.base import BaseElevationsApiClient


class PathProfileService:
    def __init__(
        self,
        input_data: InputData,
        elevations_api_client: BaseElevationsApiClient,
        block_size=256,
        resolution=0.5,
    ):
        self.elevations_api_client = elevations_api_client
        self.block_size = block_size
        self.resolution = resolution
        if (
            input_data.site_a_coordinates is None
            or input_data.site_b_coordinates is None
        ):
            raise ValueError("Cannot fetch elevations without coordinates")
        self.coord_a = input_data.site_a_coordinates
        self.coord_b = input_data.site_b_coordinates

        # Adjust S and W coordinates if necessary
        self._adjust_coordinates()

    def _adjust_coordinates(self) -> None:
        """
        Adjusts for negative coordinates if needed.
        """
        lat_a, lon_a = self.coord_a
        lat_b, lon_b = self.coord_b

        # Adjust latitude (index 0)
        if lat_a < 0 and abs(lat_a + 360 - lat_b) < 180:
            lat_a += 360
        if lat_b < 0 and abs(lat_a + 360 - lat_b) < 180:
            lat_b += 360

        # Adjust longitude (index 1)
        if lon_a < 0 and abs(lon_a + 360 - lon_b) < 180:
            lon_a += 360
        if lon_b < 0 and abs(lon_a + 360 - lon_b) < 180:
            lon_b += 360

        self.coord_a = Coordinates(lat_a, lon_a)
        self.coord_b = Coordinates(lat_b, lon_b)

    def linspace_coord(self) -> NDArray[np.floating[Any]]:
        """
        Generates a coordinate vector between two points with a given resolution.
        """

        full_distance = self.get_distance(self.coord_a, self.coord_b)
        points_num = (
            np.ceil(full_distance / (self.block_size * self.resolution))
            * self.block_size
        ).astype(int)
        lat_vector = np.linspace(self.coord_a[0], self.coord_b[0], points_num)
        lon_vector = np.linspace(self.coord_a[1], self.coord_b[1], points_num)
        return np.column_stack((lat_vector, lon_vector))

    async def get_profile(self) -> PathData:
        """
        Entry point method that creates a profile containing distance and elevation arrays.
        """
        self.coord_vect = self.linspace_coord()
        points_num = self.coord_vect.shape[0]
        print(f"Coordinates: {self.coord_a} {self.coord_b}")

        path_profile = PathData(
            coordinates=self.coord_vect,
            distances=np.zeros(points_num),
            elevations=np.zeros(points_num),
        )

        # Calculate cumulative distance along the coordinate vector
        path_profile.distances[0] = 0.0
        for i in range(1, points_num):
            path_profile.distances[i] = self.get_distance(
                self.coord_vect[0], self.coord_vect[i]
            )
        path_profile.elevations = await self.elevations_api_client.fetch_elevations(
            self.coord_vect, self.block_size
        )
        return path_profile

    @classmethod
    def get_distance(cls, coord_1: Coordinates, coord_2: Coordinates) -> float:
        """
        Calculates the distance between two coordinates in kilometers.
        """
        return 6371.21 * cls.get_angle(coord_1, coord_2)

    @staticmethod
    def get_angle(coord_1: Coordinates, coord_2: Coordinates):
        """
        Calculates the angular separation (in radians) between two coordinates.
        """
        a = (coord_1[0] * np.pi / 180.0, coord_1[1] * np.pi / 180.0)
        b = (coord_2[0] * np.pi / 180.0, coord_2[1] * np.pi / 180.0)
        return np.arccos(
            np.sin(a[0]) * np.sin(b[0])
            + np.cos(a[0]) * np.cos(b[0]) * np.cos(b[1] - a[1])
        )

    @staticmethod
    def coord_min2dec(degree: int, minutes: int, seconds=0.0):
        """
        Converts coordinates in degrees, minutes, and seconds to decimal degrees.
        """
        return degree + minutes / 60 + seconds / 3600

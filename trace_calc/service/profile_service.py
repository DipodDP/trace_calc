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

    def linspace_coord(self) -> NDArray[np.floating[Any]]:
        """
        Generates a coordinate vector between two points with a given resolution.
        """

        # Adjust S and W coordinates if necessary
        coord_a, coord_b = self._get_extended_coordinates(self.coord_a, self.coord_b)

        full_distance = self.get_distance(
            coord_a, coord_b, from_extended_coordinates=True
        )
        points_num = (
            np.ceil(full_distance / (self.block_size * self.resolution))
            * self.block_size
        ).astype(int)
        lat_vector = np.linspace(coord_a[0], coord_b[0], points_num)
        lon_vector = np.linspace(coord_a[1], coord_b[1], points_num)
        for i in range(lon_vector.size):
            # getting back W and S coords from coord vector
            if lon_vector[i] > 180:
                lon_vector[i] = self.normalize_longitude_180(lon_vector[i])

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
                self.coord_vect[0], self.coord_vect[i], from_extended_coordinates=True
            )
        path_profile.elevations = await self.elevations_api_client.fetch_elevations(
            self.coord_vect, self.block_size
        )
        return path_profile

    @classmethod
    def get_distance(
        cls, coord_1: Coordinates, coord_2: Coordinates, from_extended_coordinates=False
    ) -> float:
        """
        Calculates the distance between two coordinates in kilometers.
        """
        return 6371.21 * cls.get_angle(coord_1, coord_2, from_extended_coordinates)

    @classmethod
    def get_angle(
        cls, coord_1: Coordinates, coord_2: Coordinates, from_extended_coordinates=False
    ):
        """
        Calculates the angular separation (in radians) between two coordinates.
        """
        if not from_extended_coordinates:
            coord_1, coord_2 = cls._get_extended_coordinates(coord_1, coord_2)

        a = (coord_1[0] * np.pi / 180.0, coord_1[1] * np.pi / 180.0)
        b = (coord_2[0] * np.pi / 180.0, coord_2[1] * np.pi / 180.0)
        return np.arccos(
            np.sin(a[0]) * np.sin(b[0])
            + np.cos(a[0]) * np.cos(b[0]) * np.cos(b[1] - a[1])
        )

    @staticmethod
    def normalize_longitude_180(lon: float) -> float:
        """
        Normalize a longitude to 0...180 for east and 0...-180 for west.
        """
        return ((lon + 180) % 360) - 180

    @staticmethod
    def coord_min2dec(degree: int, minutes: int, seconds=0.0):
        """
        Converts coordinates in degrees, minutes, and seconds to decimal degrees.
        """
        return degree + minutes / 60 + seconds / 3600

    @staticmethod
    def _get_extended_coordinates(
        coord_1: Coordinates, coord_2: Coordinates
    ) -> tuple[Coordinates, Coordinates]:
        """
        Adjusts for negative coordinates if needed.
        """
        lat_a, lon_a = coord_1
        lat_b, lon_b = coord_2

        # Adjust longitude (index 1)
        if lon_a < 0 and abs(lon_a + 360 - lon_b) < 180:
            lon_a += 360
        if lon_b < 0 and abs(lon_b + 360 - lon_a) < 180:
            lon_b += 360

        return Coordinates(lat_a, lon_a), Coordinates(lat_b, lon_b)

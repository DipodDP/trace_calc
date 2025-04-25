import math
import numpy as np
from trace_calc.models.input_data import Coordinates


class CoordinatesService:
    """
    Provides functionality to calculate great circle distance and initial azimuth (bearing) between two points
    on the Earth using the full spherical equations and other coordinates-related actions.
    More info: http://gis-lab.info/qa/great-circles.html
    """

    def __init__(
        self,
        coord_a: Coordinates,
        coord_b: Coordinates,
    ):
        """
        Precompute values for two coordinates given in decimal degrees.
        """
        # Radius of the Earth (in meters)
        self.earth_radius = 6372795
        self.coord_a = coord_a
        self.coord_b = coord_b

        coord_a_adjusted, coord_b_adjusted = self.get_extended_coordinates()
        # Convert input coordinates from degrees to radians
        lat_1 = coord_a_adjusted[0] * math.pi / 180.0
        lat_2 = coord_b_adjusted[0] * math.pi / 180.0
        long_1 = coord_a_adjusted[1] * math.pi / 180.0
        long_2 = coord_b_adjusted[1] * math.pi / 180.0

        # Precompute sines and cosines of latitudes
        self.cos_lat_1 = math.cos(lat_1)
        self.cos_lat_2 = math.cos(lat_2)
        self.sin_lat_1 = math.sin(lat_1)
        self.sin_lat_2 = math.sin(lat_2)

        delta = long_2 - long_1
        self.cos_delta = math.cos(delta)
        self.sin_delta = math.sin(delta)

    def get_distance(
        self,
    ) -> float:
        """
        Calculates the distance between two coordinates in kilometers.
        """
        return self.earth_radius * self.get_angle() * 0.001

    def get_angle(
        self,
    ):
        """
        Calculates the angular separation (in radians) between two coordinates.
        """
        return np.arccos(
            self.sin_lat_1 * self.sin_lat_2
            + self.cos_lat_1 * self.cos_lat_2 * self.cos_delta
        )

    def get_azimuth(self) -> float:
        """
        Calculate the initial bearing (forward azimuth) from point A to point B.

        Azimuth is measured in degrees from North (0°) clockwise to 360°.

        Returns:
            float: Initial bearing in decimal degrees.
        """
        # Components for bearing calculation
        x = (self.cos_lat_1 * self.sin_lat_2) - (
            self.sin_lat_1 * self.cos_lat_2 * self.cos_delta
        )
        y = self.sin_delta * self.cos_lat_2

        # Raw bearing angle in degrees
        z = math.degrees(math.atan(-y / x))

        # Adjust quadrant
        if x < 0:
            z += 180.0

        # Normalize to [-180, 180) then convert to radians and unwrap to [0, 2π)
        z_norm = self.normalize_longitude_180(z)
        z_rad = -math.radians(z_norm)
        angle_rad = z_rad - ((2 * math.pi) * math.floor((z_rad / (2 * math.pi))))
        angle_degree = (angle_rad * 180.0) / math.pi

        return angle_degree

    def get_extended_coordinates(
        self,
    ) -> tuple[Coordinates, Coordinates]:
        """
        Adjusts for negative coordinates if needed.
        """
        lat_a, lon_a = self.coord_a
        lat_b, lon_b = self.coord_b

        # Adjust longitude (index 1)
        if lon_a < 0 and abs(lon_a + 360 - lon_b) < 180:
            lon_a += 360
        if lon_b < 0 and abs(lon_b + 360 - lon_a) < 180:
            lon_b += 360

        return Coordinates(lat_a, lon_a), Coordinates(lat_b, lon_b)

    @staticmethod
    def normalize_longitude_180(lon: float) -> float:
        """
        Normalize a longitude value to the range [-180, 180).

        Args:
            lon (float): Longitude in decimal degrees (unbounded).

        Returns:
            float: Normalized longitude within [-180, 180).
        """
        return ((lon + 180) % 360) - 180

    @staticmethod
    def coord_min2dec(degree: int, minutes: int, seconds: float = 0.0) -> float:
        """
        Convert coordinates given in degrees, minutes, and seconds to decimal degrees.

        Returns:
            float: Coordinate in decimal degrees.
        """
        return degree + minutes / 60 + seconds / 3600

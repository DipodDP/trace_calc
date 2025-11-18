from typing import NamedTuple

from dataclasses import dataclass

from .units import Angle, Loss, Meters


class Coordinates(NamedTuple):
    lat: float
    lon: float


@dataclass(slots=True)
class InputData:
    """
    Model that holds expected input data.

    Name of stored data, regional climate losses,
    sites coordinates:a tuple of coordinates (lat, lon), antennas heights
    """

    path_name: str
    frequency_mhz: float = 4500.0
    climate_losses: Loss = Loss(0.0)
    site_a_coordinates: Coordinates | None = None
    site_b_coordinates: Coordinates | None = None
    antenna_a_height: Meters = Meters(2)
    antenna_b_height: Meters = Meters(2)
    elevation_angle_offset: Angle = Angle(2.5)

    def __post_init__(self):
        if self.elevation_angle_offset < 0:
            raise ValueError(
                "elevation_angle_offset must be non-negative"
            )
        if self.elevation_angle_offset > 45:
            raise ValueError(
                "elevation_angle_offset must be <= 45 degrees"
            )

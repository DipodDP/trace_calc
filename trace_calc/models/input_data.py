from typing import NamedTuple

from dataclasses import dataclass


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
    climate_losses = 0.0
    site_a_coordinates: Coordinates | None = None
    site_b_coordinates: Coordinates | None = None
    antenna_a_height = 2
    antenna_b_height = 2

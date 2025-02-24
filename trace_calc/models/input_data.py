from typing import NamedTuple

from dataclasses import dataclass


class Coordinates(NamedTuple):
    lat: float
    lon: float

@dataclass
class InputData:
    path_filename: str
    Lk = 0.0
    site_a_coordinates: Coordinates | None = None
    site_b_coordinates: Coordinates | None = None
    antenna_a_height = 2
    antenna_b_height = 2
    # bot_mode = bot_mode


from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class PathData:
    """
    Model that holds path-related data.

    :param coordinates: An array of coordinates (e.g., [[lat, lon], ...])
    :param distances: An array of distances
    :param elevations: An array of elevations
    """

    coordinates: NDArray[np.floating[Any]]
    distances: NDArray[np.float64]
    elevations: NDArray[np.float64]

    def __repr__(self):
        return (
            f"PathData(coordinates={self.coordinates}, "
            f"distances={self.distances}, elevations={self.elevations})"
        )

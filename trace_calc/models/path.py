from typing import Any

import numpy as np
from numpy.typing import NDArray


# --- Domain Model ---
class PathData:
    """
    Model that holds path-related data.
    """

    def __init__(
        self,
        coordinates: NDArray[np.floating[Any]],
        distances: NDArray[np.float64],
        elevations: NDArray[np.float64],
    ):
        """
        :param coordinates: An array or list of coordinates (e.g., [[lat, lon], ...])
        :param distances: An array or list of distances
        :param elevations: An array or list of elevations
        """
        self.coordinates = coordinates
        self.distances = distances
        self.elevations = elevations

    def __repr__(self):
        return (
            f"PathData(coordinates={self.coordinates}, "
            f"distances={self.distances}, elevations={self.elevations})"
        )

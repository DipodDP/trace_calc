from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trace_calc.models.path import PathData


class BasePathStorage(ABC):
    """
    Abstract base class that defines the interface for loading path data.
    """

    @abstractmethod
    async def load(self, filename: str) -> PathData:
        """
        Asynchronously load path data from a file or another source.

        :param filename: The filename (without extension) to load data from.
        :return: An instance of PathData.
        """
        pass

    @abstractmethod
    async def store(self, filename: str, path_data: PathData) -> None:
        """
        Asynchronously store path data to a file or another storage.

        :param filename: The filename (without extension) to store the data.
        :return: An instance of PathData.
        """
        pass


class BaseElevationsApiClient(ABC):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.elevations_api_key = api_key

    @abstractmethod
    async def fetch_elevations(
        self,
        coord_vect: NDArray[np.floating[Any]],
        block_size: int,
    ) -> NDArray[np.float64]:
        """
        Fetch elevations for a given block of coordinates.
        This method must be implemented by subclasses.
        """
        pass


class BaseCalculator(ABC):
    """
    Abstract base class for performing calculations.
    """

    @abstractmethod
    def calculate(self, *args, **kwargs) -> tuple[float, ...]:
        """
        Perform calculations based on provided parameters.

        :return: Computed results.
        """
        pass


class BaseAnalyzer(ABC):
    """
    Abstract base class for performing analysis.
    """

    def __init__(self, coord_a, coord_b, Lk, path_filename, ha1, ha2):
        self.coord_a = coord_a
        self.coord_b = coord_b
        self.Lk = Lk
        self.path_filename = path_filename
        self.ha1 = ha1
        self.ha2 = ha2

    @abstractmethod
    async def analyze(self) -> dict:
        """
        Asynchronously perform analysis and return the results as a dictionary.

        :return: Dictionary of analysis results.
        """
        pass

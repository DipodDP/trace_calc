from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trace_calc.domain.units import Angle, Meters
from trace_calc.models.input_data import Coordinates, InputData
from trace_calc.models.path import HCAData, PathData


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


class BaseApiClient(ABC):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key


class BaseElevationsApiClient(BaseApiClient):
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


class BaseDeclinationsApiClient(BaseApiClient):
    @abstractmethod
    async def fetch_declinations(
        self, coordinates: Iterable[Coordinates]
    ) -> list[Angle]:
        """
        Fetch magnet declinations for the given coordinates.
        This method must be implemented by subclasses.
        """
        pass


class BaseHCACalculator(ABC):
    def __init__(self, profile: PathData, input_data: InputData):
        self.antenna_a_height: Meters = input_data.antenna_a_height
        self.antenna_b_height: Meters = input_data.antenna_b_height
        self.elevations = profile.elevations
        self.distances = profile.distances
        self.input_data = input_data

    @abstractmethod
    def calculate_hca(self) -> HCAData:
        pass


class BaseSpeedCalculator(ABC):
    """
    Abstract base class for performing calculations.
    """

    @abstractmethod
    def calculate_speed(self, *args: Any, **kwargs: Any) -> tuple[float, ...]:
        """
        Perform calculations based on provided parameters.

        :return: Computed results.
        """
        pass


class BaseAnalyzer(BaseSpeedCalculator, ABC):
    """
    Abstract base class for performing analysis.
    """

    def __init__(self, profile: PathData, input_data: InputData):
        self.profile = profile
        self.input_data = input_data

    @abstractmethod
    def analyze(self, /, **kwargs: Any) -> dict[str, Any]:
        """
        Perform analysis using speed and HCA calculators,
        and return the results as a dictionary.

        :return: Dictionary of analysis results.
        """
        pass

import numpy as np
from numpy.typing import NDArray
from typing import Any

from trace_calc.domain.models.units import Angle, Degrees # Corrected import
from trace_calc.domain.models.coordinates import Coordinates # Corrected import
from trace_calc.application.services.base import BaseDeclinationsApiClient, BaseElevationsApiClient # Corrected import


class MockMagDeclinationApiClient(BaseDeclinationsApiClient):
    async def fetch_declinations(
        self, coordinates: tuple[Coordinates, Coordinates]
    ) -> list[Angle]:
        return [Angle(Degrees(0.0)), Angle(Degrees(0.0))]


class MockElevationsApiClient(BaseElevationsApiClient):
    async def fetch_elevations(
        self,
        coord_vect: NDArray[np.floating[Any]],
        block_size: int,
    ) -> NDArray[np.float64]:
        return np.linspace(100, 200, coord_vect.shape[0])

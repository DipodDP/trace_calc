import aiofiles
import numpy as np
from numpy.typing import NDArray

from trace_calc.models.path import PathData
from trace_calc.services.base import BasePathStorage


class FilePathStorage(BasePathStorage):
    async def load(self, filename: str) -> PathData:
        async with aiofiles.open(filename + ".path", "rb") as f:
            content = await f.read()

        tmp = np.frombuffer(content).reshape((-1, 4))
        coordinates = tmp[:, :2]
        distances: NDArray[np.float64] = tmp[:, 2]
        elevations: NDArray[np.float64] = tmp[:, 3]
        return PathData(coordinates, distances, elevations)

    async def store(self, filename: str, path_data: PathData) -> None:
        tmp = np.hstack((
            path_data.coordinates,
            np.column_stack((path_data.distances, path_data.elevations)),
        ))

        async with aiofiles.open(filename + ".path", "wb") as f:
            await f.write(tmp.tobytes())

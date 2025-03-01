import aiofiles
import numpy as np

from trace_calc.models.path import PathData
from trace_calc.service.base import BasePathStorage


class FilePathStorage(BasePathStorage):
    async def load(self, filename: str) -> PathData:
        async with aiofiles.open(filename + ".path", "rb") as f:
            content = await f.read()

        tmp = np.frombuffer(content).reshape((-1, 4))
        coordinates = tmp[:, :2]
        distances = tmp[:, 2]
        elevations = tmp[:, 3]
        return PathData(coordinates, distances, elevations)

    async def store(self, filename: str, path_data: PathData) -> None:
        tmp = np.hstack((
            path_data.coordinates,
            np.column_stack((path_data.distances, path_data.elevations)),
        ))

        async with aiofiles.open(filename + ".path", "wb") as f:
            await f.write(tmp.tobytes())

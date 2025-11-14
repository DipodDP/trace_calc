import aiofiles
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from trace_calc.domain.models.path import PathData
from trace_calc.application.services.base import BasePathStorage


class FilePathStorage(BasePathStorage):
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def load(self, filename: str) -> PathData:
        file_path = self.output_dir / (filename + ".path")
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()

        tmp = np.frombuffer(content).reshape((-1, 4))
        coordinates = tmp[:, :2]
        distances: NDArray[np.float64] = tmp[:, 2]
        elevations: NDArray[np.float64] = tmp[:, 3]
        return PathData(coordinates, distances, elevations)

    async def store(self, filename: str, path_data: PathData) -> None:
        file_path = self.output_dir / (filename + ".path")
        tmp = np.hstack(
            (
                path_data.coordinates,
                np.column_stack((path_data.distances, path_data.elevations)),
            )
        )

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(tmp.tobytes())

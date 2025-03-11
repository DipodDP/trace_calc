import asyncio
import json
from collections.abc import Iterable
from typing import Any

from httpx import AsyncClient
import numpy as np
import requests
from numpy.typing import NDArray
from progressbar import ProgressBar

from trace_calc.service.base import BaseElevationsApiClient


class APIException(Exception):
    """Custom exception when API data cannot be retrieved."""

    pass


class SyncElevationsApiClient(BaseElevationsApiClient):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def elevations_api_request(self, coord_vect_block: Iterable):
        headers = {
            "X-RapidAPI-Host": self.api_url,
            "X-RapidAPI-Key": self.api_key,
        }
        querystring = {"points": "["}

        for coord in coord_vect_block:
            # getting back W and S coords from coord vector
            if coord[0] > 180:
                coord[0] = coord[0] - 360
            if coord[1] > 180:
                coord[1] = coord[1] - 360

            querystring["points"] += f"[{coord[0]:.6f},{coord[1]:.6f}],"

        querystring["points"] = querystring["points"][:-1] + "]"
        response = requests.request(
            "GET", self.api_url, headers=headers, params=querystring
        )
        resp = json.loads(response.text)

        print(f'\nQuery string: {querystring["points"][:80]}...')

        if response.status_code in [200, 301, 302]:
            resp_data = resp
            return resp_data

        else:
            raise APIException(
                f'{response.status_code} - {": ".join(list(resp.values()))}'
            )

    async def fetch_elevations(
        self, coord_vect: NDArray[np.floating[Any]], block_size: int
    ) -> NDArray[np.float64]:
        """
        Retrieves elevation data in blocks for the given coordinate vector.
        """
        assert (
            coord_vect.shape[0] % block_size == 0
        ), f"Supports only {block_size} wide requests"
        blocks_num = coord_vect.shape[0] // block_size
        print("Retrieving data...")
        bar = ProgressBar(max_value=blocks_num).start()

        # Initialize an empty NumPy array with dtype float64
        elevations: NDArray[np.float64] = np.empty(0, dtype=np.float64)

        for n in range(blocks_num):
            coord_vect_block = coord_vect[n * block_size : (n + 1) * block_size]
            elevations_block = self.elevations_api_request(coord_vect_block)
            elevations = np.append(elevations, elevations_block)
            bar.update(n + 1)
            await asyncio.sleep(1)  # Non-blocking sleep
        bar.finish()
        return elevations


class AsyncElevationsApiClient(BaseElevationsApiClient):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    async def elevations_api_request(self, coord_vect_block: Iterable) -> list[float]:
        """Asynchronous API request with httpx"""
        async with AsyncClient() as client:
            headers = {
                "X-RapidAPI-Host": self.api_url,
                "X-RapidAPI-Key": self.api_key,
            }

            # Build points list with coordinate normalization
            # coords = [
            #     [coord[0] % 360 - 180, coord[1] % 360 - 180]
            #     for coord in coord_vect_block
            # ]

            # params = {"points": json.dumps(points)}
            querystring = {"points": "["}

            # for coord in coords:
            for coord in coord_vect_block:
                # getting back W and S coords from coord vector
                if coord[0] > 180:
                    coord[0] = coord[0] - 360
                if coord[1] > 180:
                    coord[1] = coord[1] - 360

                querystring["points"] += f"[{coord[0]:.6f},{coord[1]:.6f}],"

            querystring["points"] = querystring["points"][:-1] + "]"

            print("------- Start fetching block of elevations -------")
            print(f"Query string {querystring}")
            response = await client.get(
                self.api_url, headers=headers, params=querystring, timeout=30.0
            )
            print("------- Got response -------")

            if not response.is_success:
                try:
                    error_data = response.json()
                    raise APIException(
                        f"{response.status_code} - {': '.join(error_data.values())}"
                    )
                except json.JSONDecodeError:
                    raise APIException(f"{response.status_code} - Unknown error")

            return response.json()

    async def fetch_elevations(
        self, coord_vect: NDArray[np.floating[Any]], block_size: int
    ) -> NDArray[np.float64]:
        """Asynchronous elevation data retrieval with progress tracking"""
        if coord_vect.shape[0] % block_size != 0:
            raise ValueError(
                f"Coordinate vector length must be divisible by {block_size}"
            )

        blocks_num = coord_vect.shape[0] // block_size

        print("Retrieving data...")
        coord_vect_blocks = [
            coord_vect[n * block_size : (n + 1) * block_size] for n in range(blocks_num)
        ]

        tasks = [self.elevations_api_request(block) for block in coord_vect_blocks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = []
        errors = []

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append((idx, result))
                # For API errors, consider adding retry logic here
                # if isinstance(result, APIException):
                # logger.error(f"API failed for task {idx}: {str(result)}")
            else:
                successes.append(result)

        print(f"---- Got {len(results)} blocks -----")
        print(results)

        elevations = np.append(*successes)

        return elevations

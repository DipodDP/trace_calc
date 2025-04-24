import asyncio
import json
import re
import xml.dom.minidom
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
import requests
from httpx import AsyncClient, QueryParams, Response
from numpy.typing import NDArray
from progressbar import ProgressBar

from trace_calc.models.input_data import Coordinates
from trace_calc.service.base import BaseDeclinationsApiClient, BaseElevationsApiClient


class APIException(Exception):
    """Custom exception when API data cannot be retrieved."""

    pass


class SyncElevationsApiClient(BaseElevationsApiClient):
    def elevations_api_request(self, coord_vect_block: Iterable):
        headers = {
            "X-RapidAPI-Host": self.api_url.split("/")[2],  # Getting host from API URL
            "X-RapidAPI-Key": self.api_key,
        }
        querystring = {"points": "["}

        for coord in coord_vect_block:
            querystring["points"] += f"[{coord[0]:.6f},{coord[1]:.6f}],"
        querystring["points"] = querystring["points"][:-1] + "]"

        print("------- Start fetching block of elevations -------")
        print(f"Query string: {querystring['points'][:80]}...")
        response = requests.request(
            "GET", self.api_url, headers=headers, params=querystring
        )
        print("------- Got response -------", response.status_code)

        try:
            resp = json.loads(response.text)
        except json.JSONDecodeError:
            resp = {"message": response.text}

        if response.status_code not in [200, 301, 302] or not isinstance(resp, list):
            raise APIException(
                f"{response.status_code} - {': '.join(list(resp.values()))}"
            )

        return resp

    async def fetch_elevations(
        self, coord_vect: NDArray[np.floating[Any]], block_size: int
    ) -> NDArray[np.float64]:
        """
        Retrieves elevation data in blocks for the given coordinate vector.
        """

        assert coord_vect.shape[0] % block_size == 0, (
            f"Supports only {block_size} wide requests"
        )
        blocks_num = coord_vect.shape[0] // block_size
        print("Retrieving elevation data...")
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
    async def elevations_api_request(self, coord_vect_block: Iterable) -> list[float]:
        """Asynchronous API request with httpx"""
        headers = {
            "X-RapidAPI-Host": self.api_url.split("/")[2],  # Getting host from API URL
            "X-RapidAPI-Key": self.api_key,
        }

        querystring = {"points": "["}
        for coord in coord_vect_block:
            querystring["points"] += f"[{coord[0]:.6f},{coord[1]:.6f}],"
        querystring["points"] = querystring["points"][:-1] + "]"

        print("------- Start fetching block of elevations -------")
        print(f"Query string: {querystring['points'][:80]}...")

        async with AsyncClient() as client:
            response = await client.get(
                self.api_url, headers=headers, params=querystring, timeout=10.0
            )
            print("------- Got response -------", response.status_code)

            if not response.is_success:
                try:
                    error_data = response.json()
                except json.JSONDecodeError:
                    error_data = {"message": response.text}

                raise APIException(
                    f"{response.status_code} - {': '.join(error_data.values())}"
                )

            return response.json()

    async def fetch_elevations(
        self, coord_vect: NDArray[np.floating[Any]], block_size: int
    ) -> NDArray[np.float64]:
        """
        Asynchronously retrieves elevation data in blocks for the given coordinate vector.
        """

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
            if isinstance(result, BaseException):
                errors.append((idx, result))
                # TODO: For API errors, consider adding retry logic here
                # if isinstance(result, APIException):
                # logger.error(f"API failed for task {idx}: {str(result)}")
            else:
                successes.append(result)

        print(f"---- Got {len(results)} blocks -----")

        if errors:
            raise APIException(errors)

        elevations = np.append(*successes)
        return elevations


class AsyncMagDeclinationApiClient(BaseDeclinationsApiClient):
    @staticmethod
    async def _parse_xml(response: Response) -> float:
        """
        Process XML file to get only declination info
        """

        def get_text(nodelist):
            rc = []
            for node in nodelist:
                if node.nodeType == node.TEXT_NODE:
                    rc.append(node.data)
            return "".join(rc)

        xml_response = await response.aread()
        # Process XML file into object tree and get only declination info
        dom = xml.dom.minidom.parseString(xml_response)
        my_string = get_text(dom.getElementsByTagName("declination")[0].childNodes)
        # At this point the string still contains some formatting, this removes it
        declination = float(re.findall(r"[-+]?\d*\.\d+|\d+", my_string)[0])
        # Output formatting and append line to declination file
        await response.aclose()
        return declination

    async def declination_api_request(self, coordinate: Coordinates) -> float:
        """Magnet Declination API request with httpx"""

        month = datetime.now().month
        latitude = coordinate[0]
        longitude = coordinate[1]

        params = QueryParams({
            "lat1": latitude,
            "lon1": longitude,
            "key": self.api_key,
            "resultFormat": "xml",
            "startMonth": month,
        })
        print("------- Start fetching magnet declination -------")
        async with AsyncClient() as client:
            response = await client.get(self.api_url, params=params, timeout=10.0)

            print(
                f"------- Got response for {response.request.url} -------",
                response.status_code,
            )

            if not response.is_success:
                try:
                    error_data = response.json()
                except json.JSONDecodeError:
                    error_data = {"message": response.text}

                raise APIException(
                    f"{response.status_code} - {': '.join(error_data.values())}"
                )

            return await self._parse_xml(response)

    async def fetch_declinations(
        self, coordinates: Iterable[Coordinates]
    ) -> list[float]:
        """
        Asynchronously retrieves magnet declinations data for the given coordinates.
        """

        print("Retrieving magnet declination data...")

        tasks = [self.declination_api_request(coordinate) for coordinate in coordinates]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        declinations = []
        errors = []

        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append((idx, result))
                # TODO: For API errors, consider adding retry logic here
                # if isinstance(result, APIException):
                # logger.error(f"API failed for task {idx}: {str(result)}")
            else:
                declinations.append(result)

        print(f"---- Got {len(results)} blocks -----")

        if errors:
            raise APIException(errors)

        return declinations

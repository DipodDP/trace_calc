import asyncio
import json
import re
import xml.dom.minidom
from collections.abc import Iterable
from datetime import datetime
from typing import Any, cast

import numpy as np
import requests
from httpx import AsyncClient, QueryParams, Response, Timeout
from numpy.typing import NDArray
from progressbar import ProgressBar

from trace_calc.domain.models.units import Angle, Degrees, Elevation
from trace_calc.domain.models.coordinates import Coordinates
from trace_calc.domain.interfaces import (
    BaseDeclinationsApiClient,
    BaseElevationsApiClient,
)
from trace_calc.domain.exceptions import APIException

from .decorators import async_retry
from trace_calc.logging_config import get_logger

logger = get_logger(__name__)


class SyncElevationsApiClient(BaseElevationsApiClient):
    def elevations_api_request(
        self, coord_vect_block: NDArray[np.floating[Any]]
    ) -> list[Elevation]:
        headers = {
            "X-RapidAPI-Host": self.api_url.split("/")[2],  # Getting host from API URL
            "X-RapidAPI-Key": self.api_key,
        }
        querystring = {"points": "["}

        for coord in coord_vect_block:
            querystring["points"] += f"[{coord[0]:.6f},{coord[1]:.6f}],"
        querystring["points"] = querystring["points"][:-1] + "]"

        logger.info("------- Start fetching block of elevations -------")
        logger.debug(f"Query string: {querystring['points'][:80]}...")
        response = requests.request(
            "GET", self.api_url, headers=headers, params=querystring
        )
        logger.info(f"------- Got response ------- {response.status_code}")

        try:
            resp = json.loads(response.text)
        except json.JSONDecodeError:
            resp = {"message": response.text}

        if response.status_code not in [200, 301, 302] or not isinstance(resp, list):
            raise APIException(
                f"{response.status_code} - {': '.join(list(resp.values()))}"
            )

        return [Elevation(e) for e in resp]

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
        logger.info("Retrieving elevation data...")
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
    @async_retry()
    async def elevations_api_request(
        self, coord_vect_block: NDArray[np.floating[Any]], **kwargs
    ) -> list[Elevation]:
        """Asynchronous API request with httpx"""
        headers = {
            "X-RapidAPI-Host": self.api_url.split("/")[2],  # Getting host from API URL
            "X-RapidAPI-Key": self.api_key,
        }

        querystring = {"points": "["}
        for coord in coord_vect_block:
            querystring["points"] += f"[{coord[0]:.6f},{coord[1]:.6f}],"
        querystring["points"] = querystring["points"][:-1] + "]"

        timeout = kwargs.get("timeout", 10.0)
        # Configure timeout with connect and read timeouts
        timeout_config = Timeout(timeout, connect=5.0)

        async with AsyncClient(timeout=timeout_config, follow_redirects=True) as client:
            request = client.build_request("GET", self.api_url, params=querystring, headers=headers)
            logger.info(f"HTTP Request: {request.method} {request.url}")
            response = await client.send(request)
            logger.info(f"HTTP Response: {response.status_code}")

            if not response.is_success:
                try:
                    error_data = response.json()
                except json.JSONDecodeError:
                    error_data = {"message": response.text}

                raise APIException(
                    f"{response.status_code} - {': '.join(error_data.values())}"
                )

            return [Elevation(e) for e in response.json()]

    async def fetch_elevations(
        self, coord_vect: NDArray[np.floating[Any]], block_size: int
    ) -> NDArray[np.float64]:
        """
        Asynchronously retrieves elevation data in blocks for the given coordinate vector.
        """
        logger.debug(
            f"fetch_elevations called with coord_vect.shape={coord_vect.shape}, block_size={block_size}"
        )

        if coord_vect.shape[0] == 0:
            return np.array([], dtype=np.float64)

        blocks_num = (coord_vect.shape[0] + block_size - 1) // block_size

        logger.info(
            f"Retrieving elevation data for {coord_vect.shape[0]} coordinates in {blocks_num} blocks..."
        )
        coord_vect_blocks = [
            coord_vect[n * block_size : (n + 1) * block_size] for n in range(blocks_num)
        ]
        logger.debug(
            f"Created {len(coord_vect_blocks)} block(s) with shapes: {[block.shape for block in coord_vect_blocks]}"
        )

        tasks = [self.elevations_api_request(block) for block in coord_vect_blocks]
        logger.debug(f"Created {len(tasks)} task(s) for async execution")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successes: list[list[Elevation] | None] = [None] * blocks_num
        errors = []

        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append((idx, result))
            else:
                successes[idx] = result

        logger.info(f"---- Got {len(results)} blocks -----")

        if errors:
            raise APIException(errors)

        if not all(s is not None for s in successes):
            # This case should ideally not be reached if no errors were raised
            raise Exception("Failed to fetch some elevation blocks without raising an exception.")

        return np.concatenate([np.array(s, dtype=np.float64) for s in successes])


class AsyncMagDeclinationApiClient(BaseDeclinationsApiClient):
    @staticmethod
    async def _parse_xml(response: Response) -> Angle:
        """
        Process XML file to get only declination info
        """

        def get_text(nodelist: list[xml.dom.minidom.Node]) -> str:
            rc = []
            for node in nodelist:
                if node.nodeType == node.TEXT_NODE:
                    if node.nodeValue is not None:
                        rc.append(node.nodeValue)
            return "".join(rc)

        xml_response = await response.aread()
        # Process XML file into object tree and get only declination info
        dom = xml.dom.minidom.parseString(xml_response)
        my_string = get_text(
            cast(
                list[xml.dom.minidom.Node],
                dom.getElementsByTagName("declination")[0].childNodes,
            )
        )
        # At this point the string still contains some formatting, this removes it
        declination = float(re.findall(r"[-+]?\d*\.\d+|\d+", my_string)[0])
        # Output formatting and append line to declination file
        await response.aclose()
        return Angle(Degrees(declination))

    @async_retry()
    async def declination_api_request(self, coordinate: Coordinates, **kwargs) -> Angle:
        """Magnet Declination API request with httpx"""

        month = datetime.now().month
        latitude = coordinate[0]
        longitude = coordinate[1]

        params = QueryParams(
            {
                "lat1": latitude,
                "lon1": longitude,
                "key": self.api_key,
                "resultFormat": "xml",
                "startMonth": month,
            }
        )
        timeout = kwargs.get("timeout", 10.0)
        # Configure timeout with connect and read timeouts
        timeout_config = Timeout(timeout, connect=5.0)

        async with AsyncClient(timeout=timeout_config, follow_redirects=True) as client:
            request = client.build_request("GET", self.api_url, params=params)
            logger.info(f"HTTP Request: {request.method} {request.url}")
            response = await client.send(request)
            logger.info(f"HTTP Response: {response.status_code}")

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
    ) -> list[Angle]:
        """
        Asynchronously retrieves magnet declinations data for the given coordinates.
        """

        logger.info("Retrieving magnet declination data...")

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

        logger.debug(f"---- Got {len(results)} blocks -----")

        if errors:
            raise APIException(errors)

        return declinations

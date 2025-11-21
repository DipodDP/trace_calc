import pytest
from trace_calc.application.orchestration import GeoDataService
from tests.mocks import MockMagDeclinationApiClient
from trace_calc.domain.models.coordinates import Coordinates


@pytest.mark.asyncio
async def test_geodataservice_process_runs_successfully():
    declinations_api_client = MockMagDeclinationApiClient(api_url="", api_key="")
    geo_data_service = GeoDataService(declinations_api_client)

    coord_a = Coordinates(lat=59.585, lon=154.141139)
    coord_b = Coordinates(lat=59.569594, lon=151.256373)

    await geo_data_service.process(coord_a, coord_b)

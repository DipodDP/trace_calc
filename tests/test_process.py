import pytest
from trace_calc.services.process import GeoDataService
from tests.mocks import MockMagDeclinationApiClient
from trace_calc.models.input_data import Coordinates


@pytest.mark.asyncio
async def test_geodataservice_process_does_not_raise_name_error():
    declinations_api_client = MockMagDeclinationApiClient(api_url="", api_key="")
    geo_data_service = GeoDataService(declinations_api_client)

    coord_a = Coordinates(lat=59.585, lon=154.141139)
    coord_b = Coordinates(lat=59.569594, lon=151.256373)

    try:
        await geo_data_service.process(coord_a, coord_b)
    except NameError:
        pytest.fail("NameError was raised unexpectedly")
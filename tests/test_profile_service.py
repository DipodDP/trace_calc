import numpy as np
import pytest

from trace_calc.models.input_data import Coordinates, InputData
from trace_calc.service.base import BaseElevationsApiClient
from trace_calc.service.exceptions import CoordinatesRequiredException
from trace_calc.service.profile_service import PathProfileService


# Dummy elevations API client that simulates an async API call
class DummyElevationsApiClient(BaseElevationsApiClient):
    async def fetch_elevations(self, coord_vect, block_size):
        # Return dummy elevation data based on the number of coordinates
        return np.linspace(100, 200, coord_vect.shape[0])


# Parametrized test cases for linspace_coord edge cases.
@pytest.mark.parametrize(
    "site_a, site_b, expected_behavior",
    [
        # Identical coordinates: expect all points to be identical.
        (Coordinates(10, 20), Coordinates(10, 20), "identical"),
        # Normal coordinates: expect a monotonic coordinate array.
        (Coordinates(10, 20), Coordinates(20, 30), "normal"),
        (Coordinates(5, 60), Coordinates(-5, 65), "normal"),
        (Coordinates(-5, 60), Coordinates(5, 65), "normal"),
        (Coordinates(-10, 20), Coordinates(5, 30), "normal"),
        (Coordinates(-10, -5), Coordinates(5, 30), "normal"),
        (Coordinates(83.2, -35.5), Coordinates(82.2, -17.5), "normal"),
        # Negative coordinates with triggering adjustment.
        (Coordinates(60, -179.5), Coordinates(61, 179.0), "transition_180"),
        (Coordinates(60, 179.5), Coordinates(61, -179.0), "transition_180"),
    ],
)
def test_linspace_coord_edge_cases(site_a, site_b, expected_behavior):
    input_data = InputData(
        "test",
        site_a_coordinates=site_a,
        site_b_coordinates=site_b,
    )
    service = PathProfileService(
        input_data,
        DummyElevationsApiClient("test_url", "test_key"),
        block_size=10,
        resolution=1,
    )
    coords = service.linspace_coord()
    print('\n', input_data)
    print('\nCoordinates: ', service.coord_a, service.coord_b)
    print('Coordinate vector:\n', coords)
    assert isinstance(coords, np.ndarray)
    assert coords.ndim == 2 and coords.shape[1] == 2
    # Expect the shortest path is choosen for latitude.
    if np.any(coords):
        if coords[0][0] > coords[-1][0]:
            assert coords[0][0] > coords[1][0]
            assert coords[-1][0] < coords[-2][0]
        else:
            assert coords[0][0] <= coords[1][0]
            assert coords[-1][0] >= coords[-2][0]

    if expected_behavior == "identical":
        # For identical coordinates, all rows should be the same.
        expected_point = np.array([site_a[0], site_a[1]])
        # Ensure every coordinate equals the expected point.
        assert np.all(np.isclose(coords, expected_point)), "Not all points are identical for identical inputs."

    elif expected_behavior == "normal":
        # Expect the first coordinate to match site_a and the last to match site_b.
        assert np.allclose(coords[0], np.array([site_a[0], site_a[1]]))
        assert np.allclose(coords[-1], np.array([site_b[0], site_b[1]]))
        # Expect the shortest path is choosen for longitude.
        if coords[0][1] > coords[-1][1]:
            assert coords[0][1] > coords[1][1]
            assert coords[-1][1] < coords[-2][1]
        else:
            assert coords[0][1] < coords[1][1]
            assert coords[-1][1] > coords[-2][1]

    elif expected_behavior == "transition_180":
        assert np.allclose(coords[0], np.array([site_a[0], site_a[1]]))
        assert np.allclose(coords[-1], np.array([site_b[0], site_b[1]]))
        # When transit 180 longitude, check that the shortest path for longitude is choosen.
        if coords[0][1] > coords[-1][1]:
            assert coords[0][1] < coords[1][1]
            assert coords[-1][1] > coords[-2][1]
        else:
            assert coords[0][1] > coords[1][1]
            assert coords[-1][1] < coords[-2][1]


def test_initialization_error():
    # Test that a ValueError is raised when either coordinate is missing.
    with pytest.raises(CoordinatesRequiredException):
        input_data = InputData(
            "test",
            site_a_coordinates=None,
            site_b_coordinates=None,
        )
        PathProfileService(input_data, DummyElevationsApiClient("test_api", "test_key"))


@pytest.mark.asyncio
async def test_get_profile():
    # Test the asynchronous get_profile method.
    input_data = InputData(
        "test",
        site_a_coordinates=Coordinates(10, 20),
        site_b_coordinates=Coordinates(20, 30),
    )
    service = PathProfileService(
        input_data,
        DummyElevationsApiClient("test_url", "test_key"),
        block_size=50,
        resolution=0.5,
    )
    profile = await service.get_profile()
    # Check that coordinates, distances, and elevations arrays have consistent lengths.
    assert (
        profile.coordinates.shape[0]
        == profile.distances.shape[0]
        == profile.elevations.shape[0]
    )
    # The first distance should be zero.
    assert profile.distances[0] == 0.0
    # Expect a monotonically increasing distances array.
    assert profile.distances[0] < profile.distances[1]
    assert profile.distances[-1] > profile.distances[-2]

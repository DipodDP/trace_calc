"""Tests for AsyncElevationsApiClient API request behavior"""

import pytest
import numpy as np

from trace_calc.domain.models.coordinates import Coordinates, InputData
from trace_calc.infrastructure.api.clients import AsyncElevationsApiClient
from trace_calc.application.services.profile import PathProfileService


@pytest.fixture
def sample_input_data():
    """Create sample input data with test coordinates"""
    return InputData(
        path_name="test",
        site_a_coordinates=Coordinates(59.585000, 154.141139),
        site_b_coordinates=Coordinates(59.569594, 151.256373),
    )


@pytest.fixture
def api_client():
    """Create API client for testing"""
    return AsyncElevationsApiClient(
        api_url="https://api.example.com/elevations", api_key="test_key"
    )


class ApiRequestTracker:
    """Helper class to track API requests during testing"""

    def __init__(self):
        self.call_count = 0
        self.call_sizes = []
        self.call_coordinates = []

    def create_tracked_method(self, original_method=None):
        """Create a tracked version of the API request method"""

        async def tracked_method(coord_vect_block):
            self.call_count += 1
            block_size = len(coord_vect_block)
            self.call_sizes.append(block_size)
            self.call_coordinates.append(
                {"first": coord_vect_block[0], "last": coord_vect_block[-1]}
            )
            # Return mock elevation data
            return [100.0] * block_size

        return tracked_method


@pytest.mark.asyncio
async def test_api_request_count_with_256_block_size(sample_input_data, api_client):
    """Test that correct number of API requests are made with block_size=256"""
    # Create tracker
    tracker = ApiRequestTracker()

    # Patch the API request method
    api_client.elevations_api_request = tracker.create_tracked_method()

    # Create profile service with block_size=256
    profile_service = PathProfileService(
        input_data=sample_input_data,
        elevations_api_client=api_client,
        block_size=256,
        resolution=0.5,
    )

    # Get coordinate vector
    coord_vect = profile_service.linspace_coord()
    total_coords = coord_vect.shape[0]

    # Fetch elevations
    elevations = await api_client.fetch_elevations(coord_vect, block_size=256)

    # Calculate expected number of blocks
    expected_blocks = total_coords // 256
    if total_coords % 256 != 0:
        expected_blocks += 1

    # Verify results
    assert tracker.call_count == expected_blocks, (
        f"Expected {expected_blocks} API requests, but made {tracker.call_count}"
    )
    assert len(elevations) == total_coords, (
        f"Expected {total_coords} elevations, but got {len(elevations)}"
    )
    assert sum(tracker.call_sizes) == total_coords, (
        "Total coordinates across all blocks doesn't match total"
    )


@pytest.mark.asyncio
async def test_api_request_block_sizes(sample_input_data, api_client):
    """Test that blocks are correctly sized (full blocks + potential partial block)"""
    tracker = ApiRequestTracker()
    api_client.elevations_api_request = tracker.create_tracked_method()

    block_size = 256
    profile_service = PathProfileService(
        input_data=sample_input_data,
        elevations_api_client=api_client,
        block_size=block_size,
        resolution=0.5,
    )

    coord_vect = profile_service.linspace_coord()
    total_coords = coord_vect.shape[0]

    await api_client.fetch_elevations(coord_vect, block_size=block_size)

    # Check block sizes
    for i, size in enumerate(tracker.call_sizes[:-1]):
        # All blocks except possibly the last should be full size
        assert size == block_size, f"Block {i} has size {size}, expected {block_size}"

    # Last block can be partial
    last_block_size = tracker.call_sizes[-1]
    expected_last_size = total_coords % block_size or block_size
    assert last_block_size == expected_last_size, (
        f"Last block has size {last_block_size}, expected {expected_last_size}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block_size,resolution",
    [
        (128, 0.5),
        (256, 0.5),
        (512, 0.5),
        (256, 1.0),
        (256, 0.25),
    ],
)
async def test_api_requests_with_different_parameters(
    sample_input_data, api_client, block_size, resolution
):
    """Test API requests with various block sizes and resolutions"""
    tracker = ApiRequestTracker()
    api_client.elevations_api_request = tracker.create_tracked_method()

    profile_service = PathProfileService(
        input_data=sample_input_data,
        elevations_api_client=api_client,
        block_size=block_size,
        resolution=resolution,
    )

    coord_vect = profile_service.linspace_coord()
    total_coords = coord_vect.shape[0]

    elevations = await api_client.fetch_elevations(coord_vect, block_size=block_size)

    # Verify all coordinates got elevations
    assert len(elevations) == total_coords
    assert sum(tracker.call_sizes) == total_coords

    # Verify no block exceeds the block size
    for size in tracker.call_sizes:
        assert size <= block_size, f"Block size {size} exceeds limit {block_size}"


@pytest.mark.asyncio
async def test_api_request_coordinate_continuity(sample_input_data, api_client):
    """Test that coordinates are passed to API in correct order"""
    tracker = ApiRequestTracker()
    api_client.elevations_api_request = tracker.create_tracked_method()

    profile_service = PathProfileService(
        input_data=sample_input_data,
        elevations_api_client=api_client,
        block_size=256,
        resolution=0.5,
    )

    coord_vect = profile_service.linspace_coord()
    await api_client.fetch_elevations(coord_vect, block_size=256)

    # Verify first block starts with first coordinate
    assert np.allclose(tracker.call_coordinates[0]["first"], coord_vect[0]), (
        "First block doesn't start with first coordinate"
    )

    # Verify last block ends with last coordinate
    assert np.allclose(tracker.call_coordinates[-1]["last"], coord_vect[-1]), (
        "Last block doesn't end with last coordinate"
    )

    # Verify blocks are continuous (last coord of block N == first coord of block N+1 - 1)
    if len(tracker.call_coordinates) > 1:
        cumulative_size = 0
        for i, size in enumerate(tracker.call_sizes[:-1]):
            expected_last = coord_vect[cumulative_size + size - 1]
            actual_last = tracker.call_coordinates[i]["last"]
            assert np.allclose(actual_last, expected_last), (
                f"Block {i} last coordinate doesn't match expected position"
            )
            cumulative_size += size


@pytest.mark.asyncio
async def test_single_block_request(api_client):
    """Test API request when total coordinates fit in a single block

    Note: PathProfileService generates coordinates in multiples of block_size,
    so we use a large block_size to ensure only one block is needed.
    """
    # Create input data with close coordinates
    input_data = InputData(
        path_name="test",
        site_a_coordinates=Coordinates(59.585000, 154.141139),
        site_b_coordinates=Coordinates(59.585100, 154.141239),
    )

    tracker = ApiRequestTracker()
    api_client.elevations_api_request = tracker.create_tracked_method()

    # Use a larger block size to ensure single block
    large_block_size = 1024
    profile_service = PathProfileService(
        input_data=input_data,
        elevations_api_client=api_client,
        block_size=large_block_size,
        resolution=0.5,
    )

    coord_vect = profile_service.linspace_coord()
    # With large block size and close coordinates, should get exactly one block
    assert coord_vect.shape[0] == large_block_size, (
        f"Expected {large_block_size} coordinates (one block), got {coord_vect.shape[0]}"
    )

    elevations = await api_client.fetch_elevations(
        coord_vect, block_size=large_block_size
    )

    # Should make exactly 1 API request
    assert tracker.call_count == 1, f"Expected 1 request, got {tracker.call_count}"
    assert tracker.call_sizes[0] == coord_vect.shape[0]
    assert len(elevations) == coord_vect.shape[0]

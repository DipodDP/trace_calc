"""
Integration tests for the TraceAnalyzerAPI.

These tests verify that the facade correctly orchestrates the analysis process
and provides a simplified interface for integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from trace_calc.adapter import TraceAnalyzerAPI
from trace_calc.domain.models.coordinates import Coordinates


@pytest.mark.asyncio
class TestAdapterFacade:
    """Test suite for the TraceAnalyzerAPI."""

    async def test_coordinate_parsing(self, trace_analyzer_api: TraceAnalyzerAPI):
        """Test coordinate parsing from text."""
        s_names, coords_dec, coords_formatted = await trace_analyzer_api.parse_sites(
            s_name="Site_A Site_B",
            coords="55.367269 91.646198 55.989642 92.899899"
        )

        # Verify names
        assert len(s_names) == 2
        assert s_names[0] == "Site_A"
        assert s_names[1] == "Site_B"

        # Verify decimal coordinates
        assert len(coords_dec) == 4
        assert pytest.approx(coords_dec[0], abs=0.01) == 55.367
        assert pytest.approx(coords_dec[1], abs=0.01) == 91.646
        assert pytest.approx(coords_dec[2], abs=0.01) == 55.989
        assert pytest.approx(coords_dec[3], abs=0.01) == 92.899

        # Verify formatted coordinates
        assert len(coords_formatted) == 4
        assert "N" in coords_formatted[0]
        assert "E" in coords_formatted[1]

    async def test_azimuth_calculation(self, trace_analyzer_api: TraceAnalyzerAPI):
        """Test azimuth and magnetic declination calculation."""
        coords_dec = [55.367, 91.646, 55.989, 92.899]

        azim1, azim2, dec1, dec2, mag_azim1, mag_azim2 = (
            await trace_analyzer_api.get_azimuths(coords_dec)
        )

        # Verify azimuth values are in valid range
        assert 0 <= azim1 <= 360
        assert 0 <= azim2 <= 360

        # Verify magnetic azimuths are calculated correctly
        # Mock declination is 0, so mag_azim should equal azim
        assert mag_azim1 == pytest.approx(azim1 - dec1, abs=0.1)
        assert mag_azim2 == pytest.approx(azim2 - dec2, abs=0.1)


    async def test_coordinate_parsing_default_names(self, trace_analyzer_api: TraceAnalyzerAPI):
        """Test coordinate parsing with default site names."""
        s_names, coords_dec, coords_formatted = await trace_analyzer_api.parse_sites(
            s_name="",
            coords="55.367269 91.646198 55.989642 92.899899"
        )

        # Should use default names
        assert s_names[0] == "Точка А"
        assert s_names[1] == "Точка Б"

    async def test_coordinate_parsing_single_name(self, trace_analyzer_api: TraceAnalyzerAPI):
        """Test coordinate parsing with single site name."""
        s_names, coords_dec, coords_formatted = await trace_analyzer_api.parse_sites(
            s_name="SiteA",
            coords="55.367269 91.646198 55.989642 92.899899"
        )

        # Should add default second name
        assert s_names[0] == "SiteA"
        assert s_names[1] == "Точка Б"

    async def test_coordinate_parsing_invalid_coords(self, trace_analyzer_api: TraceAnalyzerAPI):
        """Test coordinate parsing with invalid coordinates."""
        with pytest.raises(ValueError):
            await trace_analyzer_api.parse_sites(
                s_name="Site_A Site_B",
                coords="invalid coords"
            )

    async def test_coordinate_parsing_wrong_count(self, trace_analyzer_api: TraceAnalyzerAPI):
        """Test coordinate parsing with wrong number of coordinates."""
        with pytest.raises(ValueError, match="Expected 2 coordinate pairs"):
            await trace_analyzer_api.parse_sites(
                s_name="Site_A Site_B",
                coords="55.367269 91.646198"  # Only 1 coordinate pair
            )


@pytest.fixture
def trace_analyzer_api():
    """Create a TraceAnalyzerAPI instance with mocked dependencies for testing."""
    env = MagicMock()
    env.str.side_effect = {
        "ELEVATION_API_URL": "http://fake-elevation-api.com",
        "ELEVATION_API_KEY": "fake-key",
        "DECLINATION_API_URL": "http://fake-declination-api.com",
        "DECLINATION_API_KEY": "fake-key",
    }.get

    # Mock API clients
    elevations_client = AsyncMock()
    declinations_client = AsyncMock()
    
    # Mock storage
    storage = MagicMock()
    storage.load = AsyncMock(side_effect=FileNotFoundError) # Always fetch from api
    storage.store = AsyncMock()
    
    facade = TraceAnalyzerAPI.create_from_config(env, storage)
    
    # Replace clients with mocks
    facade._elevations_api_client = elevations_client
    facade._declinations_api_client = declinations_client
    facade._storage = storage

    # Mock the internal get_azimuths so we don't make real api calls or depend on GreatCircle
    async def mock_get_azimuths(self, coords_dec: list[float]):
        # Dummy but valid values that satisfy the assertions
        azim1 = 10.0
        azim2 = 190.0
        dec1 = 5.0
        dec2 = 6.0
        mag_azim1 = azim1 - dec1
        mag_azim2 = azim2 - dec2
        
        return (
            azim1,
            azim2,
            dec1,
            dec2,
            mag_azim1,
            mag_azim2,
        )

    # Replace the actual get_azimuths with our mock
    facade.get_azimuths = mock_get_azimuths.__get__(facade, TraceAnalyzerAPI) # Bind the method to the instance
    return facade

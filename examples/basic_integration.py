"""
Basic Integration Example
==========================

This example demonstrates how to use the trace_calc package programmatically
in your own applications (e.g., Telegram bots, web services, automation tools).

It shows:
- Setting up dependencies without .env file (hardcoded configuration)
- Creating and running an analysis service
- Processing results as structured data
- Error handling patterns

Requirements:
- Valid API keys for elevation and declination services
- Python 3.10+
"""

import asyncio
from trace_calc.adapter import TraceAnalyzerAPI
from trace_calc.domain.models.coordinates import Coordinates, InputData
from trace_calc.domain.models.units import Meters
from trace_calc.application.analysis import GrozaAnalysisService
from trace_calc.application.orchestration import OrchestrationService
from trace_calc.application.services.profile import PathProfileService
from trace_calc.infrastructure.api.clients import (
    AsyncElevationsApiClient,
    AsyncMagDeclinationApiClient,
)
from trace_calc.infrastructure.output.formatters import JSONOutputFormatter
from trace_calc.domain.exceptions import APIException


class Config:
    """
    Configuration class - alternative to .env file.
    In production, load these from environment variables or a secure config system.
    """
    ELEVATION_API_URL = "https://your-elevation-api.com/v1"
    ELEVATION_API_KEY = "your_elevation_api_key_here"
    DECLINATION_API_URL = "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination"
    DECLINATION_API_KEY = ""  # Declination API may not require a key


async def run_troposcatter_analysis(
    site_a_lat: float,
    site_a_lon: float,
    site_b_lat: float,
    site_b_lon: float,
    frequency_mhz: float = 5000.0,
    antenna_a_height: float = 30.0,
    antenna_b_height: float = 30.0,
    path_name: str = "example_path",
) -> dict:
    """
    Run a complete troposcatter link analysis programmatically.

    Args:
        site_a_lat: Latitude of site A (degrees)
        site_a_lon: Longitude of site A (degrees)
        site_b_lat: Latitude of site B (degrees)
        site_b_lon: Longitude of site B (degrees)
        frequency_mhz: Operating frequency in MHz (default: 5000)
        antenna_a_height: Antenna A height in meters (default: 30)
        antenna_b_height: Antenna B height in meters (default: 30)
        path_name: Name for this path analysis (default: "example_path")

    Returns:
        dict: Analysis results with link speed, loss components, and geo data

    Raises:
        APIException: If API calls fail
        ValueError: If input parameters are invalid
    """

    # Step 1: Create input data model
    input_data = InputData(
        path_name=path_name,
        site_a_coordinates=Coordinates(site_a_lat, site_a_lon),
        site_b_coordinates=Coordinates(site_b_lat, site_b_lon),
        frequency_mhz=frequency_mhz,
        antenna_a_height=Meters(antenna_a_height),
        antenna_b_height=Meters(antenna_b_height),
    )

    # Step 2: Initialize API clients
    elevations_client = AsyncElevationsApiClient(
        api_url=Config.ELEVATION_API_URL,
        api_key=Config.ELEVATION_API_KEY,
    )

    declinations_client = AsyncMagDeclinationApiClient(
        api_url=Config.DECLINATION_API_URL,
        api_key=Config.DECLINATION_API_KEY,
    )

    # Step 3: Create profile service to fetch elevation data
    profile_service = PathProfileService(
        input_data=input_data,
        elevations_api_client=elevations_client,
        block_size=256,  # Fetch elevations in blocks of 256 points
        resolution=0.05,  # 0.05 km resolution between points
    )

    # Step 4: Fetch elevation profile from API
    print(f"Fetching elevation profile for {path_name}...")
    path_data = await profile_service.get_profile()
    print(f"✓ Fetched {len(path_data.elevations)} elevation points")

    # Step 5: Choose analysis method (Groza or Sosnik)
    analysis_service = GrozaAnalysisService()

    # Step 6: Create orchestration service with dependencies
    orchestrator = OrchestrationService(
        analysis_service=analysis_service,
        profile_service=profile_service,
        declinations_api_client=declinations_client,
        output_formatter=None,  # No console output
        visualizer=None,  # No visualization
    )

    # Step 7: Run analysis
    print("Running propagation analysis...")
    result = await orchestrator.process(
        input_data=input_data,
        antenna_a_height=antenna_a_height,
        antenna_b_height=antenna_b_height,
        display_output=False,
        generate_plot=False,
    )

    # Step 8: Convert result to dictionary for easy consumption
    formatter = JSONOutputFormatter()
    geo_data = result.result.get("geo_data")
    profile_data = result.result.get("profile_data")

    # Extract structured data
    output = {
        "path_name": path_name,
        "link_speed_mbps": result.link_speed,
        "wavelength_m": result.wavelength,
        "distance_km": geo_data.get("distance_km") if geo_data else None,
        "propagation_loss": result.model_propagation_loss_parameters.get("propagation_loss"),
        "total_loss_db": result.model_propagation_loss_parameters.get("total_loss"),
        "geo_data": geo_data,
        "profile_data": profile_data,
    }

    print(f"✓ Analysis complete! Link speed: {result.link_speed:.1f} Mbps")
    return output


async def example_usage():
    """
    Example showing how to call the analysis function.
    """
    try:
        # Example coordinates (replace with real coordinates)
        result = await run_troposcatter_analysis(
            site_a_lat=55.7558,  # Moscow latitude
            site_a_lon=37.6173,  # Moscow longitude
            site_b_lat=59.9343,  # St. Petersburg latitude
            site_b_lon=30.3351,  # St. Petersburg longitude
            frequency_mhz=5000.0,
            antenna_a_height=30.0,
            antenna_b_height=30.0,
            path_name="moscow_spb",
        )

        # Access results
        print("\n=== Analysis Results ===")
        print(f"Path: {result['path_name']}")
        print(f"Link Speed: {result['link_speed_mbps']:.1f} Mbps")
        print(f"Distance: {result['distance_km']:.1f} km")
        print(f"Total Path Loss: {result['total_loss_db']:.2f} dB")

        if result['propagation_loss']:
            loss = result['propagation_loss']
            print("\nLoss Components:")
            print(f"  Free Space Loss: {loss.get('free_space_loss', 'N/A')} dB")
            print(f"  Atmospheric Loss: {loss.get('atmospheric_loss', 'N/A')} dB")
            print(f"  Diffraction Loss: {loss.get('diffraction_loss', 'N/A')} dB")
            print(f"  Refraction Loss: {loss.get('refraction_loss', 'N/A')} dB")

        return result

    except APIException as e:
        print(f"API Error: {e}")
        print("Check your API keys and network connection")
        raise

    except ValueError as e:
        print(f"Invalid input: {e}")
        raise

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


async def example_with_multiple_paths():
    """
    Example showing how to analyze multiple paths in parallel.
    """
    paths = [
        {
            "name": "path_1",
            "a_lat": 55.7558,
            "a_lon": 37.6173,
            "b_lat": 59.9343,
            "b_lon": 30.3351,
        },
        {
            "name": "path_2",
            "a_lat": 51.5074,
            "a_lon": -0.1278,
            "b_lat": 48.8566,
            "b_lon": 2.3522,
        },
    ]

    # Run analyses in parallel
    tasks = [
        run_troposcatter_analysis(
            site_a_lat=path["a_lat"],
            site_a_lon=path["a_lon"],
            site_b_lat=path["b_lat"],
            site_b_lon=path["b_lon"],
            path_name=path["name"],
        )
        for path in paths
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Path {paths[i]['name']} failed: {result}")
        else:
            print(f"Path {paths[i]['name']}: {result['link_speed_mbps']:.1f} Mbps")

    return results


async def example_with_facade():
    """
    Example showing how to use the TraceAnalyzerAPI for simplified integration.
    This is the recommended approach for most external applications.
    """
    print("\n=== Running analysis with TraceAnalyzerAPI ===")
    try:
        # The facade internally handles dependency creation from environment variables.
        # For this example, we mock the environment. In a real app, you would
        # have a .env file or your environment variables set.
        from unittest.mock import MagicMock
        env = MagicMock()
        env.str.side_effect = {
            "ELEVATION_API_URL": Config.ELEVATION_API_URL,
            "ELEVATION_API_KEY": Config.ELEVATION_API_KEY,
            "DECLINATION_API_URL": Config.DECLINATION_API_URL,
            "DECLINATION_API_KEY": Config.DECLINATION_API_KEY,
        }.get

        # 1. Create the facade from the environment
        facade = TraceAnalyzerAPI.create_from_env(env)

        # 2. Run analysis with a single method call
        (
            L0, Lmed, Lr, trace_dist, b1_max, b2_max,
            b_sum, Ltot, dL, speed, sp_pref
        ) = await facade.analyze_groza(
            coord_a=[55.7558, 37.6173],
            coord_b=[59.9343, 30.3351],
            path_filename="moscow_spb_facade",
            antenna_a_height=30.0,
            antenna_b_height=30.0,
        )

        print(f"✓ Facade analysis complete! Link speed: {speed:.1f} {sp_pref}bps")
        print(f"  Total Loss: {Ltot:.2f} dB")
        print(f"  Distance: {trace_dist:.1f} km")

    except Exception as e:
        print(f"An error occurred: {e}")
        # In a real app, you might want more specific error handling
        # for APIException, ValueError, etc.

if __name__ == "__main__":
    # Run single analysis example
    # asyncio.run(example_usage())

    # Run the simplified facade example
    asyncio.run(example_with_facade())

    # Uncomment to run multiple paths example:
    # asyncio.run(example_with_multiple_paths())

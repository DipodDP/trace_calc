"""Orchestration service (coordinates workflow with dependency injection)"""

from typing import Optional

from trace_calc.domain.models.coordinates import InputData, Coordinates
from trace_calc.domain.models.analysis import AnalysisResult
from trace_calc.application.analysis import (
    BaseAnalysisService,
)
from trace_calc.application.services.profile import PathProfileService
from trace_calc.infrastructure.output.formatters import OutputFormatter
from trace_calc.infrastructure.visualization.plotter import ProfileVisualizer
from trace_calc.domain.models.path import GeoData, PathData
from trace_calc.application.services.coordinates import CoordinatesService
from trace_calc.application.services.base import BaseDeclinationsApiClient
from trace_calc.domain.models.units import Angle, Degrees, Kilometers


class OrchestrationService:
    """
    Coordinates the complete analysis workflow.

    Uses dependency injection to decouple components and enable testing.
    All I/O dependencies (formatters, visualizers) are injected, not inherited.
    """

    def __init__(
        self,
        analysis_service: BaseAnalysisService,
        profile_service: PathProfileService,
        declinations_api_client: BaseDeclinationsApiClient,
        output_formatter: Optional[OutputFormatter] = None,
        visualizer: Optional[ProfileVisualizer] = None,
    ):
        """
        Initialize orchestration service with injected dependencies.

        Args:
            analysis_service: Strategy for propagation analysis (Groza, Sosnik, etc.)
            profile_service: Service for fetching/calculating profiles
            output_formatter: Optional formatter for console output
            visualizer: Optional visualizer for plots
        """
        self.analysis_service = analysis_service
        self.profile_service = profile_service
        self.declinations_api_client = declinations_api_client
        self.output_formatter = output_formatter
        self.visualizer = visualizer

    async def process(
        self,
        input_data: InputData,
        antenna_a_height: float,
        antenna_b_height: float,
        display_output: bool = True,
        generate_plot: bool = True,
        path: Optional[PathData] = None,
        save_plot_path: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Execute complete analysis workflow.

        Steps:
        1. Fetch elevation profile from API
        2. Calculate magnetic declination
        3. Perform propagation analysis
        4. Format output (if formatter provided)
        5. Generate visualization (if visualizer provided)

        Args:
            input_data: User input with coordinates, frequency, etc.
            antenna_a_height: Antenna A height (meters)
            antenna_b_height: Antenna B height (meters)
            display_output: Whether to print results
            generate_plot: Whether to generate plot
            path: Optional pre-loaded path data
            save_plot_path: Optional path to save the plot

        Returns:
            AnalysisResult: Complete analysis result (pure data)
        """
        # Step 1: Fetch profile data
        if path is None:
            path = await self.profile_service.get_profile()

        # Step 2: Perform analysis (pure calculation, no side effects)
        result = await self.analysis_service.analyze(
            path=path,
            input_data=input_data,
            antenna_a_height=antenna_a_height,
            antenna_b_height=antenna_b_height,
        )

        # Always calculate GeoData
        geo_data_service = GeoDataService(self.declinations_api_client)
        geo_data = await geo_data_service.process(
            input_data.site_a_coordinates,
            input_data.site_b_coordinates
        )
        result.metadata["geo_data"] = geo_data  # Add geo_data to result metadata

        # Step 3: Display output (optional, injected dependency)
        if display_output and self.output_formatter:
            # Pass input_data to formatter for coordinates and other context
            self.output_formatter.format_result(
                result, input_data=input_data, geo_data=geo_data
            )

        # Step 4: Generate visualization (optional, injected dependency)
        if generate_plot and self.visualizer:
            # The profile_data is now calculated inside the analysis service and can be accessed from the result metadata
            profile_data = result.metadata.get("profile_data")
            if profile_data:
                self.visualizer.plot_profile(
                    path,
                    profile_data,
                    result,
                    save_path=save_plot_path,
                    show=save_plot_path is None,
                )

        return result


class GeoDataService: 
    """
    Computes geographic metadata between two coordinates.
    """

    def __init__(
        self,
        declinations_api_client: BaseDeclinationsApiClient,
    ):
        self.declinations_api_client = declinations_api_client

    async def process(
        self,
        coord_a: Coordinates,
        coord_b: Coordinates,
    ) -> GeoData:
        """
        Fetch declinations, calculate distance and azimuths between two coordinates.

        Returns:
            GeoData with distance, true/magnetic azimuths, and declinations.

        Raises:
            RuntimeError: If declination client is missing.
        """

        if not self.declinations_api_client:
            raise RuntimeError("Declinations API client is missing!")

        mag_declinations: list[
            Angle
        ] = await self.declinations_api_client.fetch_declinations(
            (
                coord_a,
                coord_b,
            )
        )
        coordinates_service_a_b = CoordinatesService(coord_a, coord_b)
        coordinates_service_b_a = CoordinatesService(coord_b, coord_a)

        distance: Kilometers = coordinates_service_a_b.get_distance()
        azimuth_a_b: Angle = coordinates_service_a_b.get_azimuth()
        azimuth_b_a: Angle = coordinates_service_b_a.get_azimuth()

        return GeoData(
            distance=distance,
            mag_declination_a=Angle(Degrees(round(mag_declinations[0], 2))),
            mag_declination_b=Angle(Degrees(round(mag_declinations[1], 2))),
            true_azimuth_a_b=Angle(Degrees(round(azimuth_a_b, 2))),
            true_azimuth_b_a=Angle(Degrees(round(azimuth_b_a, 2))),
            mag_azimuth_a_b=Angle(Degrees(round(azimuth_a_b - mag_declinations[0], 2))),
            mag_azimuth_b_a=Angle(Degrees(round(azimuth_b_a - mag_declinations[1], 2))),
        )

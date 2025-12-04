"""Facade adapter for simplified API integration."""

from pathlib import Path
from environs import Env

from trace_calc.domain.models.coordinates import Coordinates, InputData
from trace_calc.domain.models.path import GeoData
from trace_calc.domain.models.units import Meters
from trace_calc.domain.constants import OUTPUT_DATA_DIR
from trace_calc.infrastructure.api.clients import (
    AsyncElevationsApiClient,
    AsyncMagDeclinationApiClient,
)
from trace_calc.infrastructure.storage import FilePathStorage
from trace_calc.application.orchestration import OrchestrationService, GeoDataService
from trace_calc.application.services.profile import PathProfileService
from trace_calc.application.analysis import (
    BaseAnalysisService,
    GrozaAnalysisService,
    SosnikAnalysisService,
)
from trace_calc.application.services.user_input_parser import CoordinateParser
from trace_calc.infrastructure.visualization.plotter import ProfileVisualizer
from trace_calc.domain.models.analysis import AnalysisResult


class TraceAnalyzerAPI:
    """
    Simplified facade for external integration with a streamlined API.

    Provides a convenient interface that hides the complexity of dependency injection
    and orchestration, offering simple function-call semantics while using
    the modern DDD architecture internally.
    """

    def __init__(
        self,
        elevations_api_client: AsyncElevationsApiClient,
        declinations_api_client: AsyncMagDeclinationApiClient,
        storage: FilePathStorage,
    ):
        """
        Initialize facade with required dependencies.

        Args:
            elevations_api_client: Client for fetching elevation profiles
            declinations_api_client: Client for fetching magnetic declinations
            storage: File storage for path data
        """
        self._elevations_api_client = elevations_api_client
        self._declinations_api_client = declinations_api_client
        self._storage = storage
        self._visualizer = ProfileVisualizer(style="default")
        self._coord_parser = CoordinateParser()

        # Cached analysis services by method
        self._analysis_services: dict[str, BaseAnalysisService] = {}

    @classmethod
    def create_from_env(cls, env: Env) -> "TraceAnalyzerAPI":
        """
        Factory method: one-line initialization from environment.

        Args:
            env: Environment variable handler (Env instance)

        Returns:
            Configured TraceAnalyzerAPI ready to use

        Example:
            >>> from environs import Env
            >>> env = Env()
            >>> env.read_env()
            >>> facade = TraceAnalyzerAPI.create_from_env(env)
        """
        # Read environment variables
        elevation_api_url = env.str("ELEVATION_API_URL")
        elevation_api_key = env.str("ELEVATION_API_KEY")
        declination_api_url = env.str("DECLINATION_API_URL")
        declination_api_key = env.str("DECLINATION_API_KEY")

        # Initialize API clients
        elevations_client = AsyncElevationsApiClient(
            elevation_api_url, elevation_api_key
        )
        declinations_client = AsyncMagDeclinationApiClient(
            declination_api_url, declination_api_key
        )

        # Initialize storage
        storage = FilePathStorage(output_dir=OUTPUT_DATA_DIR)

        return cls(elevations_client, declinations_client, storage)

    async def parse_sites(
        self, s_name: str, coords: str
    ) -> tuple[list[str], list[float], list[str]]:
        """
        Parse and validate coordinates with site names.

        Args:
            s_name: Site names separated by spaces, commas, or semicolons
            coords: Coordinate string in any supported format

        Returns:
            Tuple of (site_names, coords_decimal, coords_formatted)

        Raises:
            ValueError: If coordinate format is invalid or count is wrong
        """
        try:
            return self._coord_parser.parse_with_names(
                s_name,
                coords,
                default_name_a='Точка А',
                default_name_b='Точка Б',
            )
        except ValueError as e:
            raise ValueError(
                f"Could not parse coordinates. {str(e)}\n"
                f"Supported formats:\n"
                f"  - Decimal: 55.7558 37.6173\n"
                f"  - DMS: 55°45'25.4\"N 37°37'6.2\"E\n"
                f"  - Mixed separators: commas, spaces, semicolons"
            ) from e

    async def get_azimuths(
        self, coords_dec: list[float]
    ) -> GeoData:
        """
        Calculate geographic data (azimuths and magnetic declinations).

        Args:
            coords_dec: List of 4 floats [lat1, lon1, lat2, lon2]

        Returns:
            GeoData object with all geographic data

        Raises:
            RuntimeError: If API request fails
        """
        if len(coords_dec) != 4:
            raise ValueError(
                f"Expected 4 coordinates, got {len(coords_dec)}. "
                f"Format: [lat1, lon1, lat2, lon2]"
            )

        coord_a = Coordinates(lat=coords_dec[0], lon=coords_dec[1])
        coord_b = Coordinates(lat=coords_dec[2], lon=coords_dec[3])

        geo_service = GeoDataService(self._declinations_api_client)
        return await geo_service.process(coord_a, coord_b)

    def _get_analysis_service(self, method: str) -> BaseAnalysisService:
        """Get cached analysis service for specified method."""
        if method not in self._analysis_services:
            analysis_service = (
                GrozaAnalysisService() if method == "groza" else SosnikAnalysisService()
            )
            # Note: profile_service is created per-call since it needs input_data
            self._analysis_services[method] = analysis_service
        return self._analysis_services[method]

    async def analyze_groza(
        self,
        coord_a: list[float],
        coord_b: list[float],
        path_filename: str,
        antenna_a_height: float = 2.0,
        antenna_b_height: float = 2.0,
        geo_data: GeoData | None = None,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, str]:
        """
        Run Groza propagation analysis.

        Args:
            coord_a: [lat, lon] for site A
            coord_b: [lat, lon] for site B
            path_filename: Filename for storing/loading path data
            antenna_a_height: Antenna A height in meters (default: 2.0)
            antenna_b_height: Antenna B height in meters (default: 2.0)
            geo_data: Optional pre-calculated geographic data to avoid refetching

        Returns:
            Tuple of (L0, Lmed, Lr, trace_dist, b1_max, b2_max,
                     b_sum, Ltot, dL, speed, sp_pref)
        """
        result = await self._run_analysis(
            "groza",
            coord_a,
            coord_b,
            path_filename,
            antenna_a_height,
            antenna_b_height,
            geo_data,
        )

        # Extract Groza-specific parameters
        params = result.model_propagation_loss_parameters

        return (
            params.get("L0", 0.0),  # Free space loss
            params.get("Lmed", 0.0),  # Median loss
            params.get("Lr", 0.0),  # Roughness loss
            params.get("trace_distance_km", 0.0),  # Path distance
            result.result.get("b1_max", 0.0),  # HCA site A
            result.result.get("b2_max", 0.0),  # HCA site B
            result.result.get("b_sum", 0.0),  # HCA sum
            params.get("total_loss", 0.0),  # Total loss
            params.get("dL", 0.0),  # Differential loss
            result.link_speed,  # Speed
            result.result.get("speed_prefix", "M"),  # Speed prefix (M or k)
        )

    async def analyze_sosnik(
        self,
        coord_a: list[float],
        coord_b: list[float],
        path_filename: str,
        antenna_a_height: float = 2.0,
        antenna_b_height: float = 2.0,
        geo_data: GeoData | None = None,
    ) -> tuple[float, float, float, float, float, float, float, str]:
        """
        Run Sosnik propagation analysis.

        Args:
            coord_a: [lat, lon] for site A
            coord_b: [lat, lon] for site B
            path_filename: Filename for storing/loading path data
            antenna_a_height: Antenna A height in meters (default: 2.0)
            antenna_b_height: Antenna B height in meters (default: 2.0)
            geo_data: Optional pre-calculated geographic data to avoid refetching

        Returns:
            Tuple of (trace_dist, extra_dist, b1_max, b2_max,
                     b_sum, Lr, speed, sp_pref)
        """
        result = await self._run_analysis(
            "sosnik",
            coord_a,
            coord_b,
            path_filename,
            antenna_a_height,
            antenna_b_height,
            geo_data,
        )

        # Extract Sosnik-specific parameters
        params = result.model_propagation_loss_parameters

        return (
            params.get("trace_distance_km", 0.0),  # Path distance
            params.get("extra_distance_km", 0.0),  # Extra distance
            result.result.get("b1_max", 0.0),  # HCA site A
            result.result.get("b2_max", 0.0),  # HCA site B
            result.result.get("b_sum", 0.0),  # HCA sum
            params.get("L_correction", 0.0),  # Roughness loss
            result.link_speed,  # Speed
            result.result.get("speed_prefix", "M"),  # Speed prefix (M or k)
        )

    async def _run_analysis(
        self,
        method: str,
        coord_a: list[float],
        coord_b: list[float],
        path_filename: str,
        antenna_a_height: float,
        antenna_b_height: float,
        geo_data: GeoData | None = None,
    ) -> AnalysisResult:
        """Internal method to run analysis with specified method."""
        # Create input data
        input_data = InputData(
            path_name=path_filename,
            site_a_coordinates=Coordinates(lat=coord_a[0], lon=coord_a[1]),
            site_b_coordinates=Coordinates(lat=coord_b[0], lon=coord_b[1]),
            antenna_a_height=Meters(antenna_a_height),
            antenna_b_height=Meters(antenna_b_height),
        )

        # Try to load existing path data
        path_data = None
        try:
            path_data = await self._storage.load(path_filename)
        except (FileNotFoundError, IndexError, ValueError):
            pass  # Will fetch from API below

        # Create profile service
        profile_service = PathProfileService(
            input_data=input_data,
            elevations_api_client=self._elevations_api_client,
            block_size=256,
            resolution=0.05,
        )

        # Fetch profile if not cached
        if path_data is None:
            path_data = await profile_service.get_profile()
            await self._storage.store(path_filename, path_data)

        # Create orchestrator
        analysis_service = self._get_analysis_service(method)
        orchestrator = OrchestrationService(
            analysis_service=analysis_service,
            profile_service=profile_service,
            declinations_api_client=self._declinations_api_client,
            visualizer=self._visualizer,
        )

        # Run analysis and generate plot
        plot_path = str(self._storage.output_dir / f"{path_filename}.png")
        result = await orchestrator.process(
            input_data=input_data,
            antenna_a_height=antenna_a_height,
            antenna_b_height=antenna_b_height,
            display_output=False,  # Don't print to console
            generate_plot=True,  # Generate PNG
            path=path_data,
            save_plot_path=plot_path,
            geo_data=geo_data,
        )

        return result


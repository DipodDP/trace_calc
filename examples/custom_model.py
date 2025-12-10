"""
Custom Propagation Model Example
==================================

This example demonstrates how to create a custom propagation model by extending
the base classes provided by trace_calc.

The example implements a simplified "Distance-Based Model" that calculates
link speed based primarily on path distance and frequency. This is an
educational example, not a real propagation model.

Steps to create a custom model:
1. Create a SpeedCalculator (inherits from BaseSpeedCalculator)
2. Create an Analyzer (inherits from BaseServiceAnalyzer + your calculator)
3. Create an AnalysisService (inherits from BaseAnalysisService)
4. Use it with OrchestrationService

Requirements:
- Python 3.10+
- trace_calc package installed
"""

from typing import Tuple, Any
import math

from trace_calc.domain.interfaces import BaseSpeedCalculator
from trace_calc.domain.models.units import Loss, Speed, Kilometers
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import PathData
from trace_calc.domain.models.analysis import AnalyzerResult, PropagationLoss
from trace_calc.application.analyzers.base import BaseServiceAnalyzer
from trace_calc.application.analysis import BaseAnalysisService


# ==============================================================================
# Step 1: Create Speed Calculator
# ==============================================================================

class DistanceBasedSpeedCalculator(BaseSpeedCalculator):
    """
    Custom speed calculator based on simple distance and frequency model.

    This is a simplified educational example. Real models should use
    empirical data and validated propagation equations.
    """

    def calculate_speed(
        self,
        distance_km: Kilometers,
        frequency_mhz: float,
        free_space_loss_db: Loss,
        terrain_factor: float = 1.0,
    ) -> Tuple[Loss, Speed]:
        """
        Calculate link speed based on distance and frequency.

        Args:
            distance_km: Path distance in kilometers
            frequency_mhz: Operating frequency in MHz
            free_space_loss_db: Free space path loss in dB
            terrain_factor: Terrain roughness factor (1.0 = smooth, >1.0 = rough)

        Returns:
            Tuple of (total_loss, link_speed)
        """
        # Simple loss model: free space + distance penalty + terrain penalty
        distance_penalty = 0.1 * distance_km  # 0.1 dB per km
        terrain_penalty = 5.0 * (terrain_factor - 1.0)  # Extra loss for rough terrain
        total_loss = Loss(free_space_loss_db + distance_penalty + terrain_penalty)

        # Speed thresholds based on total loss
        # These are example values - real models use link budgets and modulation schemes
        if total_loss < 150:
            speed = Speed(1024.0)  # kbps
        elif total_loss < 180:
            speed = Speed(512.0)
        elif total_loss < 200:
            speed = Speed(256.0)
        elif total_loss < 220:
            speed = Speed(128.0)
        else:
            speed = Speed(0.0)  # Link not feasible

        return total_loss, speed


# ==============================================================================
# Step 2: Create Analyzer
# ==============================================================================

class DistanceBasedAnalyzer(BaseServiceAnalyzer, DistanceBasedSpeedCalculator):
    """
    Custom analyzer that uses distance-based propagation model.

    Inherits from:
    - BaseServiceAnalyzer: Provides profile calculation, HCA, and plotting
    - DistanceBasedSpeedCalculator: Provides custom speed calculation logic
    """

    def analyze(self, **kwargs: Any) -> AnalyzerResult:
        """
        Perform analysis using custom distance-based model.

        Returns:
            AnalyzerResult with link speed, wavelength, and model parameters
        """
        # Get path distance from profile
        distance_km = Kilometers(self.distances[-1])  # Last distance is total path length

        # Calculate frequency and wavelength
        frequency_hz = self.input_data.frequency_mhz * 1e6
        wavelength = 3e8 / frequency_hz  # Speed of light / frequency

        # Calculate free space path loss using Friis equation
        # L_fs = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
        # Simplified: L_fs = 32.45 + 20*log10(f_MHz) + 20*log10(d_km)
        free_space_loss = Loss(
            32.45 + 20 * math.log10(self.input_data.frequency_mhz) + 20 * math.log10(distance_km)
        )

        # Calculate terrain roughness factor
        # Simple example: use standard deviation of elevations as roughness indicator
        terrain_std = float(self.elevations.std())
        terrain_factor = 1.0 + (terrain_std / 100.0)  # Normalize to reasonable range

        # Calculate speed using custom model
        total_loss, link_speed = self.calculate_speed(
            distance_km=distance_km,
            frequency_mhz=self.input_data.frequency_mhz,
            free_space_loss_db=free_space_loss,
            terrain_factor=terrain_factor,
        )

        # Package model parameters for output
        model_parameters = {
            "method": "distance_based",
            "distance_km": float(distance_km),
            "free_space_loss_db": float(free_space_loss),
            "terrain_factor": terrain_factor,
            "terrain_std_dev_m": terrain_std,
            "total_loss_db": float(total_loss),
        }

        # Create propagation loss breakdown (for compatibility with output formatters)
        # In this simple model, we only have free space and terrain losses
        propagation_loss = PropagationLoss(
            free_space_loss=free_space_loss,
            atmospheric_loss=Loss(0.0),  # Not modeled in this simple example
            diffraction_loss=Loss(0.0),  # Not modeled in this simple example
            refraction_loss=Loss(5.0 * (terrain_factor - 1.0)),  # Terrain penalty
            total_loss=total_loss,
        )

        model_parameters["L_free_space"] = float(free_space_loss)
        model_parameters["L_terrain"] = float(5.0 * (terrain_factor - 1.0))
        model_parameters["Ltot"] = float(total_loss)

        return AnalyzerResult(
            link_speed=float(link_speed),
            wavelength=wavelength,
            model_parameters=model_parameters,
            hca=self.hca_data,
            profile_data=self.profile_data,
            speed_prefix="k",  # kbps
        )


# ==============================================================================
# Step 3: Create Analysis Service
# ==============================================================================

class DistanceBasedAnalysisService(BaseAnalysisService):
    """
    Analysis service for the custom distance-based model.

    This class follows the Template Method pattern, implementing abstract
    methods required by BaseAnalysisService.
    """

    def _create_analyzer(self, path: PathData, input_data: InputData):
        """Factory method to create the custom analyzer."""
        return DistanceBasedAnalyzer(path, input_data)

    def _get_propagation_loss(self, result_data: dict[str, Any]) -> PropagationLoss:
        """Extract propagation loss components from analyzer results."""
        return PropagationLoss(
            free_space_loss=result_data.get("L_free_space", 0.0),
            atmospheric_loss=Loss(0.0),
            diffraction_loss=Loss(0.0),
            refraction_loss=result_data.get("L_terrain", 0.0),
            total_loss=result_data.get("Ltot", 0.0),
        )

    def _get_total_path_loss(self, result_data: dict[str, Any]) -> Loss:
        """Extract total path loss from analyzer results."""
        return Loss(result_data.get("Ltot", 0.0))


# ==============================================================================
# Usage Example
# ==============================================================================

async def example_custom_model_usage():
    """
    Example showing how to use the custom model with OrchestrationService.
    """
    from trace_calc.domain.models.coordinates import Coordinates, InputData
    from trace_calc.domain.models.units import Meters
    from trace_calc.application.orchestration import OrchestrationService
    from trace_calc.application.services.profile import PathProfileService
    from trace_calc.infrastructure.api.clients import (
        AsyncElevationsApiClient,
        AsyncMagDeclinationApiClient,
    )

    # Setup (same as any other model)
    input_data = InputData(
        path_name="custom_model_test",
        site_a_coordinates=Coordinates(55.7558, 37.6173),
        site_b_coordinates=Coordinates(59.9343, 30.3351),
        frequency_mhz=5000.0,
        antenna_a_height=Meters(30.0),
        antenna_b_height=Meters(30.0),
    )

    # Initialize API clients (you need valid API keys)
    elevations_client = AsyncElevationsApiClient(
        api_url="your_api_url",
        api_key="your_api_key",
    )
    declinations_client = AsyncMagDeclinationApiClient(
        api_url="your_api_url",
        api_key="your_api_key",
    )

    profile_service = PathProfileService(
        input_data=input_data,
        elevations_api_client=elevations_client,
        block_size=256,
        resolution=0.05,
    )

    # Use your custom analysis service instead of Groza or Sosnik!
    custom_analysis_service = DistanceBasedAnalysisService()

    orchestrator = OrchestrationService(
        analysis_service=custom_analysis_service,  # <-- Custom model here
        profile_service=profile_service,
        declinations_api_client=declinations_client,
    )

    # Run analysis
    result = await orchestrator.process(
        input_data=input_data,
        antenna_a_height=30.0,
        antenna_b_height=30.0,
        display_output=False,
        generate_plot=False,
    )

    print(f"Custom model result: {result.link_speed} kbps")
    print(f"Model parameters: {result.model_propagation_loss_parameters}")

    return result


if __name__ == "__main__":
    import asyncio

    print("Custom Propagation Model Example")
    print("=" * 50)
    print()
    print("This example shows how to create a custom propagation model.")
    print("To run this example, you need valid API keys configured.")
    print()
    print("Key points:")
    print("1. DistanceBasedSpeedCalculator - Custom calculation logic")
    print("2. DistanceBasedAnalyzer - Combines base analyzer with custom calculator")
    print("3. DistanceBasedAnalysisService - Service wrapper for dependency injection")
    print()
    print("The custom model can be used exactly like built-in models (Groza, Sosnik)")
    print("by simply passing it to OrchestrationService.")
    print()

    # Uncomment to run (requires API keys):
    # asyncio.run(example_custom_model_usage())

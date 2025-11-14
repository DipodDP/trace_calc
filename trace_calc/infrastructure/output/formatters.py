"""Output formatting services for console display"""

from typing import Protocol, Optional
import json

from trace_calc.domain.models.analysis import AnalysisResult
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import GeoData
from trace_calc.application.services.coordinates import CoordinatesService


class OutputFormatter(Protocol):
    """Protocol for output formatting strategies"""

    def format_result(
        self,
        result: AnalysisResult,
        input_data: Optional[InputData] = None,
        geo_data: Optional[GeoData] = None,
    ) -> None:
        """Format and display analysis result with optional context"""
        ...


class ConsoleOutputFormatter:
    """Format analysis results for console output"""

    def format_result(
        self,
        result: AnalysisResult,
        input_data: Optional[InputData] = None,
        geo_data: Optional[GeoData] = None,
    ) -> None:
        """
        Print analysis result to console in human-readable format.

        Args:
            result: Analysis result to display
            input_data: Optional input data containing site coordinates
            geo_data: Optional geographic data (distance, azimuths, etc.)
        """
        method = result.metadata.get("method", "unknown").upper()

        print(f"\n{'=' * 60}")
        print(f"{method} Analysis Result")
        print(f"{'=' * 60}")

        # Site coordinates section from InputData
        if (
            input_data
            and input_data.site_a_coordinates
            and input_data.site_b_coordinates
        ):
            print("\n📍 Site Coordinates:")
            print(
                f"  Site A:                  {input_data.site_a_coordinates.lat:.6f}°, "
                f"{input_data.site_a_coordinates.lon:.6f}°"
            )
            print(
                f"  Site B:                  {input_data.site_b_coordinates.lat:.6f}°, "
                f"{input_data.site_b_coordinates.lon:.6f}°"
            )

            # Calculate distance if not provided in geo_data
            if not geo_data:
                coord_service = CoordinatesService(
                    input_data.site_a_coordinates, input_data.site_b_coordinates
                )
                distance = coord_service.get_distance()
                print(f"  Distance:                {distance:.2f} km")

        # Geographic data section (additional details)
        if geo_data:
            print("\n🌍 Geographic Data:")
            print(f"  Distance:                {geo_data.distance:.2f} km")
            print(f"  True Azimuth A→B:        {geo_data.true_azimuth_a_b:.2f}°")
            print(f"  True Azimuth B→A:        {geo_data.true_azimuth_b_a:.2f}°")
            print(f"  Mag Declination A:       {geo_data.mag_declination_a:.2f}°")
            print(f"  Mag Declination B:       {geo_data.mag_declination_b:.2f}°")

        print("\n📡 Link Parameters:")
        print(f"  Wavelength:              {result.wavelength:.4f} m")
        if "frequency_mhz" in result.metadata:
            print(
                f"  Frequency:               {result.metadata['frequency_mhz']:.1f} MHz"
            )
        if "distance_km" in result.metadata:
            print(f"  Path Distance:           {result.metadata['distance_km']:.2f} km")

        print("\n📉 Propagation Loss:")
        print(f"  Basic Transmission Loss: {result.basic_transmission_loss:.2f} dB")

        if result.propagation_loss:
            print(
                f"    ├─ Free Space Loss:    {result.propagation_loss.free_space_loss:.2f} dB"
            )
            print(
                f"    ├─ Atmospheric Loss:   {result.propagation_loss.atmospheric_loss:.2f} dB"
            )
            print(
                f"    └─ Diffraction Loss:   {result.propagation_loss.diffraction_loss:.2f} dB"
            )

        print(f"  Total Path Loss:         {result.total_path_loss:.2f} dB")

        print("\n🚀 Link Performance:")
        print(f"  Estimated Speed:         {result.link_speed:.1f} Mbps")

        if "link_margin_db" in result.metadata:
            margin = result.metadata["link_margin_db"]
            status = (
                "✅ EXCELLENT"
                if margin > 20
                else "⚠️  MARGINAL"
                if margin > 10
                else "❌ POOR"
            )
            print(f"  Link Margin:             {margin:.1f} dB ({status})")

        print(f"{'=' * 60}\n")


class JSONOutputFormatter:
    """Format analysis results as JSON (for API/automation)"""

    def format_result(
        self,
        result: AnalysisResult,
        input_data: Optional[InputData] = None,
        geo_data: Optional[GeoData] = None,
    ) -> str:
        """
        Return JSON string representation with optional context.

        Args:
            result: Analysis result to format
            input_data: Optional input data containing site coordinates
            geo_data: Optional geographic data

        Returns:
            JSON string with all available data
        """
        output_dict = result.to_dict()

        # Add site coordinates from InputData
        if input_data:
            if input_data.site_a_coordinates:
                output_dict["site_a_coordinates"] = {
                    "lat": input_data.site_a_coordinates.lat,
                    "lon": input_data.site_a_coordinates.lon,
                }
            if input_data.site_b_coordinates:
                output_dict["site_b_coordinates"] = {
                    "lat": input_data.site_b_coordinates.lat,
                    "lon": input_data.site_b_coordinates.lon,
                }

            # Calculate and add distance if both coordinates are present and no geo_data
            if (
                input_data.site_a_coordinates
                and input_data.site_b_coordinates
                and not geo_data
            ):
                coord_service = CoordinatesService(
                    input_data.site_a_coordinates, input_data.site_b_coordinates
                )
                distance = coord_service.get_distance()
                output_dict["calculated_distance_km"] = float(distance)

        # Add geographic data
        if geo_data:
            output_dict["geo_data"] = {
                "distance_km": float(geo_data.distance),
                "true_azimuth_a_b": float(geo_data.true_azimuth_a_b),
                "true_azimuth_b_a": float(geo_data.true_azimuth_b_a),
                "mag_azimuth_a_b": float(geo_data.mag_azimuth_a_b),
                "mag_azimuth_b_a": float(geo_data.mag_azimuth_b_a),
                "mag_declination_a": float(geo_data.mag_declination_a),
                "mag_declination_b": float(geo_data.mag_declination_b),
            }

        return json.dumps(output_dict, indent=2)

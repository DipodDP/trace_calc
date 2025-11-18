"""Output formatting services for console display"""

from typing import Protocol, Optional
import json

from trace_calc.domain.models.analysis import AnalysisResult
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import GeoData, ProfileData
from trace_calc.application.services.coordinates import CoordinatesService


def format_common_volume_results(profile: ProfileData) -> str:
    """
    Format Common Scatter Volume analysis results for console output.

    Args:
        profile: Complete profile data with intersections and volume

    Returns:
        Formatted string for console display
    """
    sight_lines = profile.lines_of_sight
    intersections = profile.intersections
    volume = profile.volume

    output = []
    output.append("\n=== Common Scatter Volume Analysis ===")

    # Lower sight lines
    output.append("\nLower Sight Lines:")
    output.append(
        f"  Site A -> Obstacle: slope={sight_lines.lower_a[0]:.4f}, "
        f"intercept={sight_lines.lower_a[1]/1000:.4f}km"
    )
    output.append(
        f"  Site B -> Obstacle: slope={-sight_lines.lower_b[0]:.4f}, "
        f"intercept={sight_lines.lower_b[1]/1000:.4f}km"
    )

    # Upper sight lines
    output.append("\nUpper Sight Lines:")
    output.append(
        f"  Site A (upper): slope={sight_lines.upper_a[0]:.4f}, "
        f"intercept={sight_lines.upper_a[1]/1000:.4f}km"
    )
    output.append(
        f"  Site B (upper): slope={-sight_lines.upper_b[0]:.4f}, "
        f"intercept={sight_lines.upper_b[1]/1000:.4f}km"
    )

    # Bisectors
    output.append("\nBisectors:")
    output.append(
        f"  Bisector A starts at {profile.lines_of_sight.bisector_a[0]/1000:.4f}km and ends at {profile.lines_of_sight.bisector_a[-1]/1000:.4f}km"
    )
    output.append(
        f"  Bisector B starts at {profile.lines_of_sight.bisector_b[0]/1000:.4f}km and ends at {profile.lines_of_sight.bisector_b[-1]/1000:.4f}km"
    )

    # Cross intersections
    output.append("\nCross Intersections:")
    output.append(
        f"  Upper A x Lower B: {intersections.cross_ab.distance_km:.2f} km, "
        f"{intersections.cross_ab.elevation_sea_level/1000:.4f}km ASL, "
        f"{intersections.cross_ab.elevation_terrain/1000:.4f}km above terrain"
    )
    output.append(
        f"  Upper B x Lower A: {intersections.cross_ba.distance_km:.2f} km, "
        f"{intersections.cross_ba.elevation_sea_level/1000:.4f}km ASL, "
        f"{intersections.cross_ba.elevation_terrain/1000:.4f}km above terrain"
    )

    # Volume metrics
    output.append("\nVolume Metrics:")
    output.append(
        f"  Common scatter volume: {volume.cone_intersection_volume_m3 / 1e9:.2f} km³"
    )
    output.append(
        f"  Distance from A to Upper A x Lower B: {volume.distance_a_to_cross_ab:.2f} km"
    )
    output.append(
        f"  Distance from B to Upper B x Lower A: {volume.distance_b_to_cross_ba:.2f} km"
    )
    output.append(
        f"  Distance between cross intersections: {volume.distance_between_crosses:.2f} km"
    )
    output.append(
        f"  Common volume top (upper intersection): {intersections.upper_intersection.distance_km:.2f} km, "
        f"{intersections.upper_intersection.elevation_terrain/1000:.2f} km above terrain, "
        f"{intersections.upper_intersection.elevation_sea_level/1000:.2f} km ASL"
    )
    output.append(
        f"  Common volume bottom (lower intersection): {intersections.lower_intersection.distance_km:.2f} km, "
        f"{intersections.lower_intersection.elevation_terrain/1000:.2f} km above terrain, "
        f"{intersections.lower_intersection.elevation_sea_level/1000:.2f} km ASL"
    )

    # Distance metrics to lower/upper intersections
    output.append("\nDistance Metrics:")
    output.append(
        f"  Distance from A to lower intersection: {volume.distance_a_to_lower_intersection:.2f} km"
    )
    output.append(
        f"  Distance from B to lower intersection: {volume.distance_b_to_lower_intersection:.2f} km"
    )
    output.append(
        f"  Distance from A to upper intersection: {volume.distance_a_to_upper_intersection:.2f} km"
    )
    output.append(
        f"  Distance from B to upper intersection: {volume.distance_b_to_upper_intersection:.2f} km"
    )
    output.append(
        f"  Distance between lower and upper intersections: {volume.distance_between_lower_upper_intersections:.2f} km"
    )

    return "\n".join(output)


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

        if "profile_data" in result.metadata:
            print(format_common_volume_results(result.metadata["profile_data"]))

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

        if "profile_data" in result.metadata:
            profile_data = result.metadata["profile_data"]
            
            def intersection_to_dict_km(point):
                return {
                    "distance_km": point.distance_km,
                    "elevation_sea_level_km": point.elevation_sea_level / 1000,
                    "elevation_terrain_km": point.elevation_terrain / 1000,
                }

            output_dict["profile_data"] = {
                "sight_lines": {
                    "lower_a": profile_data.lines_of_sight.lower_a.tolist(),
                    "lower_b": profile_data.lines_of_sight.lower_b.tolist(),
                    "upper_a": profile_data.lines_of_sight.upper_a.tolist(),
                    "upper_b": profile_data.lines_of_sight.upper_b.tolist(),
                    "bisector_a": profile_data.lines_of_sight.bisector_a.tolist(),
                    "bisector_b": profile_data.lines_of_sight.bisector_b.tolist(),
                },
                "intersections": {
                    "lower": intersection_to_dict_km(profile_data.intersections.lower_intersection),
                    "upper": intersection_to_dict_km(profile_data.intersections.upper_intersection),
                    "cross_ab": intersection_to_dict_km(profile_data.intersections.cross_ab),
                    "cross_ba": intersection_to_dict_km(profile_data.intersections.cross_ba),
                },
                "volume": {
                    "cone_intersection_volume_m3": profile_data.volume.cone_intersection_volume_m3,
                    "distance_a_to_cross_ab": profile_data.volume.distance_a_to_cross_ab,
                    "distance_b_to_cross_ba": profile_data.volume.distance_b_to_cross_ba,
                    "distance_between_crosses": profile_data.volume.distance_between_crosses,
                    "distance_a_to_lower_intersection": profile_data.volume.distance_a_to_lower_intersection,
                    "distance_b_to_lower_intersection": profile_data.volume.distance_b_to_lower_intersection,
                    "distance_a_to_upper_intersection": profile_data.volume.distance_a_to_upper_intersection,
                    "distance_b_to_upper_intersection": profile_data.volume.distance_b_to_upper_intersection,
                    "distance_between_lower_upper_intersections": profile_data.volume.distance_between_lower_upper_intersections,
                },
            }

        return json.dumps(output_dict, indent=2)

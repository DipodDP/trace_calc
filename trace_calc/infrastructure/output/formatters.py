"""Output formatting services for console display."""

import json
import numpy as np
from typing import Protocol

from trace_calc.domain.models.analysis import AnalysisResult
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import GeoData, ProfileData
from trace_calc.application.services.coordinates import CoordinatesService
from trace_calc.infrastructure.i18n import t


def _format_dict_floats(d, precision):
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = round(v, precision)
        elif isinstance(v, dict):
            _format_dict_floats(v, precision)
        elif isinstance(v, list):
            d[k] = [
                _format_dict_floats(i, precision)
                if isinstance(i, dict)
                else (round(i, precision) if isinstance(i, float) else i)
                for i in v
            ]
    return d


def _build_output_dict(
    result: AnalysisResult,
    input_data: InputData | None = None,
    geo_data: GeoData | None = None,
    profile_data: ProfileData | None = None,
) -> dict:
    res_data = result.to_dict()

    # If profile_data is not provided as a separate argument, try to get it from result.result
    if profile_data is None and "profile_data" in result.result:
        profile_data = result.result.pop("profile_data")

    # Format metadata and loss parameters
    metadata = _format_dict_floats(res_data["result"], 2)
    loss_params = _format_dict_floats(
        res_data.pop("model_propagation_loss_parameters"), 2
    )

    output_dict = {
        "analysis_result": {
            "link_speed": round(res_data.pop("link_speed"), 1),
            "model_propagation_loss_parameters": loss_params,
            "metadata": metadata,
        }
    }
    # Clean up redundant loss fields from model_propagation_loss_parameters
    loss_params = output_dict["analysis_result"]["model_propagation_loss_parameters"]

    # Reformat propagation_loss
    if "propagation_loss" in loss_params:
        prop_loss = loss_params["propagation_loss"]
        if "total_loss" in prop_loss:
            loss_params["total_loss"] = prop_loss.pop("total_loss")

    for key in ["L0", "Lmed", "Ld", "Lr", "Ltot", "dL", "method", "total_path_loss"]:
        loss_params.pop(key, None)

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
            output_dict["calculated_distance_km"] = round(distance, 2)

    if geo_data:
        geo_dict = geo_data.to_dict() if hasattr(geo_data, "to_dict") else geo_data
        output_dict["geo_data"] = {
            "distance_km": round(float(geo_dict["distance_km"]), 2),
            "true_azimuth_a_b": float(geo_dict["true_azimuth_a_b"]),
            "true_azimuth_b_a": float(geo_dict["true_azimuth_b_a"]),
            "mag_azimuth_a_b": float(geo_dict["mag_azimuth_a_b"]),
            "mag_azimuth_b_a": float(geo_dict["mag_azimuth_b_a"]),
            "mag_declination_a": float(geo_dict["mag_declination_a"]),
            "mag_declination_b": float(geo_dict["mag_declination_b"]),
        }

    if profile_data:

        def intersection_to_dict_km(point):
            data = {
                "distance_km": round(point.distance_km, 2),
                "elevation_sea_level_km": round(point.elevation_sea_level / 1000, 2),
                "elevation_terrain_km": round(point.elevation_terrain / 1000, 2),
            }
            if hasattr(point, "angle") and point.angle is not None:
                data["angle_deg"] = round(point.angle, 2)
            return data

        output_dict["profile_data"] = {
            "sight_lines": {
                "lower_a_slope": round(profile_data.lines_of_sight.lower_a[0], 4),
                "lower_b_slope": round(profile_data.lines_of_sight.lower_b[0], 4),
                "upper_a_slope": round(profile_data.lines_of_sight.upper_a[0], 4),
                "upper_b_slope": round(profile_data.lines_of_sight.upper_b[0], 4),
            },
            "intersections": {
                "lower": intersection_to_dict_km(profile_data.intersections.lower),
                "upper": intersection_to_dict_km(profile_data.intersections.upper),
                "cross_ab": intersection_to_dict_km(
                    profile_data.intersections.cross_ab
                ),
                "cross_ba": intersection_to_dict_km(
                    profile_data.intersections.cross_ba
                ),
                "beam_intersection_point": (
                    intersection_to_dict_km(
                        profile_data.intersections.beam_intersection_point
                    )
                    if profile_data.intersections.beam_intersection_point
                    else None
                ),
            },
            "common_volume": {
                "cone_intersection_volume_m3": round(
                    profile_data.volume.cone_intersection_volume_m3, 2
                ),
                "distance_a_to_cross_ab": round(
                    profile_data.volume.distance_a_to_cross_ab, 2
                ),
                "distance_b_to_cross_ba": round(
                    profile_data.volume.distance_b_to_cross_ba, 2
                ),
                "distance_between_crosses": round(
                    profile_data.volume.distance_between_crosses, 2
                ),
                "distance_a_to_lower_intersection": round(
                    profile_data.volume.distance_a_to_lower_intersection, 2
                ),
                "distance_b_to_lower_intersection": round(
                    profile_data.volume.distance_b_to_lower_intersection, 2
                ),
                "distance_a_to_upper_intersection": round(
                    profile_data.volume.distance_a_to_upper_intersection, 2
                ),
                "distance_b_to_upper_intersection": round(
                    profile_data.volume.distance_b_to_upper_intersection, 2
                ),
                "distance_between_lower_upper_intersections": round(
                    profile_data.volume.distance_between_lower_upper_intersections, 2
                ),
                "antenna_elevation_angle_a": round(
                    profile_data.volume.antenna_elevation_angle_a, 2
                ),
                "antenna_elevation_angle_b": round(
                    profile_data.volume.antenna_elevation_angle_b, 2
                ),
            },
        }

    return output_dict


def format_common_volume_results(profile: ProfileData) -> str:
    """
    Format Common Scatter Volume analysis results for console output.

    Args:
        profile: Complete profile data with intersections and common_volume

    Returns:
        Formatted string for console display
    """
    sight_lines = profile.lines_of_sight
    intersections = profile.intersections
    common_volume = profile.volume  # Changed from 'volume' to 'common_volume'

    output = []
    output.append(t("common_volume_analysis_title"))

    # Lower sight lines
    output.append(t("lower_sight_lines"))
    angle_a = np.degrees(np.arctan(sight_lines.lower_a[0] / 1000))
    angle_b = np.degrees(np.arctan(-sight_lines.lower_b[0] / 1000))
    output.append(t("site_a_obstacle", sight_lines.lower_a[0], angle_a))
    output.append(t("site_b_obstacle", -sight_lines.lower_b[0], angle_b))

    # Upper sight lines
    output.append(t("upper_sight_lines"))
    angle_upper_a = np.degrees(np.arctan(sight_lines.upper_a[0] / 1000))
    angle_upper_b = np.degrees(np.arctan(-sight_lines.upper_b[0] / 1000))
    output.append(t("site_a_upper", sight_lines.upper_a[0], angle_upper_a))
    output.append(t("site_b_upper", -sight_lines.upper_b[0], angle_upper_b))

    # Beam Intersection Point
    if intersections.beam_intersection_point:
        output.append(t("beam_intersection_point"))
        output.append(
            t(
                "beam_intersection_details",
                intersections.beam_intersection_point.distance_km,
                intersections.beam_intersection_point.elevation_sea_level / 1000,
                intersections.beam_intersection_point.elevation_terrain / 1000,
                intersections.beam_intersection_point.angle or 0.0,
            )
        )
    else:
        output.append(t("beam_intersection_not_found"))

    # Antenna Elevation Angles
    output.append(t("antenna_elevation_angles"))
    output.append(t("antenna_elevation_angle_a", common_volume.antenna_elevation_angle_a))
    output.append(t("antenna_elevation_angle_b", common_volume.antenna_elevation_angle_b))

    # Cross intersections
    output.append(t("cross_intersections"))
    output.append(
        t(
            "upper_a_lower_b",
            intersections.cross_ab.distance_km,
            intersections.cross_ab.elevation_sea_level / 1000,
            intersections.cross_ab.elevation_terrain / 1000,
        )
    )
    output.append(
        t(
            "upper_b_lower_a",
            intersections.cross_ba.distance_km,
            intersections.cross_ba.elevation_sea_level / 1000,
            intersections.cross_ba.elevation_terrain / 1000,
        )
    )

    # Common Volume metrics
    output.append(t("common_volume_metrics"))
    output.append(t("common_scatter_volume", common_volume.cone_intersection_volume_m3 / 1e9))
    output.append(t("dist_a_cross_ab", common_volume.distance_a_to_cross_ab))
    output.append(t("dist_b_cross_ba", common_volume.distance_b_to_cross_ba))
    output.append(t("dist_between_crosses", common_volume.distance_between_crosses))
    output.append(
        t(
            "common_volume_top",
            intersections.upper.distance_km,
            intersections.upper.elevation_terrain / 1000,
            intersections.upper.elevation_sea_level / 1000,
        )
    )
    output.append(
        t(
            "common_volume_bottom",
            intersections.lower.distance_km,
            intersections.lower.elevation_terrain / 1000,
            intersections.lower.elevation_sea_level / 1000,
        )
    )

    # Distance metrics to lower/upper intersections
    output.append(t("distance_metrics"))
    output.append(t("dist_a_lower", common_volume.distance_a_to_lower_intersection))
    output.append(t("dist_b_lower", common_volume.distance_b_to_lower_intersection))
    output.append(t("dist_a_upper", common_volume.distance_a_to_upper_intersection))
    output.append(t("dist_b_upper", common_volume.distance_b_to_upper_intersection))
    output.append(t("dist_lower_upper", common_volume.distance_between_lower_upper_intersections))

    return "\n".join(output)


class OutputFormatter(Protocol):
    """Protocol for output formatting strategies"""

    def format_result(
        self,
        result: AnalysisResult,
        input_data: InputData | None = None,
        geo_data: GeoData | None = None,
        profile_data: ProfileData | None = None,
    ) -> None:
        """Format and display analysis result with optional context"""
        ...


class ConsoleOutputFormatter:
    """Format analysis results for console output"""

    def format_result(
        self,
        result: AnalysisResult,
        input_data: InputData | None = None,
        geo_data: GeoData | None = None,
        profile_data: ProfileData | None = None,
    ) -> None:
        output_dict = _build_output_dict(result, input_data, geo_data, profile_data)

        analysis_result = output_dict.get("analysis_result", {})
        metadata = analysis_result.get("metadata", {})
        method = metadata.get("method", "unknown").upper()

        print(f"\n{'=' * 60}")
        print(t("analysis_result", method))
        print(f"{'=' * 60}")

        if "site_a_coordinates" in output_dict and "site_b_coordinates" in output_dict:
            print(t("site_coordinates"))
            print(
                f"{t('site_a')}{output_dict['site_a_coordinates']['lat']:.6f}°, "
                f"{output_dict['site_a_coordinates']['lon']:.6f}°"
            )
            print(
                f"{t('site_b')}{output_dict['site_b_coordinates']['lat']:.6f}°, "
                f"{output_dict['site_b_coordinates']['lon']:.6f}°"
            )

        if "geo_data" in output_dict:
            geo_dict = output_dict["geo_data"]
            print(t("geographic_data"))
            print(f"{t('distance')}{geo_dict['distance_km']:.2f} km")
            print(f"{t('true_azimuth_a_b')}{geo_dict['true_azimuth_a_b']:.2f}°")
            print(f"{t('true_azimuth_b_a')}{geo_dict['true_azimuth_b_a']:.2f}°")
            print(f"{t('mag_declination_a')}{geo_dict['mag_declination_a']:.2f}°")
            print(f"{t('mag_declination_b')}{geo_dict['mag_declination_b']:.2f}°")
            print(f"{t('mag_azimuth_a_b')}{geo_dict['mag_azimuth_a_b']:.2f}°")
            print(f"{t('mag_azimuth_b_a')}{geo_dict['mag_azimuth_b_a']:.2f}°")

        if "b1_max" in metadata and "b2_max" in metadata and "b_sum" in metadata:
            print(t("hca_title"))
            print(f"{t('site_a_hca')}{metadata['b1_max']:.2f}°")
            print(f"{t('site_b_hca')}{metadata['b2_max']:.2f}°")
            print(f"{t('hca_sum')}{metadata['b_sum']:.2f}°")

        print(t("link_parameters"))
        print(f"{t('wavelength')}{metadata.get('wavelength', 0):.2f} m")
        model_params = analysis_result.get("model_propagation_loss_parameters", {})
        if "frequency_mhz" in metadata:
            print(f"{t('frequency')}{metadata['frequency_mhz']:.2f} MHz")
        if "hpbw" in metadata:
            print(f"{t('hpbw')}{metadata['hpbw']:.2f}°")

        print(t("propagation_loss_parameters"))
        if model_params.get("propagation_loss"):
            prop_loss = model_params["propagation_loss"]
            print(
                f"{t('free_space_loss')}{prop_loss['free_space_loss']:.2f} dB"
            )
            print(
                f"{t('atmospheric_loss')}{prop_loss['atmospheric_loss']:.2f} dB"
            )
            print(
                f"{t('diffraction_loss')}{prop_loss['diffraction_loss']:.2f} dB"
            )
            print(
                f"{t('refraction_loss')}{prop_loss['refraction_loss']:.2f} dB"
            )

        if model_params.get("total_loss") is not None:
            print(f"{t('total_path_loss')}{model_params['total_loss']:.2f} dB")

        # Sosnik-specific parameters (no propagation_loss breakdown)
        if "extra_dist" in model_params:
            print(f"{t('extra_dist')}{model_params['extra_dist']:.2f} km")
        if "equal_dist" in model_params:
            print(f"{t('equal_dist')}{model_params['equal_dist']:.2f} km")
        if "L_correction" in model_params:
            print(f"{t('l_correction')}{model_params['L_correction']:.2f} dB")

        print(t("link_performance"))
        speed_prefix = metadata.get("speed_prefix", "M")
        speed_unit = t("mbps") if speed_prefix == "M" else t("kbps")
        print(
            f"{t('estimated_speed')}{analysis_result.get('link_speed', 0):.1f} {speed_unit}"
        )

        if (
            "profile_data" in output_dict
            and "common_volume" in output_dict["profile_data"]
        ):
            print(format_common_volume_results(profile_data))

        print(f"{'=' * 60}\n")


class JSONOutputFormatter:
    """Format analysis results as JSON (for API/automation)"""

    def format_result(
        self,
        result: AnalysisResult,
        input_data: InputData | None = None,
        geo_data: GeoData | None = None,
        profile_data: ProfileData | None = None,
    ) -> str:
        output_dict = _build_output_dict(result, input_data, geo_data, profile_data)
        return json.dumps(output_dict, indent=2)

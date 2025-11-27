"""Profile visualization service (separated from analysis logic)"""

import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from trace_calc.domain.curvature import calculate_earth_drop
from trace_calc.domain.models.path import PathData, ProfileData
from trace_calc.domain.models.analysis import AnalysisResult


class ProfileVisualizer:
    """
    Service for visualizing elevation profiles and analysis results.

    Separated from analysis logic to allow:
    - Testing analysis without matplotlib
    - Reusing visualization with different analyzers
    - Running analysis in headless environments (servers, CI)

    Renders two-panel elevation profile matching the original plotter style:
    - Panel 1: Plain (flat) elevation profile
    - Panel 2: Curved profile with Earth curvature and sight lines
    """

    def __init__(self, style: str = "default"):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style (default, seaborn, ggplot, etc.)
        """
        self.style = style
        if style != "default":
            plt.style.use(style)

    def plot_profile(
        self,
        path: PathData,
        profile: ProfileData,
        result: Optional[AnalysisResult] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Generate and save a two-panel profile chart matching original style.

        Args:
            path: Raw path data (distances, elevations)
            profile: Processed profile data (plain, curved, sight lines)
            result: Optional analysis result to display
            show: Whether to display plot interactively
            save_path: Optional path to save figure
        """
        distances = path.distances

        if distances.ndim != 1 or distances.size < 2:
            raise ValueError("`distances` must be a 1D array with at least two points.")

        # Create two-panel figure matching original dimensions
        fig, axes = plt.subplots(2, 1, figsize=(19.20, 10.8))

        # ===== Panel 1: Plain Profile =====
        elevations, zero = (
            profile.plain.elevations,
            profile.plain.baseline,
        )
        axes[0].plot(distances, elevations, "k-", linewidth=1.0, label="Terrain")
        axes[0].fill_between(distances, elevations, zero, facecolor="g", alpha=0.2)
        axes[0].grid(True)
        axes[0].set_title("Plain Elevation Profile", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Distance (km)", fontsize=10)
        axes[0].set_ylabel("Elevation (m)", fontsize=10)

        elevations_range = elevations.max() - elevations.min()
        axes[0].set_xlim(distances[0], distances[-1])
        axes[0].set_ylim(
            elevations.min() - elevations_range / 10,
            elevations.max() + elevations_range / 10,
        )

        # ===== Panel 2: Curved Profile with Sight Lines =====
        elevations_curved, zero_curved = (
            profile.curved.elevations / 1000,
            profile.curved.baseline / 1000,
        )

        sight_lines = profile.lines_of_sight
        intersections = profile.intersections

        # Plot curved elevation with fill
        axes[1].plot(
            distances, elevations_curved, "k-", linewidth=1.0, label="Terrain (curved)"
        )
        axes[1].fill_between(
            distances,
            elevations_curved,
            zero_curved,
            facecolor="g",
            alpha=0.2,
        )

        # Plot lower sight lines (using colors that work well in grayscale)
        lower_line_1 = np.polyval(sight_lines.lower_a, distances.astype(float)) / 1000
        lower_line_2 = np.polyval(sight_lines.lower_b, distances.astype(float)) / 1000
        axes[1].plot(
            distances,
            lower_line_1,
            color="C1",
            linestyle="-",
            lw=2.0,
            label="Lower sight line A",
        )
        axes[1].plot(
            distances,
            lower_line_2,
            color="C0",
            linestyle="-",
            lw=2.0,
            label="Lower sight line B",
        )

        # Plot upper sight lines (same colors as lower, but dashed)
        upper_line_1 = np.polyval(sight_lines.upper_a, distances.astype(float)) / 1000
        upper_line_2 = np.polyval(sight_lines.upper_b, distances.astype(float)) / 1000
        axes[1].plot(
            distances,
            upper_line_1,
            color="C1",
            linestyle="--",
            lw=2.0,
            alpha=0.8,
            label="Upper sight line A",
        )
        axes[1].plot(
            distances,
            upper_line_2,
            color="C0",
            linestyle="--",
            lw=2.0,
            alpha=0.8,
            label="Upper sight line B",
        )

        # Plot antenna elevation angle lines
        axes[1].plot(
            distances,
            profile.lines_of_sight.antenna_elevation_angle_a / 1000,
            color="purple",
            linestyle=":",
            lw=1.5,
            alpha=0.9,
            label="Antenna Elevation Angle A",
        )
        axes[1].plot(
            distances,
            profile.lines_of_sight.antenna_elevation_angle_b / 1000,
            color="brown",
            linestyle=":",
            lw=1.5,
            alpha=0.9,
            label="Antenna Elevation Angle B",
        )

        # Plot intersection points (using colors with good grayscale contrast)
        # Note: We add the earth drop back to the elevations for plotting purposes only.
        # This projects the physically accurate (curved sea level) coordinates back
        # onto the flat tangent plane that the sight lines are drawn on.
        lower_intersection = intersections.lower
        upper_intersection = intersections.upper
        cross_ab_intersection = intersections.cross_ab
        cross_ba_intersection = intersections.cross_ba

        lower_plot_elev = (
            lower_intersection.elevation_sea_level
            + calculate_earth_drop(np.array([lower_intersection.distance_km]))[0]
        ) / 1000
        upper_plot_elev = (
            upper_intersection.elevation_sea_level
            + calculate_earth_drop(np.array([upper_intersection.distance_km]))[0]
        ) / 1000
        cross_ab_plot_elev = (
            cross_ab_intersection.elevation_sea_level
            + calculate_earth_drop(np.array([cross_ab_intersection.distance_km]))[0]
        ) / 1000
        cross_ba_plot_elev = (
            cross_ba_intersection.elevation_sea_level
            + calculate_earth_drop(np.array([cross_ba_intersection.distance_km]))[0]
        ) / 1000

        axes[1].scatter(
            lower_intersection.distance_km,
            lower_plot_elev,
            c="darkgreen",
            s=120,
            marker="o",
            label="Lower intersection",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        axes[1].scatter(
            upper_intersection.distance_km,
            upper_plot_elev,
            c="darkred",
            s=120,
            marker="o",
            label="Upper intersection",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        axes[1].scatter(
            cross_ab_intersection.distance_km,
            cross_ab_plot_elev,
            c="goldenrod",
            s=100,
            marker="^",
            label="Cross AB (Upper A × Lower B)",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        axes[1].scatter(
            cross_ba_intersection.distance_km,
            cross_ba_plot_elev,
            c="indigo",
            s=100,
            marker="v",
            label="Cross BA (Upper B × Lower A)",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

        # Plot beam intersection point
        if intersections.beam_intersection_point:
            beam_intersection_point = intersections.beam_intersection_point
            beam_intersection_point_plot_elev = (
                beam_intersection_point.elevation_sea_level
                + calculate_earth_drop(np.array([beam_intersection_point.distance_km]))[
                    0
                ]
            ) / 1000
            axes[1].scatter(
                beam_intersection_point.distance_km,
                beam_intersection_point_plot_elev,
                c="blue",
                s=150,
                marker="X",
                label="Beam Intersection Point",
                edgecolors="black",
                linewidths=1.5,
                zorder=6,
            )

        axes[1].grid(True)
        axes[1].set_title(
            "Curved Profile with Common Scatter Volume Analysis",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_xlabel("Distance (km)", fontsize=10)
        axes[1].set_ylabel("Elevation (km)", fontsize=10)
        axes[1].legend(loc="upper right", fontsize=8, ncol=2)

        # Add volume metrics text box
        height_between_intersections = (
            profile.intersections.upper.elevation_sea_level
            - profile.intersections.lower.elevation_sea_level
        )
        metrics_text = (
            f"Common scatter volume: {profile.volume.cone_intersection_volume_m3 / 1e9:.2f} km³\n"
            f"Distance A→Cross AB: {profile.volume.distance_a_to_cross_ab:.2f} km\n"
            f"Distance B→Cross BA: {profile.volume.distance_b_to_cross_ba:.2f} km\n"
            f"Distance between crosses: {profile.volume.distance_between_crosses:.2f} km\n"
            f"Top above terrain: {profile.intersections.upper.elevation_terrain / 1000:.2f} km ("
            f"ASL: {profile.intersections.upper.elevation_sea_level / 1000:.2f} km)\n"
            f"Bottom above terrain: {profile.intersections.lower.elevation_terrain / 1000:.2f} km ("
            f"ASL: {profile.intersections.lower.elevation_sea_level / 1000:.2f} km)\n"
            f"Height: {height_between_intersections / 1000:.2f} km"
        )
        if result:
            hca_b1_max = result.result.get("b1_max", 0.0)
            hca_b2_max = result.result.get("b2_max", 0.0)
            hca_b_sum = result.result.get("b_sum", 0.0)
            hpbw_value = result.result.get("hpbw", 0.0)

            # HCA is Horizon Close Angle, BIA is Beam Intersection Angle, Θ is Beamwidgth (HPBW)"
            metrics_text += (
                f"\nSite A: HCA={hca_b1_max:.2f}°, Elev={profile.volume.antenna_elevation_angle_a:.2f}°, Θ={hpbw_value:.2f}°\n"
                f"Site B: HCA={hca_b2_max:.2f}°, Elev={profile.volume.antenna_elevation_angle_b:.2f}°, Θ={hpbw_value:.2f}°\n"
                                        f"HCA sum: {hca_b_sum:.2f}°, BIA: {intersections.beam_intersection_point.angle:.2f}°"
            )
        axes[1].text(
            0.02,
            0.98,
            metrics_text,
            transform=axes[1].transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Set Y-axis limits for curved panel
        # Calculate site positions from sight lines (where they're anchored)
        site_a_elevation = np.polyval(sight_lines.lower_a, distances[0]) / 1000
        site_b_elevation = np.polyval(sight_lines.lower_b, distances[-1]) / 1000

        all_y_values = [
            lower_plot_elev,
            upper_plot_elev,
            cross_ab_plot_elev,
            cross_ba_plot_elev,
            elevations_curved.max(),
            site_a_elevation,  # Site A antenna position
            site_b_elevation,  # Site B antenna position
            0,  # Sea level
        ]
        y_min = min(all_y_values) - 0.1  # Add 100m margin below
        y_max = max(all_y_values) + 0.3  # Add 300m margin above
        axes[1].set_ylim(y_min, y_max)

        axes[1].set_xlim(distances[0], distances[-1])

        # Save with original styling
        if save_path:
            output_dir = os.path.dirname(save_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300, facecolor="mintcream")
            plt.tight_layout()

        if show:
            plt.show()
        else:
            plt.close(fig)

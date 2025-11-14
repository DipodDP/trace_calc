"""Profile visualization service (separated from analysis logic)"""
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

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
        fig, axes = plt.subplots(2, 1, figsize=(19.20, 5.4))

        # ===== Panel 1: Plain Profile =====
        elevations, zero = profile.plain.elevations, profile.plain.baseline
        axes[0].plot(distances, elevations, "g", label="Elevation")
        axes[0].fill_between(distances, elevations, zero, facecolor="g", alpha=0.2)
        axes[0].grid(True)

        elevations_range = elevations.max() - elevations.min()
        axes[0].set_xlim(distances[0], distances[-1])
        axes[0].set_ylim(
            elevations.min() - elevations_range / 10,
            elevations.max() + elevations_range / 10,
        )

        # ===== Panel 2: Curved Profile with Sight Lines =====
        elevations_curved, zero_curved = profile.curved.elevations, profile.curved.baseline

        # Shift baseline to start at 0 for clean plotting
        shift = zero_curved[0]
        zero_curved_for_plot = zero_curved - shift
        elevations_curved_for_plot = elevations_curved - shift

        sight_1, sight_2, cross = profile.lines_of_sight
        sight_1_for_plot = np.polyval(sight_1, distances.astype(float)) - shift
        sight_2_for_plot = np.polyval(sight_2, distances.astype(float)) - shift
        cross_for_plot = (cross[0], cross[1] - shift)

        # Plot curved elevation with fill
        axes[1].plot(
            distances, elevations_curved_for_plot, "g", label="Curved Elevation"
        )
        axes[1].fill_between(
            distances,
            elevations_curved_for_plot,
            zero_curved_for_plot,
            facecolor="g",
            alpha=0.2,
        )

        # Plot sight lines
        axes[1].plot(
            distances,
            sight_1_for_plot,
            lw=1.4,
            linestyle="--",
            label="Sight Line A",
        )
        axes[1].plot(
            distances,
            sight_2_for_plot,
            lw=1.4,
            linestyle="--",
            label="Sight Line B",
        )
        axes[1].scatter(*cross_for_plot, zorder=5, label="Intersection")
        axes[1].grid(True)
        axes[1].legend()

        # Set Y-axis limits for curved panel
        y_max = cross_for_plot[1] * 1.1
        lower_limit = elevations_curved_for_plot.min() - 20
        if lower_limit > -10:
            lower_limit = -10

        axes[1].set_xlim(distances[0], distances[-1])
        axes[1].set_ylim(bottom=lower_limit, top=y_max + 20)

        # Save with original styling
        if save_path:
            fig.savefig(save_path, dpi=300, facecolor="mintcream")

        if show:
            plt.show()
        else:
            plt.close(fig)

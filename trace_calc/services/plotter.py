import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from trace_calc.models.path import ProfileData


class ProfilePlotter:
    """
    Renders elevation profiles: flat and curved with sight lines.
    """

    def __init__(self, calculations: ProfileData):
        """
        Args:
            calculations: Precomputed profile data (plain, curved, sight lines).
        """
        self.calc = calculations

    def plot(self, distances: NDArray[np.float64], path_filename: str) -> None:
        """
        Generate and save a two-panel profile chart.

        Args:
            distances: 1D array of path distances (>=2 points).
            path_filename: Base filename for saving PNG (without extension).
        Raises:
            ValueError: If distances is not a 1D array with >=2 elements.
        """
        distances = np.atleast_1d(distances)

        if distances.ndim != 1 or distances.size < 2:
            raise ValueError("`distances` must be a 1D array with at least two points.")

        fig, axes = plt.subplots(2, 1, figsize=(19.20, 5.4))

        # 1. Plain profile
        elevations: NDArray[np.float64]
        zero: NDArray[np.float64]
        elevations, zero = self.calc.plain
        axes[0].plot(distances, elevations, "g", label="Elevation")
        axes[0].fill_between(distances, elevations, zero, facecolor="g", alpha=0.2)
        axes[0].grid(True)
        elevations_range = elevations.max() - elevations.min()
        axes[0].set_xlim(distances[0], distances[-1])
        axes[0].set_ylim(
            elevations.min() - elevations_range / 10,
            elevations.max() + elevations_range / 10,
        )

        # 2. Curved profile and lines of sight
        elevations_curved: NDArray[np.float64]
        zero_curved: NDArray[np.float64]
        elevations_curved, zero_curved = self.calc.curved

        # Shift baseline to start at 0 for plotting
        shift = zero_curved[0]
        zero_curved_for_plot = zero_curved - shift
        elevations_curved_for_plot = elevations_curved - shift

        sight_1, sight_2, cross = self.calc.lines_of_sight
        sight_1_for_plot = np.polyval(sight_1, distances.astype(float)) - shift
        sight_2_for_plot = np.polyval(sight_2, distances.astype(float)) - shift
        cross_for_plot = (cross[0], cross[1] - shift)

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

        y_max = cross_for_plot[1] * 1.1

        lower_limit = elevations_curved_for_plot.min() - 20
        if lower_limit > -10:
            lower_limit = -10

        axes[1].set_xlim(distances[0], distances[-1])
        axes[1].set_ylim(bottom=lower_limit, top=y_max + 20)

        fig.savefig(f"{path_filename}.png", dpi=300, facecolor="mintcream")
        plt.close(fig)

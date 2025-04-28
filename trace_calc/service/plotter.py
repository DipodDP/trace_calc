import numpy as np
import matplotlib.pyplot as plt

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

    def plot(self, distances: np.ndarray, path_filename: str):
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
        elevations_curved, zero_curved = self.calc.curved
        sight_1, sight_2, cross = self.calc.lines_of_sight
        axes[1].plot(distances, elevations_curved, "g", label="Curved Elevation")
        axes[1].fill_between(
            distances, elevations_curved, zero_curved, facecolor="g", alpha=0.2
        )
        axes[1].plot(
            distances,
            np.polyval(sight_1, distances),
            lw=1.4,
            linestyle="--",
            label="Sight Line A",
        )
        axes[1].plot(
            distances,
            np.polyval(sight_2, distances),
            lw=1.4,
            linestyle="--",
            label="Sight Line B",
        )
        axes[1].scatter(*cross, zorder=5, label="Intersection")
        axes[1].grid(True)

        y_min = min(elevations_curved.min(), zero_curved.min())
        y_max = cross[1]
        y_range = y_max - y_min
        axes[1].set_xlim(distances[0], distances[-1])
        axes[1].set_ylim(y_min - y_range / 10, y_max + y_range / 10)

        fig.savefig(f"{path_filename}.png", dpi=300, facecolor="mintcream")
        plt.close(fig)

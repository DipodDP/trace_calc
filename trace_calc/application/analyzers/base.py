from abc import ABC, abstractmethod
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trace_calc.domain.models.units import Meters
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import HCAData, PathData, ProfileData
from trace_calc.infrastructure.visualization.plotter import ProfileVisualizer
from trace_calc.domain.constants import OUTPUT_DATA_DIR
from trace_calc.application.services.profile_data_calculator import (
    DefaultProfileDataCalculator,
)
from trace_calc.domain.interfaces import BaseAnalyzer


class PlotterMixin:
    distances: NDArray[np.float64]
    profile_data: ProfileData
    input_data: InputData
    path_data: PathData

    def draw_plot(self) -> None:
        show_plot = "PYTEST_CURRENT_TEST" not in os.environ
        visualizer = ProfileVisualizer()
        visualizer.plot_profile(
            self.path_data,
            self.profile_data,
            show=show_plot,
            save_path=f"{OUTPUT_DATA_DIR}/{self.input_data.path_name}.png",
        )


class BaseServiceAnalyzer(BaseAnalyzer, PlotterMixin):
    def __init__(self, profile: PathData, input_data: InputData):
        super().__init__(profile, input_data)

        profile_data_calculator = DefaultProfileDataCalculator(profile, input_data)
        self.profile_data = profile_data_calculator.profile_data
        self.hca_data = profile_data_calculator.hca_data
        self.path_data = profile

        hca_calc = profile_data_calculator.hca_calculator
        self.distances = hca_calc.distances
        self.elevations = hca_calc.elevations
        self.antenna_a_height = hca_calc.antenna_a_height
        self.antenna_b_height = hca_calc.antenna_b_height
        self.input_data = input_data

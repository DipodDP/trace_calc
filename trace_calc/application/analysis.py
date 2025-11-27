import logging
import os
import math
from typing import Any, Protocol
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from trace_calc.domain.models.units import (
    Angle,
    Kilometers,
    Loss,
    Meters,
    Speed,
    Degrees,
)
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import PathData, ProfileData
from trace_calc.domain.models.analysis import AnalysisResult, PropagationLoss
from trace_calc.domain.constants import OUTPUT_DATA_DIR
from trace_calc.domain.interfaces import BaseAnalyzer
from trace_calc.application.analyzers.base import BaseServiceAnalyzer


from trace_calc.logging_config import get_logger

logger = get_logger(__name__)


from trace_calc.application.analyzers.groza_analyzer import GrozaAnalyzer
from trace_calc.application.analyzers.sosnik_analyzer import SosnikAnalyzer


class BaseAnalysisService(ABC):
    """
    Base class for analysis services, providing common functionality.
    """

    async def analyze(
        self, path: PathData, input_data: InputData, **kwargs: Any
    ) -> AnalysisResult:
        analyzer = self._create_analyzer(path, input_data)
        result_data = analyzer.analyze(**kwargs)

        c = 299792458
        frequency_hz = input_data.frequency_mhz * 1e6
        wavelength_m = c / frequency_hz

        propagation_loss = self._get_propagation_loss(result_data)

        result_data["profile_data"] = analyzer.profile_data
        result_data["frequency_mhz"] = input_data.frequency_mhz
        result_data["distance_km"] = float(analyzer.distances[-1])
        result_data["hpbw"] = float(input_data.hpbw)

        return AnalysisResult(
            basic_transmission_loss=self._get_basic_transmission_loss(result_data),
            total_path_loss=self._get_total_path_loss(result_data),
            link_speed=result_data["speed"],
            wavelength=wavelength_m,
            propagation_loss=propagation_loss,
            metadata=result_data,
        )

    @abstractmethod
    def _create_analyzer(self, path: PathData, input_data: InputData) -> BaseAnalyzer:
        pass

    @abstractmethod
    def _get_propagation_loss(self, result_data: dict[str, Any]) -> PropagationLoss | None:
        pass

    @abstractmethod
    def _get_basic_transmission_loss(self, result_data: dict[str, Any]) -> Loss | None:
        pass

    @abstractmethod
    def _get_total_path_loss(self, result_data: dict[str, Any]) -> Loss | None:
        pass


class GrozaAnalysisService(BaseAnalysisService):
    """
    Analysis service specifically for the Groza model.
    """

    def _create_analyzer(self, path: PathData, input_data: InputData) -> BaseAnalyzer:
        return GrozaAnalyzer(path, input_data)

    def _get_propagation_loss(self, result_data: dict[str, Any]) -> PropagationLoss:
        return PropagationLoss(
            free_space_loss=result_data["L0"],
            atmospheric_loss=0,
            diffraction_loss=result_data["Lr"],
            total_loss=result_data["L0"] + result_data["Lr"],
        )

    def _get_basic_transmission_loss(self, result_data: dict[str, Any]) -> Loss:
        return result_data["Ltot"]

    def _get_total_path_loss(self, result_data: dict[str, Any]) -> Loss:
        return result_data["Ltot"]


class SosnikAnalysisService(BaseAnalysisService):
    """
    Analysis service specifically for the Sosnik model.
    """

    def _create_analyzer(self, path: PathData, input_data: InputData) -> BaseAnalyzer:
        return SosnikAnalyzer(path, input_data)

    def _get_propagation_loss(self, result_data: dict[str, Any]) -> PropagationLoss | None:
        return None  # Sosnik analyzer does not provide a full loss breakdown

    def _get_basic_transmission_loss(self, result_data: dict[str, Any]) -> Loss | None:
        return None

    def _get_total_path_loss(self, result_data: dict[str, Any]) -> Loss | None:
        return None

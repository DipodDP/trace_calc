from typing import Any
from abc import ABC, abstractmethod

from trace_calc.domain.models.units import Loss
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.path import PathData
from trace_calc.domain.models.analysis import AnalysisResult, PropagationLoss
from trace_calc.domain.interfaces import BaseAnalyzer
from trace_calc.logging_config import get_logger
from trace_calc.application.analyzers.groza_analyzer import GrozaAnalyzer
from trace_calc.application.analyzers.sosnik_analyzer import SosnikAnalyzer


logger = get_logger(__name__)


class BaseAnalysisService(ABC):
    """
    Base class for analysis services, providing common functionality.
    """

    async def analyze(
        self, path: PathData, input_data: InputData, **kwargs: Any
    ) -> AnalysisResult:
        analyzer = self._create_analyzer(path, input_data)
        analyzer_result = analyzer.analyze(**kwargs)

        model_params = analyzer_result.model_parameters

        result_data = {}
        if analyzer_result.hca:
            result_data.update(
                {
                    "b1_max": analyzer_result.hca.b1_max,
                    "b2_max": analyzer_result.hca.b2_max,
                    "b_sum": analyzer_result.hca.b_sum,
                }
            )

        propagation_loss_obj = self._get_propagation_loss(model_params)
        total_path_loss_val = self._get_total_path_loss(model_params)

        model_params["total_loss"] = total_path_loss_val
        result_data["frequency_mhz"] = input_data.frequency_mhz
        result_data["hpbw"] = float(input_data.hpbw)
        result_data["method"] = model_params.get("method")
        result_data["profile_data"] = analyzer_result.profile_data
        result_data["speed_prefix"] = analyzer_result.speed_prefix

        if propagation_loss_obj:
            model_params["propagation_loss"] = propagation_loss_obj

        model_params = {k: v for k, v in model_params.items() if v is not None}

        return AnalysisResult(
            link_speed=analyzer_result.link_speed,
            wavelength=analyzer_result.wavelength,
            model_propagation_loss_parameters=model_params,
            result=result_data,
        )

    @abstractmethod
    def _create_analyzer(self, path: PathData, input_data: InputData) -> BaseAnalyzer:
        pass

    @abstractmethod
    def _get_total_path_loss(self, result_data: dict[str, Any]) -> Loss | None:
        pass

    @abstractmethod
    def _get_propagation_loss(
        self, result_data: dict[str, Any]
    ) -> PropagationLoss | None:
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
            atmospheric_loss=result_data["Lmed"],
            refraction_loss=result_data["Lr"],
            diffraction_loss=result_data["Ld"],
            total_loss=result_data["Ltot"],
        )

    def _get_total_path_loss(self, result_data: dict[str, Any]) -> Loss:
        return result_data["Ltot"]


class SosnikAnalysisService(BaseAnalysisService):
    """
    Analysis service specifically for the Sosnik model.
    """

    def _create_analyzer(self, path: PathData, input_data: InputData) -> BaseAnalyzer:
        return SosnikAnalyzer(path, input_data)

    def _get_propagation_loss(
        self, result_data: dict[str, Any]
    ) -> PropagationLoss | None:
        return None

    def _get_total_path_loss(self, result_data: dict[str, Any]) -> Loss | None:
        return None

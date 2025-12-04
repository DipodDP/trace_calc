"""Domain models for propagation analysis results"""

from dataclasses import dataclass, field
from typing import Any

from trace_calc.domain.models.path import HCAData, ProfileData


@dataclass(frozen=True, slots=True)
class PropagationLoss:
    """Breakdown of signal loss components (immutable)"""

    free_space_loss: float  # dB
    atmospheric_loss: float  # dB
    diffraction_loss: float  # dB
    refraction_loss: float  # dB
    total_loss: float  # dB

    def __post_init__(self):
        """Validate that total matches sum of components"""
        expected = (
            self.free_space_loss
            + self.atmospheric_loss
            + self.diffraction_loss
            + self.refraction_loss
        )
        if abs(self.total_loss - expected) > 0.1:
            raise ValueError(
                f"Total loss {self.total_loss} dB doesn't match sum of components "
                f"({expected:.2f} dB)"
            )


@dataclass(frozen=True, slots=True)
class AnalyzerResult:
    """Explicit output structure for analyzer components"""

    model_parameters: dict[str, Any]
    link_speed: float
    wavelength: float
    hca: HCAData | None = None
    profile_data: ProfileData | None = None
    speed_prefix: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """
    Result of troposcatter propagation analysis.

    This is a pure data structure with no behavior, following
    the Domain-Driven Design principle of separating data from operations.

    All fields are immutable (frozen=True) to prevent accidental modification
    and ensure thread safety in async contexts.
    """

    link_speed: float  # Mbps (estimated throughput)
    wavelength: float  # meters (radio wavelength)
    model_propagation_loss_parameters: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""

        serializable_params = self.model_propagation_loss_parameters.copy()

        if "propagation_loss" in serializable_params and isinstance(
            serializable_params["propagation_loss"], PropagationLoss
        ):
            prop_loss = serializable_params["propagation_loss"]
            serializable_params["propagation_loss"] = {
                "free_space_loss": prop_loss.free_space_loss,
                "atmospheric_loss": prop_loss.atmospheric_loss,
                "diffraction_loss": prop_loss.diffraction_loss,
                "refraction_loss": prop_loss.refraction_loss,
                "total_loss": prop_loss.total_loss,
            }

        serializable_params = {
            k: v for k, v in serializable_params.items() if v is not None
        }

        # Clean result by removing non-serializable data
        serializable_result = {
            k: v
            for k, v in self.result.items()
            if k not in ["profile_data", "distance_km", "geo_data"]
        }
        serializable_result["wavelength"] = self.wavelength

        return {
            "link_speed": self.link_speed,
            "model_propagation_loss_parameters": serializable_params,
            "result": serializable_result,
        }

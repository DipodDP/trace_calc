"""Domain models for propagation analysis results"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PropagationLoss:
    """Breakdown of signal loss components (immutable)"""

    free_space_loss: float  # dB
    atmospheric_loss: float  # dB
    diffraction_loss: float  # dB
    total_loss: float  # dB

    def __post_init__(self):
        """Validate that total matches sum of components"""
        expected = self.free_space_loss + self.atmospheric_loss + self.diffraction_loss
        if abs(self.total_loss - expected) > 0.1:
            raise ValueError(
                f"Total loss {self.total_loss} dB doesn't match sum of components "
                f"({expected:.2f} dB)"
            )


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """
    Result of troposcatter propagation analysis.

    This is a pure data structure with no behavior, following
    the Domain-Driven Design principle of separating data from operations.

    All fields are immutable (frozen=True) to prevent accidental modification
    and ensure thread safety in async contexts.
    """

    basic_transmission_loss: float  # dB (Groza Lb or Sosnik equivalent)
    total_path_loss: float  # dB (includes all attenuation)
    link_speed: float  # Mbps (estimated throughput)
    wavelength: float  # meters (radio wavelength)
    propagation_loss: PropagationLoss | None  # Detailed loss breakdown (optional)
    metadata: dict[str, Any]  # Model-specific data (method, version, etc.)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        # Remove non-serializable objects from metadata
        metadata_serializable = dict(self.metadata)
        if "profile_data" in metadata_serializable:
            del metadata_serializable["profile_data"]

        return {
            "basic_transmission_loss": self.basic_transmission_loss,
            "total_path_loss": self.total_path_loss,
            "link_speed": self.link_speed,
            "wavelength": self.wavelength,
            "propagation_loss": (
                {
                    "free_space_loss": self.propagation_loss.free_space_loss,
                    "atmospheric_loss": self.propagation_loss.atmospheric_loss,
                    "diffraction_loss": self.propagation_loss.diffraction_loss,
                    "total_loss": self.propagation_loss.total_loss,
                }
                if self.propagation_loss
                else None
            ),
            "metadata": metadata_serializable,
        }

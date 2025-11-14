"""Propagation analysis services (pure business logic, no I/O)"""

from abc import ABC, abstractmethod

import numpy as np

from trace_calc.domain.models.path import PathData
from trace_calc.domain.models.coordinates import InputData
from trace_calc.domain.models.analysis import AnalysisResult, PropagationLoss
from trace_calc.application.services.hca import (
    HCACalculatorCC,
)  # Changed from HCAService
from trace_calc.application.services.profile_data_calculator import (
    DefaultProfileDataCalculator,
)  # Changed from ProfileCalculationService


class AnalysisService(ABC):
    """
    Abstract base class for propagation analysis strategies.

    Removed mixins (PlotterMixin, PrinterMixin) to decouple I/O from logic.
    Each analyzer now returns pure data (AnalysisResult) with no side effects.
    """

    @abstractmethod
    def analyze(
        self, path: PathData, input_data: InputData, **kwargs
    ) -> AnalysisResult:
        """
        Perform propagation analysis and return pure data.

        Returns:
            AnalysisResult: Immutable result with no side effects
        """
        pass


class GrozaAnalysisService(AnalysisService):
    """
    Groza propagation model for troposcatter analysis.

    This implementation uses empirical formulas to estimate:

    - Basic transmission loss (Lb) accounting for horizon geometry
    - Atmospheric attenuation
    - Diffraction losses over terrain obstacles
    - Link throughput estimation based on signal margin
    """

    def __init__(self):
        pass

    def analyze(
        self,
        path: PathData,
        input_data: InputData,
        antenna_a_height: float,
        antenna_b_height: float,
        transmit_power_dbm: float = 30.0,
        receiver_sensitivity_dbm: float = -85.0,
    ) -> AnalysisResult:
        """
        Perform Groza propagation analysis.

        Args:
            path: Path data for the analysis.
            input_data: User input with coordinates, frequency, etc.
            antenna_a_height: Height of antenna A above ground (meters)
            antenna_b_height: Height of antenna B above ground (meters)
            transmit_power_dbm: Transmitter power (dBm), default 30 dBm (1W)
            receiver_sensitivity_dbm: Receiver sensitivity (dBm), default -85 dBm

        Returns:
            AnalysisResult: Complete analysis with loss calculations and speed estimate
        """

        # Initialize HCA and profile data (template method)
        hca_service = HCACalculatorCC(path, input_data)
        hca_data = hca_service.calculate_hca()
        profile_calculator = DefaultProfileDataCalculator(path, input_data)
        profile_data = profile_calculator.profile_data

        # Calculate wavelength: λ = c / f
        frequency_hz = input_data.frequency_mhz * 1e6
        wavelength = 3e8 / frequency_hz  # meters

        # Extract key parameters
        distance_km = path.distances[-1]
        theta_a = (
            hca_data.b1_max
        )  # Horizon clearance angle A (Changed from hca_data.hca_a)
        theta_b = (
            hca_data.b2_max
        )  # Horizon clearance angle B (Changed from hca_data.hca_b)

        # Groza basic transmission loss formula
        # Lb = 20*log10(4πd/λ) + 30*log10(d) + 10*log10(θa*θb) + C
        # Where:
        #   d = path distance (km)
        #   λ = wavelength (m)
        #   θa, θb = horizon angles (milliradians)
        #   C = empirical constant (~-18 dB for standard atmosphere)
        free_space_loss = 20 * np.log10(4 * np.pi * distance_km * 1000 / wavelength)
        scatter_loss = 30 * np.log10(distance_km)

        # Convert theta_a and theta_b to milliradians if they are in degrees
        theta_a_mrad = theta_a * (np.pi / 180) * 1000
        theta_b_mrad = theta_b * (np.pi / 180) * 1000

        angle_factor = (
            10 * np.log10(theta_a_mrad * theta_b_mrad)
            if (theta_a_mrad > 0 and theta_b_mrad > 0)
            else 0
        )
        empirical_constant = -18.0  # Standard atmospheric conditions

        basic_loss = free_space_loss + scatter_loss + angle_factor + empirical_constant

        # Additional losses
        atmospheric_loss = self._calculate_atmospheric_loss(distance_km, frequency_hz)
        diffraction_loss = self._calculate_diffraction_loss(
            path, wavelength, antenna_a_height, antenna_b_height
        )

        # Calculate total loss for PropagationLoss dataclass
        total_loss_for_propagation_loss = (
            free_space_loss + atmospheric_loss + diffraction_loss
        )

        # Calculate total path loss for AnalysisResult
        total_path_loss = basic_loss + atmospheric_loss + diffraction_loss

        # Link margin and speed calculation
        link_margin = transmit_power_dbm - receiver_sensitivity_dbm - total_path_loss
        link_speed = self._calculate_link_speed(link_margin)

        # Create immutable result
        return AnalysisResult(
            basic_transmission_loss=basic_loss,
            total_path_loss=total_path_loss,
            link_speed=link_speed,
            wavelength=wavelength,
            propagation_loss=PropagationLoss(
                free_space_loss=free_space_loss,
                atmospheric_loss=atmospheric_loss,
                diffraction_loss=diffraction_loss,
                total_loss=total_loss_for_propagation_loss,
            ),
            metadata={
                "method": "groza",
                "version": "1.0",
                "distance_km": distance_km,
                "frequency_mhz": input_data.frequency_mhz,
                "theta_a_mrad": theta_a_mrad,
                "theta_b_mrad": theta_b_mrad,
                "link_margin_db": link_margin,
                "profile_data": profile_data,
            },
        )

    def _calculate_atmospheric_loss(
        self, distance_km: float, frequency_hz: float
    ) -> float:
        """
        Calculate atmospheric attenuation.

        Uses ITU-R P.676 approximation for oxygen and water vapor absorption.

        Reference:
            ITU-R P.676-12, "Attenuation by atmospheric gases and related effects"
        """

        # Simplified model (accurate for frequencies < 10 GHz)
        frequency_ghz = frequency_hz / 1e9

        # Specific attenuation (dB/km)
        gamma = 0.01 + 0.001 * frequency_ghz  # Approximate for standard atmosphere
        return gamma * distance_km

    def _calculate_diffraction_loss(
        self,
        path: PathData,
        wavelength: float,
        antenna_a_height: float,
        antenna_b_height: float,
    ) -> float:
        """
        Calculate single knife-edge diffraction loss.
        """
        distances_m = path.distances * 1000
        elevations_m = path.elevations

        start_elevation = elevations_m[0] + antenna_a_height
        end_elevation = elevations_m[-1] + antenna_b_height

        los_elevations = np.linspace(start_elevation, end_elevation, len(distances_m))

        obstruction_heights = elevations_m - los_elevations

        max_obstruction_idx = np.argmax(obstruction_heights)
        h = obstruction_heights[max_obstruction_idx]

        if h <= 0:
            return 0.0

        d1 = distances_m[max_obstruction_idx]
        d2 = distances_m[-1] - d1

        if d1 == 0 or d2 == 0:
            return 0.0

        v = h * np.sqrt((2 / wavelength) * (1 / d1 + 1 / d2))

        # ITU-R P.526-14 formula for J(v)
        loss = 6.9 + 20 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)

        return loss

    def _calculate_link_speed(self, link_margin_db: float) -> float:
        """
        Estimate link throughput based on SNR margin.
        Uses Shannon-Hartley theorem approximation:
        C = B * log2(1 + SNR)

        Args:
            link_margin_db: Link margin in dB

        Returns:
            Estimated throughput in Mbps
        """

        # Assume 20 MHz channel bandwidth (typical for point-to-point links)
        bandwidth_mhz = 20.0

        # Convert margin to linear SNR
        snr_linear = 10 ** (link_margin_db / 10)

        # Shannon capacity (Mbps)
        capacity = bandwidth_mhz * np.log2(1 + snr_linear)

        # Apply realistic efficiency factor (modulation/coding overhead)
        efficiency = 0.8
        return max(0.0, capacity * efficiency)


class SosnikAnalysisService(AnalysisService):
    """
    Sosnik propagation model (alternative to Groza).

    TODO: Document Sosnik model reference and differences from Groza
    """

    def __init__(self):
        pass

    def analyze(
        self, path: PathData, input_data: InputData, **kwargs
    ) -> AnalysisResult:
        """Perform Sosnik propagation analysis"""
        # TODO: Implement Sosnik-specific analysis
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "Sosnik analysis service not yet refactored. "
            "See GrozaAnalysisService for reference implementation."
        )

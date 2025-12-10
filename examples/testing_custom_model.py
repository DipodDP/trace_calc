"""
Testing Custom Models Example
===============================

This example demonstrates how to write tests for custom propagation models
and calculators using pytest.

Topics covered:
1. Setting up test fixtures for common test data
2. Testing speed calculators in isolation
3. Testing analyzers with mock data
4. Testing analysis services
5. Integration testing with the full workflow
6. Mocking API clients

Requirements:
- pytest
- pytest-asyncio
"""

import pytest
import numpy as np
from typing import Tuple
from unittest.mock import AsyncMock, Mock

from trace_calc.domain.models.coordinates import Coordinates, InputData
from trace_calc.domain.models.path import PathData, HCAData
from trace_calc.domain.models.units import Meters, Kilometers, Loss, Speed
from trace_calc.domain.models.analysis import PropagationLoss

# Import custom model from the example
# In practice, these would be in your package
from examples.custom_model import (
    DistanceBasedSpeedCalculator,
    DistanceBasedAnalyzer,
    DistanceBasedAnalysisService,
)


# ==============================================================================
# Fixtures - Reusable Test Data
# ==============================================================================

@pytest.fixture
def sample_input_data() -> InputData:
    """Fixture providing sample input data for tests."""
    return InputData(
        path_name="test_path",
        site_a_coordinates=Coordinates(55.7558, 37.6173),
        site_b_coordinates=Coordinates(59.9343, 30.3351),
        frequency_mhz=5000.0,
        antenna_a_height=Meters(30.0),
        antenna_b_height=Meters(30.0),
    )


@pytest.fixture
def sample_path_data() -> PathData:
    """Fixture providing sample path data for tests."""
    # Create a simple 100 km path with 1000 elevation points
    num_points = 1000
    distances = np.linspace(0, 100, num_points)

    # Simulate terrain: baseline + small hills
    elevations = 200.0 + 50.0 * np.sin(distances / 20.0)

    coordinates = np.array([
        [55.7558, 37.6173],  # Moscow
        [59.9343, 30.3351],  # St. Petersburg
    ])

    return PathData(
        coordinates=coordinates,
        distances=distances,
        elevations=elevations,
    )


@pytest.fixture
def sample_hca_data() -> HCAData:
    """Fixture providing sample HCA data."""
    return HCAData(
        b1_max=0.5,  # degrees
        b2_max=0.6,  # degrees
        b_sum=1.1,   # degrees
        b1_idx=100,
        b2_idx=900,
    )


# ==============================================================================
# Test Speed Calculator
# ==============================================================================

class TestDistanceBasedSpeedCalculator:
    """Test suite for the custom speed calculator."""

    @pytest.fixture
    def calculator(self) -> DistanceBasedSpeedCalculator:
        """Create calculator instance for tests."""
        return DistanceBasedSpeedCalculator()

    def test_calculate_speed_short_distance_low_loss(self, calculator):
        """Test speed calculation for short distance with low loss."""
        total_loss, link_speed = calculator.calculate_speed(
            distance_km=Kilometers(50.0),
            frequency_mhz=5000.0,
            free_space_loss_db=Loss(140.0),
            terrain_factor=1.0,
        )

        # Expected: free_space (140) + distance_penalty (5.0) + terrain (0) = 145 dB
        # This should map to 1024 kbps (< 150 dB threshold)
        assert total_loss == pytest.approx(145.0, rel=0.01)
        assert link_speed == 1024.0

    def test_calculate_speed_long_distance_high_loss(self, calculator):
        """Test speed calculation for long distance with high loss."""
        total_loss, link_speed = calculator.calculate_speed(
            distance_km=Kilometers(200.0),
            frequency_mhz=5000.0,
            free_space_loss_db=Loss(160.0),
            terrain_factor=1.5,
        )

        # Expected: 160 + 20.0 + 2.5 = 182.5 dB
        # This should map to 256 kbps (180-200 dB threshold)
        assert total_loss == pytest.approx(182.5, rel=0.01)
        assert link_speed == 256.0

    def test_calculate_speed_rough_terrain_penalty(self, calculator):
        """Test that rough terrain increases loss."""
        # Smooth terrain
        loss_smooth, speed_smooth = calculator.calculate_speed(
            distance_km=Kilometers(100.0),
            frequency_mhz=5000.0,
            free_space_loss_db=Loss(150.0),
            terrain_factor=1.0,
        )

        # Rough terrain
        loss_rough, speed_rough = calculator.calculate_speed(
            distance_km=Kilometers(100.0),
            frequency_mhz=5000.0,
            free_space_loss_db=Loss(150.0),
            terrain_factor=2.0,  # Rough terrain
        )

        # Rough terrain should have higher loss
        assert loss_rough > loss_smooth
        # May or may not affect speed depending on thresholds


# ==============================================================================
# Test Analyzer
# ==============================================================================

class TestDistanceBasedAnalyzer:
    """Test suite for the custom analyzer."""

    def test_analyzer_produces_valid_result(self, sample_path_data, sample_input_data):
        """Test that analyzer produces valid AnalyzerResult."""
        analyzer = DistanceBasedAnalyzer(sample_path_data, sample_input_data)
        result = analyzer.analyze()

        # Check result structure
        assert result.link_speed > 0
        assert result.wavelength > 0
        assert result.speed_prefix == "k"  # kbps

        # Check model parameters
        assert "method" in result.model_parameters
        assert result.model_parameters["method"] == "distance_based"
        assert "distance_km" in result.model_parameters
        assert "total_loss_db" in result.model_parameters

    def test_analyzer_uses_correct_distance(self, sample_path_data, sample_input_data):
        """Test that analyzer calculates distance correctly."""
        analyzer = DistanceBasedAnalyzer(sample_path_data, sample_input_data)
        result = analyzer.analyze()

        # Distance should be approximately 100 km (from sample_path_data)
        calculated_distance = result.model_parameters["distance_km"]
        assert calculated_distance == pytest.approx(100.0, rel=0.01)

    def test_analyzer_calculates_wavelength(self, sample_path_data, sample_input_data):
        """Test wavelength calculation."""
        analyzer = DistanceBasedAnalyzer(sample_path_data, sample_input_data)
        result = analyzer.analyze()

        # Wavelength = c / f = 3e8 / (5000e6) = 0.06 m
        expected_wavelength = 3e8 / (sample_input_data.frequency_mhz * 1e6)
        assert result.wavelength == pytest.approx(expected_wavelength, rel=0.01)


# ==============================================================================
# Test Analysis Service
# ==============================================================================

class TestDistanceBasedAnalysisService:
    """Test suite for the custom analysis service."""

    @pytest.mark.asyncio
    async def test_service_creates_correct_analyzer(self, sample_path_data, sample_input_data):
        """Test that service creates the correct analyzer type."""
        service = DistanceBasedAnalysisService()

        result = await service.analyze(
            path=sample_path_data,
            input_data=sample_input_data,
            antenna_a_height=30.0,
            antenna_b_height=30.0,
        )

        # Check that result has expected structure
        assert hasattr(result, 'link_speed')
        assert hasattr(result, 'wavelength')
        assert hasattr(result, 'model_propagation_loss_parameters')

    @pytest.mark.asyncio
    async def test_service_extracts_propagation_loss(self, sample_path_data, sample_input_data):
        """Test that service correctly extracts propagation loss."""
        service = DistanceBasedAnalysisService()

        result = await service.analyze(
            path=sample_path_data,
            input_data=sample_input_data,
            antenna_a_height=30.0,
            antenna_b_height=30.0,
        )

        # Check propagation loss
        if 'propagation_loss' in result.model_propagation_loss_parameters:
            prop_loss = result.model_propagation_loss_parameters['propagation_loss']
            assert prop_loss.free_space_loss > 0
            assert prop_loss.total_loss > 0


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestCustomModelIntegration:
    """Integration tests for the complete custom model workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_mocked_apis(self, sample_input_data):
        """Test complete workflow with mocked API clients."""
        from trace_calc.application.orchestration import OrchestrationService
        from trace_calc.application.services.profile import PathProfileService

        # Mock API clients
        mock_elevations_client = AsyncMock()
        mock_declinations_client = AsyncMock()

        # Configure mock elevations client to return sample data
        sample_elevations = 200.0 + 50.0 * np.sin(np.linspace(0, 100, 1000) / 20.0)
        mock_elevations_client.fetch_elevations = AsyncMock(return_value=sample_elevations)

        # Configure mock declinations client
        mock_declinations_client.fetch_declinations = AsyncMock(
            return_value=[1.5, 2.0]  # Sample declination values
        )

        # Create services
        profile_service = PathProfileService(
            input_data=sample_input_data,
            elevations_api_client=mock_elevations_client,
            block_size=256,
            resolution=0.05,
        )

        analysis_service = DistanceBasedAnalysisService()

        orchestrator = OrchestrationService(
            analysis_service=analysis_service,
            profile_service=profile_service,
            declinations_api_client=mock_declinations_client,
        )

        # Run analysis
        result = await orchestrator.process(
            input_data=sample_input_data,
            antenna_a_height=30.0,
            antenna_b_height=30.0,
            display_output=False,
            generate_plot=False,
        )

        # Verify result
        assert result.link_speed > 0
        assert result.wavelength > 0
        assert 'geo_data' in result.result


# ==============================================================================
# Validation Tests
# ==============================================================================

class TestCustomModelValidation:
    """Test validation and edge cases."""

    def test_speed_calculator_handles_zero_distance(self):
        """Test calculator behavior with zero distance."""
        calculator = DistanceBasedSpeedCalculator()

        total_loss, link_speed = calculator.calculate_speed(
            distance_km=Kilometers(0.0),
            frequency_mhz=5000.0,
            free_space_loss_db=Loss(100.0),
            terrain_factor=1.0,
        )

        # Should still produce valid output
        assert total_loss >= 100.0
        assert link_speed >= 0

    def test_speed_calculator_handles_extreme_loss(self):
        """Test calculator with very high loss values."""
        calculator = DistanceBasedSpeedCalculator()

        total_loss, link_speed = calculator.calculate_speed(
            distance_km=Kilometers(500.0),
            frequency_mhz=5000.0,
            free_space_loss_db=Loss(200.0),
            terrain_factor=3.0,
        )

        # Should result in zero or very low speed
        assert link_speed == 0.0  # Based on custom model thresholds


# ==============================================================================
# Running Tests
# ==============================================================================

if __name__ == "__main__":
    print("Testing Custom Model Example")
    print("=" * 50)
    print()
    print("This file contains example tests for custom propagation models.")
    print()
    print("To run these tests:")
    print("  pytest examples/testing_custom_model.py -v")
    print()
    print("To run with coverage:")
    print("  pytest examples/testing_custom_model.py --cov=examples.custom_model -v")
    print()
    print("Key testing patterns demonstrated:")
    print("1. Fixtures for reusable test data")
    print("2. Unit tests for calculators")
    print("3. Unit tests for analyzers")
    print("4. Async tests for services")
    print("5. Integration tests with mocked APIs")
    print("6. Validation and edge case testing")

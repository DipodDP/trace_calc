"""Test domain models for analysis results"""
import pytest
from trace_calc.domain.models.analysis import AnalysisResult, PropagationLoss


def test_analysis_result_creation():
    """Test creating analysis result with valid data"""
    result = AnalysisResult(
        basic_transmission_loss=120.5,
        total_path_loss=145.2,
        link_speed=100.0,
        wavelength=0.03,
        propagation_loss=None,
        metadata={"model": "groza", "version": "1.0"}
    )

    assert result.basic_transmission_loss == 120.5
    assert result.total_path_loss == 145.2
    assert result.link_speed == 100.0
    assert result.wavelength == 0.03
    assert result.metadata["model"] == "groza"


def test_analysis_result_immutable():
    """Test that AnalysisResult is immutable (frozen dataclass)"""
    result = AnalysisResult(
        basic_transmission_loss=120.5,
        total_path_loss=145.2,
        link_speed=100.0,
        wavelength=0.03,
        propagation_loss=None,
        metadata={}
    )

    with pytest.raises(AttributeError):
        result.link_speed = 200.0  # Should raise FrozenInstanceError


def test_propagation_loss_validation():
    """Test PropagationLoss dataclass with validation"""
    loss = PropagationLoss(
        free_space_loss=92.4,
        atmospheric_loss=0.5,
        diffraction_loss=12.3,
        total_loss=105.2
    )

    assert loss.free_space_loss == 92.4
    # Validate total matches sum
    expected_total = 92.4 + 0.5 + 12.3
    assert abs(loss.total_loss - expected_total) < 0.1

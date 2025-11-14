"""Test output formatting (console display)"""
import pytest
from io import StringIO
import sys

from trace_calc.infrastructure.output.formatters import ConsoleOutputFormatter
from trace_calc.domain.models.analysis import AnalysisResult, PropagationLoss


@pytest.fixture
def sample_analysis_result():
    """Create sample analysis result for testing"""
    return AnalysisResult(
        basic_transmission_loss=120.5,
        total_path_loss=145.2,
        link_speed=87.3,
        wavelength=0.3,
        propagation_loss=PropagationLoss(
            free_space_loss=92.4,
            atmospheric_loss=0.5,
            diffraction_loss=12.3,
            total_loss=105.2,
        ),
        metadata={
            "method": "groza",
            "distance_km": 100.5,
            "frequency_mhz": 1000.0,
        }
    )


def test_console_formatter_prints_summary(sample_analysis_result):
    """Test that formatter prints readable summary"""
    formatter = ConsoleOutputFormatter()

    # Capture output
    captured_output = StringIO()
    sys.stdout = captured_output

    formatter.format_result(sample_analysis_result)

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    # Verify key information is printed
    assert "GROZA Analysis Result" in output  # Changed to uppercase
    assert "120.5" in output  # Basic loss
    assert "145.2" in output  # Total loss
    assert "87.3" in output   # Link speed


def test_console_formatter_handles_missing_loss_breakdown(sample_analysis_result):
    """Test formatter handles missing propagation_loss gracefully"""
    result_without_loss = AnalysisResult(
        basic_transmission_loss=120.5,
        total_path_loss=145.2,
        link_speed=87.3,
        wavelength=0.3,
        propagation_loss=None,  # Missing!
        metadata={"method": "groza"}
    )

    formatter = ConsoleOutputFormatter()

    # Should not raise exception
    captured_output = StringIO()
    sys.stdout = captured_output
    formatter.format_result(result_without_loss)
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    assert "120.5" in output  # Still prints basic info

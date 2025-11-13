import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from trace_calc.models.path import ProfileData, ProfileViewData
from trace_calc.services.plotter import ProfilePlotter


@patch("matplotlib.pyplot.subplots")
def test_plotter_y_limits(mock_subplots):
    # Create dummy data
    distances = np.linspace(0, 100, 100)
    elevations = np.ones_like(distances) * 100
    baseline = np.zeros_like(distances)
    
    # Simulate a curve that starts at -20
    curve = np.linspace(-20, 0, 50)
    curve = np.concatenate((curve, np.flip(curve)))
    
    elevations_curved = elevations + curve
    baseline_curved = baseline + curve
    
    sight_1_coeffs = np.polyfit([0, 50], [80, 150], 1)
    sight_2_coeffs = np.polyfit([100, 50], [80, 150], 1)
    
    cross_x = 50
    cross_y = 150
    cross = (cross_x, cross_y)

    profile_data = ProfileData(
        plain=ProfileViewData(elevations, baseline),
        curved=ProfileViewData(elevations_curved, baseline_curved),
        lines_of_sight=(sight_1_coeffs, sight_2_coeffs, cross),
    )

    plotter = ProfilePlotter(profile_data)

    # Mock the figure and axes objects
    fig = MagicMock()
    ax1 = MagicMock()
    ax2 = MagicMock()
    mock_subplots.return_value = (fig, [ax1, ax2])

    # Call the plot method
    plotter.plot(distances, "test_plot")

    # Get the call arguments for set_ylim on the second axes
    _, kwargs = ax2.set_ylim.call_args

    # Assert the y-limits
    shift = baseline_curved[0] # -20
    
    # lower_limit = (elevations_curved - shift).min() - 20
    # (100 + curve - (-20)).min() - 20 = (120 + curve).min() - 20 = (120 - 20) - 20 = 80
    # if 80 > -10, lower_limit = -10
    
    y_max = cross[1] - shift # 150 - (-20) = 170
    top = y_max + 20 # 190

    assert kwargs["bottom"] == -10
    assert kwargs["top"] == pytest.approx(207.0)

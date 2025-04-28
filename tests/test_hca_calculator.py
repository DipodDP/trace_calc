import numpy as np
from trace_calc.models.input_data import InputData
from trace_calc.models.path import PathData
from trace_calc.service.hca_calculator import HCACalculatorDefault, HCACalculatorFFT

# Set fixed seed for reproducibility
np.random.seed(42)

# Create a random NDArray with shape (100,) of type float64 (default for np.random.rand)
random_array = np.random.rand(256)
# Scale the array values to be from 10 to 250
min_val, max_val = 135, 175
scaled_array = random_array * (max_val - min_val) + min_val
# Define a simple moving average filter with a window size
window_size = 5
kernel = np.ones(window_size) / window_size

# Use mode 'same' to maintain the original array size
test_elevations = np.convolve(scaled_array, kernel, mode="same")
test_elevations[0] = test_elevations[0] + 70
test_elevations[-1] = test_elevations[-1] + 70

test_profile = PathData(
    coordinates=np.linspace(123.10, 234.50, 256),
    distances=np.linspace(0, 100, 256),
    elevations=test_elevations,
)


def test_hca_calculator_default():
    result = HCACalculatorDefault(test_profile, InputData("test_file"))
    assert f"{result.hca_data.b_sum:.3f}" == "0.552"


def test_hca_calculator_fft():
    result = HCACalculatorFFT(test_profile, InputData("test_file"))
    assert f"{result.hca_data.b_sum:.3f}" == "0.352"

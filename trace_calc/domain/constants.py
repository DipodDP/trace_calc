"""Constants used across the application."""

# Output directory
OUTPUT_DATA_DIR = "output_data"

# Physical constants
EARTH_RADIUS_KM = 6371.0  # Earth's mean radius in kilometers
SPEED_OF_LIGHT = 299792458  # Speed of light in m/s

# Curvature constants (used in domain calculations)
CURVATURE_SCALE = 12.742  # Empirical curvature scale factor
GEOMETRIC_CURVATURE_SCALE = CURVATURE_SCALE * 1000  # For geometric calculations

"""Constants used across the application."""

import os
from pathlib import Path

# Output directory - configurable via environment variable
# Default: 'output_data' in current working directory
# Best practice: Set OUTPUT_DATA_DIR in .env to absolute path in your project root
OUTPUT_DATA_DIR = os.getenv("OUTPUT_DATA_DIR", str(Path.cwd() / "output_data"))

# Physical constants
EARTH_RADIUS_KM = 6371.0  # Earth's mean radius in kilometers
SPEED_OF_LIGHT = 299792458  # Speed of light in m/s

# Curvature constants (used in domain calculations)
CURVATURE_SCALE = 12.742  # Empirical curvature scale factor

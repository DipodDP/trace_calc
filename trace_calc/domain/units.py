# trace_calc/domain/units.py
from typing import NewType

# Base numeric types
Meters = NewType("Meters", float)
Kilometers = NewType("Kilometers", float)
Degrees = NewType("Degrees", float)

# Semantic types
Distance = NewType("Distance", Kilometers)
Elevation = NewType("Elevation", Meters)
Angle = NewType("Angle", Degrees)
Loss = NewType("Loss", float) # in dB
Speed = NewType("Speed", float) # in Mbps or kbps

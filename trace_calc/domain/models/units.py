# trace_calc/domain/units.py
"""
Type-safe unit definitions for troposcatter calculations.

This module uses NewType to create distinct types for different units,
helping catch unit conversion errors at type-checking time.

Usage:
    from trace_calc.domain.models.units import Meters, Kilometers, Degrees

    def calculate_distance(height: Meters, angle: Degrees) -> Kilometers:
        # Type checker ensures units are correct
        return Kilometers(height * math.tan(math.radians(angle)) / 1000)
"""

from typing import NewType

# Base physical units
Meters = NewType("Meters", float)  # Distance in meters
Kilometers = NewType("Kilometers", float)  # Distance in kilometers
Degrees = NewType("Degrees", float)  # Angle in degrees

# Semantic types (domain-specific meanings)
Distance = NewType("Distance", Kilometers)  # Path distance
Elevation = NewType("Elevation", Meters)  # Terrain elevation above sea level
Angle = NewType("Angle", Degrees)  # Angular measurement
Loss = NewType("Loss", float)  # Signal loss in dB
Speed = NewType("Speed", float)  # Link speed in Mbps or kbps

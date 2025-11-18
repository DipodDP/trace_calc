# PLAN

## Overview

This plan adds advanced terrain visibility analysis to the existing radio path profiling system. The enhancement calculates multiple sight lines with angular offsets, computes their intersections, and determines volumetric metrics for the region bounded by these lines.

### Current State

The system currently:
- Calculates lower sight lines from Site A and Site B to terrain obstacles
- Computes single intersection point between lower lines
- Applies Earth curvature corrections (geometric and empirical)
- Uses HCA (Horizon Clearance Angle) for obstacle identification
- Follows clean architecture: Domain → Application → Infrastructure

### Enhancement Goals

Add capability to:
1. Generate upper sight lines with configurable elevation angle offset
2. Calculate 4 intersection points: Lower_A×Lower_B, Upper_A×Upper_B, Upper_A×Lower_B, Upper_B×Lower_A
3. Compute intersection heights relative to sea level and terrain
4. Calculate volume of 3D cone intersection region
5. Measure distances from origin points to cross-intersections
6. Measure distance between cross-intersection points
7. Visualize all lines and intersections
8. Output comprehensive metrics in console and JSON formats

---

## Architecture

### Layer Organization

**Domain Layer** (Pure business logic, no dependencies)
- `trace_calc/domain/geometry.py` - NEW: Geometric calculations
- `trace_calc/domain/models/path.py` - MODIFIED: Extended data structures
- `trace_calc/domain/models/input.py` - MODIFIED: Add angle offset parameter

**Application Layer** (Use cases, orchestration)
- `trace_calc/application/services/profile_data_calculator.py` - MODIFIED: Extended calculations

**Infrastructure Layer** (External I/O, frameworks)
- `trace_calc/infrastructure/visualization/plotter.py` - MODIFIED: Enhanced visualization
- `trace_calc/infrastructure/output/console_output.py` - MODIFIED: Extended output
- `trace_calc/infrastructure/output/json_output.py` - MODIFIED: Extended JSON schema

**Test Layer**
- `tests/unit/domain/test_geometry.py` - NEW: Geometry unit tests
- `tests/unit/application/test_profile_data_calculator.py` - MODIFIED: Extended tests
- `tests/integration/test_extended_visibility.py` - NEW: Integration tests

### Design Principles

1. **Immutability**: All data structures use NamedTuple or frozen dataclass
2. **Type Safety**: Strong typing with NewType for domain units
3. **Separation of Concerns**: Geometric logic isolated in domain layer
4. **Testability**: Pure functions with deterministic outputs
5. **Backward Compatibility**: Existing functionality preserved, new features additive

---

## Data Structures

### New Domain Models

#### IntersectionPoint
```
Purpose: Represents a point where two sight lines intersect
Type: NamedTuple
Location: trace_calc/domain/models/path.py

Fields:
  - distance_km: float
    Description: Distance along path from Site A
    Unit: kilometers
    Range: [0, total_path_distance]
    Precision: 3 decimal places (meter-level)

  - elevation_sea_level: float
    Description: Absolute height above sea level
    Unit: meters
    Range: [-500, 10000] (Dead Sea to above Everest)
    Precision: 2 decimal places (centimeter-level)

  - elevation_terrain: float
    Description: Relative height above terrain at this distance
    Unit: meters
    Range: [-1000, 10000]
    Precision: 2 decimal places
    Calculation: elevation_sea_level - interpolated_terrain_elevation
```

#### SightLinesData
```
Purpose: Container for all four sight line equations
Type: NamedTuple
Location: trace_calc/domain/models/path.py

Fields:
  - lower_a: NDArray[np.float64]
    Description: Lower sight line from Site A
    Shape: (2,)
    Format: [k, b] representing y = k*x + b
    Unit: k in meters/km, b in meters

  - lower_b: NDArray[np.float64]
    Description: Lower sight line from Site B
    Shape: (2,)
    Format: Same as lower_a

  - upper_a: NDArray[np.float64]
    Description: Upper sight line from Site A (rotated from lower_a)
    Shape: (2,)
    Format: Same as lower_a

  - upper_b: NDArray[np.float64]
    Description: Upper sight line from Site B (rotated from lower_b)
    Shape: (2,)
    Format: Same as lower_a

Constraints:
  - All slopes must be finite (no vertical lines)
  - Slopes typically in range [-10, 10] for realistic terrain
```

#### IntersectionsData
```
Purpose: All intersection points between sight lines
Type: NamedTuple
Location: trace_calc/domain/models/path.py

Fields:
  - lower_intersection: IntersectionPoint
    Description: Where lower_a crosses lower_b
    Semantics: Current system's primary intersection point

  - upper_intersection: IntersectionPoint
    Description: Where upper_a crosses upper_b
    Semantics: Intersection of elevated sight lines

  - cross_ab: IntersectionPoint
    Description: Where upper_a crosses lower_b
    Semantics: One boundary of analysis region

  - cross_ba: IntersectionPoint
    Description: Where upper_b crosses lower_a
    Semantics: Other boundary of analysis region

Invariants:
  - All intersections must have distance_km within path bounds
  - cross_ab and cross_ba typically bracket lower and upper intersections
  - All elevation values must be finite
```

#### VolumeData
```
Purpose: Volumetric and distance metrics for analysis region
Type: NamedTuple
Location: trace_calc/domain/models/path.py

Fields:
  - cone_intersection_volume_m3: float
    Description: Volume of 3D region bounded by four lines
    Unit: cubic meters
    Range: [0, 1e12] (sanity limit: 1000 km³)
    Precision: Whole cubic meters (0 decimal places)
    Method: Numerical integration with circular cross-section approximation

  - distance_a_to_cross_ab: float
    Description: Distance from Site A to cross_ab intersection
    Unit: kilometers
    Range: [0, total_path_distance]
    Calculation: cross_ab.distance_km (direct)

  - distance_b_to_cross_ba: float
    Description: Distance from Site B to cross_ba intersection
    Unit: kilometers
    Range: [0, total_path_distance]
    Calculation: total_path_distance - cross_ba.distance_km

  - distance_between_crosses: float
    Description: Separation between cross_ab and cross_ba
    Unit: kilometers
    Range: [0, total_path_distance]
    Calculation: |cross_ab.distance_km - cross_ba.distance_km|

Constraints:
  - volume >= 0 (non-negative)
  - All distances >= 0 and <= total_path_distance
```

### Modified Domain Models

#### ProfileData (EXTENDED)
```
Location: trace_calc/domain/models/path.py
Change Type: Field modifications

OLD Structure:
  - plain: ProfileViewData
  - curved: ProfileViewData
  - lines_of_sight: tuple[NDArray, NDArray, tuple[float, float]]

NEW Structure:
  - plain: ProfileViewData (UNCHANGED)
  - curved: ProfileViewData (UNCHANGED)
  - lines_of_sight: SightLinesData (TYPE CHANGED)
  - intersections: IntersectionsData (NEW FIELD)
  - volume: VolumeData (NEW FIELD)

Migration:
  - Old code accessing lines_of_sight as tuple must be updated
  - New fields optional initially for backward compatibility testing
```

#### CalculationInput (EXTENDED)
```
Location: trace_calc/domain/models/input.py
Change Type: Add field with validation

NEW Field:
  - elevation_angle_offset: Angle = Angle(2.5)
    Description: Angular offset for upper sight lines
    Default: 2.5 degrees
    Range: [0.0, 45.0] degrees
    Validation: Must be non-negative and <= 45
    Purpose: Controls separation between lower and upper lines

Validation Logic (in __post_init__):
  if elevation_angle_offset < 0:
    raise ValueError("elevation_angle_offset must be non-negative")
  if elevation_angle_offset > 45:
    raise ValueError("elevation_angle_offset must be <= 45 degrees")

User Control:
  - User can specify any value in valid range
  - Default suitable for most radio propagation analysis
  - Zero value produces upper == lower (for testing)
```

---

## Functions and Algorithms

### Geometry Module (NEW)

Location: `trace_calc/domain/geometry.py`
Purpose: Pure geometric calculations with no business logic dependencies

#### rotate_line_by_angle
```
Purpose: Rotate a line equation by given angle around pivot point

Input Parameters:
  - line_coeffs: NDArray[np.float64]
    Shape: (2,)
    Content: [k, b] for y = k*x + b
  - pivot_point: tuple[float, float]
    Format: (x0, y0) in km, meters
    Semantics: Point that must remain on rotated line
  - angle_degrees: float
    Unit: degrees
    Sign: positive = counterclockwise
    Range: [-90, 90] (enforced)

Output:
  - NDArray[np.float64]
    Shape: (2,)
    Content: [k_new, b_new] for rotated line

Algorithm:
  1. Validate inputs:
     - line_coeffs.shape == (2,)
     - |k| < 1e6 (not vertical)
     - angle in [-90, 90]

  2. Extract current parameters:
     k, b = line_coeffs
     x0, y0 = pivot_point

  3. CRITICAL: Convert slope units from m/km to m/m for angle calculation:
     k_actual = k / 1000  # Slope is stored as m/km, convert to geometric slope m/m

  4. Calculate current angle:
     α = arctan(k_actual)

  5. Determine rotation direction:
     θ = angle_degrees × π/180
     IF k < 0 (descending line):
       # For descending lines, subtract offset to make less steep (keeps line above)
       α_new = α - θ
     ELSE (ascending line or horizontal):
       # For ascending lines, add offset to make steeper
       α_new = α + θ

  6. Calculate new geometric slope:
     k_new_actual = tan(α_new)

  7. Convert slope back to m/km units:
     k_new = k_new_actual × 1000

  8. Validate new slope:
     if |k_new| > 1e6: raise ValueError

  9. Calculate new intercept:
     Line must pass through pivot:
     y0 = k_new × x0 + b_new
     b_new = y0 - k_new × x0

  10. Return [k_new, b_new]

Edge Cases:
  - Input slope too steep (|k| > 1e6): ValueError "Cannot rotate vertical line"
  - Result slope too steep (|k_new| > 1e6): ValueError "Near-vertical result"
  - Angle out of range: ValueError
  - 90° rotation: tan(α + 90°) → ±∞, caught by steep check

Critical Implementation Notes:
  - **Unit Conversion is Essential**: Slopes are stored as m/km but angles must be
    calculated from geometric slopes (m/m). Failure to convert will result in
    angles near 90° and incorrect rotations.
  - **Descending Lines Need Opposite Rotation**: For lines with negative slopes
    (descending from right to left), subtracting the angle offset keeps the upper
    line geometrically above the lower line. Adding the offset would flip the line
    to ascending and place it below.
  - **Verification**: After rotation, verify that upper line is above lower line
    at multiple points along the path.

Performance:
  - Time: O(1), < 1 microsecond
  - Space: O(1), allocates single 2-element array
```

#### find_line_intersection
```
Purpose: Calculate intersection point of two non-parallel lines

Input Parameters:
  - line1_coeffs: NDArray[np.float64]
    Shape: (2,)
    Content: [k1, b1]
  - line2_coeffs: NDArray[np.float64]
    Shape: (2,)
    Content: [k2, b2]

Output:
  - tuple[float, float]
    Format: (x, y) intersection coordinates

Algorithm:
  1. Validate inputs:
     - Both arrays shape (2,)

  2. Extract coefficients:
     k1, b1 = line1_coeffs
     k2, b2 = line2_coeffs

  3. Check parallelism:
     if |k1 - k2| < 1e-9:
       if |b1 - b2| < 1e-9: raise ValueError "Coincident"
       else: raise ValueError "Parallel"

  4. Calculate intersection:
     Solve: k1×x + b1 = k2×x + b2
     x = (b2 - b1) / (k1 - k2)
     y = k1 × x + b1

  5. Validate result:
     if not finite(x) or not finite(y): raise ValueError

  6. Return (x, y)

Edge Cases:
  - Parallel lines (k1 ≈ k2, b1 ≠ b2): ValueError
  - Coincident lines (k1 ≈ k2, b1 ≈ b2): ValueError
  - Near-vertical lines (large k values): Handled by finite check
  - Intersection at infinity: Caught by parallel check

Mathematical Properties:
  - Solution unique for non-parallel lines
  - Symmetric: intersection(L1, L2) == intersection(L2, L1)
  - Commutative in result location

Performance:
  - Time: O(1), < 1 microsecond
  - Space: O(1), returns tuple
```

#### calculate_height_above_terrain
```
Purpose: Interpolate terrain elevation at distance and compute relative height

Input Parameters:
  - distance_km: float
    Description: Query point along path
    Range: [distances[0], distances[-1]]
  - elevation_sea_level: float
    Description: Absolute elevation at query point
    Unit: meters
  - distances: NDArray[np.float64]
    Description: Path distance array
    Properties: Monotonically increasing
  - elevations: NDArray[np.float64]
    Description: Terrain elevation array
    Properties: Same shape as distances

Output:
  - float
    Description: Height above terrain
    Unit: meters
    Sign: Positive = above, negative = below

Algorithm:
  1. Validate inputs:
     - distances.shape == elevations.shape
     - distance_km in [distances[0], distances[-1]]

  2. Find bracketing indices:
     idx_right = searchsorted(distances, distance_km)

  3. Handle boundary cases:
     if idx_right == 0:
       terrain_elev = elevations[0]
     elif idx_right >= len(distances):
       terrain_elev = elevations[-1]

  4. Otherwise interpolate:
     idx_left = idx_right - 1
     t = (distance_km - distances[idx_left]) /
         (distances[idx_right] - distances[idx_left])
     terrain_elev = elevations[idx_left] +
                    t × (elevations[idx_right] - elevations[idx_left])

  5. Calculate relative height:
     height = elevation_sea_level - terrain_elev

  6. Validate result:
     if not finite(height): raise ValueError

  7. Return height

Edge Cases:
  - Query at exact array point: Returns exact elevation (no interpolation)
  - Query outside bounds: ValueError
  - Non-monotonic distances: searchsorted gives undefined result (validate earlier)
  - NaN in elevations: Propagates to result, caught by finite check

Interpolation Method:
  - Linear interpolation (order 1)
  - Sufficient for typical path profiles sampled every 100-500m
  - Could upgrade to cubic spline if higher accuracy needed

Performance:
  - Time: O(log n) due to searchsorted
  - Space: O(1)
  - Typical: < 10 microseconds for n=1000
```

#### calculate_cone_intersection_volume
```
Purpose: Numerically integrate volume of 3D region bounded by four sight lines

Input Parameters:
  - lower_a, lower_b, upper_a, upper_b: NDArray[np.float64]
    Description: Four line coefficients [k, b]
  - distances: NDArray[np.float64]
    Description: Path distance array
  - lower_intersection_x: float
    Description: x-coordinate of lower intersection
  - upper_intersection_x: float
    Description: x-coordinate of upper intersection
  - cross_ab_x: float
    Description: x-coordinate of upper_a × lower_b
  - cross_ba_x: float
    Description: x-coordinate of upper_b × lower_a

Output:
  - float
    Description: Volume in cubic meters
    Range: [0, 1e12]

Algorithm:
  1. Determine integration bounds:
     x_min = min(cross_ba_x, cross_ab_x)
     x_max = max(cross_ba_x, cross_ab_x)

  2. Handle degenerate case:
     if |x_max - x_min| < 1e-6:
       return 0.0

  3. Generate sample points:
     n_samples = max(100, int((x_max - x_min) × 10))
     # At least 10 samples per km
     x_samples = linspace(x_min, x_max, n_samples)

  4. For each sample x_i:
     a. Evaluate all four lines:
        y_lower_a = k_lower_a × x_i + b_lower_a
        y_lower_b = k_lower_b × x_i + b_lower_b
        y_upper_a = k_upper_a × x_i + b_upper_a
        y_upper_b = k_upper_b × x_i + b_upper_b

     b. Determine vertical extent:
        y_bottom = min(y_lower_a, y_lower_b)
        y_top = max(y_upper_a, y_upper_b)
        height = y_top - y_bottom

     c. Validate height:
        if height < 0: raise ValueError "Invalid geometry"

     d. Calculate cross-sectional area:
        # Simplified circular approximation
        # Assumes radial symmetry around path axis
        area = π × height²

     e. Validate area:
        if area > 1e10: raise ValueError "Integration unstable"

     f. Store area[i] = area

  5. Numerical integration:
     volume = trapz(areas, x_samples × 1000)
     # Multiply x by 1000 to convert km to m

  6. Validate result:
     if volume < 0: raise ValueError
     if volume > 1e12: raise ValueError "Sanity check failed"

  7. Return volume

Volume Model:
  - Treats sight lines as generatrices of cones
  - Assumes circular cross-sections perpendicular to path
  - Approximation valid for:
    * Small angular spreads (< 10°)
    * Straight paths (< 100 km)
    * Modest terrain variations

Limitations:
  - Ignores Earth curvature in 3D (applies only to 2D profile)
  - Assumes radial symmetry (may underestimate for asymmetric terrain)
  - Does not account for actual Fresnel zone shape (ellipsoid)

Refinement Options:
  - Use actual azimuth variations for elliptical cross-sections
  - Apply 3D curvature correction
  - Sample terrain in 3D grid for more accurate bounds

Edge Cases:
  - Degenerate geometry (all intersections at same point): volume = 0
  - Lines diverge (no bounded region): Caught by negative height check
  - Integration numerical instability: Caught by area limit check

Performance:
  - Time: O(n_samples) ≈ O(path_length)
  - Typical: 10-1000 samples, 1-100 ms
  - Space: O(n_samples) for temporary arrays
  - Vectorized numpy operations for efficiency
```

#### calculate_distance_between_points
```
Purpose: Euclidean distance between two points in mixed units

Input Parameters:
  - point_a: tuple[float, float]
    Format: (x, y) where x in km, y in meters
  - point_b: tuple[float, float]
    Format: (x, y) where x in km, y in meters

Output:
  - float
    Description: Distance in kilometers

Algorithm:
  1. Extract coordinates:
     x1, y1 = point_a
     x2, y2 = point_b

  2. Calculate deltas:
     dx = x2 - x1  # km
     dy = (y2 - y1) / 1000  # convert meters to km

  3. Calculate distance:
     distance = sqrt(dx² + dy²)

  4. Return distance

Edge Cases:
  - Same point: returns 0.0
  - Very small distances: Limited by float precision (~1e-15 km = 1 femtometer)

Performance:
  - Time: O(1), < 1 microsecond
  - Space: O(1)

Note: This is straight-line distance, not great circle distance
```

### ProfileDataCalculator Extensions (MODIFIED)

Location: `trace_calc/application/services/profile_data_calculator.py`

#### _calculate_upper_lines (NEW METHOD)
```
Purpose: Generate upper sight line by rotating lower line around pivot

Input Parameters:
  - lower_line_coeffs: NDArray[np.float64]
    Description: Lower sight line [k, b]
  - pivot_point: tuple[float, float]
    Description: Site coordinates (distance, elevation)
  - angle_offset: Angle
    Description: Rotation angle

Output:
  - NDArray[np.float64]
    Description: Upper sight line [k_upper, b_upper]

Algorithm:
  1. Call geometry module:
     try:
       upper_coeffs = geometry.rotate_line_by_angle(
         lower_line_coeffs,
         pivot_point,
         float(angle_offset)
       )
     except ValueError as e:
       raise ValueError(f"Failed to calculate upper line: {e}")

  2. Return upper_coeffs

Error Handling:
  - Wraps geometry module errors with context
  - Preserves original exception as cause
```

#### _calculate_all_intersections (NEW METHOD)
```
Purpose: Calculate all four intersection points between sight lines

Input Parameters:
  - sight_lines: SightLinesData
    Description: All four line coefficients
  - distances: NDArray[np.float64]
    Description: Path distance array
  - elevations: NDArray[np.float64]
    Description: Terrain elevation array (curvature-corrected)

Output:
  - IntersectionsData
    Description: All four intersection points

Algorithm:
  1. Calculate lower intersection:
     x, y = geometry.find_line_intersection(
       sight_lines.lower_a, sight_lines.lower_b
     )
     Validate: x in [0, distances[-1]]
     h = geometry.calculate_height_above_terrain(x, y, distances, elevations)
     lower_int = IntersectionPoint(x, y, h)

  2. Calculate upper intersection:
     [Same process with upper_a, upper_b]

  3. Calculate cross_ab:
     [Same process with upper_a, lower_b]

  4. Calculate cross_ba:
     [Same process with upper_b, lower_a]

  5. Return IntersectionsData(lower_int, upper_int, cross_ab, cross_ba)

Validation:
  - Each intersection must be within path bounds
  - All coordinates must be finite
  - Raises ValueError with specific context if validation fails

Error Handling:
  - Catches geometry module errors
  - Adds context about which intersection failed
  - Preserves chain for debugging
```

#### _calculate_volume_metrics (NEW METHOD)
```
Purpose: Calculate volume and distance metrics

Input Parameters:
  - sight_lines: SightLinesData
  - distances: NDArray[np.float64]
  - intersections: IntersectionsData

Output:
  - VolumeData

Algorithm:
  1. Calculate volume:
     volume = geometry.calculate_cone_intersection_volume(
       sight_lines.lower_a,
       sight_lines.lower_b,
       sight_lines.upper_a,
       sight_lines.upper_b,
       distances,
       intersections.lower_intersection.distance_km,
       intersections.upper_intersection.distance_km,
       intersections.cross_ab.distance_km,
       intersections.cross_ba.distance_km
     )

  2. Calculate distance from A to cross_ab:
     distance_a = intersections.cross_ab.distance_km

  3. Calculate distance from B to cross_ba:
     total_distance = distances[-1]
     distance_b = total_distance - intersections.cross_ba.distance_km

  4. Calculate distance between crosses:
     distance_between = abs(
       intersections.cross_ab.distance_km -
       intersections.cross_ba.distance_km
     )

  5. Validate all metrics:
     if any distance < 0: raise ValueError

  6. Return VolumeData(volume, distance_a, distance_b, distance_between)
```

#### calculate_all (MODIFIED METHOD)
```
Change Type: Add parameter, extend logic, change return type

NEW Parameter:
  - angle_offset: Angle = Angle(0.0)
    Default: 0.0 (upper == lower for backward compatibility)

Algorithm Changes:
  [After existing lower sight line calculation]

  NEW CODE BLOCK:
    # Determine pivot points (site locations with antenna heights)
    pivot_a = (distances[0], elevations_curved[0] + offset_start)
    pivot_b = (distances[-1], elevations_curved[-1] + offset_end)

    # Calculate upper sight lines
    coeff1_upper = self._calculate_upper_lines(coeff1, pivot_a, angle_offset)
    coeff2_upper = self._calculate_upper_lines(coeff2, pivot_b, angle_offset)

    # Assemble sight lines data
    sight_lines = SightLinesData(coeff1, coeff2, coeff1_upper, coeff2_upper)

    # Calculate all intersections
    intersections = self._calculate_all_intersections(
      sight_lines, distances, elevations_curved
    )

    # Calculate volume metrics
    volume = self._calculate_volume_metrics(
      sight_lines, distances, intersections
    )

  [Modify return statement]
    return ProfileData(
      plain=plain,
      curved=curved,
      lines_of_sight=sight_lines,  # Changed from tuple to SightLinesData
      intersections=intersections,  # NEW
      volume=volume  # NEW
    )

Backward Compatibility:
  - Default angle_offset=0.0 produces upper == lower
  - Existing callers without angle_offset parameter continue to work
  - New fields always present in ProfileData (may have trivial values)

Dependencies:
  - Requires offset_start, offset_end from existing code
  - Requires elevations_curved from existing code
  - Must be called after lower sight line calculation
```

---

## Visualization Enhancements

Location: `trace_calc/infrastructure/visualization/plotter.py`

### Changes to Profile Plot

Current State:
- Panel 1: Plain elevation profile
- Panel 2: Curved profile with lower sight lines and intersection

Enhanced State:
- Panel 1: Unchanged
- Panel 2: Curved profile with:
  * Lower sight lines (solid, current colors)
  * Upper sight lines (dashed, matching colors)
  * Lower intersection (green circle)
  * Upper intersection (purple circle)
  * Cross AB intersection (orange triangle)
  * Cross BA intersection (cyan inverted triangle)
  * Enhanced legend

Implementation:
```
After plotting lower sight lines in Panel 2:

1. Extract sight lines:
   sight_lines = profile.lines_of_sight

2. Calculate upper line arrays:
   upper_line_1 = polyval(sight_lines.upper_a, distances)
   upper_line_2 = polyval(sight_lines.upper_b, distances)

3. Plot upper lines:
   ax.plot(distances, upper_line_1, '--', color='red',
           linewidth=1.5, label='Upper sight line A', alpha=0.7)
   ax.plot(distances, upper_line_2, '--', color='blue',
           linewidth=1.5, label='Upper sight line B', alpha=0.7)

4. Extract intersections:
   intersections = profile.intersections

5. Plot intersection points:
   # Lower intersection (existing, may need relabel)
   ax.scatter(
     intersections.lower_intersection.distance_km,
     intersections.lower_intersection.elevation_sea_level,
     c='green', s=100, marker='o', label='Lower intersection', zorder=5
   )

   # Upper intersection
   ax.scatter(
     intersections.upper_intersection.distance_km,
     intersections.upper_intersection.elevation_sea_level,
     c='purple', s=100, marker='o', label='Upper intersection', zorder=5
   )

   # Cross AB
   ax.scatter(
     intersections.cross_ab.distance_km,
     intersections.cross_ab.elevation_sea_level,
     c='orange', s=80, marker='^', label='Cross AB', zorder=5
   )

   # Cross BA
   ax.scatter(
     intersections.cross_ba.distance_km,
     intersections.cross_ba.elevation_sea_level,
     c='cyan', s=80, marker='v', label='Cross BA', zorder=5
   )

6. Update legend:
   ax.legend(loc='best', fontsize=8)

7. Scale y-axis to include all intersections:
   # Collect all y-values that need to be visible
   all_y_values = [
     intersections.lower_intersection.elevation_sea_level,
     intersections.upper_intersection.elevation_sea_level,
     intersections.cross_ab.elevation_sea_level,
     intersections.cross_ba.elevation_sea_level,
     elevations_curved.max(),
     site_a_elevation + antenna_a_height,
     site_b_elevation + antenna_b_height,
     0  # Sea level
   ]
   y_min = min(all_y_values) - 100  # 100m margin
   y_max = max(all_y_values) + 100  # 100m margin
   ax.set_ylim(y_min, y_max)

CRITICAL: Baseline Handling and Sea Level Reference
  - **DO NOT apply baseline shift**: Previous implementations shifted elevations by
    subtracting the first point's elevation to start plots at 0. This is INCORRECT
    as it obscures the true sea level reference.
  - **Use actual elevations**: All elevations (terrain, sight lines, intersections)
    must be plotted at their true elevation above sea level (ASL).
  - **Sea level at 0m**: The baseline (y=0) represents sea level. Terrain should be
    filled from 0 to elevation values.
  - **Y-axis scaling**: Scale the y-axis to include all intersection points,
    especially the upper intersection which may be thousands of meters above terrain.
  - **Why this matters**: Extended visibility analysis creates intersection points
    that can be far above the terrain. Without proper scaling and sea level reference,
    the plot becomes confusing and appears to show negative elevations.

Optional Enhancements:
  - Fill region between lines with semi-transparent color
  - Add text annotations with intersection coordinates
  - Draw vertical lines from intersections to terrain
  - Add grid for easier coordinate reading
```

---

## Output Format Specifications

### Console Output

Location: `trace_calc/infrastructure/output/console_output.py`

New Function: `format_extended_visibility_results(profile: ProfileData) -> str`

Output Format:
```
=== Extended Terrain Visibility Analysis ===

Lower Sight Lines:
  Site A → Obstacle: slope=0.0234, intercept=123.45m
  Site B → Obstacle: slope=-0.0187, intercept=456.78m
  Intersection: 12.345 km, 234.56m ASL, +45.67m above terrain

Upper Sight Lines (offset: 2.5°):
  Site A (upper): slope=0.0278, intercept=125.34m
  Site B (upper): slope=-0.0231, intercept=458.90m
  Intersection: 13.456 km, 267.89m ASL, +52.34m above terrain

Cross Intersections:
  Upper A × Lower B: 14.567 km, 245.67m ASL, +38.90m above terrain
  Upper B × Lower A: 11.234 km, 223.45m ASL, +41.23m above terrain

Volume Metrics:
  Cone intersection volume: 2,450,000 m³
  Distance from A to Upper A × Lower B: 14.567 km
  Distance from B to Upper B × Lower A: 8.766 km
  Distance between cross intersections: 3.333 km
```

Formatting Rules:
- Slopes: 4 decimal places
- Elevations: 2 decimal places + "m" suffix
- Distances: 3 decimal places + "km" suffix
- Volume: comma-separated, 0 decimal places + "m³" suffix
- Heights above terrain: Sign prefix (+/-), 2 decimal places

### JSON Output

Location: `trace_calc/infrastructure/output/json_output.py`

Extension to Existing JSON Structure:

```json
{
  "profile": {
    "plain": { ... existing ... },
    "curved": { ... existing ... },
    "sight_lines": {
      "lower_a": [k, b],
      "lower_b": [k, b],
      "upper_a": [k, b],
      "upper_b": [k, b]
    },
    "intersections": {
      "lower": {
        "distance_km": float,
        "elevation_sea_level": float,
        "elevation_terrain": float
      },
      "upper": {
        "distance_km": float,
        "elevation_sea_level": float,
        "elevation_terrain": float
      },
      "cross_ab": {
        "distance_km": float,
        "elevation_sea_level": float,
        "elevation_terrain": float
      },
      "cross_ba": {
        "distance_km": float,
        "elevation_sea_level": float,
        "elevation_terrain": float
      }
    },
    "volume": {
      "cone_intersection_volume_m3": float,
      "distance_a_to_cross_ab": float,
      "distance_b_to_cross_ba": float,
      "distance_between_crosses": float
    }
  },
  "input": {
    ... existing fields ...,
    "elevation_angle_offset": float
  }
}
```

Implementation:
- Convert numpy arrays to lists using .tolist()
- Round floats to appropriate precision
- Ensure all values are JSON-serializable
- Validate schema after generation

---

## Testing Strategy

### Unit Tests

#### Geometry Module Tests
Location: `tests/unit/domain/test_geometry.py`

Test Classes:
1. TestRotateLineByAngle
   - test_rotate_45_degrees: Verify correct rotation
   - test_rotate_with_non_origin_pivot: Verify pivot remains on line
   - test_rotate_negative_angle: Verify negative rotation
   - test_rotate_vertical_line_raises: Verify error handling
   - test_rotate_to_near_vertical_raises: Verify steep result detection
   - test_invalid_angle_raises: Verify angle validation

2. TestFindLineIntersection
   - test_simple_intersection: Basic case
   - test_perpendicular_lines: Orthogonal lines
   - test_parallel_lines_raise: Error case
   - test_coincident_lines_raise: Edge case

3. TestCalculateHeightAboveTerrain
   - test_exact_match: No interpolation needed
   - test_interpolation: Linear interpolation
   - test_out_of_bounds_raises: Boundary validation

4. TestCalculateConeIntersectionVolume
   - test_simple_volume: Basic volume calculation
   - test_degenerate_case_zero_volume: Edge case
   - test_negative_height_raises: Invalid geometry detection

5. TestCalculateDistanceBetweenPoints
   - test_simple_distance: 3-4-5 triangle
   - test_same_point: Zero distance

#### ProfileDataCalculator Tests
Location: `tests/unit/application/test_profile_data_calculator.py`

New/Modified Tests:
- test_calculate_upper_lines: Verify upper line generation
- test_calculate_all_intersections: Verify all 4 intersections computed
- test_calculate_volume_metrics: Verify volume and distances
- test_calculate_all_with_angle_offset: Full integration with angle
- test_calculate_all_zero_angle_offset: Backward compatibility
- test_calculate_all_default_angle: Verify default parameter

### Integration Tests

Location: `tests/integration/test_extended_visibility.py`

Test Cases:
1. test_end_to_end_with_real_coordinates
   - Use actual coordinate data
   - Verify complete calculation pipeline
   - Check all outputs present and valid

2. test_visualization_generation
   - Create profile with new data
   - Generate plot
   - Verify plot contains all elements

3. test_console_output_formatting
   - Generate profile
   - Format console output
   - Verify all sections present

4. test_json_output_schema
   - Generate profile
   - Serialize to JSON
   - Validate against schema
   - Verify round-trip (serialize/deserialize)

5. test_different_angle_offsets
   - Test with 0°, 1°, 2.5°, 5° offsets
   - Verify monotonic relationships
   - Verify boundary cases

### Regression Tests

Ensure existing functionality unchanged:
- Run all existing test suite
- Verify backward compatibility
- Test with angle_offset not provided (uses default)
- Test with angle_offset=0.0 (upper == lower)

---

## Edge Cases and Failure Scenarios

### Input Validation Failures

1. **Invalid angle_offset**
   - Negative value: Raise ValueError in __post_init__
   - Too large (> 45°): Raise ValueError in __post_init__
   - Non-numeric: Type error from Angle wrapper

2. **Geometric Impossibilities**
   - Parallel sight lines: Rare, but handle with ValueError
   - Vertical sight lines: Prevented by slope validation
   - Lines diverge (no intersection): ValueError with explanation

### Calculation Failures

1. **Intersection Outside Path**
   - Cross intersections beyond path endpoints
   - Handle: Raise ValueError with specific intersection name
   - User action: Adjust angle_offset or path length

2. **Negative Volume**
   - Indicates invalid geometry (lines don't form closed region)
   - Handle: Raise ValueError during integration
   - Debug: Log line coefficients and intersections

3. **Numerical Instability**
   - Very steep lines (large k values): Prevented by validation
   - Very small regions (numerical precision): Set volume = 0 if < 1e-6
   - Integration divergence: Caught by area sanity checks

### Data Quality Issues

1. **Sparse Elevation Data**
   - Few points → poor interpolation
   - Handle: Warn if < 50 points for path > 10 km
   - Mitigation: Request denser sampling from API

2. **Unrealistic Terrain**
   - Sudden cliffs, data errors
   - Handle: Log warnings for large elevation jumps
   - Validation: Check |Δelev| < 1000m between adjacent points

3. **Extreme Path Conditions**
   - Very long paths (> 100 km): Volume approximation less accurate
   - Very short paths (< 1 km): May have degenerate intersections
   - Handle: Document limitations in warnings

---

## Performance Considerations

### Computational Complexity

1. **Geometry Operations**: O(1)
   - Line rotation: Single trigonometric calculation
   - Line intersection: Simple algebra
   - Target: < 10 microseconds total

2. **Terrain Interpolation**: O(log n)
   - Binary search for bracketing indices
   - Target: < 10 microseconds per query

3. **Volume Integration**: O(m)
   - m = number of integration samples ≈ 10 × path_length_km
   - Typical: 100-1000 samples
   - Target: < 100 ms for paths up to 100 km

4. **Total Addition**: O(n log n) dominated by existing calculations
   - New features add approximately 5-10% overhead
   - Acceptable for batch processing

### Memory Requirements

1. **New Data Structures**: ~500 bytes
   - IntersectionPoint × 4: 4 × 32 bytes
   - SightLinesData: 4 × 16 bytes
   - VolumeData: 32 bytes
   - Negligible compared to elevation arrays

2. **Integration Temporary Arrays**: O(m)
   - Areas array: m × 8 bytes
   - Sample points: m × 8 bytes
   - Typical: 1000 × 16 bytes = 16 KB
   - Acceptable memory footprint

### Optimization Opportunities

1. **Vectorization**
   - Use numpy array operations throughout
   - Avoid Python loops in volume integration
   - Already applied in proposed implementation

2. **Caching**
   - Cache intersection calculations if used multiple times
   - Not necessary for single-pass calculation
   - Consider if adding iterative angle sweep feature

3. **Parallel Processing**
   - Volume integration slices independent
   - Could parallelize with multiprocessing
   - Overhead likely not worth it for typical scales
   - Keep serial for simplicity

---

## Migration and Compatibility

### Breaking Changes

1. **ProfileData.lines_of_sight Type Change**
   - Old: tuple[NDArray, NDArray, tuple[float, float]]
   - New: SightLinesData
   - Impact: Code accessing as tuple will break
   - Migration: Update to use .lower_a, .lower_b attributes

### Backward Compatibility Strategy

1. **Default Parameters**
   - angle_offset defaults to 0.0
   - Zero offset produces upper == lower lines
   - Existing callers continue to work

2. **Graceful Degradation**
   - New fields always present in ProfileData
   - With angle_offset=0, intersections duplicate lower values
   - Volume calculated but may be near-zero

3. **Testing Strategy**
   - Run all existing tests without modification
   - Add new tests alongside existing ones
   - Verify no regressions in output format

### Rollout Plan

1. **Phase 1**: Add geometry module (no dependencies)
2. **Phase 2**: Extend domain models (backward compatible)
3. **Phase 3**: Update ProfileDataCalculator (default parameters)
4. **Phase 4**: Update visualization and output
5. **Phase 5**: Update tests and documentation
6. **Phase 6**: Enable angle_offset in user interface

---

## Dependencies

### Required Python Packages

All dependencies already present:
- numpy: Array operations, trigonometry
- math: Additional math functions
- typing: Type annotations
- dataclasses: Data structure definitions

No new external dependencies required.

### Internal Module Dependencies

New dependencies (all internal):
- profile_data_calculator → domain.geometry
- visualization.plotter → domain.models.path (extended types)
- output modules → domain.models.path (extended types)

Dependency graph remains acyclic (clean architecture preserved).

---

## Validation Rules

### Input Validation

1. **CalculationInput.elevation_angle_offset**
   - Type: Angle (float wrapper)
   - Range: [0.0, 45.0] degrees
   - Enforcement: __post_init__ validation
   - Error: ValueError with descriptive message

### Intermediate Validation

1. **After Line Rotation**
   - Slope finite and < 1e6
   - Intercept finite
   - Line passes through pivot (verify algebraically)

2. **After Intersection Calculation**
   - Coordinates finite (not inf, not nan)
   - Distance within path bounds [0, total_distance]
   - Elevation within reasonable range [-500, 10000]

3. **During Volume Integration**
   - Each slice area non-negative
   - Each slice area < 1e10 (sanity limit)
   - No inf or nan in intermediate calculations

### Output Validation

1. **IntersectionPoint**
   - All fields finite
   - distance_km in [0, total_distance]
   - Elevations in reasonable ranges

2. **VolumeData**
   - Volume >= 0
   - Volume < 1e12 (sanity limit)
   - All distances >= 0
   - Distances consistent with path length

### Validation Strategy

- Fail fast: Raise ValueError immediately on violation
- Descriptive messages: Include context (which intersection, what value)
- Preserve context: Use exception chaining (raise ... from e)
- Log warnings: For near-boundary conditions without failing

---

## Pseudo-Code Example

### Complete Calculation Flow

```
FUNCTION calculate_extended_visibility_profile(
    coordinates_a, coordinates_b,
    antenna_height_a, antenna_height_b,
    angle_offset
):
    # Phase 1: Existing path profile calculation
    path_data = fetch_elevation_profile(coordinates_a, coordinates_b)
    distances = path_data.distances
    elevations = path_data.elevations

    # Phase 2: HCA calculation (existing)
    hca_data = calculate_hca(
        elevations, distances,
        antenna_height_a, antenna_height_b
    )
    i1 = hca_data.b1_idx  # Obstacle from A
    i2 = hca_data.b2_idx  # Obstacle from B

    # Phase 3: Apply curvature (existing)
    elevations_curved = elevations + calculate_geometric_curvature(distances)

    # Phase 4: Calculate lower sight lines (existing)
    p0 = (distances[0], elevations_curved[0] + antenna_height_a)
    p1 = (distances[i1], elevations_curved[i1])
    lower_a = line_through_points(p0, p1)

    p2 = (distances[-1], elevations_curved[-1] + antenna_height_b)
    p3 = (distances[i2], elevations_curved[i2])
    lower_b = line_through_points(p2, p3)

    # Phase 5: Calculate upper sight lines (NEW)
    pivot_a = p0
    pivot_b = p2

    upper_a = rotate_line_by_angle(lower_a, pivot_a, angle_offset)
    upper_b = rotate_line_by_angle(lower_b, pivot_b, angle_offset)

    sight_lines = SightLinesData(lower_a, lower_b, upper_a, upper_b)

    # Phase 6: Calculate intersections (NEW)
    # Lower intersection
    x_low, y_low = find_line_intersection(lower_a, lower_b)
    h_low = calculate_height_above_terrain(
        x_low, y_low, distances, elevations_curved
    )
    lower_int = IntersectionPoint(x_low, y_low, h_low)

    # Upper intersection
    x_up, y_up = find_line_intersection(upper_a, upper_b)
    h_up = calculate_height_above_terrain(
        x_up, y_up, distances, elevations_curved
    )
    upper_int = IntersectionPoint(x_up, y_up, h_up)

    # Cross AB intersection
    x_ab, y_ab = find_line_intersection(upper_a, lower_b)
    h_ab = calculate_height_above_terrain(
        x_ab, y_ab, distances, elevations_curved
    )
    cross_ab = IntersectionPoint(x_ab, y_ab, h_ab)

    # Cross BA intersection
    x_ba, y_ba = find_line_intersection(upper_b, lower_a)
    h_ba = calculate_height_above_terrain(
        x_ba, y_ba, distances, elevations_curved
    )
    cross_ba = IntersectionPoint(x_ba, y_ba, h_ba)

    intersections = IntersectionsData(
        lower_int, upper_int, cross_ab, cross_ba
    )

    # Phase 7: Calculate volume (NEW)
    volume = calculate_cone_intersection_volume(
        sight_lines, distances, intersections
    )

    distance_a_to_ab = x_ab
    distance_b_to_ba = distances[-1] - x_ba
    distance_between = |x_ab - x_ba|

    volume_data = VolumeData(
        volume, distance_a_to_ab, distance_b_to_ba, distance_between
    )

    # Phase 8: Assemble result
    RETURN ProfileData(
        plain=plain_profile,
        curved=curved_profile,
        lines_of_sight=sight_lines,
        intersections=intersections,
        volume=volume_data
    )
END FUNCTION
```

---

# GEMINI_GUIDE

## Implementation Instructions for Gemini

This guide provides step-by-step instructions for implementing the extended terrain visibility analysis feature. Follow each step exactly in the order presented.

---

## Prerequisites

Before starting:
1. Ensure working directory is project root: `/home/dp/projects/python/trace_calc/architecture-refactor-v1`
2. Ensure git working tree is clean or on feature branch
3. Ensure Python environment is activated
4. Verify current tests pass: `python -m pytest tests/`

---

## Step 1: Create Geometry Module

**Task:** Create new domain module with pure geometric functions

**Action:** Create file `trace_calc/domain/geometry.py`

**Content:** Copy the complete geometry module code from the PLAN section. Include all five functions:
- rotate_line_by_angle
- find_line_intersection
- calculate_height_above_terrain
- calculate_cone_intersection_volume
- calculate_distance_between_points

**Required Imports:**
```python
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
```

**Verification:**
- File exists at exact path
- Run: `python -m py_compile trace_calc/domain/geometry.py`
- Should complete without errors
- Run: `python -c "from trace_calc.domain import geometry; print(dir(geometry))"`
- Should list all five functions

**Pitfalls to Avoid:**
- Do not import from other trace_calc modules (keep it pure)
- Do not add any business logic
- Maintain exact function signatures as specified

---

## Step 2: Extend Domain Models - Add New Types

**Task:** Add four new NamedTuple classes to path models

**Action:** Edit file `trace_calc/domain/models/path.py`

**Location:** After existing imports, before any existing class definitions

**Add These Classes (in order):**

1. IntersectionPoint
2. SightLinesData
3. IntersectionsData
4. VolumeData

**Ensure Each Has:**
- All fields with correct types
- Docstrings for class and each field
- Correct NamedTuple syntax

**Verification:**
- Run: `python -m py_compile trace_calc/domain/models/path.py`
- Run: `python -c "from trace_calc.domain.models.path import IntersectionPoint, SightLinesData, IntersectionsData, VolumeData; print('OK')"`
- Should print "OK"

**Pitfalls to Avoid:**
- Do not modify existing classes yet (that's Step 3)
- Maintain correct indentation
- Use NDArray[np.float64] not just np.ndarray for type hints

---

## Step 3: Extend Domain Models - Modify ProfileData

**Task:** Change ProfileData class structure

**Action:** Edit file `trace_calc/domain/models/path.py`

**Find:** The existing ProfileData class definition (should be a @dataclass)

**Replace:** The entire class definition with new version that has 5 fields:
- plain: ProfileViewData (unchanged)
- curved: ProfileViewData (unchanged)
- lines_of_sight: SightLinesData (type changed from tuple)
- intersections: IntersectionsData (new)
- volume: VolumeData (new)

**Verification:**
- Run: `python -m py_compile trace_calc/domain/models/path.py`
- Run: `python -c "from trace_calc.domain.models.path import ProfileData; import inspect; print(inspect.signature(ProfileData))"`
- Should show 5 parameters

**Pitfalls to Avoid:**
- Do not delete the @dataclass(slots=True) decorator
- Keep all docstrings
- Do not change plain or curved fields

---

## Step 4: Extend Input Model

**Task:** Add elevation_angle_offset field with validation

**Action:** Edit file `trace_calc/domain/models/input.py`

**Find:** The CalculationInput class

**Add Field:** After all existing fields, before end of class:
```python
elevation_angle_offset: Angle = Angle(2.5)
```

**Add or Extend Method:** `__post_init__`

If method doesn't exist, create it. If it exists, add this validation code to it:
```python
if self.elevation_angle_offset < 0:
    raise ValueError(
        f"elevation_angle_offset must be non-negative, got {self.elevation_angle_offset}"
    )
if self.elevation_angle_offset > 45:
    raise ValueError(
        f"elevation_angle_offset must be <= 45 degrees, got {self.elevation_angle_offset}"
    )
```

**Verification:**
- Run: `python -m py_compile trace_calc/domain/models/input.py`
- Test validation: `python -c "from trace_calc.domain.models.input import CalculationInput; from trace_calc.domain.models.units import Angle; from trace_calc.domain.models.geo import GeoData; from trace_calc.domain.models.units import Frequency, Meters; ci = CalculationInput(GeoData(0,0,0), GeoData(0,0,0), Frequency(1000), Meters(10), Meters(10), elevation_angle_offset=Angle(-1))"`
- Should raise ValueError

**Pitfalls to Avoid:**
- Do not remove existing __post_init__ code if present
- Use Angle() wrapper for the default value
- Import Angle if not already imported

---

## Step 5: Add ProfileDataCalculator Methods - Part 1

**Task:** Add imports to ProfileDataCalculator

**Action:** Edit file `trace_calc/application/services/profile_data_calculator.py`

**Find:** The import section at top of file

**Add These Imports (if not present):**
```python
from trace_calc.domain import geometry
from trace_calc.domain.models.path import (
    IntersectionPoint,
    IntersectionsData,
    ProfileData,
    SightLinesData,
    VolumeData,
)
from trace_calc.domain.models.units import Angle
```

**Note:** Some of these may already be imported. Add only missing ones.

**Verification:**
- Run: `python -m py_compile trace_calc/application/services/profile_data_calculator.py`
- Should complete without import errors

---

## Step 6: Add ProfileDataCalculator Methods - Part 2

**Task:** Add _calculate_upper_lines method

**Action:** Edit file `trace_calc/application/services/profile_data_calculator.py`

**Find:** The ProfileDataCalculator class

**Add Method:** After existing methods, before the calculate_all method:

```python
def _calculate_upper_lines(
    self,
    lower_line_coeffs: NDArray[np.float64],
    pivot_point: tuple[float, float],
    angle_offset: Angle
) -> NDArray[np.float64]:
    """
    Calculate upper sight line by rotating lower line.

    Args:
        lower_line_coeffs: [k, b] for lower sight line
        pivot_point: (distance, elevation) of site
        angle_offset: Angular offset in degrees

    Returns:
        [k_upper, b_upper] for upper sight line

    Raises:
        ValueError: If rotation produces invalid line
    """
    try:
        upper_coeffs = geometry.rotate_line_by_angle(
            lower_line_coeffs,
            pivot_point,
            float(angle_offset)
        )
        return upper_coeffs
    except ValueError as e:
        raise ValueError(f"Failed to calculate upper line: {e}") from e
```

**Verification:**
- Run: `python -m py_compile trace_calc/application/services/profile_data_calculator.py`
- Method should be part of class (check indentation)

---

## Step 7: Add ProfileDataCalculator Methods - Part 3

**Task:** Add _calculate_all_intersections method

**Action:** Edit file `trace_calc/application/services/profile_data_calculator.py`

**Add Method:** After _calculate_upper_lines method:

```python
def _calculate_all_intersections(
    self,
    sight_lines: SightLinesData,
    distances: NDArray[np.float64],
    elevations: NDArray[np.float64]
) -> IntersectionsData:
    """
    Calculate all intersection points between sight lines.

    Args:
        sight_lines: All four line coefficients
        distances: Path distances (km)
        elevations: Terrain elevations (meters)

    Returns:
        IntersectionsData with all 4 intersections

    Raises:
        ValueError: If any intersection is invalid or outside path
    """
    # Lower intersection
    x, y = geometry.find_line_intersection(sight_lines.lower_a, sight_lines.lower_b)
    if x < 0 or x > distances[-1]:
        raise ValueError(f"Lower intersection at {x} km is outside path bounds")
    h_terrain = geometry.calculate_height_above_terrain(x, y, distances, elevations)
    lower_int = IntersectionPoint(x, y, h_terrain)

    # Upper intersection
    x, y = geometry.find_line_intersection(sight_lines.upper_a, sight_lines.upper_b)
    if x < 0 or x > distances[-1]:
        raise ValueError(f"Upper intersection at {x} km is outside path bounds")
    h_terrain = geometry.calculate_height_above_terrain(x, y, distances, elevations)
    upper_int = IntersectionPoint(x, y, h_terrain)

    # Cross AB intersection
    x, y = geometry.find_line_intersection(sight_lines.upper_a, sight_lines.lower_b)
    if x < 0 or x > distances[-1]:
        raise ValueError(f"Cross AB intersection at {x} km is outside path bounds")
    h_terrain = geometry.calculate_height_above_terrain(x, y, distances, elevations)
    cross_ab = IntersectionPoint(x, y, h_terrain)

    # Cross BA intersection
    x, y = geometry.find_line_intersection(sight_lines.upper_b, sight_lines.lower_a)
    if x < 0 or x > distances[-1]:
        raise ValueError(f"Cross BA intersection at {x} km is outside path bounds")
    h_terrain = geometry.calculate_height_above_terrain(x, y, distances, elevations)
    cross_ba = IntersectionPoint(x, y, h_terrain)

    return IntersectionsData(lower_int, upper_int, cross_ab, cross_ba)
```

**Verification:**
- Run: `python -m py_compile trace_calc/application/services/profile_data_calculator.py`

---

## Step 8: Add ProfileDataCalculator Methods - Part 4

**Task:** Add _calculate_volume_metrics method

**Action:** Edit file `trace_calc/application/services/profile_data_calculator.py`

**Add Method:** After _calculate_all_intersections method:

```python
def _calculate_volume_metrics(
    self,
    sight_lines: SightLinesData,
    distances: NDArray[np.float64],
    intersections: IntersectionsData
) -> VolumeData:
    """
    Calculate volume and distance metrics.

    Args:
        sight_lines: All four lines
        distances: Path distances
        intersections: All intersection points

    Returns:
        VolumeData with volume and distance metrics

    Raises:
        ValueError: If volume calculation fails
    """
    # Calculate volume
    volume = geometry.calculate_cone_intersection_volume(
        sight_lines.lower_a,
        sight_lines.lower_b,
        sight_lines.upper_a,
        sight_lines.upper_b,
        distances,
        intersections.lower_intersection.distance_km,
        intersections.upper_intersection.distance_km,
        intersections.cross_ab.distance_km,
        intersections.cross_ba.distance_km
    )

    # Calculate distances
    distance_a_to_cross_ab = intersections.cross_ab.distance_km
    total_distance = distances[-1]
    distance_b_to_cross_ba = total_distance - intersections.cross_ba.distance_km
    distance_between = abs(
        intersections.cross_ab.distance_km - intersections.cross_ba.distance_km
    )

    # Validate
    if distance_a_to_cross_ab < 0 or distance_b_to_cross_ba < 0:
        raise ValueError("Distance metrics produced negative values")

    return VolumeData(
        volume,
        distance_a_to_cross_ab,
        distance_b_to_cross_ba,
        distance_between
    )
```

**Verification:**
- Run: `python -m py_compile trace_calc/application/services/profile_data_calculator.py`

---

## Step 9: Modify ProfileDataCalculator.calculate_all Method

**Task:** Extend calculate_all method with new calculations

**Action:** Edit file `trace_calc/application/services/profile_data_calculator.py`

**Find:** The calculate_all method signature

**Modify Signature:** Add parameter with default:
```python
def calculate_all(
    self,
    hca_indices: tuple[int, int],
    height_offsets: tuple[Meters, Meters],
    angle_offset: Angle = Angle(0.0)
) -> ProfileData:
```

**Find:** The section where lower sight lines are calculated. Look for variables named something like:
- coeff1, coeff2 (line coefficients)
- offset_start, offset_end (antenna heights)
- elevations_curved (curvature-corrected elevations)
- distances (distance array)

**After Lower Line Calculation, Insert This Code:**

```python
# Calculate upper sight lines
pivot_a = (distances[0], elevations_curved[0] + offset_start)
pivot_b = (distances[-1], elevations_curved[-1] + offset_end)

coeff1_upper = self._calculate_upper_lines(coeff1, pivot_a, angle_offset)
coeff2_upper = self._calculate_upper_lines(coeff2, pivot_b, angle_offset)

# Assemble sight lines data
sight_lines = SightLinesData(coeff1, coeff2, coeff1_upper, coeff2_upper)

# Calculate all intersections
intersections = self._calculate_all_intersections(
    sight_lines,
    distances,
    elevations_curved
)

# Calculate volume metrics
volume = self._calculate_volume_metrics(
    sight_lines,
    distances,
    intersections
)
```

**Find:** The return statement at end of method (should create ProfileData)

**Modify Return Statement:**
```python
return ProfileData(
    plain=plain,
    curved=curved,
    lines_of_sight=sight_lines,
    intersections=intersections,
    volume=volume
)
```

**Note:** Adjust variable names (plain, curved) to match what's actually used in the existing code.

**Verification:**
- Run: `python -m py_compile trace_calc/application/services/profile_data_calculator.py`
- Should compile without errors

**Pitfalls to Avoid:**
- Ensure offset_start and offset_end variable names match existing code
- Ensure elevations_curved variable name matches existing code
- Do not delete any existing code, only add and modify return statement

---

## Step 10: Update Visualization - Add Imports

**Task:** Add required imports to plotter module

**Action:** Edit file `trace_calc/infrastructure/visualization/plotter.py`

**Find:** Import section

**Add Imports:**
```python
from trace_calc.domain.models.path import IntersectionsData, SightLinesData
```

**Verification:**
- Run: `python -m py_compile trace_calc/infrastructure/visualization/plotter.py`

---

## Step 11: Update Visualization - Extend Plotting

**Task:** Add upper lines and intersection markers to plot

**Action:** Edit file `trace_calc/infrastructure/visualization/plotter.py`

**Find:** The method that creates the curved profile plot (usually in second panel/subplot)

**Locate:** The section where lower sight lines are plotted. Look for code plotting lines using coeff1, coeff2 or similar.

**After Lower Sight Line Plotting, Add:**

```python
# Plot upper sight lines
sight_lines = profile.lines_of_sight

upper_line_1 = np.polyval(sight_lines.upper_a, distances)
upper_line_2 = np.polyval(sight_lines.upper_b, distances)

ax.plot(distances, upper_line_1, '--', color='red', linewidth=1.5,
        label='Upper sight line A', alpha=0.7)
ax.plot(distances, upper_line_2, '--', color='blue', linewidth=1.5,
        label='Upper sight line B', alpha=0.7)

# Plot intersection points
intersections = profile.intersections

# Lower intersection
ax.scatter(
    intersections.lower_intersection.distance_km,
    intersections.lower_intersection.elevation_sea_level,
    c='green', s=100, marker='o', label='Lower intersection', zorder=5
)

# Upper intersection
ax.scatter(
    intersections.upper_intersection.distance_km,
    intersections.upper_intersection.elevation_sea_level,
    c='purple', s=100, marker='o', label='Upper intersection', zorder=5
)

# Cross intersections
ax.scatter(
    intersections.cross_ab.distance_km,
    intersections.cross_ab.elevation_sea_level,
    c='orange', s=80, marker='^', label='Cross AB', zorder=5
)
ax.scatter(
    intersections.cross_ba.distance_km,
    intersections.cross_ba.elevation_sea_level,
    c='cyan', s=80, marker='v', label='Cross BA', zorder=5
)

# Update legend
ax.legend(loc='best', fontsize=8)
```

**Note:** Adjust variable names (ax, distances) to match existing code.

**Verification:**
- Run: `python -m py_compile trace_calc/infrastructure/visualization/plotter.py`
- Run a test that generates a plot to verify visual output

**Pitfalls to Avoid:**
- Ensure 'ax' variable name matches what's used for the subplot axis
- Ensure 'distances' variable name matches existing code
- Do not remove existing plot elements
- Ensure legend() call replaces or comes after any existing legend()

---

## Step 12: Update Console Output - Add Imports

**Task:** Add imports for new types

**Action:** Edit file `trace_calc/infrastructure/output/console_output.py`

**Find:** Import section

**Add Imports:**
```python
from trace_calc.domain.models.path import IntersectionsData, ProfileData, VolumeData
```

**Note:** ProfileData may already be imported; if so, just add the other two.

**Verification:**
- Run: `python -m py_compile trace_calc/infrastructure/output/console_output.py`

---

## Step 13: Update Console Output - Add Formatting Function

**Task:** Add function to format extended visibility results

**Action:** Edit file `trace_calc/infrastructure/output/console_output.py`

**Location:** After existing functions

**Add Function:**

```python
def format_extended_visibility_results(profile: ProfileData) -> str:
    """
    Format extended visibility analysis results for console output.

    Args:
        profile: Complete profile data with intersections and volume

    Returns:
        Formatted string for console display
    """
    sight_lines = profile.lines_of_sight
    intersections = profile.intersections
    volume = profile.volume

    output = []
    output.append("\n=== Extended Terrain Visibility Analysis ===\n")

    # Lower sight lines
    output.append("Lower Sight Lines:")
    output.append(f"  Site A → Obstacle: slope={sight_lines.lower_a[0]:.4f}, "
                  f"intercept={sight_lines.lower_a[1]:.2f}m")
    output.append(f"  Site B → Obstacle: slope={sight_lines.lower_b[0]:.4f}, "
                  f"intercept={sight_lines.lower_b[1]:.2f}m")
    output.append(f"  Intersection: {intersections.lower_intersection.distance_km:.3f} km, "
                  f"{intersections.lower_intersection.elevation_sea_level:.2f}m ASL, "
                  f"+{intersections.lower_intersection.elevation_terrain:.2f}m above terrain\n")

    # Upper sight lines
    output.append("Upper Sight Lines:")
    output.append(f"  Site A (upper): slope={sight_lines.upper_a[0]:.4f}, "
                  f"intercept={sight_lines.upper_a[1]:.2f}m")
    output.append(f"  Site B (upper): slope={sight_lines.upper_b[0]:.4f}, "
                  f"intercept={sight_lines.upper_b[1]:.2f}m")
    output.append(f"  Intersection: {intersections.upper_intersection.distance_km:.3f} km, "
                  f"{intersections.upper_intersection.elevation_sea_level:.2f}m ASL, "
                  f"+{intersections.upper_intersection.elevation_terrain:.2f}m above terrain\n")

    # Cross intersections
    output.append("Cross Intersections:")
    output.append(f"  Upper A × Lower B: {intersections.cross_ab.distance_km:.3f} km, "
                  f"{intersections.cross_ab.elevation_sea_level:.2f}m ASL, "
                  f"+{intersections.cross_ab.elevation_terrain:.2f}m above terrain")
    output.append(f"  Upper B × Lower A: {intersections.cross_ba.distance_km:.3f} km, "
                  f"{intersections.cross_ba.elevation_sea_level:.2f}m ASL, "
                  f"+{intersections.cross_ba.elevation_terrain:.2f}m above terrain\n")

    # Volume metrics
    output.append("Volume Metrics:")
    output.append(f"  Cone intersection volume: {volume.cone_intersection_volume_m3:,.0f} m³")
    output.append(f"  Distance from A to Upper A × Lower B: {volume.distance_a_to_cross_ab:.3f} km")
    output.append(f"  Distance from B to Upper B × Lower A: {volume.distance_b_to_cross_ba:.3f} km")
    output.append(f"  Distance between cross intersections: {volume.distance_between_crosses:.3f} km")

    return "\n".join(output)
```

**Verification:**
- Run: `python -m py_compile trace_calc/infrastructure/output/console_output.py`

**Usage Note:** This function can be called from orchestration code to display results.

---

## Step 14: Update JSON Output

**Task:** Extend JSON serialization to include new fields

**Action:** Edit file `trace_calc/infrastructure/output/json_output.py`

**Find:** The function that serializes ProfileData to JSON (look for json.dumps or dict construction)

**Find:** The dictionary/JSON structure being built for profile data

**Add These Fields to the JSON Dictionary:**

```python
"sight_lines": {
    "lower_a": profile.lines_of_sight.lower_a.tolist(),
    "lower_b": profile.lines_of_sight.lower_b.tolist(),
    "upper_a": profile.lines_of_sight.upper_a.tolist(),
    "upper_b": profile.lines_of_sight.upper_b.tolist()
},
"intersections": {
    "lower": {
        "distance_km": profile.intersections.lower_intersection.distance_km,
        "elevation_sea_level": profile.intersections.lower_intersection.elevation_sea_level,
        "elevation_terrain": profile.intersections.lower_intersection.elevation_terrain
    },
    "upper": {
        "distance_km": profile.intersections.upper_intersection.distance_km,
        "elevation_sea_level": profile.intersections.upper_intersection.elevation_sea_level,
        "elevation_terrain": profile.intersections.upper_intersection.elevation_terrain
    },
    "cross_ab": {
        "distance_km": profile.intersections.cross_ab.distance_km,
        "elevation_sea_level": profile.intersections.cross_ab.elevation_sea_level,
        "elevation_terrain": profile.intersections.cross_ab.elevation_terrain
    },
    "cross_ba": {
        "distance_km": profile.intersections.cross_ba.distance_km,
        "elevation_sea_level": profile.intersections.cross_ba.elevation_sea_level,
        "elevation_terrain": profile.intersections.cross_ba.elevation_terrain
    }
},
"volume": {
    "cone_intersection_volume_m3": profile.volume.cone_intersection_volume_m3,
    "distance_a_to_cross_ab": profile.volume.distance_a_to_cross_ab,
    "distance_b_to_cross_ba": profile.volume.distance_b_to_cross_ba,
    "distance_between_crosses": profile.volume.distance_between_crosses
}
```

**Important:** Use .tolist() on numpy arrays to convert to JSON-serializable lists.

**Verification:**
- Run: `python -m py_compile trace_calc/infrastructure/output/json_output.py`
- Test JSON generation to ensure valid JSON output

---

## Step 15: Create Geometry Unit Tests

**Task:** Create comprehensive unit tests for geometry module

**Action:** Create file `tests/unit/domain/test_geometry.py`

**Content:** Include all test classes specified in PLAN:
- TestRotateLineByAngle (6 tests)
- TestFindLineIntersection (4 tests)
- TestCalculateHeightAboveTerrain (3 tests)
- TestCalculateConeIntersectionVolume (2 tests)
- TestCalculateDistanceBetweenPoints (2 tests)

**Required Imports:**
```python
import math

import numpy as np
import pytest

from trace_calc.domain import geometry
```

**Verification:**
- Run: `python -m pytest tests/unit/domain/test_geometry.py -v`
- All tests should pass
- If any fail, debug before proceeding

**Pitfalls to Avoid:**
- Use pytest.raises for exception tests
- Use appropriate tolerance for float comparisons (e.g., abs(x - y) < 1e-6)
- Ensure test names are descriptive

---

## Step 16: Update ProfileDataCalculator Tests

**Task:** Add tests for new calculator methods

**Action:** Edit file `tests/unit/application/test_profile_data_calculator.py`

**Add Test Methods:**

```python
def test_calculate_upper_lines(self):
    """Test upper line calculation."""
    # Setup: Create mock calculator with dependencies
    # Call _calculate_upper_lines
    # Verify: Result has shape (2,) and different from input
    pass  # Implement based on existing test patterns

def test_calculate_all_intersections(self):
    """Test all intersections calculation."""
    # Setup: Mock sight lines, distances, elevations
    # Call _calculate_all_intersections
    # Verify: Returns IntersectionsData with 4 points
    pass  # Implement based on existing test patterns

def test_calculate_volume_metrics(self):
    """Test volume and distance metrics."""
    # Setup: Mock data
    # Call _calculate_volume_metrics
    # Verify: Returns VolumeData with non-negative values
    pass  # Implement based on existing test patterns

def test_calculate_all_with_angle_offset(self):
    """Test full calculation with angle offset."""
    # Setup: Mock all dependencies
    # Call calculate_all with angle_offset=2.5
    # Verify: ProfileData has all new fields populated
    pass  # Implement based on existing test patterns

def test_calculate_all_zero_angle_offset(self):
    """Test backward compatibility with zero offset."""
    # Setup: Mock dependencies
    # Call calculate_all with angle_offset=0.0
    # Verify: Upper lines equal lower lines
    pass  # Implement based on existing test patterns
```

**Note:** Implement these tests following the patterns of existing tests in the file.

**Verification:**
- Run: `python -m pytest tests/unit/application/test_profile_data_calculator.py -v`
- New tests should pass

---

## Step 17: Create Integration Tests

**Task:** Create end-to-end integration tests

**Action:** Create file `tests/integration/test_extended_visibility.py`

**Content:** Include tests for:
1. End-to-end calculation with real coordinates
2. Visualization generation
3. Console output formatting
4. JSON output validation
5. Different angle offset values

**Required Imports:**
```python
import json

import numpy as np
import pytest

from trace_calc.domain.models.input import CalculationInput
from trace_calc.domain.models.units import Angle
# Additional imports as needed
```

**Verification:**
- Run: `python -m pytest tests/integration/test_extended_visibility.py -v`
- All tests should pass

---

## Step 18: Run Full Test Suite

**Task:** Verify all tests pass including existing and new

**Action:** Run complete test suite

**Commands:**
```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run all integration tests
python -m pytest tests/integration/ -v

# Run everything
python -m pytest tests/ -v

# Check coverage (optional)
python -m pytest tests/ --cov=trace_calc --cov-report=html
```

**Expected Result:**
- All tests pass
- No regressions in existing functionality
- New features covered by tests

**If Tests Fail:**
1. Review failure messages
2. Check if failure is in existing or new code
3. Debug and fix before proceeding
4. Re-run tests

---

## Step 19: Verify Cross-Module Consistency

**Task:** Ensure all modules are consistent and importable

**Verification Commands:**

```bash
# Compile all Python files
python -m compileall trace_calc/

# Check imports
python -c "from trace_calc.domain import geometry; print('geometry OK')"
python -c "from trace_calc.domain.models.path import IntersectionPoint, SightLinesData, IntersectionsData, VolumeData; print('models OK')"
python -c "from trace_calc.application.services.profile_data_calculator import ProfileDataCalculator; print('calculator OK')"

# Check for import errors
python -m pylint trace_calc/ --errors-only
```

**Expected:** All commands complete without errors

---

## Step 20: Update Documentation

**Task:** Document the new functionality

**Actions:**

1. **Update README (if present):**
   - Add section describing extended visibility analysis
   - Document elevation_angle_offset parameter
   - Provide usage examples

2. **Update API Documentation:**
   - Document new CalculationInput field
   - Document new ProfileData fields
   - Document output format changes

3. **Add Docstring Examples:**
   - Add usage examples to key functions
   - Show typical angle_offset values

**Files to Check:**
- README.md
- docs/ folder (if present)
- Any API documentation files

---

## Step 21: Test with Real Data

**Task:** Run system with actual path data

**Action:** Execute full calculation with real coordinates

**Example Test:**
```python
from trace_calc.domain.models.input import CalculationInput
from trace_calc.domain.models.geo import GeoData
from trace_calc.domain.models.units import Angle, Frequency, Meters

# Create input with real coordinates
input_data = CalculationInput(
    site_a=GeoData(lat=40.0, lon=-74.0, elevation=0),
    site_b=GeoData(lat=40.1, lon=-74.1, elevation=0),
    frequency=Frequency(1000),
    antenna_a_height=Meters(30),
    antenna_b_height=Meters(30),
    elevation_angle_offset=Angle(2.5)
)

# Run calculation (adjust based on actual API)
# result = orchestration_service.calculate(input_data)

# Verify output
# print(console_output.format_extended_visibility_results(result.profile))
```

**Verification:**
- Calculation completes without errors
- All intersection points within path bounds
- Volume is reasonable (not negative, not absurdly large)
- Visualization displays correctly
- Console output is readable
- JSON output is valid

---

## Step 22: Final Validation Checklist

**Verify Each Item:**

- [ ] All new files created:
  - trace_calc/domain/geometry.py
  - tests/unit/domain/test_geometry.py
  - tests/integration/test_extended_visibility.py

- [ ] All modified files updated correctly:
  - trace_calc/domain/models/path.py
  - trace_calc/domain/models/input.py
  - trace_calc/application/services/profile_data_calculator.py
  - trace_calc/infrastructure/visualization/plotter.py
  - trace_calc/infrastructure/output/console_output.py
  - trace_calc/infrastructure/output/json_output.py

- [ ] All tests pass:
  - Unit tests for geometry
  - Unit tests for calculator
  - Integration tests
  - Existing regression tests

- [ ] Code quality:
  - No syntax errors
  - All imports resolve
  - Type hints correct
  - Docstrings present

- [ ] Functionality:
  - Upper lines calculated correctly
  - All 4 intersections computed
  - Volume calculation produces reasonable results
  - Visualization shows all elements
  - Output formats include new data

- [ ] Backward compatibility:
  - Existing code works with default angle_offset
  - No breaking changes to public APIs
  - Existing tests still pass

**If ALL items checked:** Implementation complete!

**If ANY items unchecked:** Return to relevant step and fix.

---

## Troubleshooting Guide

### Common Issues and Solutions

**Issue: Import errors after adding geometry module**
- Solution: Ensure __init__.py exists in trace_calc/domain/
- Check: Python path includes project root

**Issue: Tests fail with "parallel lines" error**
- Solution: Check that test data produces non-parallel lines
- Verify: Slopes differ by more than 1e-9

**Issue: Intersection outside path bounds**
- Solution: Verify angle_offset is reasonable (< 10°)
- Check: Path is long enough for given angle

**Issue: Volume calculation returns huge number**
- Solution: Review integration bounds
- Check: Cross intersections are in correct order
- Verify: Line coefficients are reasonable

**Issue: Visualization doesn't show upper lines**
- Solution: Verify code added to correct subplot (panel 2)
- Check: Variable names match existing code (ax, distances)

**Issue: JSON serialization fails**
- Solution: Use .tolist() on all numpy arrays
- Verify: All numeric values are JSON-serializable

**Issue: Type errors with Angle wrapper**
- Solution: Use float(angle_offset) when passing to functions expecting float
- Check: Angle imported from correct module

---

## Post-Implementation Tasks

After successful implementation:

1. **Commit Changes:**
   ```bash
   git add .
   git commit -m "feat: Add extended terrain visibility analysis

   - Add upper sight lines with configurable angle offset
   - Calculate 4 intersection points
   - Compute cone intersection volume
   - Enhance visualization with all intersections
   - Extend output formats (console and JSON)"
   ```

2. **Update Changelog:**
   - Document new features
   - Note any API changes
   - List new configuration options

3. **Create Pull Request:**
   - Reference this implementation plan
   - Include test results
   - Add screenshots of visualization

4. **Performance Testing:**
   - Benchmark with various path lengths
   - Verify acceptable performance (<200ms for typical paths)
   - Profile if needed

5. **User Documentation:**
   - Create user guide for angle_offset parameter
   - Document output interpretation
   - Provide example use cases

---

## CRITICAL IMPLEMENTATION CORRECTIONS

### Summary of Essential Fixes

During prototype development with real terrain data, several critical issues were discovered
that MUST be addressed in the implementation. These corrections are NOT optional - they are
required for correct functionality.

#### 1. Unit Conversion for Angle Calculations

**Problem:**
Line slopes are stored as **meters/kilometer** (m/km), but trigonometric functions require
geometric slopes in **meters/meter** (m/m). Failure to convert units results in angles near
90° instead of the actual small angles (typically 0-5°).

**Solution:**
```python
# WRONG - directly using m/km slope
angle = np.arctan(slope_m_per_km)  # Results in ~87° for slope=19.3

# CORRECT - convert to m/m first
slope_m_per_m = slope_m_per_km / 1000
angle = np.arctan(slope_m_per_m)  # Results in ~1.1° for slope=19.3
```

**Impact:** Without this conversion, the 2.5° angular offset becomes effectively meaningless,
and the upper/lower line separation will be visually incorrect by orders of magnitude.

#### 2. Opposite Rotation for Descending Lines

**Problem:**
For descending lines (negative slope, from Site B looking towards Site A), rotating
counterclockwise by +2.5° can flip the line to ascending, placing the "upper" line
geometrically BELOW the "lower" line.

**Solution:**
```python
# Determine rotation direction based on slope sign
if slope < 0:  # Descending line
    # Subtract offset to make less steep (keeps descending, moves up)
    new_angle = current_angle - angle_offset
else:  # Ascending or horizontal
    # Add offset to make steeper (moves up)
    new_angle = current_angle + angle_offset
```

**Why this works:**
- Ascending line: angle +2.5° → steeper ascent → line moves up ✓
- Descending line: angle -2.5° → shallower descent → line moves up ✓

**Impact:** Without this correction, the blue dashed line (upper from Site B) will appear
below the blue solid line (lower from Site B), violating the fundamental requirement that
upper lines must be above lower lines.

#### 3. No Baseline Shift in Visualization

**Problem:**
Some implementations shift all elevations by subtracting the first point's elevation to
make plots "start at zero". This obscures the sea level reference and can make terrain
appear to be below sea level when it's not.

**Solution:**
```python
# WRONG - shifting baseline
shift = elevations[0]
elevations_plot = elevations - shift  # Confuses sea level reference

# CORRECT - use actual elevations
elevations_plot = elevations  # Preserves sea level at y=0
```

**Impact:** Extended visibility creates intersection points thousands of meters above
terrain. Without proper sea level reference, plots become confusing and appear to show
negative elevations.

#### 4. Y-Axis Scaling for Intersection Visibility

**Problem:**
Auto-scaling to terrain alone (0-669m in test data) hides the upper intersection point
(~4721m in test data), making the extended visibility analysis invisible.

**Solution:**
```python
# Collect ALL relevant y-values
all_y = [
    intersections.lower_intersection.elevation_sea_level,
    intersections.upper_intersection.elevation_sea_level,
    intersections.cross_ab.elevation_sea_level,
    intersections.cross_ba.elevation_sea_level,
    elevations.max(),
    site_a_elev + antenna_a,
    site_b_elev + antenna_b,
    0  # Sea level
]
ax.set_ylim(min(all_y) - 100, max(all_y) + 100)
```

**Impact:** Without explicit scaling, the primary feature of extended visibility (the upper
intersection and visibility cone) will be cut off and invisible in plots.

### Verification Checklist

Before considering the implementation complete, verify:

- [ ] **Unit conversion**: Print actual angles - should be 0-5°, NOT 85-90°
- [ ] **Angular separation**: Both sites show exactly Δ=2.5° (or specified offset)
- [ ] **Upper above lower**: At x=midpoint, verify upper_line_y > lower_line_y for both A and B
- [ ] **Sea level reference**: Baseline (y=0) represents sea level, terrain fills upward
- [ ] **All intersections visible**: Plot y-limits include upper intersection point
- [ ] **Real data test**: Test with actual terrain data, not just synthetic flat/simple profiles

### Test Data Validation

The test.path file provides real-world validation:
- Path: 162.47 km
- Terrain: 0-669m elevation
- Expected upper intersection: ~4721m (much higher than terrain)
- Expected angles: Site A ~1.1°→3.6°, Site B ~-0.5°→-3.0°

If these values are drastically different (e.g., angles near 90°), the implementation is incorrect.

---

## Notes for Gemini

**Strict Adherence:**
- Follow steps in exact order
- Do not skip steps
- Do not merge steps
- Complete each verification before proceeding

**Error Handling:**
- If any step fails, stop immediately
- Report exact error message
- Do not attempt to fix without guidance
- Do not continue to next step

**Code Modifications:**
- Only modify files explicitly listed
- Only add/change code in specified locations
- Preserve all existing functionality
- Maintain existing code style

**Variable Names:**
- Use exact variable names from existing code
- Adjust examples in guide to match actual codebase
- Do not rename existing variables

**Testing:**
- Run verification command after each step
- Do not proceed if verification fails
- Run full test suite before declaring complete

**Communication:**
- Report progress after each step
- Note any deviations or issues
- Confirm completion with checklist

**Success Criteria:**
- All files created/modified as specified
- All tests pass (existing and new)
- No regressions in functionality
- System produces correct output for test cases
- Documentation updated

This implementation should take approximately 2-4 hours depending on codebase familiarity and testing thoroughness.

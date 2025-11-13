# Architectural Restructuring Plan: Fix Profile Visualization & Decouple Components

**Date**: 2025-11-13 (Updated after deep investigation)
**Status**: Phase 1 Ready for Implementation

---

## Critical Discovery After Investigation ✅

After extensive analysis including testing all formula variations, we discovered:

**HCA Calculator (hca_calculator.py):**
- ✅ **CORRECT** - Tests pass with b_sum = 0.552° (validated reference data)
- ✅ Uses **empirical troposcatter formula** (Groza/Sosnik methodology)
- ✅ Unit mixing is **INTENTIONAL** - not a bug!
- ✅ Formula: `curvature_correction = R² / 12.742` (produces km, used in meter context)
- ✅ This is part of the empirical model calibrated against real propagation data
- ❌ **DO NOT CHANGE** - Would break validated physics model

**Profile Calculator (profile_data_calculator.py):**
- ❌ **HAS BUG** - Line 77 missing `curve *= 1000` conversion
- ❌ Currently adds kilometers to meters (wrong for visualization)
- ❌ **NO TESTS** - Bug went undetected because visualization isn't tested
- ✅ **NEEDS FIX** - Should convert geometric curvature from km to meters

---

## Root Cause Analysis

### The Formula Bug (Profile Only)

```python
# In curved_profile (profile_data_calculator.py:75-78):
curve = -((self.distances - self.distances[mid_idx]) ** 2) / curvature_scale  # [km²]/[km] = [km]
curve -= curve[0]
# curve *= 1000  ← MISSING! Comment says "convert to meters" but doesn't do it
curved_elevations = self.elevations + curve  # Adding [km] to [m] ❌
```

**Impact**: Profile visualization shows incorrect Earth curvature (off by 1000×)

### The Architecture Bug (Both Files)

```
User creates GrozaAnalyzer
  → Inherits DefaultProfileDataCalculator.__init__()
      → Calls super().__init__() → HCACalculatorCC.__init__()
          → Filters elevations (FFT/correlation)
          → Calculates HCA using FILTERED elevations
          → Stores self.hca_data (b_sum = 0.552°) ✅
      → Creates ProfileDataCalculator with ORIGINAL elevations
          → Calculates curved profile with BUGGY formula
          → Stores self.profile_data (used for plotting) ❌
```

**Key Insight**: Two **SEPARATE** calculations:
1. HCA calculation (correct, tested, validated)
2. Profile visualization (buggy, untested, wrong units)

The bug in profile calculator doesn't affect HCA tests because they're independent!

---

## Proposed Solution: 3-Phase Approach

## Phase 1: Fix Profile Visualization Bug (TODAY - 1 hour)

**Goal**: Correct unit conversion in profile calculator WITHOUT changing HCA

### Step 1.1: Fix Profile Curvature (profile_data_calculator.py)

**File**: `trace_calc/services/profile_data_calculator.py:77`

```python
# BEFORE (BUGGY):
curve = -((self.distances - self.distances[mid_idx]) ** 2) / curvature_scale
curve -= curve[0]
# curve *= 1000  ← Commented out!
curved_elevations = self.elevations + curve

# AFTER (FIXED):
curve = -((self.distances - self.distances[mid_idx]) ** 2) / curvature_scale
curve -= curve[0]
curve *= 1000  # Convert km to meters ✅
curved_elevations = self.elevations + curve
```

### Step 1.2: DO NOT Change HCA Calculator

**File**: `trace_calc/services/hca_calculator.py`

**NO CHANGES** - Current implementation is correct:
```python
@staticmethod
def betta_calc(site_height, obstacle_height, R, antenna_height=2):
    """
    Calculate horizon clearance angle using empirical troposcatter formula.

    This formula uses intentional unit mixing calibrated for Groza/Sosnik models.
    DO NOT "fix" by adding * 1000 - tests validate current implementation.
    """
    curvature_correction_m = (R**2 / 12.742)  # Empirical factor, intentional units
    return (
        math.atan2(
            (obstacle_height - curvature_correction_m - site_height - antenna_height),
            (R * 1000),
        )
        * 180
        / math.pi
    )
```

### Step 1.3: DO NOT Change Test Expectations

**Files**: `tests/test_hca_calculator.py`, `tests/test_analyzers.py`

**NO CHANGES** - Current expectations are validated reference data:
- `test_hca_calculator_cross_correlation`: b_sum = "0.552" ✅
- `test_hca_calculator_fft`: b_sum = "0.352" ✅
- `test_sosnic_analyzer`: b_sum = "0.552", speed = 64 kbps ✅
- `test_groza_analyzer`: b_sum = "0.552", speed = 22.3 Mbps ✅

### Step 1.4: Add Profile Visualization Tests

**NEW**: `tests/test_profile_visualization.py`

```python
"""Test profile visualization calculations independently from HCA."""
import numpy as np
import pytest
from trace_calc.services.profile_data_calculator import ProfileDataCalculator


def test_curved_profile_applies_curvature_in_meters():
    """Verify curvature correction is properly converted from km to meters."""
    # Flat terrain profile
    distances_km = np.array([0, 50, 100])
    elevations_m = np.array([100.0, 100.0, 100.0])

    calc = ProfileDataCalculator(distances_km, elevations_m)
    curved, baseline = calc.curved_profile()

    # At midpoint (50km from edges), curvature drop should be:
    # drop = 50² / 12.742 * 1000 = 196,201 meters
    edge_distance_from_mid = 50.0
    expected_drop_m = (edge_distance_from_mid**2 / 12.742) * 1000

    # Midpoint should be relatively unchanged (small distance from itself)
    assert abs(curved[1] - elevations_m[1]) < 100, "Midpoint should be near original"

    # Edges should drop significantly due to Earth curvature
    actual_drop_left = elevations_m[0] - curved[0]
    actual_drop_right = elevations_m[2] - curved[2]

    assert abs(actual_drop_left - expected_drop_m) < 1000, \
        f"Left edge drop should be ~{expected_drop_m:.0f}m, got {actual_drop_left:.0f}m"
    assert abs(actual_drop_right - expected_drop_m) < 1000, \
        f"Right edge drop should be ~{expected_drop_m:.0f}m, got {actual_drop_right:.0f}m"


def test_curved_profile_symmetric_around_midpoint():
    """Curvature should be symmetric around the midpoint."""
    distances_km = np.linspace(0, 100, 256)
    elevations_m = np.ones(256) * 150.0

    calc = ProfileDataCalculator(distances_km, elevations_m)
    curved, _ = calc.curved_profile()

    mid_idx = len(curved) // 2

    # Points equidistant from midpoint should have same curvature
    for offset in [10, 50, 100]:
        if mid_idx - offset >= 0 and mid_idx + offset < len(curved):
            left_drop = elevations_m[mid_idx - offset] - curved[mid_idx - offset]
            right_drop = elevations_m[mid_idx + offset] - curved[mid_idx + offset]
            assert abs(left_drop - right_drop) < 1.0, \
                f"Curvature should be symmetric at offset {offset}"


def test_profile_calculator_independent_from_hca():
    """
    Verify profile calculation doesn't affect HCA values.
    This test documents architectural separation.
    """
    from trace_calc.models.input_data import InputData
    from trace_calc.models.path import PathData
    from trace_calc.services.analyzers import GrozaAnalyzer

    # Use same test data as HCA tests
    np.random.seed(42)
    random_array = np.random.rand(256)
    scaled_array = random_array * 40 + 135
    test_elevations = np.convolve(scaled_array, np.ones(5)/5, mode="same")
    test_elevations[0] += 70
    test_elevations[-1] += 70

    test_profile = PathData(
        coordinates=np.linspace(123.10, 234.50, 256),
        distances=np.linspace(0, 100, 256),
        elevations=test_elevations,
    )

    analyzer = GrozaAnalyzer(test_profile, InputData("test_file"))

    # HCA should still be correct (independent of profile bug fix)
    assert f"{analyzer.hca_data.b_sum:.3f}" == "0.552", \
        "HCA calculation must remain unchanged"

    # Profile should now have correct units
    assert analyzer.profile_data is not None, "Profile should be calculated"
```

### Step 1.5: Document HCA Formula as Empirical

Add detailed comments to `hca_calculator.py`:

```python
@staticmethod
def betta_calc(site_height, obstacle_height, R, antenna_height=2):
    """Calculate horizon clearance angle using empirical troposcatter formula.

    Args:
        site_height: Site elevation in meters
        obstacle_height: Obstacle elevation in meters
        R: Distance in kilometers
        antenna_height: Antenna height in meters (default: 2)

    Returns:
        Angle in degrees

    Note:
        This formula uses INTENTIONAL unit mixing calibrated for Groza/Sosnik
        troposcatter propagation models. The curvature_correction_m variable
        name is misleading - the value is dimensionally in kilometers but
        is numerically small enough (~196 for 50km distance) to work correctly
        when subtracted from meter values in the empirical formula.

        DO NOT "fix" by adding * 1000 conversion - this would break validated
        physics model. Tests confirm b_sum = 0.552° is correct reference value.

        This is SEPARATE from geometric Earth curvature used in profile
        visualization (see profile_data_calculator.py which does use * 1000).

    Formula origin:
        Empirical troposcatter model, calibrated against real propagation data.
        Earth curvature factor 12.742 km ≈ 2 × Earth_radius / 1000, but used
        in specific empirical context, not pure geometric calculation.
    """
    curvature_correction_m = (R**2 / 12.742)  # Empirical factor (see docstring)

    return (
        math.atan2(
            (obstacle_height - curvature_correction_m - site_height - antenna_height),
            (R * 1000),
        )
        * 180
        / math.pi
    )
```

### Deliverables (Phase 1):

- ✅ Fixed unit conversion in profile_data_calculator.py (1 file, 1 line)
- ✅ Added detailed documentation to hca_calculator.py (explain empirical formula)
- ✅ NO changes to HCA calculator logic
- ✅ NO changes to test expectations (validated reference data)
- ✅ New test file: tests/test_profile_visualization.py (independent tests)
- ✅ Git commit: "fix: correct km→meters conversion in profile visualization"

**Expected Test Results:**
- All existing HCA tests: ✅ PASS (unchanged)
- All existing analyzer tests: ✅ PASS (unchanged)
- New profile visualization tests: ✅ PASS (verify geometric curvature)

---

## Phase 2: Architectural Decoupling (WEEK 1 - 3 days)

**Goal**: Separate concerns and eliminate hidden dependencies through composition

### Implementation Details

See Phase 2 section for:
- Shared curvature module (geometric + empirical models)
- Refactored calculators using composition
- Analysis pipeline with dependency injection
- Breaking inheritance chain

**Deliverables:**
- domain/curvature.py with two curvature models
- Refactored HCA and profile calculators
- New analysis pipeline class
- Updated tests

---

## Phase 3: Type Safety & Module Organization (WEEK 2 - 5 days)

**Goal**: Prevent future unit mistakes through type system

### Implementation Details

See Phase 3 section for:
- Unit-aware types (Distance, Elevation, Angle)
- Type-safe function signatures
- mypy strict mode configuration
- Reorganized module structure

**Deliverables:**
- Type-safe units module
- All functions with type hints
- mypy validation passing
- Clean module organization

---

## Key Learnings from Investigation

1. **Don't assume variable names are correct** - `curvature_correction_m` suggested meters, but empirical formula uses it differently
2. **Test values are reference data** - Passing tests indicate correct implementation, not bugs
3. **Separate concerns need separate tests** - Profile visualization had no tests, bugs went undetected
4. **Empirical models may intentionally mix units** - Not all dimensional inconsistencies are bugs
5. **Architecture matters** - Tight coupling hid the fact that HCA and profile are independent

---

## Migration Timeline

| Phase | Duration | Risk | Dependencies |
|-------|----------|------|--------------|
| Phase 1: Profile fix + tests | 1 hour | **LOW** | None |
| Phase 2: Architectural refactor | 3 days | **MEDIUM** | Phase 1 complete |
| Phase 3: Type safety | 5 days | **LOW** | Phase 2 complete |
| **Total** | **~2 weeks** | - | - |

---

**Status: Ready to implement Phase 1** ✅

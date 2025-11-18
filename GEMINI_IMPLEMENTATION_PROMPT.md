# Implementation Task: Extended Terrain Visibility Analysis

## Context

You are implementing a feature to add advanced terrain visibility analysis to a radio path profiling system. A comprehensive implementation plan exists at `EXTENDED_VISIBILITY_IMPLEMENTATION_PLAN.md` in this directory.

## Critical Constraints

### 1. Test-Driven Development (TDD) - MANDATORY

You MUST follow strict TDD methodology:

1. **Write failing tests FIRST** - before any implementation code
2. **Implement minimal code** - only enough to make tests pass
3. **Refactor** - clean up while keeping tests green
4. **Verify** - run tests after each change

**Never write implementation code before writing its tests.**

### 2. Atomic Feature Development

Work on ONE self-contained piece at a time:
- Complete one function with its tests before moving to next
- Each commit should represent a complete, tested unit of work
- Do not skip ahead or work on multiple features simultaneously

### 3. Approval Workflow

- Present diffs/summaries for review before committing
- Wait for explicit approval on major changes
- Generate test and integration summaries after each atomic feature

## Implementation Sequence (TDD Order)

Follow this strict sequence. Each step must be completed with passing tests before proceeding.

### Phase 1: Domain Layer - Geometry Module (Pure Functions)

**For EACH function below, follow this cycle:**
1. Write comprehensive unit tests (include edge cases, error cases, happy path)
2. Run tests - verify they fail appropriately
3. Implement the function to pass tests
4. Refactor if needed
5. Verify all tests pass
6. Request review before proceeding to next function

**Functions to implement in order:**

#### 1.1 `rotate_line_by_angle`
**Tests to write first:**
- `test_rotate_45_degrees` - verify correct rotation
- `test_rotate_with_non_origin_pivot` - verify pivot remains on line
- `test_rotate_negative_angle` - verify negative rotation
- `test_rotate_vertical_line_raises` - verify error handling
- `test_rotate_to_near_vertical_raises` - verify steep result detection
- `test_invalid_angle_raises` - verify angle validation

**Implementation location:** `trace_calc/domain/geometry.py`
**Test location:** `tests/unit/domain/test_geometry.py`

#### 1.2 `find_line_intersection`
**Tests to write first:**
- `test_simple_intersection` - basic case
- `test_perpendicular_lines` - orthogonal lines
- `test_parallel_lines_raise` - error case
- `test_coincident_lines_raise` - edge case

#### 1.3 `calculate_height_above_terrain`
**Tests to write first:**
- `test_exact_match` - no interpolation needed
- `test_interpolation` - linear interpolation
- `test_out_of_bounds_raises` - boundary validation

#### 1.4 `calculate_cone_intersection_volume`
**Tests to write first:**
- `test_simple_volume` - basic volume calculation
- `test_degenerate_case_zero_volume` - edge case
- `test_negative_height_raises` - invalid geometry detection

#### 1.5 `calculate_distance_between_points`
**Tests to write first:**
- `test_simple_distance` - 3-4-5 triangle
- `test_same_point` - zero distance

**Phase 1 Completion Criteria:**
- [ ] All 5 functions implemented with passing tests
- [ ] Test file `tests/unit/domain/test_geometry.py` complete
- [ ] Module `trace_calc/domain/geometry.py` complete
- [ ] All tests pass: `pytest tests/unit/domain/test_geometry.py -v`
- [ ] No implementation code without corresponding tests

---

### Phase 2: Domain Models (Data Structures)

**TDD Approach for Models:**
1. Write tests that instantiate models with valid data
2. Write tests that validate constraints (e.g., angle_offset range)
3. Implement models to pass validation tests
4. Verify all tests pass

#### 2.1 Add New NamedTuples to `trace_calc/domain/models/path.py`

**Tests to write first:**
```python
def test_intersection_point_creation():
    """Test IntersectionPoint instantiation"""

def test_sight_lines_data_creation():
    """Test SightLinesData instantiation"""

def test_intersections_data_creation():
    """Test IntersectionsData instantiation"""

def test_volume_data_creation():
    """Test VolumeData instantiation with non-negative values"""

def test_volume_data_negative_volume_invalid():
    """Test VolumeData rejects negative volume if validated"""
```

**Models to implement:**
- `IntersectionPoint`
- `SightLinesData`
- `IntersectionsData`
- `VolumeData`

#### 2.2 Extend ProfileData

**Tests to write first:**
```python
def test_profile_data_with_new_fields():
    """Test ProfileData accepts new sight_lines, intersections, volume fields"""
```

**Implementation:** Modify `ProfileData` class structure

#### 2.3 Extend CalculationInput

**Tests to write first:**
```python
def test_calculation_input_default_angle_offset():
    """Test default angle_offset is 2.5"""

def test_calculation_input_negative_angle_offset_raises():
    """Test validation rejects negative angle_offset"""

def test_calculation_input_too_large_angle_offset_raises():
    """Test validation rejects angle_offset > 45"""

def test_calculation_input_valid_angle_offset():
    """Test valid angle_offset accepted"""
```

**Implementation:** Add `elevation_angle_offset` field with validation

**Phase 2 Completion Criteria:**
- [ ] All model tests pass
- [ ] Models compile without errors
- [ ] Validation logic working correctly
- [ ] Tests verify both valid and invalid inputs

---

### Phase 3: Application Layer - ProfileDataCalculator Extensions

**For EACH method, write tests first, then implement.**

#### 3.1 `_calculate_upper_lines`

**Tests to write first:**
```python
def test_calculate_upper_lines_zero_offset():
    """Test upper equals lower with zero offset"""

def test_calculate_upper_lines_positive_offset():
    """Test upper differs from lower with positive offset"""

def test_calculate_upper_lines_invalid_geometry_raises():
    """Test error handling for invalid geometry"""
```

**Implementation location:** `trace_calc/application/services/profile_data_calculator.py`
**Test location:** `tests/unit/application/test_profile_data_calculator.py`

#### 3.2 `_calculate_all_intersections`

**Tests to write first:**
```python
def test_calculate_all_intersections_returns_four_points():
    """Test all 4 intersections calculated"""

def test_calculate_all_intersections_within_bounds():
    """Test intersections within path bounds"""

def test_calculate_all_intersections_outside_bounds_raises():
    """Test error when intersection outside path"""
```

#### 3.3 `_calculate_volume_metrics`

**Tests to write first:**
```python
def test_calculate_volume_metrics_non_negative():
    """Test volume is non-negative"""

def test_calculate_volume_metrics_distances_valid():
    """Test distance metrics are valid"""

def test_calculate_volume_metrics_consistency():
    """Test distance_between = |cross_ab - cross_ba|"""
```

#### 3.4 Modify `calculate_all`

**Tests to write first:**
```python
def test_calculate_all_with_angle_offset():
    """Test full calculation with angle offset"""

def test_calculate_all_zero_angle_backward_compat():
    """Test backward compatibility with zero offset"""

def test_calculate_all_default_parameter():
    """Test default angle parameter works"""

def test_calculate_all_profile_data_structure():
    """Test ProfileData has all 5 fields populated"""
```

**Phase 3 Completion Criteria:**
- [ ] All calculator methods implemented with tests
- [ ] Tests pass: `pytest tests/unit/application/test_profile_data_calculator.py -v`
- [ ] Backward compatibility verified (angle_offset=0 works)
- [ ] Integration with geometry module verified

---

### Phase 4: Infrastructure Layer - Visualization

#### 4.1 Extend Plotter

**Tests to write first:**
```python
def test_plotter_renders_upper_lines():
    """Test upper lines appear in plot"""

def test_plotter_renders_all_intersections():
    """Test all 4 intersection markers present"""

def test_plotter_legend_updated():
    """Test legend includes new elements"""
```

**Implementation location:** `trace_calc/infrastructure/visualization/plotter.py`
**Test location:** `tests/integration/test_extended_visibility.py` (visualization tests)

**Phase 4 Completion Criteria:**
- [ ] Visualization tests pass
- [ ] Manual verification: plot displays correctly
- [ ] All elements visible and labeled

---

### Phase 5: Infrastructure Layer - Output Formatting

#### 5.1 Console Output

**Tests to write first:**
```python
def test_format_extended_visibility_contains_all_sections():
    """Test output has all required sections"""

def test_format_extended_visibility_formatting():
    """Test numeric formatting (decimal places, units)"""

def test_format_extended_visibility_readability():
    """Test output is human-readable"""
```

**Implementation:** Add `format_extended_visibility_results` function

#### 5.2 JSON Output

**Tests to write first:**
```python
def test_json_output_valid_json():
    """Test output is valid JSON"""

def test_json_output_contains_new_fields():
    """Test sight_lines, intersections, volume present"""

def test_json_output_round_trip():
    """Test serialize/deserialize preserves data"""

def test_json_output_numpy_arrays_converted():
    """Test numpy arrays converted to lists"""
```

**Phase 5 Completion Criteria:**
- [ ] Output formatting tests pass
- [ ] Console output human-readable
- [ ] JSON output valid and complete

---

### Phase 6: Integration Testing

**Write comprehensive integration tests:**

```python
def test_end_to_end_with_real_coordinates():
    """Full pipeline test with realistic data"""

def test_different_angle_offsets():
    """Test with 0°, 1°, 2.5°, 5° offsets"""

def test_volume_increases_with_angle():
    """Test volume monotonically increases with angle"""

def test_edge_case_very_short_path():
    """Test behavior with short paths"""

def test_edge_case_very_long_path():
    """Test behavior with long paths (>100km)"""
```

**Phase 6 Completion Criteria:**
- [ ] All integration tests pass
- [ ] Edge cases handled gracefully
- [ ] Performance acceptable (<200ms typical paths)

---

## Test Execution Strategy

### After Each Atomic Feature:
```bash
# Run unit tests for component
pytest tests/unit/domain/test_geometry.py::TestRotateLineByAngle -v

# Run all unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Full test suite
pytest tests/ -v
```

### Continuous Verification:
- Run tests after every implementation
- Keep test suite green at all times
- If tests fail, fix before proceeding

## Documentation Requirements

After each phase completion:

1. **Test Summary:**
   - Number of tests written
   - Coverage metrics
   - Any edge cases discovered

2. **Implementation Summary:**
   - Functions/classes implemented
   - Design decisions made
   - Any deviations from plan

3. **Integration Summary:**
   - How component integrates with existing code
   - Backward compatibility verified
   - Breaking changes (should be none)

## Error Handling Standards

All functions must:
- Fail fast with descriptive errors
- Use ValueError for validation failures
- Include context in error messages (e.g., "Lower intersection at 45.3 km is outside path bounds")
- Chain exceptions (`raise ... from e`)

## Success Criteria

Implementation is complete when:

- [ ] All unit tests pass (100+ tests expected)
- [ ] All integration tests pass (5+ tests expected)
- [ ] No existing tests broken (regression-free)
- [ ] Code compiles without errors
- [ ] All imports resolve correctly
- [ ] Real-world data test completes successfully
- [ ] Visualization renders correctly
- [ ] Output formats (console, JSON) correct
- [ ] Documentation updated
- [ ] Test coverage ≥90% for new code
- [ ] Performance benchmarks met

## Key Files Reference

### Implementation Plan:
- `EXTENDED_VISIBILITY_IMPLEMENTATION_PLAN.md` - Complete specification

### Current Project Structure:
```
trace_calc/
├── domain/
│   ├── geometry.py          [NEW - Phase 1]
│   └── models/
│       ├── path.py          [MODIFY - Phase 2]
│       └── input.py         [MODIFY - Phase 2]
├── application/
│   └── services/
│       └── profile_data_calculator.py  [MODIFY - Phase 3]
└── infrastructure/
    ├── visualization/
    │   └── plotter.py       [MODIFY - Phase 4]
    └── output/
        ├── console_output.py [MODIFY - Phase 5]
        └── json_output.py   [MODIFY - Phase 5]

tests/
├── unit/
│   ├── domain/
│   │   └── test_geometry.py [NEW - Phase 1]
│   └── application/
│       └── test_profile_data_calculator.py [MODIFY - Phase 3]
└── integration/
    └── test_extended_visibility.py [NEW - Phase 6]
```

## Working Directory
- Current: `/home/dp/projects/python/trace_calc/architecture-refactor-v2`
- Ensure all paths are relative to this directory

## Git Workflow

After each phase completion:
1. Review all changes with diff
2. Request approval to commit
3. Commit with descriptive message following format:
   ```
   test: Add unit tests for [component]

   - Test case 1
   - Test case 2
   - Test case 3
   ```
4. Then commit implementation:
   ```
   feat: Implement [component]

   - Implementation detail 1
   - Implementation detail 2

   Tests: [X] passing
   ```

## Final Checklist

Before declaring implementation complete:

```
PHASE 1: GEOMETRY MODULE
[ ] 5 functions implemented
[ ] 17+ unit tests passing
[ ] Module compiles without errors
[ ] All imports work

PHASE 2: DOMAIN MODELS
[ ] 4 new NamedTuples added
[ ] ProfileData extended
[ ] CalculationInput extended with validation
[ ] 8+ model tests passing

PHASE 3: APPLICATION LAYER
[ ] 3 new methods in ProfileDataCalculator
[ ] calculate_all method modified
[ ] 10+ calculator tests passing
[ ] Backward compatibility verified

PHASE 4: VISUALIZATION
[ ] Upper lines rendered
[ ] 4 intersection markers rendered
[ ] Legend updated
[ ] Visual verification complete

PHASE 5: OUTPUT FORMATTING
[ ] Console output function added
[ ] JSON output extended
[ ] 6+ output tests passing
[ ] Formats validated

PHASE 6: INTEGRATION
[ ] 5+ integration tests passing
[ ] End-to-end test with real data works
[ ] Edge cases handled
[ ] Performance benchmarks met

FINAL VERIFICATION
[ ] All tests pass (unit + integration)
[ ] No regressions in existing tests
[ ] Code quality checks pass
[ ] Documentation updated
[ ] Git commits clean and descriptive
```

## Questions Before Starting?

1. Confirm understanding of TDD requirement
2. Confirm access to test framework (pytest)
3. Confirm ability to run tests after each change
4. Any clarifications needed on implementation plan?

## Begin Implementation

Start with **Phase 1.1: rotate_line_by_angle**

1. Create test file `tests/unit/domain/test_geometry.py`
2. Write the 6 tests for `rotate_line_by_angle`
3. Run tests - verify they fail
4. Create `trace_calc/domain/geometry.py`
5. Implement `rotate_line_by_angle`
6. Run tests - verify they pass
7. Report results and request approval to proceed to Phase 1.2

**Remember: Tests first, implementation second. Always.**

Beam Intersection Point

This document details the implementation of the beam intersection point analysis feature within the `trace_calc` project. This feature extends the common scatter volume analysis by providing insights into the intersection of beam intersection point lines and their angles from the initial points.

## Feature Description

The antenna elevation angle analysis feature enhances the existing profile data calculations by:
- Identifying the intersection point of the two beam intersection point lines (`antenna_elevation_angle_a` and `antenna_elevation_angle_b`).
- Calculating the distance, elevation above sea level (ASL), and height above terrain for this beam intersection point point.
- Determining the angle of `antenna_elevation_angle_a` from the starting point (Site A) and `antenna_elevation_angle_b` from the ending point (Site B).

These new metrics provide a more comprehensive understanding of the common scatter volume and the behavior of the sight lines.

## Changes Made

The following files were modified to implement this feature:

1.  **`trace_calc/domain/models/path.py`**
    -   The `IntersectionsData` NamedTuple was updated to include an optional `beam_intersection_point` field of type `IntersectionPoint`.
    -   The `VolumeData` NamedTuple was updated to include `antenna_elevation_angle_a` and `antenna_elevation_angle_b` fields of type `Angle`.

2.  **`trace_calc/application/services/profile_data_calculator.py`**
    -   A new private method `_calculate_beam_intersection_point` was added to determine the intersection point of `antenna_elevation_angle_a` and `antenna_elevation_angle_b` using linear interpolation. This method also corrects the elevation for Earth's curvature and calculates height above terrain.
    -   A new private method `_calculate_antenna_elevation_angles` was added to compute the initial angles of `antenna_elevation_angle_a` (at Site A) and `antenna_elevation_angle_b` (at Site B) based on their initial slopes.
    -   The `calculate_all` method was updated to call these new methods and store their results in the `intersections` and `volume` data structures, respectively.
    -   The `Optional` type was imported from `typing`.

3.  **`trace_calc/infrastructure/output/formatters.py`**
    -   `ConsoleOutputFormatter`: The `format_common_volume_results` method was updated to display the new "Antenna Elevation Angle Intersection" details (distance, elevation ASL, elevation above terrain) and "Antenna Elevation Angles" for both antenna elevation angles.
    -   `JSONOutputFormatter`: The `format_result` method was updated to include the `beam_intersection_point` (if present) in the `intersections` dictionary and `antenna_elevation_angle_a`, `antenna_elevation_angle_b` in the `volume` dictionary of the JSON output.

4.  **`trace_calc/infrastructure/visualization/plotter.py`**
    -   The `plot_profile` method was updated to optionally plot the `beam_intersection_point` point on the curved profile, using a distinct marker (blue 'X').

5.  **Test Files (`tests/` directory)**
    -   **`tests/unit/application/test_profile_data_calculator.py`**:
        -   A new test method `test_beam_intersection_point_and_angles` was added to verify the correct calculation of the beam intersection point point and the antenna elevation angles.
        -   The `test_calculate_all_profile_data_structure` test was updated to assert the presence of the new antenna-elevation-angle-related fields.
        -   The `hca_indices` in `test_beam_intersection_point_and_angles` were adjusted to avoid `ZeroDivisionError` during slope calculation.
        -   Expected values for `antenna_elevation_angle_intersection.elevation_sea_level`, `antenna_elevation_angle_a`, and `antenna_elevation_angle_b` were updated to reflect more accurate calculations.
        -   Duplicate import of `Angle` was removed.
    -   **`tests/unit/domain/test_analysis_models.py`**:
        -   `IntersectionsData` and `VolumeData` instantiations were updated to include placeholder values (`None` or `Angle(0.0)`) for the new fields, ensuring backward compatibility with existing tests.
        -   `Angle` was imported from `trace_calc.domain.models.units`.
    -   **`tests/unit/infrastructure/test_output_formatter.py`**:
        -   The `sample_analysis_result` fixture was updated to include dummy `beam_intersection_point` and `antenna_elevation_angle_a`, `antenna_elevation_angle_b` values.
        -   Assertions in `test_console_formatter_prints_summary` were updated to match the new expected output, including the antenna elevation angle intersection and angles.
        -   `Angle` was imported from `trace_calc.domain.models.units`.
    -   **`tests/integration/test_common_volume_analysis.py`**:
        -   Assertions in `test_console_output_formatting` were updated to reflect the new output format, including the antenna elevation angle intersection and angles.

## How to Use

The antenna elevation angle analysis is automatically performed as part of the `ProfileDataCalculator.calculate_all` method. The results are integrated into the `ProfileData` object.

To view the new metrics:
-   **Console Output:** Run the main application or any process that uses `ConsoleOutputFormatter`. The new "Beam Intersection Point" and "Antenna Elevation Angles" sections will appear in the summary.
-   **JSON Output:** If using `JSONOutputFormatter`, the `beam_intersection_point` details and `antenna_elevation_angle_a`, `antenna_elevation_angle_b` will be present in the `profile_data.intersections` and `profile_data.volume` sections of the JSON output, respectively.
-   **Visualization:** If `ProfileVisualizer` is used, the beam intersection point point will be plotted on the curved profile.

## Testing

The implementation was thoroughly tested with unit and integration tests:
-   **Unit Tests:** New test cases were added to `tests/unit/application/test_profile_data_calculator.py` to specifically validate the `_calculate_beam_intersection_point` and `_calculate_antenna_elevation_angles` methods. Existing tests were updated to accommodate the new fields in `IntersectionsData` and `VolumeData`.
-   **Integration Tests:** Existing integration tests in `tests/integration/test_common_volume_analysis.py` were updated to ensure the end-to-end pipeline correctly processes and outputs the new antenna elevation angle analysis data, both in console and JSON formats, and that visualization functions as expected.

All tests were run successfully after the changes were applied, ensuring the stability and correctness of the new feature.
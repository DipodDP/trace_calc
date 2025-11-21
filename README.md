# Troposcatter Trace Calculation and Advanced Link Analysis

![Example Plot](output_data/example_plot.png)

Visualization of a common volume analysis, showing the terrain profile, multiple sight lines, and key intersection points.

## Overview

This project provides tools for calculating troposcatter radio link profiles, including terrain analysis and advanced common volume calculations. It helps in designing and evaluating communication links by providing detailed insights into propagation paths, signal interference areas, and geographical data.

## Prerequirements

You need to get an API key for using the [Elevations API on RapidAPI](https://rapidapi.com/toursprung-toursprung-default/api/maptoolkit/playground/apiendpoint_6da0665d-de84-4227-a41d-accd9c7974d7) and an API key for using the [Geomagnetic Declination API](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#declination). These should be configured in your `.env` file.

## Key Features

### Common Volume Analysis (CVA)

This project features advanced Common Volume Analysis, which calculates the shared illuminated volume between two antennas, considering their half-power beamwidths (HPBW). It provides detailed metrics essential for enhanced troposcatter link design, including:

*   **Multiple Sight Lines**: Generates lower and upper sight lines with configurable angular offsets.
*   **Four Intersection Points**: Identifies:
    *   The primary intersection of lower sight lines.
    *   The intersection of upper sight lines.
    *   Two "cross" intersections where the upper line from one site meets the lower line from the other (Upper A × Lower B, Upper B × Lower A).
*   **Volumetric Data**: Computes the 3D volume of the intersection region, providing insights into the propagation environment.
*   **Beam Intersection Point Analysis**: A dedicated analysis to identify the intersection point of two beam intersection point lines, calculating its distance, elevation above sea level (ASL), and height above terrain.

### Enhanced Output & Geographic Data

Analysis outputs now include comprehensive geographic and link information:

*   **Detailed Site Coordinates**: Console and JSON outputs display precise latitude and longitude for both sites.
*   **Geographic Metrics**: Automatically calculated path distance, azimuths, and magnetic declinations (if available).
*   **Robust Calculations**: Underlying architectural improvements ensure accurate Earth curvature corrections for profile visualization and robust Horizon Clearance Angle (HCA) calculations.

## Using

Install dependencies
```sh
poetry install
```
Activate virtual environment
```sh
poetry shell
```
Run tests (verbose)
```sh
pytest -v -s
```
Run script
```sh
python trace_calc/main.py
```
## Configuration

To configure the analysis parameters (e.g., coordinates, antenna heights, HPBW, angular offset for CVA), please adjust the `.env` file or provide arguments via the command line as supported by `main.py`.

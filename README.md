# Troposcatter trace calculation

## Prerequirements

You need to get an API key for using the [Elevations API on RapidAPI](https://rapidapi.com/toursprung-toursprung-default/api/maptoolkit/playground/apiendpoint_6da0665d-de84-4227-a41d-accd9c7974d7) and an API key for using the [Geomagnetic Declination API](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#declination). These should be configured in your `.env` file.

## Common Volume Analysis

This project now supports Common Volume Analysis, which calculates the shared illuminated volume between two antennas, considering their half-power beamwidths (HPBW). It provides detailed metrics on sight lines, intersection points (including a dedicated beam intersection point), and volumetric data for enhanced troposcatter link design.

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
To configure the analysis parameters (e.g., coordinates, antenna heights, HPBW), please adjust the `.env` file or provide arguments via the command line as supported by `main.py`.

## Example Plot

Below is an example visualization of a common volume analysis, showing the terrain profile, sight lines, and key intersection points.

![Example Plot](output_data/example_plot.png)
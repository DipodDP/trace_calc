# Troposcatter Trace Calculation and Advanced Link Analysis

![Example Plot](output_data/example_plot.png)
*Visualization of a common volume analysis, showing the terrain profile, sight lines, and key intersection points.*

## Overview

This project provides an asynchronous command-line tool for calculating troposcatter radio link profiles, including terrain analysis and advanced common volume calculations. It helps in designing and evaluating communication links by providing detailed insights into propagation paths, signal interference areas, and geographical data.

The application runs asynchronously, allowing for efficient I/O operations (like API calls for elevation data) without blocking the main thread.

## Prerequirements

You need to get an API key for using the [Elevations API on RapidAPI](https://rapidapi.com/toursprung-toursprung-default/api/maptoolkit/playground/apiendpoint_6da0665d-de84-4227-a41d-accd9c7974d7) and an API key for using the [Geomagnetic Declination API](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#declination). These should be configured in your `.env` file.

## Key Features

### Common Volume Analysis (CVA)

This project features advanced Common Volume Analysis, which calculates the shared illuminated volume between two antennas, considering their half-power beamwidths (HPBW). It provides detailed metrics essential for enhanced troposcatter link design, including:

*   **Multiple Sight Lines**: Generates lower and upper sight lines with configurable angular offsets.
*   **Four Intersection Points**: Identifies key intersections between sight lines.
*   **Volumetric Data**: Computes the 3D volume of the intersection region.
*   **Beam Intersection Point Analysis**: A dedicated analysis to identify the intersection point of two beam intersection point lines.

### Structured Output & Geographic Data

Analysis outputs are provided in a structured format, available as both console output and a detailed JSON file. This includes:

*   **Detailed Site Coordinates**: Precise latitude and longitude for both sites.
*   **Geographic Metrics**: Automatically calculated path distance, azimuths, and magnetic declinations.
*   **Model-Specific Parameters**: Detailed parameters from the propagation model used (e.g., Groza, Sosnik).
*   **Profile Data**: Detailed information about sight lines, intersections, and common volume metrics.

## Setup

Install dependencies:
```sh
poetry install
```
Activate virtual environment:
```sh
poetry shell
```
Run tests (verbose):
```sh
pytest -v -s
```

## Usage

The main script is `trace_calc/main.py`. It can be run with several command-line arguments to control the analysis.

### Command-Line Interface

```sh
python trace_calc/main.py [--method <name>] [--save-json]
```

**Arguments:**

*   `--method <name>`: Specifies the analysis method to use.
    *   Choices: `groza` (default), `sosnik`.
*   `--save-json`: If provided, saves the full analysis results to a JSON file in the `output_data/` directory. The filename will be based on the path name you provide.

**Interactive Prompts:**

When you run the script, you will be prompted to enter:
1.  **Stored file name**: The base name of a `.path` file (without the extension) located in the `output_data` directory. If the file exists, it will be loaded. If not, the script will fetch the elevation profile from the API and save it.
2.  **Antenna heights**: You can specify the heights for antenna A and B, or press Enter to use the default values.

**Example:**

To run an analysis using the `sosnik` method and save the results to a JSON file:
```sh
python trace_calc/main.py --method sosnik --save-json
```

### JSON Output

When using the `--save-json` flag, a detailed JSON file is generated in the `output_data/` directory. The file contains a comprehensive breakdown of the analysis, including:
- `analysis_result`: Link speed and model-specific parameters.
- `site_a_coordinates`, `site_b_coordinates`: Latitude and longitude for each site.
- `geo_data`: Distance, azimuths, and magnetic declinations.
- `profile_data`: Detailed data on sight lines, intersections, and common volume calculations.

This structured output is ideal for programmatic analysis or integration with other tools.

## Configuration

*   **API Keys**: Add your `ELEVATION_API_URL`, `ELEVATION_API_KEY`, `DECLINATION_API_URL`, and `DECLINATION_API_KEY` to a `.env` file in the project root.
*   **Analysis Parameters**: Other parameters like coordinates, antenna heights, and HPBW are provided through interactive prompts or loaded from `.path` files.
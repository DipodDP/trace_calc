# Troposcatter Trace Calculation and Advanced Link Analysis

![Example Plot](output_data/example_plot.png)
*Visualization of a common volume analysis, showing the terrain profile, sight lines, and key intersection points.*

## Overview

**Package Name:** `trace-calc` (v0.3.22)

This project provides an asynchronous command-line tool for calculating troposcatter radio link profiles, including terrain analysis and advanced common volume calculations. It helps in designing and evaluating communication links by providing detailed insights into propagation paths, signal interference areas, and geographical data.

The application runs asynchronously, allowing for efficient I/O operations (like API calls for elevation data) without blocking the main thread.

**Designed for Extensibility and Integration:**

This application is built for both standalone CLI use and integration as a Python package. It features a clean, extensible architecture based on Domain-Driven Design (DDD) principles, making it easy to:
- **Integrate into other applications** - Telegram bots, web services, automation tools, etc.
- **Create custom propagation models** - Extend with your own calculation methods
- **Implement alternative data sources** - Replace API clients with custom implementations
- **Extend with custom calculators** - Add new HCA calculators, profile calculators, and more

See the [Using as a Package](#using-as-a-package) section for integration examples and [Extending with Custom Models](#extending-with-custom-models) for creating custom propagation models.

## Prerequisites

**For CLI Usage and API-based Analysis:**

You need to get an API key for using the [Elevations API on RapidAPI](https://rapidapi.com/toursprung-toursprung-default/api/maptoolkit/playground/apiendpoint_6da0665d-de84-4227-a41d-accd9c7974d7) and an API key for using the [Geomagnetic Declination API](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#declination). These should be configured in your `.env` file.

**Note:** API keys are only required if you're fetching elevation data from external services. Custom implementations can use alternative data sources (local files, databases, etc.) without requiring these API keys.

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

---

## Using as a Package

The `trace_calc` package can be integrated into your Python applications for programmatic analysis of troposcatter links.

### Installation

**From source (development):**
```bash
git clone <repository-url>
cd trace_calc
poetry install
```

**In your project (using poetry):**
```bash
poetry add /path/to/trace_calc
```

### Basic Programmatic Usage

```python
import asyncio
from trace_calc.domain.models.coordinates import Coordinates, InputData
from trace_calc.domain.models.units import Meters
from trace_calc.application.analysis import GrozaAnalysisService
from trace_calc.application.orchestration import OrchestrationService
from trace_calc.application.services.profile import PathProfileService
from trace_calc.infrastructure.api.clients import (
    AsyncElevationsApiClient,
    AsyncMagDeclinationApiClient,
)

async def analyze_link():
    # 1. Create input data
    input_data = InputData(
        path_name="my_path",
        site_a_coordinates=Coordinates(55.7558, 37.6173),
        site_b_coordinates=Coordinates(59.9343, 30.3351),
        frequency_mhz=5000.0,
        antenna_a_height=Meters(30.0),
        antenna_b_height=Meters(30.0),
    )

    # 2. Initialize API clients
    elevations_client = AsyncElevationsApiClient(api_url, api_key)
    declinations_client = AsyncMagDeclinationApiClient(api_url, api_key)

    # 3. Create services
    profile_service = PathProfileService(
        input_data=input_data,
        elevations_api_client=elevations_client,
        block_size=256,
        resolution=0.05,
    )

    analysis_service = GrozaAnalysisService()  # or SosnikAnalysisService()

    orchestrator = OrchestrationService(
        analysis_service=analysis_service,
        profile_service=profile_service,
        declinations_api_client=declinations_client,
    )

    # 4. Run analysis
    result = await orchestrator.process(
        input_data=input_data,
        antenna_a_height=30.0,
        antenna_b_height=30.0,
        display_output=False,
        generate_plot=False,
    )

    # 5. Access results
    print(f"Link Speed: {result.link_speed} Mbps")
    print(f"Total Loss: {result.model_propagation_loss_parameters['total_loss']} dB")

    return result

asyncio.run(analyze_link())
```

### Accessing Structured Results

Results are returned as immutable dataclasses with `to_dict()` methods for JSON serialization:

```python
# Get structured data
geo_data = result.result['geo_data']
profile_data = result.result['profile_data']

# Convert to JSON
from trace_calc.infrastructure.output.formatters import JSONOutputFormatter
formatter = JSONOutputFormatter()
json_output = formatter.format_result(result, input_data, geo_data, profile_data)
```

**See [examples/basic_integration.py](examples/basic_integration.py) for a complete working example.**

---

## Architecture Overview

The application follows a **three-layer Domain-Driven Design (DDD)** architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      Infrastructure Layer                    │
│  (External systems, APIs, file I/O, visualization)          │
│  • AsyncElevationsApiClient, AsyncMagDeclinationApiClient   │
│  • FilePathStorage (async file operations)                  │
│  • ConsoleOutputFormatter, JSONOutputFormatter              │
│  • ProfileVisualizer (matplotlib-based plotting)            │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  (Business logic, workflow orchestration, services)         │
│  • OrchestrationService (dependency injection, workflow)    │
│  • GrozaAnalysisService, SosnikAnalysisService (strategy)   │
│  • PathProfileService, CoordinatesService, GeoDataService   │
│  • GrozaAnalyzer, SosnikAnalyzer (concrete implementations) │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                        Domain Layer                          │
│  (Pure business logic, models, interfaces, validators)      │
│  • Models: InputData, PathData, AnalysisResult, GeoData     │
│  • Units: Type-safe Meters, Kilometers, Degrees, Loss       │
│  • Interfaces: BaseAnalyzer, BaseApiClient, BaseStorage     │
│  • Validators, Constants, Domain Functions                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `trace_calc.domain.models` | Data models (InputData, PathData, AnalysisResult, etc.) |
| `trace_calc.domain.interfaces` | Abstract base classes defining contracts |
| `trace_calc.domain.units` | Type-safe unit definitions (Meters, Kilometers, etc.) |
| `trace_calc.application.analysis` | Analysis services (Groza, Sosnik strategies) |
| `trace_calc.application.orchestration` | Workflow coordination with DI |
| `trace_calc.application.services` | Profile, coordinates, HCA services |
| `trace_calc.infrastructure.api` | External API clients (async with retry logic) |
| `trace_calc.infrastructure.storage` | File-based persistence |
| `trace_calc.infrastructure.output` | Output formatters (console, JSON) |

**For detailed API documentation, see the docstrings in each module.**

---

## Extending with Custom Models

The application is designed for easy extension with custom propagation models.

### Creating a Custom Propagation Model

Follow these three steps:

#### Step 1: Create a Speed Calculator

```python
from trace_calc.domain.interfaces import BaseSpeedCalculator
from trace_calc.domain.models.units import Loss, Speed
from typing import Tuple

class MySpeedCalculator(BaseSpeedCalculator):
    def calculate_speed(self, *args, **kwargs) -> Tuple[float, ...]:
        # Your custom speed calculation logic
        total_loss = Loss(...)
        link_speed = Speed(...)
        return total_loss, link_speed
```

#### Step 2: Create an Analyzer

```python
from trace_calc.application.analyzers.base import BaseServiceAnalyzer
from trace_calc.domain.models.analysis import AnalyzerResult

class MyAnalyzer(BaseServiceAnalyzer, MySpeedCalculator):
    def analyze(self, **kwargs) -> AnalyzerResult:
        # Calculate model parameters
        # Call self.calculate_speed(...)
        # Return AnalyzerResult with results
        pass
```

#### Step 3: Create an Analysis Service

```python
from trace_calc.application.analysis import BaseAnalysisService

class MyAnalysisService(BaseAnalysisService):
    def _create_analyzer(self, path, input_data):
        return MyAnalyzer(path, input_data)

    def _get_propagation_loss(self, result_data):
        # Extract loss components from result
        return PropagationLoss(...)

    def _get_total_path_loss(self, result_data):
        # Extract total loss
        return Loss(result_data['total_loss'])
```

#### Step 4: Use Your Custom Model

```python
# Simply pass your service to OrchestrationService
my_service = MyAnalysisService()

orchestrator = OrchestrationService(
    analysis_service=my_service,  # <-- Your custom model here
    profile_service=profile_service,
    declinations_api_client=declinations_client,
)

result = await orchestrator.process(...)
```

### Other Extension Points

**Custom HCA Calculators:**
```python
from trace_calc.application.services.hca import HCACalculator

class MyHCACalculator(HCACalculator):
    def __init__(self, profile, input_data):
        super().__init__(profile, input_data)
        # Apply custom filtering to self.elevations
        self.elevations = self.custom_filter(self.elevations)
```

**Custom Output Formatters:**
```python
from trace_calc.infrastructure.output.formatters import OutputFormatter

class MyFormatter(OutputFormatter):
    def format_result(self, result, input_data, geo_data, profile_data):
        # Custom output format
        pass
```

**Custom API Clients:**
```python
from trace_calc.domain.interfaces import BaseElevationsApiClient

class MyElevationsClient(BaseElevationsApiClient):
    async def fetch_elevations(self, coord_vect, block_size):
        # Fetch from alternative source (database, local files, etc.)
        pass
```

**See [examples/custom_model.py](examples/custom_model.py) and [examples/custom_hca_calculator.py](examples/custom_hca_calculator.py) for complete working examples.**

---

## Testing Custom Implementations

### Test Structure

Use pytest with fixtures for reusable test data:

```python
import pytest
from trace_calc.domain.models.path import PathData
from trace_calc.domain.models.coordinates import InputData

@pytest.fixture
def sample_input_data():
    return InputData(
        path_name="test",
        site_a_coordinates=Coordinates(55.7558, 37.6173),
        site_b_coordinates=Coordinates(59.9343, 30.3351),
        frequency_mhz=5000.0,
    )

@pytest.fixture
def sample_path_data():
    return PathData(
        coordinates=np.array([[55.7558, 37.6173], [59.9343, 30.3351]]),
        distances=np.linspace(0, 100, 1000),
        elevations=np.random.randn(1000) * 50 + 200,
    )
```

### Testing Speed Calculators

```python
def test_speed_calculator():
    calculator = MySpeedCalculator()
    total_loss, speed = calculator.calculate_speed(...)

    assert total_loss > 0
    assert speed >= 0
```

### Testing Analyzers

```python
def test_analyzer(sample_path_data, sample_input_data):
    analyzer = MyAnalyzer(sample_path_data, sample_input_data)
    result = analyzer.analyze()

    assert result.link_speed > 0
    assert result.wavelength > 0
    assert "method" in result.model_parameters
```

### Testing with Mocked APIs

```python
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_with_mocked_api():
    # Mock API client
    mock_client = AsyncMock()
    mock_client.fetch_elevations = AsyncMock(return_value=np.array([...]))

    # Use in service
    service = PathProfileService(
        input_data=input_data,
        elevations_api_client=mock_client,
    )

    result = await service.get_profile()
    assert result is not None
```

**See [examples/testing_custom_model.py](examples/testing_custom_model.py) for comprehensive test examples.**

---

## API Quick Reference

### Public Data Models

Located in `trace_calc.domain.models`:

| Model | Purpose | Source |
|-------|---------|--------|
| `InputData` | User input (coordinates, frequency, antenna heights) | [coordinates.py](trace_calc/domain/models/coordinates.py) |
| `PathData` | Path coordinates, distances, elevations | [path.py](trace_calc/domain/models/path.py) |
| `ProfileData` | Sight lines, intersections, volume data | [path.py](trace_calc/domain/models/path.py) |
| `HCAData` | Horizon clearance angles (b1_max, b2_max, b_sum) | [path.py](trace_calc/domain/models/path.py) |
| `GeoData` | Geographic metadata (distance, azimuths, declinations) | [path.py](trace_calc/domain/models/path.py) |
| `AnalysisResult` | Complete analysis result | [analysis.py](trace_calc/domain/models/analysis.py) |
| `PropagationLoss` | Loss components breakdown | [analysis.py](trace_calc/domain/models/analysis.py) |

### Type-Safe Units

Located in `trace_calc.domain.models.units`:

- `Meters`, `Kilometers` - Distances
- `Degrees`, `Angle` - Angular measurements
- `Loss` - Path loss in dB
- `Speed` - Link speed

### Abstract Base Classes

Located in `trace_calc.domain.interfaces`:

| Interface | Purpose | Key Methods |
|-----------|---------|-------------|
| `BaseAnalyzer` | Propagation model analyzer | `analyze(**kwargs)` |
| `BaseSpeedCalculator` | Speed calculation logic | `calculate_speed(*args)` |
| `BaseHCACalculator` | HCA calculation | `calculate_hca()` |
| `BaseElevationsApiClient` | Elevation data source | `fetch_elevations(coord_vect, block_size)` |
| `BaseDeclinationsApiClient` | Declination data source | `fetch_declinations(coordinates)` |
| `BasePathStorage` | Path data persistence | `load(filename)`, `store(filename, data)` |

### Main Services

Located in `trace_calc.application`:

| Service | Purpose | Source |
|---------|---------|--------|
| `OrchestrationService` | Workflow coordination (DI container) | [orchestration.py](trace_calc/application/orchestration.py) |
| `GrozaAnalysisService` | Groza propagation model | [analysis.py](trace_calc/application/analysis.py) |
| `SosnikAnalysisService` | Sosnik propagation model | [analysis.py](trace_calc/application/analysis.py) |
| `PathProfileService` | Elevation profile fetching | [services/profile.py](trace_calc/application/services/profile.py) |
| `CoordinatesService` | Geographic calculations | [services/coordinates.py](trace_calc/application/services/coordinates.py) |
| `GeoDataService` | Geographic metadata | [orchestration.py](trace_calc/application/orchestration.py) |

**Note:** This package uses strict type checking with `mypy`. All public interfaces are fully type-annotated.

---

## Design Patterns Used

This project demonstrates several software design patterns:

### Strategy Pattern
- **Where:** Analysis services (Groza, Sosnik)
- **Why:** Allows switching between different propagation models without changing client code
- **Example:** `OrchestrationService` accepts any `BaseAnalysisService` implementation

### Template Method Pattern
- **Where:** `BaseAnalysisService`, `BaseServiceAnalyzer`
- **Why:** Defines common workflow while allowing subclasses to override specific steps
- **Example:** `BaseAnalysisService.analyze()` calls abstract methods implemented by Groza/Sosnik

### Dependency Injection Pattern
- **Where:** `OrchestrationService`, `AppDependencies`
- **Why:** Decouples components, enables testing, allows swapping implementations
- **Example:** API clients, formatters, and visualizers are injected, not created internally

### Factory Pattern
- **Where:** `get_analysis_service(method)`, `_create_analyzer()`
- **Why:** Encapsulates object creation logic
- **Example:** Factory creates appropriate analyzer based on method name

### Repository Pattern
- **Where:** `FilePathStorage`
- **Why:** Abstracts data persistence, allows switching storage backends
- **Example:** `BasePathStorage` interface with file-based implementation

### Mixin Pattern
- **Where:** `PlotterMixin`, elevation filter mixins
- **Why:** Composes reusable functionality without deep inheritance
- **Example:** `BaseServiceAnalyzer` uses `PlotterMixin` for visualization capability

### Type-Safe Units Pattern
- **Where:** `trace_calc.domain.models.units`
- **Why:** Prevents unit conversion errors at compile time
- **Example:** `Meters`, `Kilometers` use Python `NewType` for type safety

### Immutable Data Structures
- **Where:** All domain models (frozen dataclasses)
- **Why:** Thread-safe, prevents accidental mutations, safe for async contexts
- **Example:** `AnalysisResult`, `PathData`, `GeoData` are all frozen

These patterns work together to create a maintainable, testable, and extensible codebase.
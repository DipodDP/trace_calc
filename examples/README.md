# Examples Guide

This directory contains comprehensive examples demonstrating how to use and extend the `trace_calc` package.

## Overview

The examples are organized by complexity and use case:

1. **Basic Integration** - Using the package in your applications
2. **Custom Models** - Creating custom propagation models
3. **Custom HCA Calculators** - Implementing alternative elevation filtering
4. **Testing** - Writing tests for custom implementations

## Prerequisites

All examples require:
- Python 3.10 or higher
- `trace_calc` package installed
- Valid API keys for elevation and declination services (for examples that fetch real data)

Optional dependencies for some examples:
- `scipy` - For Savitzky-Golay filtering in `custom_hca_calculator.py`
- `pytest` and `pytest-asyncio` - For running test examples

## Example Files

### 1. basic_integration.py

**Purpose:** Demonstrates programmatic usage of the package in other applications (e.g., Telegram bots, web services).

**What it shows:**
- Setting up dependencies without .env file
- Creating input data programmatically
- Running analysis asynchronously
- Processing structured results
- Error handling patterns
- Analyzing multiple paths in parallel

**How to run:**
```bash
# Edit the file to add your API keys in the Config class
python examples/basic_integration.py
```

**Expected output:**
- Elevation profile fetch progress
- Analysis completion message
- Link speed and loss components
- Distance and geographic data

**Key takeaways:**
- How to integrate trace_calc into your Python applications
- How to access analysis results as structured data
- Async/await patterns for non-blocking execution

---

### 2. custom_model.py

**Purpose:** Complete example of creating a custom propagation model.

**What it shows:**
- Creating a `SpeedCalculator` (calculation logic)
- Creating an `Analyzer` (combines calculator with profile analysis)
- Creating an `AnalysisService` (service wrapper for dependency injection)
- Using the custom model with `OrchestrationService`

**How to run:**
```bash
# This is primarily an educational example
# Uncomment the asyncio.run() line at the bottom and add API keys to run
python examples/custom_model.py
```

**Expected output:**
- Educational comments explaining each component
- Example of how the custom model integrates with the system

**Key takeaways:**
- Three-step process for creating custom models
- How to implement the required abstract methods
- Pattern for defining custom speed calculation logic
- How custom models integrate seamlessly with existing infrastructure

**The example model:**
- Simple distance-based propagation (educational, not for production)
- Shows free space loss calculation
- Demonstrates terrain roughness factor
- Uses threshold-based speed determination

---

### 3. custom_hca_calculator.py

**Purpose:** Examples of custom Horizon Clearance Angle (HCA) calculators with different elevation filtering strategies.

**What it shows:**
- Savitzky-Golay filter (preserves peaks while smoothing)
- Moving average filter (simple and fast)
- Median filter (removes spike noise)
- Adaptive filter (chooses strategy based on terrain roughness)

**How to run:**
```bash
python examples/custom_hca_calculator.py
```

**Expected output:**
- Comparison of HCA results from different filtering methods
- b1_max, b2_max, and b_sum values for each filter

**Key takeaways:**
- Mixin pattern for elevation filtering
- How to extend HCACalculator with custom filtering
- When to use different filtering strategies
- Creating adaptive algorithms based on terrain characteristics

**Note:** The Savitzky-Golay example requires scipy. Install with:
```bash
pip install scipy
```

---

### 4. testing_custom_model.py

**Purpose:** Comprehensive test examples for custom models.

**What it shows:**
- Pytest fixtures for reusable test data
- Unit tests for speed calculators
- Unit tests for analyzers
- Async tests for analysis services
- Integration tests with mocked API clients
- Validation and edge case testing

**How to run:**
```bash
# Run all tests
pytest examples/testing_custom_model.py -v

# Run with coverage
pytest examples/testing_custom_model.py --cov=examples.custom_model -v

# Run specific test class
pytest examples/testing_custom_model.py::TestDistanceBasedSpeedCalculator -v
```

**Expected output:**
- Test results showing passed/failed tests
- Coverage report (if using --cov)

**Key takeaways:**
- How to structure tests for custom models
- Creating fixtures for common test data
- Mocking API clients for integration tests
- Testing async code with pytest-asyncio
- Validation strategies for calculators

---

## Common Patterns

### Dependency Injection

All examples follow the dependency injection pattern used throughout trace_calc:

```python
# Create dependencies
elevations_client = AsyncElevationsApiClient(api_url, api_key)
declinations_client = AsyncMagDeclinationApiClient(api_url, api_key)

# Inject into services
orchestrator = OrchestrationService(
    analysis_service=your_custom_service,
    profile_service=profile_service,
    declinations_api_client=declinations_client,
)
```

### Type Safety

All examples maintain strict type checking:

```python
from trace_calc.domain.models.units import Meters, Kilometers, Loss, Speed

# Type-safe units prevent errors
antenna_height = Meters(30.0)  # Not just 30.0
distance = Kilometers(100.0)   # Not just 100.0
```

### Async/Await

Examples use async/await for non-blocking I/O:

```python
async def main():
    result = await orchestrator.process(...)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Troubleshooting

### Import Errors

If you get import errors, ensure trace_calc is installed:
```bash
cd /path/to/trace_calc
poetry install
poetry shell
```

### API Key Errors

Examples that fetch real data need valid API keys:
- Elevation API: https://rapidapi.com/toursprung-toursprung-default/api/maptoolkit
- Declination API: https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml

### Missing Dependencies

Some examples have optional dependencies:
```bash
pip install scipy  # For custom_hca_calculator.py
pip install pytest pytest-asyncio  # For testing_custom_model.py
```

## Next Steps

After reviewing these examples:

1. **For integration:** Start with `basic_integration.py` and adapt it to your use case
2. **For custom models:** Study `custom_model.py` and implement your own propagation model
3. **For custom HCA:** Review `custom_hca_calculator.py` and choose appropriate filtering
4. **For testing:** Use `testing_custom_model.py` as a template for your test suite

## Additional Resources

- Main README: `../README.md` - Complete package documentation
- Architecture Overview: See main README "Architecture Overview" section
- API Reference: See main README "API Quick Reference" section
- Tests directory: `../tests/` - More test examples for built-in models

## Contributing

If you create useful examples or find issues with existing ones, please contribute back to the project!

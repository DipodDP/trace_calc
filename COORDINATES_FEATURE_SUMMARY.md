# ✨ Site Coordinates Output - Summary

## What Changed?

Site coordinates and geographic data are now displayed in all analysis outputs using **clean architecture principles**.

## 🎯 Key Principle

**Get data from its source, don't copy it around!**

- ✅ Coordinates come from `InputData` dataclass
- ✅ Geographic data comes from `GeoData` dataclass
- ✅ Formatters receive these objects directly
- ❌ Analysis services don't copy data to metadata

## Quick Example

**Usage:**
```python
# Create input with coordinates
input_data = InputData(
    path_name='NYC_to_LA',
    site_a_coordinates=Coordinates(lat=40.7128, lon=-74.006),
    site_b_coordinates=Coordinates(lat=34.0522, lon=-118.2437),
)

# Analyze (returns pure calculation results)
result = await analysis_service.analyze(path, input_data)

# Format (formatter gets coordinates from input_data)
formatter.format_result(result, input_data=input_data)
```

**Output:**
```
============================================================
GROZA Analysis Result
============================================================

📍 Site Coordinates:
  Site A:                  40.712800°, -74.006000°
  Site B:                  34.052200°, -118.243700°
  Distance:                3936.87 km

📡 Link Parameters:
  Wavelength:              0.0667 m
  ...
```

## Files Changed

| File | What Changed |
|------|-------------|
| **formatters.py** | Accept `input_data` and `geo_data` parameters |
| **orchestration.py** | Pass `input_data` to formatter |
| **analysis.py** | Simplified `to_dict()` |
| **test_coordinates_output.py** | Updated tests for new API |

**Files NOT changed:**
- ✅ Analysis services remain pure!
- ✅ Domain models stay focused!

## Architecture Benefits

### Before ❌
```
Analysis → Copies coordinates to metadata → Formatter extracts from metadata
           (Tight coupling, duplication)
```

### After ✅
```
InputData (has coordinates)
    ↓
Orchestration passes to both Analysis + Formatter
    ↓
Analysis: Uses for calculation
Formatter: Uses for display
(Loose coupling, single source of truth)
```

## New Output Features

### Console
- 📍 **Site Coordinates** section with:
  - Site A and B lat/lon (6 decimal places)
  - **Distance** between sites (automatically calculated)
- 🌍 **Geographic Data** section (if `geo_data` provided)
  - Distance, azimuths, magnetic declinations

### JSON
- `site_a_coordinates`: `{"lat": ..., "lon": ...}`
- `site_b_coordinates`: `{"lat": ..., "lon": ...}`
- `calculated_distance_km`: Automatically calculated distance
- `geo_data`: Full geographic information object (if provided)

## Testing

```bash
python test_coordinates_output.py
```

Expected: ✅ All tests passed!

## Documentation

- **[COORDINATES_OUTPUT_REFACTORED.md](COORDINATES_OUTPUT_REFACTORED.md)** - Full technical details
- **[EXAMPLE_OUTPUT.md](EXAMPLE_OUTPUT.md)** - Visual examples and use cases

## Clean Architecture Principles Applied

1. ✅ **Separation of Concerns** - Analysis calculates, formatters display
2. ✅ **Single Source of Truth** - Coordinates live in `InputData`
3. ✅ **Dependency Inversion** - Formatters depend on domain models
4. ✅ **Open/Closed** - Easy to add new formatters
5. ✅ **Interface Segregation** - Optional parameters for optional features

## For Developers

**Adding new output format?**
```python
class MyCustomFormatter:
    def format_result(
        self,
        result: AnalysisResult,
        input_data: Optional[InputData] = None,
        geo_data: Optional[GeoData] = None,
    ):
        # Access coordinates directly from input_data
        if input_data and input_data.site_a_coordinates:
            # Your formatting logic here
            ...
```

**Need more geographic data in output?**
1. Add it to `GeoData` dataclass (domain model)
2. Update formatters to display it
3. No changes needed to analysis services!

---

**Bottom Line:** Coordinates are now properly sourced from domain models using dependency injection, following clean architecture principles. Analysis services stay pure and focused on calculations.

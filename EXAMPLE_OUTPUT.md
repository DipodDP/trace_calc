# Example Output with Site Coordinates

## Console Output Example

### Before Changes
```
============================================================
GROZA Analysis Result
============================================================

📡 Link Parameters:
  Wavelength:              0.0667 m
  Frequency:               4500.0 MHz
  Path Distance:           3936.87 km

📉 Propagation Loss:
  Basic Transmission Loss: 150.50 dB
    ├─ Free Space Loss:    130.00 dB
    ├─ Atmospheric Loss:   5.20 dB
    └─ Diffraction Loss:   20.00 dB
  Total Path Loss:         155.20 dB

🚀 Link Performance:
  Estimated Speed:         25.8 Mbps
  Link Margin:             15.3 dB (⚠️  MARGINAL)
============================================================
```

### After Changes ✨
```
============================================================
GROZA Analysis Result
============================================================

📍 Site Coordinates:                           ← NEW SECTION!
  Site A:                  40.712800°, -74.006000°
  Site B:                  34.052200°, -118.243700°

📡 Link Parameters:
  Wavelength:              0.0667 m
  Frequency:               4500.0 MHz
  Path Distance:           3936.87 km

📉 Propagation Loss:
  Basic Transmission Loss: 150.50 dB
    ├─ Free Space Loss:    130.00 dB
    ├─ Atmospheric Loss:   5.20 dB
    └─ Diffraction Loss:   20.00 dB
  Total Path Loss:         155.20 dB

🚀 Link Performance:
  Estimated Speed:         25.8 Mbps
  Link Margin:             15.3 dB (⚠️  MARGINAL)
============================================================
```

## JSON Output Example

### Before Changes
```json
{
  "basic_transmission_loss": 150.5,
  "total_path_loss": 155.2,
  "link_speed": 25.8,
  "wavelength": 0.0667,
  "propagation_loss": {
    "free_space_loss": 130.0,
    "atmospheric_loss": 5.2,
    "diffraction_loss": 20.0,
    "total_loss": 155.2
  },
  "metadata": {
    "method": "groza",
    "version": "1.0",
    "distance_km": 3936.87,
    "frequency_mhz": 4500.0,
    "theta_a_mrad": 5.234,
    "theta_b_mrad": 4.892,
    "link_margin_db": 15.3
  }
}
```

### After Changes ✨
```json
{
  "basic_transmission_loss": 150.5,
  "total_path_loss": 155.2,
  "link_speed": 25.8,
  "wavelength": 0.0667,
  "propagation_loss": {
    "free_space_loss": 130.0,
    "atmospheric_loss": 5.2,
    "diffraction_loss": 20.0,
    "total_loss": 155.2
  },
  "metadata": {
    "method": "groza",
    "version": "1.0",
    "distance_km": 3936.87,
    "frequency_mhz": 4500.0,
    "theta_a_mrad": 5.234,
    "theta_b_mrad": 4.892,
    "link_margin_db": 15.3,
    "site_a_coordinates": {              ← NEW!
      "lat": 40.7128,
      "lon": -74.006
    },
    "site_b_coordinates": {              ← NEW!
      "lat": 34.0522,
      "lon": -118.2437
    }
  }
}
```

## Real-World Example: NYC to LA Link

### Site Details
- **Site A**: New York City
  - Latitude: 40.712800°N
  - Longitude: 74.006000°W

- **Site B**: Los Angeles
  - Latitude: 34.052200°N
  - Longitude: 118.243700°W

### Analysis Output
```
============================================================
GROZA Analysis Result
============================================================

📍 Site Coordinates:
  Site A:                  40.712800°, -74.006000°
  Site B:                  34.052200°, -118.243700°

📡 Link Parameters:
  Wavelength:              0.0667 m
  Frequency:               4500.0 MHz
  Path Distance:           3936.87 km

📉 Propagation Loss:
  Basic Transmission Loss: 198.35 dB
    ├─ Free Space Loss:    175.22 dB
    ├─ Atmospheric Loss:   15.48 dB
    └─ Diffraction Loss:   7.65 dB
  Total Path Loss:         198.35 dB

🚀 Link Performance:
  Estimated Speed:         0.3 Mbps
  Link Margin:             -83.4 dB (❌ POOR)
============================================================

Note: This is a very long path (3,936 km) which results in high
propagation loss. Troposcatter links are typically used for shorter
distances (100-400 km).
```

## Use Cases

### 1. Report Generation
The coordinates make it easy to:
- Generate PDF reports with exact site locations
- Create maps showing the link path
- Document installations for maintenance teams

### 2. Database Storage
JSON format allows easy storage in databases:
```sql
INSERT INTO link_analyses (
    site_a_lat, site_a_lon,
    site_b_lat, site_b_lon,
    distance_km, total_loss, link_speed
) VALUES (
    40.7128, -74.0060,
    34.0522, -118.2437,
    3936.87, 198.35, 0.3
);
```

### 3. API Integration
External systems can extract coordinates:
```python
import json
import requests

response = requests.get('/api/analyze')
result = json.loads(response.text)

site_a = result['metadata']['site_a_coordinates']
site_b = result['metadata']['site_b_coordinates']

print(f"Link from ({site_a['lat']}, {site_a['lon']}) "
      f"to ({site_b['lat']}, {site_b['lon']})")
```

### 4. GIS/Mapping Tools
Coordinates can be imported into GIS software:
```python
# Export to KML for Google Earth
kml = f"""
<Placemark>
  <name>Site A</name>
  <Point>
    <coordinates>{site_a['lon']},{site_a['lat']},0</coordinates>
  </Point>
</Placemark>
<Placemark>
  <name>Site B</name>
  <Point>
    <coordinates>{site_b['lon']},{site_b['lat']},0</coordinates>
  </Point>
</Placemark>
"""
```

## Precision

Coordinates are displayed with **6 decimal places** for precision:
- 6 decimal places ≈ 0.11 meters precision at the equator
- Sufficient for site location identification
- Follows common GPS coordinate conventions

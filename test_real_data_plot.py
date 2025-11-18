#!/usr/bin/env python3
"""
Test script to plot real terrain data from test.path file.

Reads binary path data and creates curved profile with:
- Sites at endpoints
- Lower sight lines just above terrain
- Extended visibility features
"""

import struct
import numpy as np
import matplotlib.pyplot as plt


def read_path_file(file_path: str) -> tuple:
    """
    Read binary path data file.

    Format: latitude (f8), longitude (f8), distance (f8), elevation (f8)
    Each record is 32 bytes (4 * 8-byte doubles)

    Returns:
        (coordinates, distances, elevations)
    """
    with open(file_path, 'rb') as f:
        data = f.read()

    # Each record is 4 doubles (32 bytes): lat, lon, distance, elevation
    record_size = 32
    num_records = len(data) // record_size

    print(f"File size: {len(data)} bytes")
    print(f"Record size: {record_size} bytes")
    print(f"Number of records: {num_records}")

    latitudes = []
    longitudes = []
    elevations = []
    distances = []

    for i in range(num_records):
        offset = i * record_size
        record = data[offset:offset + record_size]

        # Unpack 4 doubles (little-endian): lat, lon, distance, elevation
        lat, lon, dist, elev = struct.unpack('<dddd', record)

        latitudes.append(lat)
        longitudes.append(lon)
        elevations.append(elev)
        distances.append(dist)

    # Convert to numpy arrays
    coordinates = np.column_stack([latitudes, longitudes])
    distances_km = np.array(distances)
    elevations_m = np.array(elevations)

    print(f"\nData ranges:")
    print(f"  Latitudes: {min(latitudes):.6f} to {max(latitudes):.6f}")
    print(f"  Longitudes: {min(longitudes):.6f} to {max(longitudes):.6f}")
    print(f"  Elevations: {min(elevations_m):.1f} to {max(elevations_m):.1f} m")
    print(f"  Distances: {min(distances_km):.3f} to {max(distances_km):.3f} km")

    return coordinates, distances_km, elevations_m


def apply_earth_curvature(distances_km: np.ndarray) -> np.ndarray:
    """
    Apply geometric Earth curvature correction.

    Args:
        distances_km: Distance array in kilometers

    Returns:
        Curvature correction array in meters (negative values)
    """
    # Earth radius in km
    R = 6371.0

    # Calculate reference point (middle of path)
    ref_distance = distances_km[len(distances_km) // 2]

    # Curvature correction: h = d^2 / (2*R)
    # Relative to reference point
    curvature = -(distances_km - ref_distance) ** 2 / (2 * R * 1000)

    return curvature


def calculate_lower_sight_lines(
    distances: np.ndarray,
    elevations_curved: np.ndarray,
    antenna_a_height: float = 30.0,
    antenna_b_height: float = 30.0,
    obstacle_margin: float = 5.0
) -> tuple:
    """
    Calculate lower sight lines that clear the terrain.

    Uses HCA (Horizon Clearance Angle) method to find critical obstacles.

    Args:
        distances: Distance array (km)
        elevations_curved: Curved elevations (m)
        antenna_a_height: Site A antenna height (m)
        antenna_b_height: Site B antenna height (m)
        obstacle_margin: Height above terrain for obstacle points (m)

    Returns:
        (lower_a, lower_b, obstacle_idx_a, obstacle_idx_b)
    """
    # Site positions
    site_a_x = distances[0]
    site_b_x = distances[-1]
    site_a_y = elevations_curved[0] + antenna_a_height
    site_b_y = elevations_curved[-1] + antenna_b_height

    # Find obstacles using HCA method (Horizon Clearance Angle)
    # From Site A: find point with maximum elevation angle
    max_angle_a = -np.inf
    obstacle_idx_a = len(distances) // 2

    for i in range(1, len(distances) - 1):
        # Elevation angle from site A to point i
        dx = distances[i] - site_a_x
        if dx > 0:  # Only look forward from A
            angle = np.arctan((elevations_curved[i] - site_a_y) / (dx * 1000))  # Convert km to m
            if angle > max_angle_a:
                max_angle_a = angle
                obstacle_idx_a = i

    # From Site B: find point with maximum elevation angle
    max_angle_b = -np.inf
    obstacle_idx_b = len(distances) // 2

    for i in range(1, len(distances) - 1):
        # Elevation angle from site B to point i
        dx = site_b_x - distances[i]
        if dx > 0:  # Only look backward from B (towards A)
            angle = np.arctan((elevations_curved[i] - site_b_y) / (dx * 1000))  # Convert km to m
            if angle > max_angle_b:
                max_angle_b = angle
                obstacle_idx_b = i

    # Calculate sight lines through obstacle points (with margin)
    obstacle_a_x = distances[obstacle_idx_a]
    obstacle_a_y = elevations_curved[obstacle_idx_a] + obstacle_margin

    obstacle_b_x = distances[obstacle_idx_b]
    obstacle_b_y = elevations_curved[obstacle_idx_b] + obstacle_margin

    # Lower line A: from site A through obstacle A
    k_lower_a = (obstacle_a_y - site_a_y) / (obstacle_a_x - site_a_x)
    b_lower_a = site_a_y - k_lower_a * site_a_x
    lower_a = np.array([k_lower_a, b_lower_a])

    # Lower line B: from site B through obstacle B
    k_lower_b = (obstacle_b_y - site_b_y) / (obstacle_b_x - site_b_x)
    b_lower_b = site_b_y - k_lower_b * site_b_x
    lower_b = np.array([k_lower_b, b_lower_b])

    print(f"\nSight line calculation:")
    print(f"  Site A: ({site_a_x:.3f} km, {site_a_y:.1f} m)")
    print(f"  Obstacle A: ({obstacle_a_x:.3f} km, {obstacle_a_y:.1f} m) [index {obstacle_idx_a}]")
    print(f"  Lower line A: y = {k_lower_a:.4f}*x + {b_lower_a:.2f}")
    print(f"  Site B: ({site_b_x:.3f} km, {site_b_y:.1f} m)")
    print(f"  Obstacle B: ({obstacle_b_x:.3f} km, {obstacle_b_y:.1f} m) [index {obstacle_idx_b}]")
    print(f"  Lower line B: y = {k_lower_b:.4f}*x + {b_lower_b:.2f}")

    return lower_a, lower_b, obstacle_idx_a, obstacle_idx_b


def calculate_upper_sight_lines(
    lower_a: np.ndarray,
    lower_b: np.ndarray,
    site_a_pos: tuple,
    site_b_pos: tuple,
    angle_offset_deg: float = 2.5
) -> tuple:
    """
    Calculate upper sight lines by rotating lower lines.

    Args:
        lower_a: Lower line A coefficients [k, b]
        lower_b: Lower line B coefficients [k, b]
        site_a_pos: Site A position (x, y)
        site_b_pos: Site B position (x, y)
        angle_offset_deg: Angular offset in degrees

    Returns:
        (upper_a, upper_b)
    """
    angle_rad = np.deg2rad(angle_offset_deg)

    # Rotate lower_a around site A (counterclockwise = upward)
    # IMPORTANT: slope is in m/km, so convert to actual geometric slope (m/m)
    k_lower_a = lower_a[0]
    slope_actual_a = k_lower_a / 1000  # Convert m/km to m/m
    current_angle_a = np.arctan(slope_actual_a)
    new_angle_a = current_angle_a + angle_rad
    slope_actual_upper_a = np.tan(new_angle_a)
    k_upper_a = slope_actual_upper_a * 1000  # Convert back to m/km
    b_upper_a = site_a_pos[1] - k_upper_a * site_a_pos[0]
    upper_a = np.array([k_upper_a, b_upper_a])

    # Rotate lower_b around site B
    # IMPORTANT: slope is in m/km, so convert to actual geometric slope (m/m)
    k_lower_b = lower_b[0]
    slope_actual_b = k_lower_b / 1000  # Convert m/km to m/m
    current_angle_b = np.arctan(slope_actual_b)

    # Determine rotation direction based on slope
    if k_lower_b < 0:  # Descending line
        new_angle_b = current_angle_b - angle_rad
    else:  # Ascending line or horizontal
        new_angle_b = current_angle_b + angle_rad

    slope_actual_upper_b = np.tan(new_angle_b)
    k_upper_b = slope_actual_upper_b * 1000  # Convert back to m/km
    b_upper_b = site_b_pos[1] - k_upper_b * site_b_pos[0]
    upper_b = np.array([k_upper_b, b_upper_b])

    # Print angular separation
    print(f"\nUpper sight lines (offset: {angle_offset_deg}°):")
    print(f"  Site A angle: {np.rad2deg(current_angle_a):.2f}° → {np.rad2deg(new_angle_a):.2f}° (Δ={np.rad2deg(new_angle_a - current_angle_a):.2f}°)")
    print(f"  Lower line A: slope={k_lower_a:.4f} m/km")
    print(f"  Upper line A: slope={k_upper_a:.4f} m/km")
    print(f"  Site B angle: {np.rad2deg(current_angle_b):.2f}° → {np.rad2deg(new_angle_b):.2f}° (Δ={np.rad2deg(new_angle_b - current_angle_b):.2f}°)")
    print(f"  Lower line B: slope={k_lower_b:.4f} m/km")
    print(f"  Upper line B: slope={k_upper_b:.4f} m/km")

    return upper_a, upper_b


def calculate_line_intersection(line1: np.ndarray, line2: np.ndarray) -> tuple:
    """Calculate intersection of two lines."""
    k1, b1 = line1
    k2, b2 = line2

    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1

    return x, y


def plot_real_data_profile(
    distances: np.ndarray,
    elevations: np.ndarray,
    elevations_curved: np.ndarray,
    lower_a: np.ndarray,
    lower_b: np.ndarray,
    upper_a: np.ndarray,
    upper_b: np.ndarray,
    antenna_a_height: float,
    antenna_b_height: float,
    obstacle_idx_a: int,
    obstacle_idx_b: int,
    save_path: str = "test.png"
):
    """
    Create detailed profile plot with real data.
    """
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(19.20, 10.8))

    # ========================================================================
    # PANEL 1: PLAIN ELEVATION PROFILE
    # ========================================================================

    axes[0].plot(distances, elevations, 'k-', linewidth=1.0, label='Terrain')
    axes[0].fill_between(distances, 0, elevations, alpha=0.3, color='tan')

    # Mark sites
    axes[0].scatter([distances[0]], [elevations[0] + antenna_a_height],
                    c='red', s=150, marker='^', label='Site A', zorder=5,
                    edgecolors='black', linewidths=1.5)
    axes[0].scatter([distances[-1]], [elevations[-1] + antenna_b_height],
                    c='blue', s=150, marker='^', label='Site B', zorder=5,
                    edgecolors='black', linewidths=1.5)

    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Distance (km)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Elevation (m)', fontsize=11, fontweight='bold')
    axes[0].set_title('Plain Elevation Profile (Real Data)', fontsize=13, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)

    # ========================================================================
    # PANEL 2: CURVED PROFILE WITH EXTENDED VISIBILITY
    # ========================================================================

    # Use actual elevations (no shift) to show true sea level reference
    shift = 0
    elevations_for_plot = elevations_curved

    # Plot terrain
    axes[1].plot(distances, elevations_for_plot, 'k-', linewidth=1.0,
                 label='Terrain (curved)', zorder=1)
    axes[1].fill_between(distances, 0, elevations_for_plot, alpha=0.3,
                         color='tan', zorder=0)

    # Plot sites (actual elevations)
    site_a_y_plot = elevations_curved[0] + antenna_a_height
    site_b_y_plot = elevations_curved[-1] + antenna_b_height

    axes[1].scatter([distances[0]], [site_a_y_plot],
                    c='red', s=150, marker='^', label='Site A', zorder=6,
                    edgecolors='black', linewidths=1.5)
    axes[1].scatter([distances[-1]], [site_b_y_plot],
                    c='blue', s=150, marker='^', label='Site B', zorder=6,
                    edgecolors='black', linewidths=1.5)

    # Mark obstacle points (actual elevations)
    axes[1].scatter([distances[obstacle_idx_a]],
                    [elevations_curved[obstacle_idx_a]],
                    c='orange', s=80, marker='o', label='Obstacle A', zorder=5,
                    edgecolors='black', linewidths=0.8)
    axes[1].scatter([distances[obstacle_idx_b]],
                    [elevations_curved[obstacle_idx_b]],
                    c='cyan', s=80, marker='o', label='Obstacle B', zorder=5,
                    edgecolors='black', linewidths=0.8)

    # Plot sight lines (actual elevations)
    lower_line_a = np.polyval(lower_a, distances)
    lower_line_b = np.polyval(lower_b, distances)
    upper_line_a = np.polyval(upper_a, distances)
    upper_line_b = np.polyval(upper_b, distances)

    axes[1].plot(distances, lower_line_a, 'r-', linewidth=1.8,
                 label='Lower sight line A', zorder=3, alpha=0.9)
    axes[1].plot(distances, lower_line_b, 'b-', linewidth=1.8,
                 label='Lower sight line B', zorder=3, alpha=0.9)
    axes[1].plot(distances, upper_line_a, 'r--', linewidth=1.8, alpha=0.7,
                 label='Upper sight line A', zorder=3)
    axes[1].plot(distances, upper_line_b, 'b--', linewidth=1.8, alpha=0.7,
                 label='Upper sight line B', zorder=3)

    # Calculate and plot intersections
    x_lower, y_lower = calculate_line_intersection(lower_a, lower_b)
    x_upper, y_upper = calculate_line_intersection(upper_a, upper_b)
    x_cross_ab, y_cross_ab = calculate_line_intersection(upper_a, lower_b)
    x_cross_ba, y_cross_ba = calculate_line_intersection(upper_b, lower_a)

    # Scale y-axis to include upper intersection with some margin
    all_y_values = [
        y_lower,
        y_upper,
        y_cross_ab,
        y_cross_ba,
        elevations_for_plot.max(),
        site_a_y_plot,
        site_b_y_plot,
        0  # Include sea level
    ]
    y_min = min(all_y_values) - 100  # Add 100m margin below
    y_max = max(all_y_values) + 100  # Add 100m margin above
    axes[1].set_ylim(y_min, y_max)

    axes[1].scatter([x_lower], [y_lower], c='green', s=120, marker='o',
                    label='Lower intersection', zorder=5,
                    edgecolors='black', linewidths=0.8)
    axes[1].scatter([x_upper], [y_upper], c='purple', s=120, marker='o',
                    label='Upper intersection', zorder=5,
                    edgecolors='black', linewidths=0.8)
    axes[1].scatter([x_cross_ab], [y_cross_ab], c='orange', s=100, marker='^',
                    label='Cross AB', zorder=5,
                    edgecolors='black', linewidths=0.8)
    axes[1].scatter([x_cross_ba], [y_cross_ba], c='cyan', s=100, marker='v',
                    label='Cross BA', zorder=5,
                    edgecolors='black', linewidths=0.8)

    # Add metrics annotation
    metrics_text = (
        f"Path length: {distances[-1]:.2f} km\n"
        f"Lower intersection: {x_lower:.2f} km, {y_lower:.1f} m\n"
        f"Upper intersection: {x_upper:.2f} km, {y_upper:.1f} m\n"
        f"Antenna heights: A={antenna_a_height:.0f}m, B={antenna_b_height:.0f}m"
    )

    axes[1].text(0.02, 0.98, metrics_text,
                transform=axes[1].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
                family='monospace')

    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Distance (km)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Elevation (m)', fontsize=11, fontweight='bold')
    axes[1].set_title('Curved Profile with Extended Visibility Analysis (Real Data)',
                      fontsize=13, fontweight='bold')
    axes[1].legend(loc='best', fontsize=8, ncol=2)

    # Save and display
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("REAL DATA PROFILE VISUALIZATION")
    print("="*70)

    # Read binary data
    print("\nReading test.path file...")
    coordinates, distances_km, elevations_m = read_path_file("test.path")

    # Apply Earth curvature
    print("\nApplying Earth curvature correction...")
    curvature = apply_earth_curvature(distances_km)
    elevations_curved = elevations_m + curvature
    print(f"  Curvature range: {curvature.min():.2f} to {curvature.max():.2f} m")

    # Calculate sight lines
    print("\nCalculating sight lines...")
    antenna_a = 30.0  # meters
    antenna_b = 30.0  # meters

    lower_a, lower_b, obs_idx_a, obs_idx_b = calculate_lower_sight_lines(
        distances_km, elevations_curved, antenna_a, antenna_b, obstacle_margin=5.0
    )

    # Calculate upper sight lines
    site_a_pos = (distances_km[0], elevations_curved[0] + antenna_a)
    site_b_pos = (distances_km[-1], elevations_curved[-1] + antenna_b)

    upper_a, upper_b = calculate_upper_sight_lines(
        lower_a, lower_b, site_a_pos, site_b_pos, angle_offset_deg=2.5
    )

    # Create plot
    print("\nGenerating visualization...")
    plot_real_data_profile(
        distances_km, elevations_m, elevations_curved,
        lower_a, lower_b, upper_a, upper_b,
        antenna_a, antenna_b, obs_idx_a, obs_idx_b,
        save_path="test.png"
    )

    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

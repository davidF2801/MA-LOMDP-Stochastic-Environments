"""Visualize agent orbit trajectories in 3D."""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

# Import trajectory utilities
try:
    if __package__:
        from .agents.trajectory import Trajectory
    else:
        raise ImportError
except ImportError:
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment_simulation.agents.trajectory import Trajectory


def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude (degrees) to 3D Cartesian coordinates on a sphere.
    
    Args:
        lat: Latitude in degrees (can be array)
        lon: Longitude in degrees (can be array)
        radius: Radius of the sphere
    
    Returns:
        Tuple of (x, y, z) arrays
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def angular_distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle angular distance (degrees) between two points on a sphere.
    
    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)
    
    Returns:
        Angular distance in degrees
    """
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = np.mod(lon2_rad - lon1_rad + np.pi, 2 * np.pi) - np.pi
    
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def is_ground_station_visited(
    ground_station_lat: float,
    ground_station_lon: float,
    trajectory: Trajectory,
    field_of_regard_deg: float,
) -> bool:
    """
    Check if a ground station is visited by a satellite at any point in its orbit.
    
    Args:
        ground_station_lat: Ground station latitude (degrees)
        ground_station_lon: Ground station longitude (degrees)
        trajectory: Satellite trajectory
        field_of_regard_deg: Satellite's field of regard radius (degrees)
    
    Returns:
        True if ground station is within FOR at any point in the orbit
    """
    for step in range(trajectory.period):
        sat_lat, sat_lon = trajectory.position_at(step)
        distance = angular_distance_deg(sat_lat, sat_lon, ground_station_lat, ground_station_lon)
        if distance <= field_of_regard_deg:
            return True
    return False


def refine_intersection(
    trajectory1: Trajectory,
    trajectory2: Trajectory,
    initial_step1: float,
    initial_step2: float,
    orbit_period: int,
    tolerance: float = 0.01,
    max_iterations: int = 50,
) -> tuple[float, float, float, float] | None:
    """
    Refine an intersection point using numerical optimization.
    
    Finds the exact intersection by minimizing the distance between orbits.
    
    Args:
        trajectory1: First satellite trajectory
        trajectory2: Second satellite trajectory
        initial_step1: Initial guess for step on trajectory1
        initial_step2: Initial guess for step on trajectory2
        orbit_period: Number of steps in orbit period
        tolerance: Convergence tolerance (degrees)
        max_iterations: Maximum number of iterations
    
    Returns:
        Tuple of (refined_step1, refined_step2, lat, lon) or None if no convergence
    """
    step1 = initial_step1 % orbit_period
    step2 = initial_step2 % orbit_period
    
    for iteration in range(max_iterations):
        # Get positions at current steps
        lat1, lon1 = trajectory1.position_at(int(step1))
        lat2, lon2 = trajectory2.position_at(int(step2))
        
        # Current distance
        distance = angular_distance_deg(lat1, lon1, lat2, lon2)
        
        if distance < tolerance:
            # Converged! Use midpoint for exact intersection
            mid_lat = (lat1 + lat2) / 2.0
            
            # Handle longitude wrapping
            lon_diff = abs(lon2 - lon1)
            if lon_diff > 180:
                mid_lon = ((lon1 + lon2 + 360) / 2.0) % 360.0
            else:
                mid_lon = (lon1 + lon2) / 2.0
            
            return (step1, step2, mid_lat, mid_lon)
        
        # Compute gradient by sampling nearby points
        # Try small perturbations to find which direction minimizes distance
        best_step1 = step1
        best_step2 = step2
        best_distance = distance
        
        # Sample nearby points
        delta = 0.5
        for ds1 in [-delta, 0, delta]:
            for ds2 in [-delta, 0, delta]:
                if ds1 == 0 and ds2 == 0:
                    continue
                
                test_step1 = (step1 + ds1) % orbit_period
                test_step2 = (step2 + ds2) % orbit_period
                
                test_lat1, test_lon1 = trajectory1.position_at(int(test_step1))
                test_lat2, test_lon2 = trajectory2.position_at(int(test_step2))
                test_distance = angular_distance_deg(test_lat1, test_lon1, test_lat2, test_lon2)
                
                if test_distance < best_distance:
                    best_distance = test_distance
                    best_step1 = test_step1
                    best_step2 = test_step2
        
        # If no improvement, use gradient descent with smaller step
        if best_distance >= distance:
            # Try smaller steps
            delta = 0.1
            for ds1 in [-delta, 0, delta]:
                for ds2 in [-delta, 0, delta]:
                    if ds1 == 0 and ds2 == 0:
                        continue
                    
                    test_step1 = (step1 + ds1) % orbit_period
                    test_step2 = (step2 + ds2) % orbit_period
                    
                    test_lat1, test_lon1 = trajectory1.position_at(int(test_step1))
                    test_lat2, test_lon2 = trajectory2.position_at(int(test_step2))
                    test_distance = angular_distance_deg(test_lat1, test_lon1, test_lat2, test_lon2)
                    
                    if test_distance < best_distance:
                        best_distance = test_distance
                        best_step1 = test_step1
                        best_step2 = test_step2
        
        # If still no improvement, we're at a minimum
        if best_distance >= distance:
            # Use midpoint of current best
            mid_lat = (lat1 + lat2) / 2.0
            lon_diff = abs(lon2 - lon1)
            if lon_diff > 180:
                mid_lon = ((lon1 + lon2 + 360) / 2.0) % 360.0
            else:
                mid_lon = (lon1 + lon2) / 2.0
            return (step1, step2, mid_lat, mid_lon)
        
        step1 = best_step1
        step2 = best_step2
    
    # Didn't converge, but return best estimate
    lat1, lon1 = trajectory1.position_at(int(step1))
    lat2, lon2 = trajectory2.position_at(int(step2))
    mid_lat = (lat1 + lat2) / 2.0
    lon_diff = abs(lon2 - lon1)
    if lon_diff > 180:
        mid_lon = ((lon1 + lon2 + 360) / 2.0) % 360.0
    else:
        mid_lon = (lon1 + lon2) / 2.0
    return (step1, step2, mid_lat, mid_lon)


def find_orbit_intersections(
    trajectory1: Trajectory,
    trajectory2: Trajectory,
    orbit_period: int,
    max_distance_deg: float = 2.0,
) -> List[tuple[float, float]]:
    """
    Find exact intersection points between two orbits.
    
    Uses numerical refinement to compute precise intersection locations.
    
    Args:
        trajectory1: First satellite trajectory
        trajectory2: Second satellite trajectory
        orbit_period: Number of steps in orbit period
        max_distance_deg: Maximum angular distance to consider an intersection (degrees)
    
    Returns:
        List of (lat, lon) tuples representing exact intersection points
    """
    intersections = []
    
    # First pass: find candidate intersections at high resolution
    sample_step = max(1, orbit_period // 200)  # Very high resolution for initial search
    
    # Track which intersections we've found (to avoid duplicates)
    found_intersections = []
    
    for step1 in range(0, orbit_period, sample_step):
        lat1, lon1 = trajectory1.position_at(step1)
        
        # Find closest point on trajectory2
        min_distance = float('inf')
        closest_step2 = -1
        
        for step2 in range(orbit_period):
            lat2, lon2 = trajectory2.position_at(step2)
            distance = angular_distance_deg(lat1, lon1, lat2, lon2)
            
            if distance < min_distance:
                min_distance = distance
                closest_step2 = step2
        
        # If orbits are close enough, refine the intersection
        if min_distance <= max_distance_deg:
            # Refine using numerical optimization
            refined = refine_intersection(
                trajectory1, trajectory2, step1, closest_step2, orbit_period,
                tolerance=0.1, max_iterations=100
            )
            
            if refined is not None:
                refined_step1, refined_step2, refined_lat, refined_lon = refined
                
                # Verify the refined point is actually an intersection
                final_lat1, final_lon1 = trajectory1.position_at(int(refined_step1))
                final_lat2, final_lon2 = trajectory2.position_at(int(refined_step2))
                final_distance = angular_distance_deg(final_lat1, final_lon1, final_lat2, final_lon2)
                
                if final_distance <= max_distance_deg * 1.5:  # Allow slightly more tolerance
                    # Check if this intersection is far enough from existing ones
                    is_new = True
                    for existing_lat, existing_lon in intersections:
                        if angular_distance_deg(refined_lat, refined_lon, existing_lat, existing_lon) < max_distance_deg * 1.5:
                            is_new = False
                            break
                    
                    if is_new:
                        intersections.append((refined_lat, refined_lon))
                        found_intersections.append((refined_step1, refined_step2, refined_lat, refined_lon))
    
    return intersections


def find_ground_stations(
    satellite_configs: List[Dict],
    orbit_period: int,
    num_stations: int = 3,
    min_satellites_per_station: int = 2,
) -> List[Dict]:
    """
    Find ground station locations at orbit intersection points.
    Ground stations must be visited by at least min_satellites_per_station satellites.
    
    Args:
        satellite_configs: List of satellite configuration dictionaries
        orbit_period: Number of steps in orbit period
        num_stations: Number of ground stations to find
        min_satellites_per_station: Minimum number of satellites that must visit each station
    
    Returns:
        List of ground station dictionaries with keys: name, lat, lon, visited_by (list of satellite names)
    """
    # Create trajectories for all satellites
    trajectories = []
    for config in satellite_configs:
        trajectory = Trajectory.circular_orbit(
            period=orbit_period,
            inclination_deg=config["inclination_deg"],
            phase_deg=config["phase_deg"],
            latitude_offset=config["latitude_offset"],
        )
        trajectories.append((config["name"], trajectory, config.get("field_of_regard_deg", 20.0)))
    
    # Find all intersection points between pairs of orbits
    # Organize intersections by pair to ensure we get one from each pair
    intersections_by_pair = {}
    all_intersection_candidates = []
    
    print(f"Finding orbit intersection points...")
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            sat1_name, traj1, for1 = trajectories[i]
            sat2_name, traj2, for2 = trajectories[j]
            
            pair_key = (i, j)
            pair_name = f"{sat1_name}-{sat2_name}"
            
            # Find intersections between this pair of orbits
            # Use tighter tolerance for better accuracy
            intersections = find_orbit_intersections(traj1, traj2, orbit_period, max_distance_deg=8.0)
            
            if intersections:
                print(f"  Found {len(intersections)} intersection points between {sat1_name} and {sat2_name}")
                # Store intersections for this pair
                intersections_by_pair[pair_key] = intersections
                all_intersection_candidates.extend(intersections)
            else:
                print(f"  No intersection points found between {sat1_name} and {sat2_name}")
                intersections_by_pair[pair_key] = []
    
    # Remove duplicates (intersections that are very close together)
    unique_candidates = []
    for lat, lon in all_intersection_candidates:
        is_duplicate = False
        for u_lat, u_lon in unique_candidates:
            if angular_distance_deg(lat, lon, u_lat, u_lon) < 3.0:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_candidates.append((lat, lon))
    
    print(f"Found {len(unique_candidates)} unique intersection points")
    
    # Ensure we have at least one ground station from each orbit pair
    ground_stations = []
    pairs_covered = set()
    
    # First pass: select one intersection from each pair
    for pair_key, intersections in intersections_by_pair.items():
        if len(intersections) == 0:
            continue
        
        if pair_key in pairs_covered:
            continue
        
        # Try each intersection from this pair until we find one that works
        # Sort intersections by how close the two orbits are (prefer tighter intersections)
        intersection_distances = []
        for gs_lat, gs_lon in intersections:
            i, j = pair_key
            traj1 = trajectories[i][1]
            traj2 = trajectories[j][1]
            
            # Find minimum distance between orbits at this intersection point
            min_dist = float('inf')
            for step in range(orbit_period):
                lat1, lon1 = traj1.position_at(step)
                dist1 = angular_distance_deg(lat1, lon1, gs_lat, gs_lon)
                if dist1 < min_dist:
                    min_dist = dist1
            
            intersection_distances.append((min_dist, gs_lat, gs_lon))
        
        # Sort by distance (tighter intersections first)
        intersection_distances.sort(key=lambda x: x[0])
        
        for min_dist, gs_lat, gs_lon in intersection_distances:
            if len(ground_stations) >= num_stations:
                break
            
            # Check which satellites can visit this intersection point
            visited_by = []
            for sat_name, trajectory, for_deg in trajectories:
                if is_ground_station_visited(gs_lat, gs_lon, trajectory, for_deg):
                    visited_by.append(sat_name)
            
            # Verify that both satellites from this pair can visit it
            i, j = pair_key
            sat1_name = trajectories[i][0]
            sat2_name = trajectories[j][0]
            
            both_can_visit = sat1_name in visited_by and sat2_name in visited_by
            
            # Only consider if visited by enough satellites AND both satellites from the pair can visit
            if len(visited_by) >= min_satellites_per_station and both_can_visit:
                # Check if this candidate is too close to already selected stations
                too_close = False
                for existing in ground_stations:
                    if angular_distance_deg(gs_lat, gs_lon, existing["lat"], existing["lon"]) < 10.0:
                        too_close = True
                        break
                
                if not too_close:
                    ground_stations.append({
                        "name": f"Ground Station {len(ground_stations) + 1} ({sat1_name}-{sat2_name})",
                        "lat": gs_lat,
                        "lon": gs_lon,
                        "visited_by": visited_by,
                    })
                    pairs_covered.add(pair_key)
                    print(f"  Selected intersection from {sat1_name}-{sat2_name}: "
                          f"lat={gs_lat:.2f}°, lon={gs_lon:.2f}° "
                          f"(min orbit distance: {min_dist:.2f}°, "
                          f"visited by {len(visited_by)} satellites: {', '.join(visited_by)})")
                    break
        
        if pair_key not in pairs_covered:
            print(f"  Warning: Could not find valid ground station for pair {pair_key}")
    
    # Second pass: fill remaining slots with any good intersections
    for gs_lat, gs_lon in unique_candidates:
        if len(ground_stations) >= num_stations:
            break
        
        # Skip if we already have this intersection
        already_included = False
        for existing in ground_stations:
            if angular_distance_deg(gs_lat, gs_lon, existing["lat"], existing["lon"]) < 3.0:
                already_included = True
                break
        
        if already_included:
            continue
        
        # Check which satellites can visit this intersection point
        visited_by = []
        for sat_name, trajectory, for_deg in trajectories:
            if is_ground_station_visited(gs_lat, gs_lon, trajectory, for_deg):
                visited_by.append(sat_name)
        
        # Only consider if visited by enough satellites
        if len(visited_by) >= min_satellites_per_station:
            # Check if this candidate is too close to already selected stations
            too_close = False
            for existing in ground_stations:
                if angular_distance_deg(gs_lat, gs_lon, existing["lat"], existing["lon"]) < 10.0:
                    too_close = True
                    break
            
            if not too_close:
                ground_stations.append({
                    "name": f"Ground Station {len(ground_stations) + 1}",
                    "lat": gs_lat,
                    "lon": gs_lon,
                    "visited_by": visited_by,
                })
    
    # If we didn't find enough at intersections, also check points where all orbits come close
    if len(ground_stations) < num_stations:
        print(f"Found {len(ground_stations)} ground stations at intersections.")
        print(f"Searching for points where multiple orbits come close...")
        
        # Find points where multiple orbits are close together
        sample_step = max(1, orbit_period // 50)
        
        for step in range(0, orbit_period, sample_step):
            if len(ground_stations) >= num_stations:
                break
            
            # Get all satellite positions at this step
            positions = []
            for sat_name, trajectory, _ in trajectories:
                lat, lon = trajectory.position_at(step)
                positions.append((sat_name, lat, lon))
            
            # Check if these positions are close together (within ~10 degrees)
            for i, (name1, lat1, lon1) in enumerate(positions):
                for j in range(i + 1, len(positions)):
                    name2, lat2, lon2 = positions[j]
                    distance = angular_distance_deg(lat1, lon1, lat2, lon2)
                    
                    # If two satellites are close, use midpoint as candidate
                    if distance <= 10.0:
                        mid_lat = (lat1 + lat2) / 2.0
                        mid_lon = (lon1 + lon2) / 2.0
                        
                        # Check if too close to existing stations
                        too_close = False
                        for existing in ground_stations:
                            if angular_distance_deg(mid_lat, mid_lon, existing["lat"], existing["lon"]) < 10.0:
                                too_close = True
                                break
                        
                        if too_close:
                            continue
                        
                        # Check how many satellites can visit this point
                        visited_by = []
                        for sat_name, trajectory, for_deg in trajectories:
                            if is_ground_station_visited(mid_lat, mid_lon, trajectory, for_deg):
                                visited_by.append(sat_name)
                        
                        if len(visited_by) >= min_satellites_per_station:
                            ground_stations.append({
                                "name": f"Ground Station {len(ground_stations) + 1}",
                                "lat": mid_lat,
                                "lon": mid_lon,
                                "visited_by": visited_by,
                            })
                            
                            if len(ground_stations) >= num_stations:
                                break
    
    return ground_stations


def plot_agent_orbits(
    agent_configs: List[Dict],
    orbit_period: int = 240,
    radius: float = 1.0,
    show_sphere: bool = True,
    show_for_samples: bool = False,
    num_for_samples: int = 8,
    for_opacity: float = 0.2,
    ground_stations: List[Dict] | None = None,
    save_path: str | None = None,
    dpi: int = 150,
    title: str | None = None,
    show_grid: bool = False,
    grid_height: int = 15,
    grid_width: int = 30,
):
    """
    Create a 3D visualization of agent orbit trajectories.
    
    Args:
        agent_configs: List of agent configuration dictionaries with keys:
            - name: Agent name
            - inclination_deg: Orbit inclination amplitude
            - phase_deg: Starting longitude offset
            - latitude_offset: Vertical offset of orbit
            - field_of_regard_deg: Observation radius
            - color: Hex color for visualization
        orbit_period: Number of discrete steps in one orbit period
        radius: Radius of the sphere for visualization
        show_sphere: Whether to show a wireframe sphere as reference
        show_for_samples: Whether to show field of regard at sample points along orbit
        num_for_samples: Number of sample points along orbit to show FOR
        for_opacity: Opacity of field of regard visualizations
        ground_stations: List of ground station dictionaries
        save_path: Optional path to save the figure
        dpi: Resolution for saved figure
        title: Optional title for the plot
        show_grid: Whether to show the simulation grid points
        grid_height: Grid height (latitude resolution) - matches main_mc_agents.py
        grid_width: Grid width (longitude resolution) - matches main_mc_agents.py
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    
    # Create a sphere wireframe for reference
    if show_sphere:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x_sphere = radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(
            x_sphere, y_sphere, z_sphere,
            color="#e0e0e0",
            alpha=0.15,
            linewidth=0.5,
            edgecolor="#b0b0b0",
            zorder=0
        )
    
    # Plot each agent's orbit
    for config in agent_configs:
        # Create trajectory from config
        trajectory = Trajectory.circular_orbit(
            period=orbit_period,
            inclination_deg=config["inclination_deg"],
            phase_deg=config["phase_deg"],
            latitude_offset=config["latitude_offset"],
        )
        
        # Convert trajectory to 3D coordinates
        x, y, z = latlon_to_xyz(trajectory.latitudes, trajectory.longitudes, radius)
        
        # Plot the orbit trajectory
        ax.plot(
            x, y, z,
            color=config["color"],
            linewidth=2.5,
            label=config["name"],
            zorder=2
        )
        
        # Optionally show field of regard at sample points
        if show_for_samples:
            field_of_regard_deg = config.get("field_of_regard_deg", 20.0)
            
            # Sample evenly spaced points along the orbit
            sample_indices = np.linspace(0, orbit_period - 1, num_for_samples, dtype=int)
            
            for idx in sample_indices:
                lat, lon = trajectory.position_at(idx)
                
                # Create a circle of points at the FOR distance
                # Use a more efficient method: create a circle in local coordinates
                # then rotate to the agent's position
                num_circle_points = 30
                angles = np.linspace(0, 2 * np.pi, num_circle_points)
                
                # Create circle at FOR distance in local coordinates
                # This is a circle on the sphere at constant angular distance
                lat_rad = np.deg2rad(lat)
                lon_rad = np.deg2rad(lon)
                for_rad = np.deg2rad(field_of_regard_deg)
                
                # Create circle points using spherical coordinates
                # For a circle at constant angular distance from a point on a sphere
                for_lat_list = []
                for_lon_list = []
                
                for angle in angles:
                    # Create a point on the circle
                    # Simplified: create points that form a circle at the FOR distance
                    # Using a small circle approximation
                    dlat = for_rad * np.cos(angle) / np.cos(lat_rad)
                    dlon = for_rad * np.sin(angle) / np.cos(lat_rad)
                    
                    circle_lat = lat + np.rad2deg(dlat)
                    circle_lon = lon + np.rad2deg(dlon)
                    
                    circle_lat = np.clip(circle_lat, -90, 90)
                    circle_lon = circle_lon % 360
                    
                    for_lat_list.append(circle_lat)
                    for_lon_list.append(circle_lon)
                
                if for_lat_list:
                    for_lat = np.array(for_lat_list)
                    for_lon = np.array(for_lon_list)
                    for_x, for_y, for_z = latlon_to_xyz(for_lat, for_lon, radius)
                    
                    # Draw the circle as a line with markers
                    ax.plot(
                        for_x, for_y, for_z,
                        color=config["color"],
                        alpha=for_opacity,
                        linewidth=1.5,
                        linestyle="--",
                        zorder=1
                    )
                    ax.scatter(
                        for_x, for_y, for_z,
                        color=config["color"],
                        alpha=for_opacity * 0.7,
                        s=20,
                        zorder=1
                    )
        
        # Mark starting point
        start_x, start_y, start_z = latlon_to_xyz(
            trajectory.latitudes[0], trajectory.longitudes[0], radius
        )
        ax.scatter(
            [start_x], [start_y], [start_z],
            color=config["color"],
            s=100,
            marker="o",
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
            label=f"{config['name']} (start)"
        )
    
    # Plot grid points if requested (matching main_mc_agents.py grid)
    if show_grid:
        # Create the same grid as in main_mc_agents.py
        lat_centers = 90.0 - (np.arange(grid_height) + 0.5) * (180.0 / grid_height)
        lon_centers = (np.arange(grid_width) + 0.5) * (360.0 / grid_width)
        lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing="ij")
        
        # Convert grid points to 3D coordinates
        grid_x, grid_y, grid_z = latlon_to_xyz(lat_grid.flatten(), lon_grid.flatten(), radius)
        
        # Plot grid points as small gray dots
        ax.scatter(
            grid_x, grid_y, grid_z,
            color="#888888",
            s=8,
            alpha=0.3,
            marker=".",
            zorder=0.5,
            label=f"Grid ({grid_height}x{grid_width})"
        )
    
    # Plot ground stations if provided
    if ground_stations:
        for i, station in enumerate(ground_stations):
            gs_x, gs_y, gs_z = latlon_to_xyz(station["lat"], station["lon"], radius)
            
            # Plot ground station as a large marker
            ax.scatter(
                [gs_x], [gs_y], [gs_z],
                color="#00ff00",  # Green
                s=300,
                marker="^",  # Triangle pointing up
                edgecolor="black",
                linewidth=2,
                zorder=4,
                label=f"{station['name']} ({len(station['visited_by'])} sats)"
            )
            
            # Draw lines to visiting satellites at sample points
            for sat_config in agent_configs:
                if sat_config["name"] in station["visited_by"]:
                    # Find a point along the satellite's orbit where it's closest to the station
                    trajectory = Trajectory.circular_orbit(
                        period=orbit_period,
                        inclination_deg=sat_config["inclination_deg"],
                        phase_deg=sat_config["phase_deg"],
                        latitude_offset=sat_config["latitude_offset"],
                    )
                    
                    min_distance = float('inf')
                    closest_step = 0
                    for step in range(0, trajectory.period, max(1, trajectory.period // 20)):
                        sat_lat, sat_lon = trajectory.position_at(step)
                        distance = angular_distance_deg(
                            sat_lat, sat_lon, station["lat"], station["lon"]
                        )
                        if distance < min_distance:
                            min_distance = distance
                            closest_step = step
                    
                    # Only draw line if satellite is within FOR
                    if min_distance <= sat_config.get("field_of_regard_deg", 20.0):
                        sat_lat, sat_lon = trajectory.position_at(closest_step)
                        sat_x, sat_y, sat_z = latlon_to_xyz(sat_lat, sat_lon, radius)
                        
                        # Draw dashed line from satellite to ground station
                        ax.plot(
                            [sat_x, gs_x],
                            [sat_y, gs_y],
                            [sat_z, gs_z],
                            color=sat_config["color"],
                            linestyle=":",
                            linewidth=1.5,
                            alpha=0.4,
                            zorder=1
                        )
    
    # Set axis labels
    ax.set_xlabel("X", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y", fontsize=11, fontweight="bold")
    ax.set_zlabel("Z", fontsize=11, fontweight="bold")
    
    # Set equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) * 0.5
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) * 0.5
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=10)
    
    # Add title
    if title:
        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    else:
        plt.suptitle("Agent Orbit Trajectories", fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Orbit visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_default_satellite_configs() -> List[Dict]:
    """
    Get default satellite configurations for 3 satellites.
    
    These are designed to provide good coverage and ensure ground stations
    can be visited by at least 2 satellites.
    """
    return [
        {
            "name": "Satellite-1",
            "inclination_deg": 40.0,
            "phase_deg": 0.0,
            "latitude_offset": 0.0,
            "field_of_regard_deg": 25.0,
            "color": "#ff6b6b",  # Red
        },
        {
            "name": "Satellite-2",
            "inclination_deg": 40.0,
            "phase_deg": 120.0,
            "latitude_offset": 0.0,
            "field_of_regard_deg": 25.0,
            "color": "#4ecdc4",  # Turquoise
        },
        {
            "name": "Satellite-3",
            "inclination_deg": 40.0,
            "phase_deg": 240.0,
            "latitude_offset": 0.0,
            "field_of_regard_deg": 25.0,
            "color": "#ffd166",  # Gold
        },
    ]


def get_default_greedy_configs() -> List[Dict]:
    """Get default greedy agent configurations from main_mc_agents.py."""
    return [
        {
            "name": "Greedy-1",
            "inclination_deg": 25.0,
            "phase_deg": 0.0,
            "latitude_offset": 0.0,
            "field_of_regard_deg": 20.0,
            "color": "#ffd166",
        },
        {
            "name": "Greedy-2",
            "inclination_deg": 55.0,
            "phase_deg": 120.0,
            "latitude_offset": 5.0,
            "field_of_regard_deg": 20.0,
            "color": "#4ecdc4",
        },
        {
            "name": "Greedy-3",
            "inclination_deg": 10.0,
            "phase_deg": 240.0,
            "latitude_offset": -8.0,
            "field_of_regard_deg": 20.0,
            "color": "#ff6b6b",
        },
    ]


def get_default_monte_carlo_configs() -> List[Dict]:
    """Get default Monte Carlo agent configurations from main_mc_agents.py."""
    return [
        {
            "name": "Aurora-1",
            "inclination_deg": 25.0,
            "phase_deg": 0.0,
            "latitude_offset": 0.0,
            "field_of_regard_deg": 20.0,
            "color": "#ffd166",
        },
        {
            "name": "Borealis-2",
            "inclination_deg": 55.0,
            "phase_deg": 120.0,
            "latitude_offset": 5.0,
            "field_of_regard_deg": 20.0,
            "color": "#4ecdc4",
        },
        {
            "name": "Zenith-3",
            "inclination_deg": 10.0,
            "phase_deg": 240.0,
            "latitude_offset": -8.0,
            "field_of_regard_deg": 20.0,
            "color": "#ff6b6b",
        },
    ]


def get_default_sharing_configs() -> List[Dict]:
    """Get default Sharing Monte Carlo agent configurations from main_mc_agents.py."""
    # These are the exact same as monte_carlo configs in main_mc_agents.py
    return [
        {
            "name": "Aurora-1",
            "inclination_deg": 25.0,
            "phase_deg": 0.0,
            "latitude_offset": 0.0,
            "field_of_regard_deg": 20.0,
            "color": "#ffd166",
        },
        {
            "name": "Borealis-2",
            "inclination_deg": 55.0,
            "phase_deg": 120.0,
            "latitude_offset": 5.0,
            "field_of_regard_deg": 20.0,
            "color": "#4ecdc4",
        },
        {
            "name": "Zenith-3",
            "inclination_deg": 10.0,
            "phase_deg": 240.0,
            "latitude_offset": -8.0,
            "field_of_regard_deg": 20.0,
            "color": "#ff6b6b",
        },
    ]


def get_default_random_configs() -> List[Dict]:
    """Get default random agent configurations from main_mc_agents.py."""
    return [
        {
            "name": "Random-1",
            "inclination_deg": 25.0,
            "phase_deg": 0.0,
            "latitude_offset": 0.0,
            "field_of_regard_deg": 20.0,
            "color": "#ffd166",
        },
        {
            "name": "Random-2",
            "inclination_deg": 55.0,
            "phase_deg": 120.0,
            "latitude_offset": 5.0,
            "field_of_regard_deg": 20.0,
            "color": "#4ecdc4",
        },
        {
            "name": "Random-3",
            "inclination_deg": 10.0,
            "phase_deg": 240.0,
            "latitude_offset": -8.0,
            "field_of_regard_deg": 20.0,
            "color": "#ff6b6b",
        },
    ]


def main():
    """Main function to visualize orbits."""
    parser = argparse.ArgumentParser(description="Visualize agent orbit trajectories in 3D")
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["greedy", "monte_carlo", "random", "sharing", "satellites", "all"],
        default="sharing",
        help="Agent type to visualize (default: sharing)"
    )
    parser.add_argument(
        "--ground-stations",
        type=int,
        default=3,
        help="Number of ground stations to find and visualize (default: 3)"
    )
    parser.add_argument(
        "--min-satellites-per-station",
        type=int,
        default=2,
        help="Minimum number of satellites that must visit each ground station (default: 2)"
    )
    parser.add_argument(
        "--orbit-period",
        type=int,
        default=240,
        help="Number of steps in one orbit period (default: 240)"
    )
    parser.add_argument(
        "--show-for",
        action="store_true",
        help="Show field of regard at sample points along orbits"
    )
    parser.add_argument(
        "--num-for-samples",
        type=int,
        default=8,
        help="Number of sample points to show FOR (default: 8)"
    )
    parser.add_argument(
        "--no-sphere",
        action="store_true",
        help="Don't show the reference sphere wireframe"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the visualization (default: display interactively)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for saved figure (default: 150)"
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Show the simulation grid points (matching main_mc_agents.py)"
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=15,
        help="Grid height (latitude resolution) - matches main_mc_agents.py debug mode (default: 15)"
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=30,
        help="Grid width (longitude resolution) - matches main_mc_agents.py debug mode (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Get agent configurations
    ground_stations = None
    if args.agent_type == "satellites":
        configs = get_default_satellite_configs()
        title = "Satellite Orbits with Ground Stations"
        
        # Find ground stations visited by at least min_satellites_per_station satellites
        print(f"Finding {args.ground_stations} ground stations visited by ≥{args.min_satellites_per_station} satellites...")
        ground_stations = find_ground_stations(
            configs,
            args.orbit_period,
            num_stations=args.ground_stations,
            min_satellites_per_station=args.min_satellites_per_station,
        )
        
        if ground_stations:
            print(f"\nFound {len(ground_stations)} ground stations:")
            for station in ground_stations:
                print(f"  - {station['name']}: "
                      f"lat={station['lat']:.1f}°, lon={station['lon']:.1f}° "
                      f"visited by {len(station['visited_by'])} satellites: {', '.join(station['visited_by'])}")
        else:
            print("Warning: No ground stations found that meet the requirements.")
    elif args.agent_type == "greedy":
        configs = get_default_greedy_configs()
        title = "Greedy Agent Orbit Trajectories"
    elif args.agent_type == "monte_carlo":
        configs = get_default_monte_carlo_configs()
        title = "Monte Carlo Agent Orbit Trajectories"
    elif args.agent_type == "random":
        configs = get_default_random_configs()
        title = "Random Agent Orbit Trajectories"
    elif args.agent_type == "sharing":
        configs = get_default_sharing_configs()
        title = "Sharing Monte Carlo Agent Orbit Trajectories"
        
        # Find ground stations visited by at least min_satellites_per_station satellites
        print(f"Finding {args.ground_stations} ground stations visited by ≥{args.min_satellites_per_station} satellites...")
        ground_stations = find_ground_stations(
            configs,
            args.orbit_period,
            num_stations=args.ground_stations,
            min_satellites_per_station=args.min_satellites_per_station,
        )
        
        if ground_stations:
            print(f"\nFound {len(ground_stations)} ground stations:")
            for station in ground_stations:
                print(f"  - {station['name']}: "
                      f"lat={station['lat']:.1f}°, lon={station['lon']:.1f}° "
                      f"visited by {len(station['visited_by'])} satellites: {', '.join(station['visited_by'])}")
        else:
            print("Warning: No ground stations found that meet the requirements.")
    else:  # all
        configs = (
            get_default_greedy_configs() +
            get_default_monte_carlo_configs() +
            get_default_random_configs() +
            get_default_sharing_configs()
        )
        title = "All Agent Orbit Trajectories"
    
    print(f"Visualizing {len(configs)} agent orbits...")
    for config in configs:
        print(f"  - {config['name']}: "
              f"inc={config['inclination_deg']}°, "
              f"phase={config['phase_deg']}°, "
              f"offset={config['latitude_offset']}°, "
              f"FOR={config['field_of_regard_deg']}°")
    
    # Create visualization
    plot_agent_orbits(
        agent_configs=configs,
        orbit_period=args.orbit_period,
        show_sphere=not args.no_sphere,
        show_for_samples=args.show_for,
        num_for_samples=args.num_for_samples,
        ground_stations=ground_stations,
        save_path=args.save,
        dpi=args.dpi,
        title=title,
        show_grid=args.show_grid,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()


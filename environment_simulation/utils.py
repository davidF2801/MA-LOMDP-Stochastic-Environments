"""Utility functions for environment_simulation package.

This module provides helpers for:
- Computing great-circle distances
- Finding orbit intersections
- Creating ground station locations from agent orbit configurations
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .agents.trajectory import Trajectory
from .agents.ground_station import GroundStation, create_ground_stations_from_dicts


def angular_distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle angular distance (degrees) between two points on a sphere."""
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
    """Check if a ground station is visited by a satellite at any point in its orbit."""
    # Sample points along the orbit
    sample_step = max(1, trajectory.period // 40)
    for step in range(0, trajectory.period, sample_step):
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
    tolerance: float = 0.05,
    max_iterations: int = 80,
) -> tuple[float, float, float, float] | None:
    """Refine an intersection point between two trajectories using local search.

    Returns (step1, step2, lat, lon) or None if no convergence.
    """
    step1 = initial_step1 % orbit_period
    step2 = initial_step2 % orbit_period

    for _ in range(max_iterations):
        lat1, lon1 = trajectory1.position_at(int(step1))
        lat2, lon2 = trajectory2.position_at(int(step2))

        distance = angular_distance_deg(lat1, lon1, lat2, lon2)
        if distance < tolerance:
            # Converged: use midpoint as intersection
            mid_lat = (lat1 + lat2) / 2.0
            lon_diff = abs(lon2 - lon1)
            if lon_diff > 180:
                mid_lon = ((lon1 + lon2 + 360.0) / 2.0) % 360.0
            else:
                mid_lon = (lon1 + lon2) / 2.0
            return (step1, step2, mid_lat, mid_lon)

        # Explore neighborhood in step space
        best_step1 = step1
        best_step2 = step2
        best_distance = distance

        delta = 0.5
        for ds1 in (-delta, 0.0, delta):
            for ds2 in (-delta, 0.0, delta):
                if ds1 == 0.0 and ds2 == 0.0:
                    continue
                t1 = (step1 + ds1) % orbit_period
                t2 = (step2 + ds2) % orbit_period
                test_lat1, test_lon1 = trajectory1.position_at(int(t1))
                test_lat2, test_lon2 = trajectory2.position_at(int(t2))
                test_distance = angular_distance_deg(test_lat1, test_lon1, test_lat2, test_lon2)
                if test_distance < best_distance:
                    best_distance = test_distance
                    best_step1 = t1
                    best_step2 = t2

        if best_distance >= distance:
            # Use midpoint and stop
            mid_lat = (lat1 + lat2) / 2.0
            lon_diff = abs(lon2 - lon1)
            if lon_diff > 180:
                mid_lon = ((lon1 + lon2 + 360.0) / 2.0) % 360.0
            else:
                mid_lon = (lon1 + lon2) / 2.0
            return (step1, step2, mid_lat, mid_lon)

        step1 = best_step1
        step2 = best_step2

    # Did not converge; return best estimate
    lat1, lon1 = trajectory1.position_at(int(step1))
    lat2, lon2 = trajectory2.position_at(int(step2))
    mid_lat = (lat1 + lat2) / 2.0
    lon_diff = abs(lon2 - lon1)
    if lon_diff > 180:
        mid_lon = ((lon1 + lon2 + 360.0) / 2.0) % 360.0
    else:
        mid_lon = (lon1 + lon2) / 2.0
    return (step1, step2, mid_lat, mid_lon)


def find_orbit_intersections(
    trajectory1: Trajectory,
    trajectory2: Trajectory,
    orbit_period: int,
    max_distance_deg: float = 2.0,
) -> List[tuple[float, float]]:
    """Find approximate intersection points between two orbits.

    Returns a list of (lat, lon) points lying on both trajectories, computed
    by scanning over steps of trajectory1 and finding closest points on trajectory2,
    then refining with local search.
    """
    intersections: List[tuple[float, float]] = []

    # Initial coarse search: scan trajectory1 in small step increments
    sample_step = max(1, orbit_period // 200)

    for step1 in range(0, orbit_period, sample_step):
        lat1, lon1 = trajectory1.position_at(step1)

        # Find closest point on trajectory2 (full scan over its period)
        min_distance = float("inf")
        closest_step2 = 0
        for step2 in range(orbit_period):
            lat2, lon2 = trajectory2.position_at(step2)
            distance = angular_distance_deg(lat1, lon1, lat2, lon2)
            if distance < min_distance:
                min_distance = distance
                closest_step2 = step2

        # If orbits are close enough, refine
        if min_distance <= max_distance_deg:
            refined = refine_intersection(
                trajectory1, trajectory2, step1, closest_step2, orbit_period, tolerance=0.1, max_iterations=80
            )
            if refined is None:
                continue

            _, _, refined_lat, refined_lon = refined

            # Ensure uniqueness (avoid duplicates)
            is_new = True
            for existing_lat, existing_lon in intersections:
                if angular_distance_deg(refined_lat, refined_lon, existing_lat, existing_lon) < max_distance_deg:
                    is_new = False
                    break

            if is_new:
                intersections.append((refined_lat, refined_lon))

    return intersections


def create_ground_stations_from_agent_configs(
    agent_configs: List[Dict[str, Any]],
    orbit_period: int = 240,
    num_stations: int = 3,
    min_satellites_per_station: int = 2,
    communication_range_deg: float = 5.0,
) -> List[GroundStation]:
    """Create ground stations at orbit intersection points from agent configs.

    This uses a geometric notion of orbit intersection: we find points on the
    sphere where pairs of orbits pass very close (independently of time),
    then place ground stations there. Each ground station must be visited
    by at least `min_satellites_per_station` satellites.
    """
    # Build trajectories for each agent
    trajectories: List[tuple[str, Trajectory, float]] = []
    for config in agent_configs:
        traj = Trajectory.circular_orbit(
            period=orbit_period,
            inclination_deg=config["inclination_deg"],
            phase_deg=config["phase_deg"],
            latitude_offset=config["latitude_offset"],
        )
        for_deg = float(config.get("field_of_regard_deg", 20.0))
        trajectories.append((str(config["name"]), traj, for_deg))

    # Collect intersection candidates from all pairs
    intersection_candidates: List[Dict[str, Any]] = []
    n = len(trajectories)
    for i in range(n):
        for j in range(i + 1, n):
            name1, traj1, _ = trajectories[i]
            name2, traj2, _ = trajectories[j]

            # Use a larger max_distance to find more intersection opportunities
            # This helps ensure we find intersections for all orbit pairs
            pair_intersections = find_orbit_intersections(traj1, traj2, orbit_period, max_distance_deg=6.0)
            for lat, lon in pair_intersections:
                intersection_candidates.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "pair": (name1, name2),
                    }
                )

    # Deduplicate candidates that are too close
    # Use a larger threshold (at least 2x FOR) to ensure stations are well-separated
    # This prevents one agent from seeing multiple stations simultaneously
    min_separation_deg = 40.0  # At least 2x the FOR (20°) to ensure separation
    unique_candidates: List[Dict[str, Any]] = []
    for cand in intersection_candidates:
        lat, lon = cand["lat"], cand["lon"]
        is_duplicate = False
        for existing in unique_candidates:
            if angular_distance_deg(lat, lon, existing["lat"], existing["lon"]) < min_separation_deg:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_candidates.append(cand)

    # Evaluate each candidate: which satellites can visit?
    # Prioritize stations that more agents can reach (especially to ensure all agents can reach at least one)
    ground_station_dicts: List[Dict[str, Any]] = []
    used_pairs: set = set()  # Track which orbit pairs we've used
    
    # First pass: prioritize stations with more agents (to ensure coverage)
    # Sort candidates by number of agents that can visit them
    candidates_with_visitors = []
    for cand in unique_candidates:
        lat = cand["lat"]
        lon = cand["lon"]
        visited_by: List[str] = []
        # Use communication range instead of FOR to check if agents can visit
        for sat_name, traj, for_deg in trajectories:
            if is_ground_station_visited(lat, lon, traj, communication_range_deg):
                visited_by.append(sat_name)
        
        if len(visited_by) >= min_satellites_per_station:
            candidates_with_visitors.append((cand, visited_by, len(visited_by)))
    
    # Track which agents have at least one station they can reach
    agents_with_stations = set()
    all_agent_names = {name for name, _, _ in trajectories}
    
    # Sort candidates to prioritize:
    # 1. Stations that include Polaris-4 or Vega-5 (if they don't have stations yet)
    # 2. Stations that help uncovered agents
    # 3. Stations with more total visitors
    def sort_key(item):
        cand, visited_by, num_visitors = item
        visited_set = set(visited_by)
        # Check if this includes Polaris-4 or Vega-5
        has_polaris = "Polaris-4" in visited_set and "Polaris-4" not in agents_with_stations
        has_vega = "Vega-5" in visited_set and "Vega-5" not in agents_with_stations
        new_agents = visited_set - agents_with_stations
        num_new = len(new_agents)
        
        # Priority: Polaris/Vega > other new agents > total visitors
        return (has_polaris or has_vega, num_new, num_visitors)
    
    candidates_with_visitors.sort(key=sort_key, reverse=True)
    
    # First pass: Ensure each agent gets at least one station they can reach
    # Prioritize stations that include agents without any stations yet
    remaining_candidates = []
    for cand, visited_by, num_visitors in candidates_with_visitors:
        if len(ground_station_dicts) >= num_stations:
            break

        lat = cand["lat"]
        lon = cand["lon"]
        pair = cand["pair"]
        name1, name2 = pair
        
        # Check if this station is too close to existing ones
        too_close = False
        for existing in ground_station_dicts:
            if angular_distance_deg(lat, lon, existing["lat"], existing["lon"]) < min_separation_deg:
                too_close = True
                break
        
        if too_close:
            remaining_candidates.append((cand, visited_by, num_visitors))
            continue
        
        # Check if this station helps agents that don't have stations yet
        new_agents = set(visited_by) - agents_with_stations
        helps_uncovered = len(new_agents) > 0
        
        # Prioritize stations that help uncovered agents, or stations with more total visitors
        if helps_uncovered or len(ground_station_dicts) == 0:
            ground_station_dicts.append(
                {
                    "name": f"Ground Station {len(ground_station_dicts) + 1} ({name1}-{name2})",
                    "lat": lat,
                    "lon": lon,
                    "visited_by": visited_by,
                }
            )
            used_pairs.add(pair)
            agents_with_stations.update(visited_by)
        else:
            remaining_candidates.append((cand, visited_by, num_visitors))
    
    # Second pass: Fill remaining slots with stations sorted by number of visitors
    remaining_candidates.sort(key=lambda x: x[2], reverse=True)
    for cand, visited_by, num_visitors in remaining_candidates:
        if len(ground_station_dicts) >= num_stations:
            break

        lat = cand["lat"]
        lon = cand["lon"]
        pair = cand["pair"]
        name1, name2 = pair
        
        # Check if this station is too close to existing ones
        too_close = False
        for existing in ground_station_dicts:
            if angular_distance_deg(lat, lon, existing["lat"], existing["lon"]) < min_separation_deg:
                too_close = True
                break
        
        if not too_close:
            ground_station_dicts.append(
                {
                    "name": f"Ground Station {len(ground_station_dicts) + 1} ({name1}-{name2})",
                    "lat": lat,
                    "lon": lon,
                    "visited_by": visited_by,
                }
            )
            used_pairs.add(pair)
            agents_with_stations.update(visited_by)

    # If still not enough, relax criteria slightly by re-using remaining candidates
    if len(ground_station_dicts) < num_stations:
        for cand in unique_candidates:
            if len(ground_station_dicts) >= num_stations:
                break

            lat = cand["lat"]
            lon = cand["lon"]

            # Skip if already very close to an existing GS
            too_close = False
            for existing in ground_station_dicts:
                if angular_distance_deg(lat, lon, existing["lat"], existing["lon"]) < 5.0:
                    too_close = True
                    break
            if too_close:
                continue

            visited_by: List[str] = []
            # Use communication range instead of FOR to check if agents can visit
            for sat_name, traj, for_deg in trajectories:
                if is_ground_station_visited(lat, lon, traj, communication_range_deg):
                    visited_by.append(sat_name)

            if len(visited_by) >= min_satellites_per_station:
                ground_station_dicts.append(
                    {
                        "name": f"Ground Station {len(ground_station_dicts) + 1}",
                        "lat": lat,
                        "lon": lon,
                        "visited_by": visited_by,
                    }
                )

    # DEBUG: Print ground station info before conversion
    if ground_station_dicts:
        print(f"\n[GS Creation] Created {len(ground_station_dicts)} ground stations:")
        for gs_dict in ground_station_dicts:
            visited_by = gs_dict.get("visited_by", [])
            print(f"  - {gs_dict['name']}: lat={gs_dict['lat']:.2f}°, lon={gs_dict['lon']:.2f}° "
                  f"| Should be visited by: {', '.join(visited_by)}")
            
            # DEBUG: Show minimum distance each agent reaches during full orbit
            print(f"    Distance analysis (min distance each agent reaches during orbit):")
            for sat_name, traj, for_deg in trajectories:
                min_dist = float('inf')
                min_step = -1
                # Check every step in the orbit
                for step in range(traj.period):
                    sat_lat, sat_lon = traj.position_at(step)
                    dist = angular_distance_deg(sat_lat, sat_lon, gs_dict['lat'], gs_dict['lon'])
                    if dist < min_dist:
                        min_dist = dist
                        min_step = step
                in_range = min_dist <= communication_range_deg
                print(f"      {sat_name}: min_distance={min_dist:.2f}° (at step {min_step}) "
                      f"{'✓ within comm range' if in_range else f'✗ outside comm range (need ≤{communication_range_deg}°)'}")
        print()
    
    # Convert dicts to GroundStation instances
    ground_stations: List[GroundStation] = create_ground_stations_from_dicts(ground_station_dicts)

    return ground_stations



"""Abstract agent class for planner testing framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .belief import Belief
from .trajectory import Trajectory


def _angular_distance_deg(lat1: float, lon1: float, lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
    """Great-circle distance (degrees) between a point and arrays of points."""
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat_grid)
    lon2_rad = np.deg2rad(lon_grid)

    dlat = lat2_rad - lat1_rad
    dlon = np.mod(lon2_rad - lon1_rad + np.pi, 2 * np.pi) - np.pi

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


@dataclass
class Agent(ABC):
    """
    Abstract agent class for planner testing framework.
    
    Agents maintain a belief over the environment state and can act based on their policy.
    Belief updates are limited to cells within the agent's field of regard along its trajectory.
    """

    name: str
    trajectory: Trajectory
    belief: Belief
    field_of_regard_deg: float = 10.0
    color: str = "#ffdd57"
    # Policy storage (subclasses can use this to store policy data)
    _policy: Optional[Any] = field(default=None, init=False, repr=False)

    def position(self, step: int) -> tuple[float, float]:
        """Latitude/longitude (degrees) at a given time step."""
        return self.trajectory.position_at(step)
    
    def compute_all_for_locations(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        max_steps: Optional[int] = None,
    ) -> set[tuple[float, float]]:
        """
        Pre-compute all (lat, lon) locations that will be in the field of regard
        over the entire trajectory period.
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            max_steps: Maximum number of steps to check. If None, uses trajectory period.
        
        Returns:
            Set of (lat, lon) tuples that will be in the FOR at some point
        """
        if max_steps is None:
            max_steps = self.trajectory.period
        
        all_locations = set()
        
        # Check FOR at each step in the trajectory period
        for step in range(max_steps):
            for_mask = self.coverage_mask(lat_grid, lon_grid, step)
            for y in range(lat_grid.shape[0]):
                for x in range(lat_grid.shape[1]):
                    if for_mask[y, x]:
                        lat, lon = float(lat_grid[y, x]), float(lon_grid[y, x])
                        all_locations.add((lat, lon))
        
        return all_locations

    def coverage_mask(self, lat_grid: np.ndarray, lon_grid: np.ndarray, step: int) -> np.ndarray:
        """
        Boolean mask of cells inside the field of regard at the given step.
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            
        Returns:
            Boolean (H, W) mask indicating which cells are within field of regard
        """
        lat, lon = self.position(step)
        distances = _angular_distance_deg(lat, lon, lat_grid, lon_grid)
        return distances <= max(0.0, float(self.field_of_regard_deg))
    
    @staticmethod
    def points_to_mask(
        observation_points: set[tuple[float, float]],
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Convert set of (lat, lon) points to boolean mask.
        
        Args:
            observation_points: Set of (lat, lon) tuples
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            height: Grid height
            width: Grid width
            
        Returns:
            Boolean (H, W) mask indicating which cells are in observation_points
        """
        mask = np.zeros((height, width), dtype=bool)
        
        # Create a mapping from (lat, lon) to (y, x) for efficient lookup
        latlon_to_grid = {}
        for y in range(height):
            for x in range(width):
                lat, lon = float(lat_grid[y, x]), float(lon_grid[y, x])
                latlon_to_grid[(lat, lon)] = (y, x)
        
        # Mark cells in observation_points
        for lat, lon in observation_points:
            lat_key = float(lat)
            lon_key = float(lon)
            # Try exact match first
            grid_coords = latlon_to_grid.get((lat_key, lon_key))
            # If not found, try tolerance-based search
            if grid_coords is None:
                for (g_lat, g_lon), (gy, gx) in latlon_to_grid.items():
                    if abs(g_lat - lat_key) < 1e-6 and abs(g_lon - lon_key) < 1e-6:
                        grid_coords = (gy, gx)
                        break
            
            if grid_coords is not None:
                y, x = grid_coords
                mask[y, x] = True
        
        return mask
    
    @staticmethod
    def mask_to_points(
        observation_mask: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
    ) -> set[tuple[float, float]]:
        """
        Convert boolean mask to set of (lat, lon) points.
        
        Args:
            observation_mask: Boolean (H, W) mask
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            
        Returns:
            Set of (lat, lon) tuples for cells where mask is True
        """
        observation_points = set()
        for y in range(lat_grid.shape[0]):
            for x in range(lon_grid.shape[1]):
                if observation_mask[y, x]:
                    lat, lon = lat_grid[y, x], lon_grid[y, x]
                    observation_points.add((float(lat), float(lon)))
        return observation_points

    def update_belief_from_observation(
        self, 
        lat_grid: np.ndarray, 
        lon_grid: np.ndarray, 
        step: int, 
        observation: np.ndarray,
        observation_points: Optional[set[tuple[float, float]]] = None
    ) -> None:
        """
        Update belief based on observations.
        
        For observed cells: b_{t+1}(j, x') = δ[x' = o_j]
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            observation: (H, W) array of observed states (0=no event, 1=event)
            observation_points: Optional set of (lat, lon) tuples that were actually observed (action/FOV).
                              If None, uses field of regard at current step (FOR).
        """
        if observation_points is None:
            # Convert FOR mask to set of (lat, lon) points
            observation_mask = self.coverage_mask(lat_grid, lon_grid, step)
            observation_points = self.mask_to_points(observation_mask, lat_grid, lon_grid)
        else:
            # Convert observation_points to mask
            observation_mask = self.points_to_mask(
                observation_points, lat_grid, lon_grid, self.belief.height, self.belief.width
            )
        
        self.belief.update_from_observation(observation_mask, observation, lat_grid, lon_grid)

    def evolve_belief_with_environment(
        self,
        env: "Environment",  # type: ignore
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        step: int,
        observation_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Evolve belief using the environment's transition kernel.
        
        This implements the belief update formula:
        - For observed cells: b_{t+1}(j, x') = δ[x' = o_j] (handled separately)
        - For unobserved cells: b_{t+1}(j, x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * b_t(j, x_j) * Π_{l∈N(j)} b_t(l, x_l)
        
        Also marks cells in the field of regard as valid for belief tracking.
        
        Args:
            env: Environment object with transition kernel
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            observation_mask: Optional (H, W) mask of observed cells. If None, uses field of regard.
        """
        from environment_simulation.environment import Environment
        
        if observation_mask is None:
            observation_mask = self.coverage_mask(lat_grid, lon_grid, step)
        else:
            observation_mask = np.asarray(observation_mask, dtype=bool)
        
        # All FOR locations should already be in belief (pre-computed at initialization)
        # But ensure any new locations are added (for safety, though they should all be pre-computed)
        field_mask = self.coverage_mask(lat_grid, lon_grid, step)
        for y in range(field_mask.shape[0]):
            for x in range(field_mask.shape[1]):
                if field_mask[y, x]:
                    lat, lon = lat_grid[y, x], lon_grid[y, x]
                    if not self.belief.has_location(lat, lon):
                        # This shouldn't happen if FOR was pre-computed, but add it for safety
                        self.belief.set_probability(lat, lon, self.belief.prior_probability)
        
        # Create neighbor getter function using environment's topology
        def neighbor_getter(y: int, x: int) -> list[tuple[int, int]]:
            return list(env._iter_neighbors(y, x))
        
        # Evolve beliefs for unobserved cells using transition kernel
        self.belief.evolve_with_transition_kernel(
            env, observation_mask, lat_grid, lon_grid, neighbor_getter
        )

    def information_gain(
        self, 
        lat_grid: np.ndarray, 
        lon_grid: np.ndarray, 
        step: int,
        observation_points: Optional[set[tuple[float, float]]] = None
    ) -> float:
        """
        Compute information gain from observing cells in field of regard.
        
        Information gain is: G(b) = H_prior(b) - H_post(b)
        Since we assume perfect observations, H_post = 0 for observed cells.
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            observation_points: Optional set of (lat, lon) tuples to observe.
                              If None, uses field of regard at current step.
        
        Returns:
            Information gain (entropy reduction) in bits
        """
        if observation_points is None:
            # Convert FOR mask to set of (lat, lon) points
            for_mask = self.coverage_mask(lat_grid, lon_grid, step)
            observation_points = self.mask_to_points(for_mask, lat_grid, lon_grid)
        else:
            observation_points = set(observation_points)
        
        # Create mask for entropy computation
        observation_mask = self.points_to_mask(
            observation_points, lat_grid, lon_grid, self.belief.height, self.belief.width
        )
        
        # Prior entropy for cells that will be observed
        prior_entropy = self.belief.entropy(observation_mask, lat_grid, lon_grid)
        
        # After perfect observation, entropy is 0 for observed cells
        # So information gain equals prior entropy
        return prior_entropy

    def expected_event_value(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        step: int,
        event_utility: dict[int, float],
        observation_points: Optional[set[tuple[float, float]]] = None
    ) -> float:
        """
        Compute expected value of detecting events in field of regard.
        
        E[value] = Σ_{x in X} b(x) * f(x)
        where f(x) is the utility of observing state x.
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            event_utility: Dictionary mapping state (0 or 1) to utility value
            observation_points: Optional set of (lat, lon) tuples to consider.
                              If None, uses field of regard at current step.
        
        Returns:
            Expected event value
        """
        if observation_points is None:
            # Convert FOR mask to set of (lat, lon) points
            for_mask = self.coverage_mask(lat_grid, lon_grid, step)
            observation_points = self.mask_to_points(for_mask, lat_grid, lon_grid)
        else:
            observation_points = set(observation_points)
        
        utility_0 = event_utility.get(0, 0.0)
        utility_1 = event_utility.get(1, 0.0)
        
        # Compute expected value for each observed cell
        expected_value = 0.0
        for lat, lon in observation_points:
            # Get probability from belief (or use prior if not in belief)
            prob = self.belief.get_probability(lat, lon)
            # E[value] = P(0) * f(0) + P(1) * f(1)
            cell_value = (1.0 - prob) * utility_0 + prob * utility_1
            expected_value += cell_value
        
        return float(expected_value)

    @abstractmethod
    def compute_reward(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        step: int,
        observation_mask: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute reward for observing cells at the current step.
        
        Subclasses should implement their own reward function.
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            observation_mask: Optional (H, W) mask of cells to observe.
                            If None, uses field of regard at current step.
            **kwargs: Additional arguments for reward computation (subclass-specific)
        
        Returns:
            Reward value
        """
        pass

    @abstractmethod
    def act(self, step: int, lat_grid: np.ndarray, lon_grid: np.ndarray) -> dict[str, Any]:
        """
        Choose an action based on the current belief and policy.
        
        Args:
            step: Current time step
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            
        Returns:
            Dictionary containing action information (subclasses define structure)
        """
        pass

    @abstractmethod
    def compute_policy(self, horizon: int, lat_grid: np.ndarray, lon_grid: np.ndarray) -> None:
        """
        Compute or update the agent's policy.
        
        Args:
            horizon: Planning horizon (number of steps ahead)
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
        """
        pass

    def save_policy(self, filepath: str) -> None:
        """
        Save the current policy to disk.
        
        Args:
            filepath: Path to save the policy
        """
        if self._policy is None:
            raise ValueError("No policy to save. Call compute_policy() first.")
        # Default implementation: save as numpy array if policy is numpy-compatible
        # Subclasses should override for custom policy formats
        try:
            np.save(filepath, self._policy)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot save policy to {filepath}: {e}. Override save_policy() for custom formats.")

    def load_policy(self, filepath: str) -> None:
        """
        Load a policy from disk.
        
        Args:
            filepath: Path to load the policy from
        """
        try:
            self._policy = np.load(filepath, allow_pickle=True)
        except Exception as e:
            raise ValueError(f"Cannot load policy from {filepath}: {e}")

    def get_belief_in_field_of_regard(
        self, lat_grid: np.ndarray, lon_grid: np.ndarray, step: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get belief probabilities and validity for cells in current field of regard.
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            
        Returns:
            probabilities: Probability values in field of regard (1D array)
            valid: Boolean array indicating which cells have valid beliefs (1D array)
        """
        region_mask = self.coverage_mask(lat_grid, lon_grid, step)
        return self.belief.get_belief_in_region(region_mask)

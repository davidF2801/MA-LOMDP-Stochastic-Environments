"""Greedy agent implementation that selects actions with highest immediate reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .agent import Agent
from .belief import Belief
from .trajectory import Trajectory


@dataclass
class GreedyAgent(Agent):
    """
    Greedy agent that selects the action with the highest immediate reward.
    
    Unlike Monte Carlo agents, this agent does not perform lookahead planning.
    It simply evaluates all possible actions (contiguous subsets) in the field of regard
    and selects the one with the highest immediate reward at each time step.
    """

    w_h: float = 1.0  # Weight for information gain term
    w_v: float = 1.0  # Weight for event detection value term
    event_utility: dict[int, float] = None  # Will default to {0: 0.0, 1: 1.0}
    max_observation_ratio: float = 0.2  # Maximum fraction of field of regard to observe (default 20%)
    
    def __post_init__(self):
        """Initialize default event utility if not provided."""
        if self.event_utility is None:
            self.event_utility = {0: 0.0, 1: 1.0}

    def act(
        self,
        step: int,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        env: Optional["Environment"] = None,  # type: ignore
        horizon: Optional[int] = None,
        max_observation_cells: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Choose the action with the highest immediate reward.
        
        Evaluates all possible contiguous subsets within the field of regard
        and selects the one with the highest immediate reward.
        
        Args:
            step: Current time step
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            env: Optional environment (not used by greedy agent for planning)
            horizon: Optional planning horizon (not used by greedy agent)
            max_observation_cells: Optional maximum number of cells to observe.
                                 If None, uses max_observation_ratio * FOR_size
        
        Returns:
            Dictionary containing action information:
            - observation_points: Set of (lat, lon) tuples representing cells to observe
            - expected_reward: Immediate reward for selected action
        """
        # Get field of regard
        field_mask = self.coverage_mask(lat_grid, lon_grid, step)
        
        # Check if field of regard has any cells
        if not np.any(field_mask):
            # No cells in field of regard, return empty action
            return {
                "observation_points": set(),
                "expected_reward": 0.0,
            }
        
        # Get all possible contiguous subsets
        all_subsets = self._get_all_contiguous_subsets(
            field_mask, lat_grid, lon_grid, max_cells=max_observation_cells
        )
        
        if not all_subsets:
            # Fallback: if no subsets found, return entire field of regard
            for_points = self.mask_to_points(field_mask, lat_grid, lon_grid)
            return {
                "observation_points": for_points,
                "expected_reward": 0.0,
            }
        
        # Evaluate each subset and select the one with highest immediate reward
        best_subset = None
        best_reward = float('-inf')
        
        for subset_points in all_subsets:
            # Compute immediate reward for this subset
            reward = self.compute_reward(
                lat_grid,
                lon_grid,
                step,
                observation_points=subset_points,
            )
            
            if reward > best_reward:
                best_reward = reward
                best_subset = subset_points
        
        # If no valid subset found, use first one as fallback
        if best_subset is None:
            best_subset = all_subsets[0]
            best_reward = self.compute_reward(
                lat_grid,
                lon_grid,
                step,
                observation_points=best_subset,
            )
        
        return {
            "observation_points": best_subset,
            "expected_reward": best_reward,
        }
    
    def _get_all_contiguous_subsets(
        self,
        field_mask: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        max_cells: Optional[int] = None,
    ) -> list[set[tuple[float, float]]]:
        """
        Get multiple contiguous subsets from the field of regard.
        
        Uses the same logic as MonteCarloAgent to generate candidate actions.
        
        Returns a list of sets of (lat, lon) tuples, each representing a contiguous subset.
        """
        rng = np.random.default_rng()
        
        # Calculate max cells based on ratio if not specified
        total_field_cells = np.sum(field_mask)
        if max_cells is None:
            max_cells = max(1, int(total_field_cells * self.max_observation_ratio))
        
        # Get all cells in the field of regard
        for_cells = []
        for y in range(field_mask.shape[0]):
            for x in range(field_mask.shape[1]):
                if field_mask[y, x]:
                    lat, lon = lat_grid[y, x], lon_grid[y, x]
                    for_cells.append((y, x, float(lat), float(lon)))
        
        if not for_cells:
            return []
        
        # If FOR is smaller than max_cells, return the entire FOR as a single subset
        if len(for_cells) <= max_cells:
            subset = set((lat, lon) for _, _, lat, lon in for_cells)
            return [subset]
        
        # Generate multiple contiguous subsets by starting from different seed points
        # Use BFS to grow connected regions from each seed
        H, W = field_mask.shape
        subsets = []
        num_seeds = min(10, len(for_cells))  # Try up to 10 different seed points
        
        # Sample seed points (can be random or spaced)
        seed_indices = rng.choice(len(for_cells), size=num_seeds, replace=False)
        
        def grow_region_from_seed(start_y: int, start_x: int, max_size: int) -> set[tuple[float, float]]:
            """Grow a connected region from a seed point using BFS."""
            if not field_mask[start_y, start_x]:
                return set()
            
            visited = np.zeros((H, W), dtype=bool)
            queue = [(start_y, start_x)]
            visited[start_y, start_x] = True
            region = set()
            
            # 8-connected neighbors
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            
            while queue and len(region) < max_size:
                cy, cx = queue.pop(0)
                lat, lon = lat_grid[cy, cx], lon_grid[cy, cx]
                region.add((float(lat), float(lon)))
                
                # Add neighbors that are in the FOR and not yet visited
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < H and 0 <= nx < W and 
                        field_mask[ny, nx] and 
                        not visited[ny, nx] and
                        len(region) < max_size):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            
            return region
        
        # Generate subsets from different seeds
        seen_subsets = set()  # Use frozenset to track unique subsets
        for seed_idx in seed_indices:
            seed_y, seed_x, seed_lat, seed_lon = for_cells[seed_idx]
            subset = grow_region_from_seed(seed_y, seed_x, max_cells)
            
            if subset and len(subset) >= 1:  # At least 1 cell
                # Convert to frozenset for comparison (order doesn't matter)
                subset_frozen = frozenset(subset)
                if subset_frozen not in seen_subsets:
                    seen_subsets.add(subset_frozen)
                    subsets.append(subset)
        
        # If we didn't get enough subsets, try some additional strategies
        if len(subsets) < 3:
            # Strategy 1: Try seeds from different regions (spatial distribution)
            if len(for_cells) > 4:
                # Simple spatial clustering: pick seeds from corners/center
                corner_indices = [
                    0,  # First cell
                    len(for_cells) // 4,  # ~25%
                    len(for_cells) // 2,  # ~50%
                    3 * len(for_cells) // 4,  # ~75%
                    len(for_cells) - 1,  # Last cell
                ]
                for idx in corner_indices:
                    if idx < len(for_cells):
                        seed_y, seed_x, seed_lat, seed_lon = for_cells[idx]
                        subset = grow_region_from_seed(seed_y, seed_x, max_cells)
                        if subset:
                            subset_frozen = frozenset(subset)
                            if subset_frozen not in seen_subsets:
                                seen_subsets.add(subset_frozen)
                                subsets.append(subset)
                                if len(subsets) >= 10:  # Enough subsets
                                    break
        
        # Ensure we have at least one subset
        if not subsets:
            # Fallback: return a single subset from the center
            center_idx = len(for_cells) // 2
            seed_y, seed_x, seed_lat, seed_lon = for_cells[center_idx]
            subset = grow_region_from_seed(seed_y, seed_x, max_cells)
            if subset:
                subsets.append(subset)
            else:
                # Last resort: return a small subset from the first cell
                if for_cells:
                    seed_y, seed_x, seed_lat, seed_lon = for_cells[0]
                    subset = {(seed_lat, seed_lon)}
                    subsets.append(subset)
        
        return subsets
    
    def compute_reward(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        step: int,
        observation_mask: Optional[np.ndarray] = None,
        observation_points: Optional[set[tuple[float, float]]] = None,
        w_h: Optional[float] = None,
        w_v: Optional[float] = None,
        event_utility: Optional[dict[int, float]] = None,
    ) -> float:
        """
        Compute reward for observing cells at the current step.
        
        Implements the reward function:
        R(τ,b,a) = Σ_{k∈U_t} [w_h * G(b_k) + w_v * Σ_x b_k(x) f(x)]
        
        where:
        - U_t is the set of cells observed at time t
        - G(b_k) = H_prior(b_k) - H_post(b_k) (information gain)
        - H_post(b_k) = 0 for perfect observations
        - f(x) is the normalized utility of observing state x
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            observation_mask: Optional (H, W) mask of cells to observe (for compatibility)
            observation_points: Optional set of (lat, lon) tuples to observe.
                              If None, uses field of regard at current step.
            w_h: Weight for information gain term. If None, uses self.w_h
            w_v: Weight for event detection value term. If None, uses self.w_v
            event_utility: Dictionary mapping state (0 or 1) to utility value.
                          If None, uses self.event_utility
        
        Returns:
            Reward value
        """
        if observation_points is None:
            # Convert FOR mask to set of (lat, lon) points
            for_mask = self.coverage_mask(lat_grid, lon_grid, step)
            observation_points = self.mask_to_points(for_mask, lat_grid, lon_grid)
        else:
            observation_points = set(observation_points)
        
        if w_h is None:
            w_h = self.w_h
        if w_v is None:
            w_v = self.w_v
        if event_utility is None:
            event_utility = self.event_utility
        
        # Compute information gain (entropy reduction)
        info_gain = self.information_gain(lat_grid, lon_grid, step, observation_points)
        
        # Compute expected event value
        event_value = self.expected_event_value(
            lat_grid, lon_grid, step, event_utility, observation_points
        )
        
        # Total reward
        reward = w_h * info_gain + w_v * event_value
        
        return reward
    
    def compute_policy(self, horizon: int, lat_grid: np.ndarray, lon_grid: np.ndarray) -> None:
        """
        Compute or update the agent's policy.
        
        Greedy agent doesn't compute a policy ahead of time, so this is a no-op.
        
        Args:
            horizon: Planning horizon (number of steps ahead)
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
        """
        pass


"""Random agent implementation for baseline comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .agent import Agent
from .belief import Belief
from .trajectory import Trajectory


@dataclass
class RandomAgent(Agent):
    """
    Random agent that selects random actions from its field of regard.
    
    This agent serves as a baseline for comparison with planning agents.
    It randomly selects a contiguous subset of cells within its field of regard
    to observe at each time step.
    """

    max_observation_ratio: float = 0.2  # Maximum fraction of field of regard to observe (default 20%)
    seed: Optional[int] = None  # Optional seed for reproducibility
    
    def __post_init__(self):
        """Initialize random number generator."""
        if self.seed is not None:
            self._rng = np.random.Generator(np.random.PCG64(self.seed))
        else:
            self._rng = np.random.default_rng()

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
        Choose a random action from the field of regard.
        
        Randomly selects a contiguous subset of cells within the field of regard
        to observe. The size of the subset is limited by max_observation_ratio.
        
        Args:
            step: Current time step
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            env: Optional environment (not used by random agent)
            horizon: Optional planning horizon (not used by random agent)
            max_observation_cells: Optional maximum number of cells to observe.
                                 If None, uses max_observation_ratio * FOR_size
        
        Returns:
            Dictionary with 'observation_points' (set of (lat, lon) tuples) and 'expected_reward' (0.0)
        """
        # Get field of regard
        for_mask = self.coverage_mask(lat_grid, lon_grid, step)
        for_points = self.mask_to_points(for_mask, lat_grid, lon_grid)
        
        if len(for_points) == 0:
            # No cells in field of regard
            return {
                "observation_points": set(),
                "expected_reward": 0.0,
            }
        
        # Determine maximum number of cells to observe
        if max_observation_cells is None:
            max_cells = max(1, int(len(for_points) * self.max_observation_ratio))
        else:
            max_cells = max(1, min(max_observation_cells, len(for_points)))
        
        # Randomly select a contiguous subset
        observation_points = self._select_random_contiguous_subset(
            for_points, for_mask, lat_grid, lon_grid, max_cells
        )
        
        return {
            "observation_points": observation_points,
            "expected_reward": 0.0,  # Random agent doesn't compute expected reward
        }
    
    def _select_random_contiguous_subset(
        self,
        for_points: set[tuple[float, float]],
        for_mask: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        max_cells: int,
    ) -> set[tuple[float, float]]:
        """
        Randomly select a contiguous subset of points from the field of regard.
        
        Uses a seed point and grows a connected region using BFS.
        
        Args:
            for_points: Set of (lat, lon) points in field of regard
            for_mask: Boolean mask of field of regard
            lat_grid: (H, W) array of latitude values
            lon_grid: (H, W) array of longitude values
            max_cells: Maximum number of cells to include
        
        Returns:
            Set of (lat, lon) points forming a contiguous subset
        """
        if len(for_points) == 0:
            return set()
        
        # Convert FOR points to grid coordinates
        grid_to_point = {}
        for_points_list = list(for_points)
        for lat, lon in for_points_list:
            # Find grid coordinates for this point using tolerance
            dist = np.abs(lat_grid - lat) + np.abs(lon_grid - lon)
            y, x = np.unravel_index(np.argmin(dist), lat_grid.shape)
            # Only add if it's actually in the mask
            if for_mask[y, x]:
                grid_to_point[(y, x)] = (lat, lon)
        
        if len(grid_to_point) == 0:
            return set()
        
        # Randomly select a seed point
        seed_cells = list(grid_to_point.keys())
        if len(seed_cells) == 0:
            return set()
        
        # Grow a connected region from a random seed using BFS
        def get_neighbors(y: int, x: int, mask: np.ndarray) -> list[tuple[int, int]]:
            """Get 4-connected neighbors that are in the mask."""
            neighbors = []
            H, W = mask.shape
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                # Handle wrapping for spherical topology
                if ny < 0:
                    ny = H - 1  # Wrap around pole
                elif ny >= H:
                    ny = 0
                if nx < 0:
                    nx = W - 1  # Wrap longitude
                elif nx >= W:
                    nx = 0
                if mask[ny, nx]:
                    neighbors.append((ny, nx))
            return neighbors
        
        # Start from random seed
        start_y, start_x = seed_cells[self._rng.integers(len(seed_cells))]
        visited = set()
        queue = [(start_y, start_x)]
        region_cells = []
        
        # BFS to grow connected region
        while queue and len(region_cells) < max_cells:
            y, x = queue.pop(0)
            if (y, x) in visited:
                continue
            visited.add((y, x))
            if (y, x) in grid_to_point:
                region_cells.append((y, x))
            if len(region_cells) >= max_cells:
                break
            # Add neighbors
            for ny, nx in get_neighbors(y, x, for_mask):
                if (ny, nx) not in visited:
                    queue.append((ny, nx))
        
        # Convert grid coordinates back to (lat, lon) points
        observation_points = {grid_to_point[(y, x)] for y, x in region_cells if (y, x) in grid_to_point}
        
        return observation_points
    
    def _fallback_random_subset(
        self,
        for_points: set[tuple[float, float]],
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        max_cells: int,
    ) -> set[tuple[float, float]]:
        """
        Fallback method to select a random subset when scipy is not available.
        
        Args:
            for_points: Set of (lat, lon) points in field of regard
            lat_grid: (H, W) array of latitude values
            lon_grid: (H, W) array of longitude values
            max_cells: Maximum number of cells to include
        
        Returns:
            Set of (lat, lon) points (may not be contiguous without scipy)
        """
        if len(for_points) == 0:
            return set()
        
        # Simple random sampling (not guaranteed to be contiguous)
        for_points_list = list(for_points)
        num_to_select = min(max_cells, len(for_points_list))
        selected_indices = self._rng.choice(
            len(for_points_list), size=num_to_select, replace=False
        )
        return {for_points_list[i] for i in selected_indices}
    
    def compute_reward(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        step: int,
        observation_mask: Optional[np.ndarray] = None,
        observation_points: Optional[set[tuple[float, float]]] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute reward for random agent.
        
        Random agent doesn't compute reward, so this returns 0.0.
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            observation_mask: Optional (H, W) mask of cells to observe (for compatibility)
            observation_points: Optional set of (lat, lon) tuples to observe
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Reward value (always 0.0)
        """
        return 0.0
    
    def compute_policy(self, horizon: int, lat_grid: np.ndarray, lon_grid: np.ndarray) -> None:
        """
        Compute or update the agent's policy.
        
        Random agent doesn't compute a policy, so this is a no-op.
        
        Args:
            horizon: Planning horizon (number of steps ahead)
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
        """
        pass


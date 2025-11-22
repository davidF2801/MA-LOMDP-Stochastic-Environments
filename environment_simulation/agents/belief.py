"""Belief representation for agents over the environment state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Callable

import numpy as np

if TYPE_CHECKING:
    from environment_simulation.environment import Environment


@dataclass
class Belief:
    """
    Belief over the environment state.
    
    Represents the agent's belief about the probability of events at each (lat, lon) point.
    Since each agent has a fixed periodic trajectory and a fixed sensor Field of Regard (FOR),
    we only maintain beliefs for cells that the agent will actually observe.
    
    The belief is stored as a dictionary mapping (lat, lon) -> P(E_j(t) = 1), where:
    - Keys: (lat, lon) tuples representing cell locations
    - Values: P(E_j(t) = 1) = probability of event at that location
    - For binary states {0, 1}: P(E_j(t) = 0) = 1 - P(E_j(t) = 1)
    
    The belief is updated according to:
    - If cell j is observed: b_{t+1}^e(j, x') = δ[x' = o_j] (delta function)
    - If cell j is not observed: b_{t+1}^e(j, x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * 
      b_t^e(j, x_j) * Π_{l∈N(j)} b_t^e(l, x_l)
    """

    height: int
    width: int
    # Dictionary mapping (lat, lon) -> P(E_j = 1)
    # Only contains cells that have valid beliefs (observed or in field of regard)
    probabilities: dict[tuple[float, float], float] = field(init=False, default_factory=dict)
    # Set of (lat, lon) tuples that have valid beliefs
    valid_locations: set[tuple[float, float]] = field(init=False, default_factory=set)
    # Prior probability for unobserved cells
    prior_probability: float = 0.5

    def __post_init__(self):
        """Initialize belief dictionary."""
        self.probabilities = {}
        self.valid_locations = set()
    
    def get_probability(self, lat: float, lon: float) -> float:
        """
        Get probability P(E_j = 1) for a specific (lat, lon) location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            Probability P(E_j = 1). Returns prior_probability if location not in belief.
        """
        key = (lat, lon)
        return self.probabilities.get(key, self.prior_probability)
    
    def set_probability(self, lat: float, lon: float, prob: float) -> None:
        """
        Set probability P(E_j = 1) for a specific (lat, lon) location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            prob: Probability P(E_j = 1), must be in [0, 1]
        """
        key = (lat, lon)
        prob = float(np.clip(prob, 0.0, 1.0))
        self.probabilities[key] = prob
        self.valid_locations.add(key)
    
    def has_location(self, lat: float, lon: float) -> bool:
        """Check if location (lat, lon) is in the belief."""
        return (lat, lon) in self.valid_locations

    def update_from_observation(
        self, 
        observation_mask: np.ndarray, 
        observations: np.ndarray | dict[tuple[float, float], int],
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
    ) -> None:
        """
        Update belief with observations (perfect observation model).
        
        For observed cells, set belief to delta function: b_{t+1}(j, x') = δ[x' = o_j]
        
        Args:
            observation_mask: Boolean (H, W) mask indicating which cells were observed
            observations: Either:
                - Integer (H, W) array of observed states (0=no event, 1=event), OR
                - Dictionary mapping (lat, lon) -> observed state (0 or 1) for cells in observation_mask
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
        """
        observation_mask = np.asarray(observation_mask, dtype=bool)
        assert observation_mask.shape == (self.height, self.width), "observation_mask must be (H,W)"
        
        if isinstance(observations, dict):
            # Update from dictionary: (lat, lon) -> state
            for (lat, lon), observed_state in observations.items():
                # Set to delta function: P(E_j = 1) = observed_state (0 or 1)
                self.set_probability(lat, lon, float(observed_state))
        else:
            # Update from array: find (lat, lon) for each observed cell
            observations = np.asarray(observations, dtype=int)
            assert observations.shape == (self.height, self.width), "observations must be (H,W)"
            
            for y in range(self.height):
                for x in range(self.width):
                    if observation_mask[y, x]:
                        lat, lon = lat_grid[y, x], lon_grid[y, x]
                        observed_state = int(observations[y, x])
                        # Set to delta function: P(E_j = 1) = observed_state (0 or 1)
                        self.set_probability(lat, lon, float(observed_state))

    def get_belief_in_region(
        self, 
        region_mask: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get belief probabilities and validity for a specific region.
        
        Args:
            region_mask: Boolean (H, W) mask indicating the region of interest
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            
        Returns:
            probabilities: Probability values in the region (1D array)
            valid: Boolean array indicating which cells in the region have valid beliefs (1D array)
        """
        region_mask = np.asarray(region_mask, dtype=bool)
        assert region_mask.shape == (self.height, self.width), "region_mask must be (H,W)"
        
        probs = []
        valid = []
        
        for y in range(self.height):
            for x in range(self.width):
                if region_mask[y, x]:
                    lat, lon = lat_grid[y, x], lon_grid[y, x]
                    if self.has_location(lat, lon):
                        probs.append(self.get_probability(lat, lon))
                        valid.append(True)
                    else:
                        probs.append(self.prior_probability)
                        valid.append(False)
        
        return np.array(probs), np.array(valid, dtype=bool)

    def entropy(
        self, 
        cell_mask: Optional[np.ndarray] = None,
        lat_grid: Optional[np.ndarray] = None,
        lon_grid: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Shannon entropy of the belief.
        
        For a binary state, entropy is: H(b) = -p*log2(p) - (1-p)*log2(1-p)
        where p is the probability of event=1.
        
        Args:
            cell_mask: Optional (H, W) mask indicating which cells to include.
                      If None, computes entropy over all valid locations.
            lat_grid: (H, W) array of latitude values (required if cell_mask is provided)
            lon_grid: (H, W) array of longitude values (required if cell_mask is provided)
        
        Returns:
            Total entropy (sum over cells)
        """
        if cell_mask is None:
            # Compute entropy over all valid locations
            probs = np.array([self.probabilities[key] for key in self.valid_locations])
        else:
            # Compute entropy over masked region
            if lat_grid is None or lon_grid is None:
                raise ValueError("lat_grid and lon_grid must be provided when cell_mask is specified")
            cell_mask = np.asarray(cell_mask, dtype=bool)
            probs = []
            for y in range(self.height):
                for x in range(self.width):
                    if cell_mask[y, x]:
                        lat, lon = lat_grid[y, x], lon_grid[y, x]
                        probs.append(self.get_probability(lat, lon))
            probs = np.array(probs)
        
        if len(probs) == 0:
            return 0.0
        
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
        entropy = -probs * np.log2(probs) - (1.0 - probs) * np.log2(1.0 - probs)
        return float(np.sum(entropy))

    def reset(self) -> None:
        """Reset belief to empty state."""
        self.probabilities.clear()
        self.valid_locations.clear()

    def evolve_with_transition_kernel(
        self,
        env: "Environment",
        observation_mask: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        neighbor_getter: Callable[[int, int], list[tuple[int, int]]],
    ) -> None:
        """
        Evolve belief using the environment's transition kernel.
        
        For observed cells: b_{t+1}(j, x') = δ[x' = o_j] (handled separately)
        For unobserved cells: b_{t+1}(j, x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * b_t(j, x_j) * Π_{l∈N(j)} b_t(l, x_l)
        
        Since the transition kernel only depends on the count of active neighbors (not full configuration),
        we approximate using the expected number of active neighbors.
        
        Args:
            env: Environment object with transition kernel
            observation_mask: (H, W) boolean mask of observed cells (these are not evolved)
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            neighbor_getter: Function (y, x) -> list of (ny, nx) neighbor coordinates
        """
        
        observation_mask = np.asarray(observation_mask, dtype=bool)
        assert observation_mask.shape == (self.height, self.width), "observation_mask must be (H,W)"
        
        # Create a copy to update - maintain ALL locations in belief
        new_probs = {}
        
        # Create a mapping from (lat, lon) to (y, x) for efficient lookup
        # Use tolerance-based matching to handle floating point precision
        latlon_to_grid = {}
        for y in range(self.height):
            for x in range(self.width):
                lat, lon = float(lat_grid[y, x]), float(lon_grid[y, x])
                latlon_to_grid[(lat, lon)] = (y, x)
        
        # Iterate over ALL locations in belief (all pre-computed FOR locations)
        for lat, lon in list(self.valid_locations):
            # Get grid coordinates for this (lat, lon) - use tolerance-based lookup
            grid_coords = None
            lat_key = float(lat)
            lon_key = float(lon)
            
            # Try exact match first
            grid_coords = latlon_to_grid.get((lat_key, lon_key))
            
            # If not found, try tolerance-based search (for floating point precision)
            if grid_coords is None:
                for (g_lat, g_lon), (gy, gx) in latlon_to_grid.items():
                    if abs(g_lat - lat_key) < 1e-6 and abs(g_lon - lon_key) < 1e-6:
                        grid_coords = (gy, gx)
                        break
            
            if grid_coords is None:
                # Location not in grid - keep current probability (shouldn't happen if pre-computed correctly)
                new_probs[(lat, lon)] = self.probabilities.get((lat, lon), self.prior_probability)
                continue
            
            y, x = grid_coords
            
            # Skip observed cells (they are updated separately via update_from_observation)
            if observation_mask[y, x]:
                # Keep observed cells at their current value (will be updated separately)
                new_probs[(lat, lon)] = self.probabilities.get((lat, lon), self.prior_probability)
                continue
            
            # Get neighbors
            neighbors = list(neighbor_getter(y, x))
            
            # Compute expected number of active neighbors
            # Use beliefs of neighbor locations (if they're in our belief)
            expected_active = 0.0
            for ny, nx in neighbors:
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    n_lat, n_lon = lat_grid[ny, nx], lon_grid[ny, nx]
                    if self.has_location(n_lat, n_lon):
                        expected_active += self.get_probability(n_lat, n_lon)
            
            # Approximate transition using expected active neighbors
            # For binary state: P(x'=1) = P(x=0) * P(1|0, E[active]) + P(x=1) * P(1|1, E[active])
            p_current = self.get_probability(lat, lon)
            p_current_0 = 1.0 - p_current
            p_current_1 = p_current
            
            # Use integer approximation of expected active neighbors for transition kernel
            active_count = int(np.round(expected_active))
            active_count = max(0, min(active_count, len(neighbors)))  # Clip to valid range
            
            # Compute transition probabilities
            p_transition_0_to_1 = env.transition_probability(y, x, 0, active_count)
            p_transition_1_to_1 = env.transition_probability(y, x, 1, active_count)
            
            # Update belief: P(x'=1) = P(x=0) * P(1|0) + P(x=1) * P(1|1)
            new_prob = p_current_0 * p_transition_0_to_1 + p_current_1 * p_transition_1_to_1
            new_probs[(lat, lon)] = float(np.clip(new_prob, 0.0, 1.0))
        
        # Update probabilities - maintain all locations
        self.probabilities = new_probs

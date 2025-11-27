"""Belief representation for agents over the environment state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Callable
import itertools

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

if TYPE_CHECKING:
    from environment_simulation.environment import Environment


# Numba-optimized helper: generate neighbor state combination from index
@njit(cache=True)
def _get_combination_from_index(idx: int, num_neighbors: int, num_states: int) -> np.ndarray:
    """Convert linear index to neighbor state combination (base num_states representation)."""
    combination = np.zeros(num_neighbors, dtype=np.int64)
    for i in range(num_neighbors):
        combination[i] = idx % num_states
        idx = idx // num_states
    return combination


# Numba-optimized transition probability functions
@njit(cache=True)
def _dbn2_transition_prob_numba(
    next_state: int,
    current_state: int,
    active_neighbors: int,
    birth_rate: float,
    death_rate: float,
    neighbor_influence: float,
) -> float:
    """
    Numba-optimized DBN-2 transition probability computation.
    
    Returns P(x'=next_state | current_state, active_neighbors).
    """
    if current_state == 1:  # EVENT_PRESENT
        # survive with prob 1 - death_rate
        p_event = max(0.0, min(1.0, 1.0 - death_rate))
    else:  # NO_EVENT
        # birth from background + neighbor influence
        p_event = max(0.0, min(1.0, birth_rate + neighbor_influence * active_neighbors))
    
    return p_event if next_state == 1 else (1.0 - p_event)


@njit(cache=True)
def _rsp_transition_prob_numba_satellite(
    next_state: int,
    current_state: int,
    active_neighbors: int,
    num_neighbors: int,
    lam: float,
    beta0: float,
    alpha: float,
    delta: float,
) -> float:
    """
    Numba-optimized RSP transition probability computation for satellite case.
    
    Matches the Julia implementation and graph case: uses exponential contagion.
    
    Returns P(x'=next_state | current_state, active_neighbors).
    
    Formula matches Julia get_transition_probability_rsp:
    - Normalize active neighbors: norm_active = active_neighbors / num_neighbors
    - Contagion: contagion = 1.0 - exp(-alpha * norm_active)
    - For current_state=0: p_event = 1.0 - exp(-(beta0 + lam + contagion))
    - For current_state=1: p_event = delta (persistence)
    """
    norm_active = active_neighbors / num_neighbors if num_neighbors > 0 else 0.0
    contagion = 1.0 - np.exp(-alpha * norm_active)
    
    if current_state == 0:  # NO_EVENT → {NO_EVENT, EVENT}
        p_event = 1.0 - np.exp(-(beta0 + lam + contagion))
        return p_event if next_state == 1 else (1.0 - p_event)
    else:  # current_state == 1: EVENT → {NO_EVENT, EVENT}
        mu = 1.0 - delta
        return delta if next_state == 1 else mu


# Numba-optimized belief evolution for a single cell - ENTIRE computation in Numba
@njit(cache=True, parallel=False)
def _evolve_cell_belief_numba_full(
    current_belief: np.ndarray,  # Current belief for all states of this cell [num_states]
    neighbor_beliefs: np.ndarray,  # Beliefs for neighbors [num_neighbors, num_states]
    num_states: int,
    num_neighbors: int,
    mode: int,  # 0=dbn2, 1=rsp, 2=viirs_table (not supported in Numba)
    # DBN-2 parameters
    birth_rate: float,
    death_rate: float,
    neighbor_influence: float,
    # RSP parameters
    lam: float,
    beta0: float,
    alpha: float,
    delta: float,
) -> np.ndarray:
    """
    Fully Numba-optimized single cell belief evolution.
    Computes everything inside Numba, including generating combinations and computing transitions.
    
    Computes: b_{t+1}(j, x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * b_t(j, x_j) * Π_{l∈N(j)} b_t(l, x_l)
    
    Where:
    - x' is the next state (0 to num_states-1)
    - x_j is the current state of cell j (0 to num_states-1)
    - x_N(j) is the configuration of neighbor states (all 2^num_neighbors combinations for binary states)
    - φ_j(x'; x_j, x_N(j)) is the transition probability
    - b_t(j, x_j) is the current belief P(cell j = x_j)
    - b_t(l, x_l) is the current belief P(neighbor l = x_l)
    
    Args:
        mode: 0 for dbn2, 1 for rsp, 2 for viirs_table (not supported, will fall back)
    """
    new_belief = np.zeros(num_states, dtype=np.float64)
    num_combinations = num_states ** num_neighbors
    
    # For each next state x'
    for next_state in range(num_states):
        prob = 0.0
        
        # Sum over all current states x_j
        for current_state in range(num_states):
            # Sum over all neighbor configurations (generated on-the-fly in Numba)
            for combo_idx in range(num_combinations):
                # Generate neighbor state combination from index (in Numba)
                combination = _get_combination_from_index(combo_idx, num_neighbors, num_states)
                
                # Count active neighbors (state 1) from combination
                active_neighbors = 0
                for i in range(num_neighbors):
                    if combination[i] == 1:
                        active_neighbors += 1
                
                # Compute transition probability (in Numba)
                if mode == 0:  # DBN-2
                    transition_prob = _dbn2_transition_prob_numba(
                        next_state, current_state, active_neighbors,
                        birth_rate, death_rate, neighbor_influence
                    )
                elif mode == 1:  # RSP
                    transition_prob = _rsp_transition_prob_numba_satellite(
                        next_state, current_state, active_neighbors, num_neighbors,
                        lam, beta0, alpha, delta
                    )
                else:  # viirs_table or unknown - fallback to uniform
                    transition_prob = 1.0 / num_states
                
                # Compute b_t(j, x_j) * Π_{l∈N(j)} b_t(l, x_l)
                prob_current = current_belief[current_state]
                
                # Product of neighbor beliefs
                prob_neighbor_config = 1.0
                for neighbor_idx in range(num_neighbors):
                    neighbor_state = combination[neighbor_idx]
                    prob_neighbor_config *= neighbor_beliefs[neighbor_idx, neighbor_state]
                
                # Add to sum
                prob += transition_prob * prob_current * prob_neighbor_config
        
        new_belief[next_state] = prob
    
    # Normalize
    total = np.sum(new_belief)
    if total > 1e-10:
        new_belief = new_belief / total
    else:
        # Uniform if total too small
        new_belief[:] = 1.0 / num_states
    
    return new_belief


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
    # Number of states (M) - currently binary (2) but kept for future generalization
    num_states: int = 2

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
        Evolve belief using the environment's transition kernel (full mean field update).
        
        Implements the mean field update formula:
        b_{t+1}(j, x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * b_t(j, x_j) * Π_{l∈N(j)} b_t(l, x_l)
        
        Where:
        - x' is the next state (0 or 1 for binary)
        - x_j is the current state of cell j
        - x_N(j) is the configuration of neighbor states (all 2^8 = 256 combinations for 8 neighbors)
        - φ_j(x'; x_j, x_N(j)) is the transition probability
        - b_t(j, x_j) is the current belief P(cell j = x_j)
        - b_t(l, x_l) is the current belief P(neighbor l = x_l)
        
        Uses Numba-optimized functions for performance.
        
        Args:
            env: Environment object with transition kernel
            observation_mask: (H, W) boolean mask of observed cells (these are not evolved)
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            neighbor_getter: Function (y, x) -> list of (ny, nx) neighbor coordinates
        """
        observation_mask = np.asarray(observation_mask, dtype=bool)
        assert observation_mask.shape == (self.height, self.width), "observation_mask must be (H,W)"
        
        # Determine environment mode for Numba function
        mode_map = {"dbn2": 0, "rsp": 1, "viirs_table": 2}
        mode = mode_map.get(env.mode, 1)  # Default to RSP if unknown
        
        # Create a copy to update - maintain ALL locations in belief
        new_probs = {}
        
        # Create a mapping from (lat, lon) to (y, x) for efficient lookup
        latlon_to_grid = {}
        for y in range(self.height):
            for x in range(self.width):
                lat, lon = float(lat_grid[y, x]), float(lon_grid[y, x])
                latlon_to_grid[(lat, lon)] = (y, x)
        
        # Iterate over ALL locations in belief (all reachable/observed cells)
        for lat, lon in list(self.valid_locations):
            # Get grid coordinates for this (lat, lon)
            grid_coords = latlon_to_grid.get((lat, lon))
            if grid_coords is None:
                # Location not in grid - keep current probability
                new_probs[(lat, lon)] = self.probabilities.get((lat, lon), self.prior_probability)
                continue
            
            y, x = grid_coords
            
            # Skip observed cells (they are updated separately via update_from_observation)
            if observation_mask[y, x]:
                # Keep observed cells at their current value (will be updated separately)
                new_probs[(lat, lon)] = self.probabilities.get((lat, lon), self.prior_probability)
                continue
            
            # Get neighbors for this cell
            neighbors = list(neighbor_getter(y, x))
            num_neighbors = len(neighbors)
            
            if num_neighbors == 0:
                # No neighbors - keep current belief (or use uniform prior)
                new_probs[(lat, lon)] = self.probabilities.get((lat, lon), self.prior_probability)
                continue
            
            # Prepare current cell belief [num_states] - binary states
            current_belief = np.zeros(self.num_states, dtype=np.float64)
            p_current = self.get_probability(lat, lon)
            current_belief[0] = 1.0 - p_current  # P(E=0)
            current_belief[1] = p_current  # P(E=1)
            
            # Prepare neighbor beliefs [num_neighbors, num_states]
            neighbor_beliefs = np.zeros((num_neighbors, self.num_states), dtype=np.float64)
            for i, (ny, nx) in enumerate(neighbors):
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    n_lat, n_lon = lat_grid[ny, nx], lon_grid[ny, nx]
                    if self.has_location(n_lat, n_lon):
                        p_neighbor = self.get_probability(n_lat, n_lon)
                    else:
                        # Neighbor not in belief - use prior
                        p_neighbor = self.prior_probability
                    neighbor_beliefs[i, 0] = 1.0 - p_neighbor  # P(E=0)
                    neighbor_beliefs[i, 1] = p_neighbor  # P(E=1)
                else:
                    # Out of bounds neighbor - assume no event
                    neighbor_beliefs[i, 0] = 1.0
                    neighbor_beliefs[i, 1] = 0.0
            
            # Get environment parameters for this cell
            if env.mode == "dbn2":
                birth_rate = env.birth_rate
                death_rate = env.death_rate
                neighbor_influence = env.neighbor_influence
                # RSP params not used for DBN-2
                lam = beta0 = alpha = delta = 0.0
            elif env.mode == "rsp":
                # Get per-cell RSP parameters
                lam_map = env.ignition_map if env.ignition_map is not None else None
                beta0_map = env.beta0_map if env.beta0_map is not None else None
                alpha_map = env.alpha_map if env.alpha_map is not None else None
                persistence_map = env.persistence_map if env.persistence_map is not None else None
                
                lam = float(lam_map[y, x]) if lam_map is not None else env.rsp.lam
                beta0 = float(beta0_map[y, x]) if beta0_map is not None else env.rsp.beta0
                alpha = float(alpha_map[y, x]) if alpha_map is not None else env.rsp.alpha
                delta = float(persistence_map[y, x]) if persistence_map is not None else env.rsp.delta
                
                # DBN-2 params not used for RSP
                birth_rate = death_rate = neighbor_influence = 0.0
            else:  # viirs_table or unknown
                # Not supported in Numba path - fall back to approximation
                # Keep current belief unchanged for now (could implement fallback)
                new_probs[(lat, lon)] = self.probabilities.get((lat, lon), self.prior_probability)
                continue
            
            # Check if cell is blocked (can't evolve)
            if env.blocked_mask is not None and env.blocked_mask[y, x]:
                # Blocked cells stay at 0 (no event)
                new_probs[(lat, lon)] = 0.0
                continue
            
            # Use Numba-optimized belief evolution if available
            if NUMBA_AVAILABLE:
                try:
                    new_belief = _evolve_cell_belief_numba_full(
                        current_belief,
                        neighbor_beliefs,
                        self.num_states,
                        num_neighbors,
                        mode,
                        birth_rate,
                        death_rate,
                        neighbor_influence,
                        lam,
                        beta0,
                        alpha,
                        delta,
                    )
                    # Extract P(E=1) from new belief
                    new_probs[(lat, lon)] = float(np.clip(new_belief[1], 0.0, 1.0))
                    continue
                except Exception as e:
                    # Fall back to Python implementation if Numba fails
                    import warnings
                    warnings.warn(f"Numba belief evolution failed, falling back to Python: {e}")
            
            # Fallback Python implementation (slower but correct)
            # This implements the full mean field formula without Numba
            new_prob_0 = 0.0
            new_prob_1 = 0.0
            
            num_combinations = self.num_states ** num_neighbors
            
            # For each next state x' (0 or 1)
            for next_state in [0, 1]:
                prob = 0.0
                
                # Sum over all current states x_j
                for current_state in [0, 1]:
                    p_current_state = current_belief[current_state]
                    
                    # Sum over all neighbor configurations
                    for combo_idx in range(num_combinations):
                        # Generate neighbor state combination
                        combination = []
                        temp_idx = combo_idx
                        for _ in range(num_neighbors):
                            combination.append(temp_idx % self.num_states)
                            temp_idx //= self.num_states
                        
                        # Count active neighbors (state 1)
                        active_neighbors = sum(1 for s in combination if s == 1)
                        
                        # Compute transition probability using env method
                        transition_prob = env.transition_probability(y, x, current_state, active_neighbors)
                        # env.transition_probability returns P(x'=1), so adjust for next_state
                        if next_state == 0:
                            transition_prob = 1.0 - transition_prob
                        
                        # Compute product of neighbor beliefs for this configuration
                        prob_neighbor_config = 1.0
                        for i, neighbor_state in enumerate(combination):
                            prob_neighbor_config *= neighbor_beliefs[i, neighbor_state]
                        
                        # Add to sum
                        prob += transition_prob * p_current_state * prob_neighbor_config
                
                if next_state == 0:
                    new_prob_0 = prob
                else:
                    new_prob_1 = prob
            
            # Normalize
            total = new_prob_0 + new_prob_1
            if total > 1e-10:
                new_probs[(lat, lon)] = float(np.clip(new_prob_1 / total, 0.0, 1.0))
            else:
                # Uniform if total too small
                new_probs[(lat, lon)] = self.prior_probability
        
        # Update probabilities - maintain all locations
        self.probabilities = new_probs

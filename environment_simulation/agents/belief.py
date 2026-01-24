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
    
    For binary states (num_states=2):
    - The belief is stored as a dictionary mapping (lat, lon) -> P(E_j(t) = 1)
    - P(E_j(t) = 0) = 1 - P(E_j(t) = 1)
    
    For 3-state kernel mode (num_states=3):
    - The belief stores probabilities for all states: (lat, lon, state) -> P(state)
    - States: 0=unburned, 1=burning, 2=burned
    - Backward compatibility: _binary_probs stores P(state=1) for compatibility
    
    The belief is updated according to:
    - If cell j is observed: b_{t+1}^e(j, x') = δ[x' = o_j] (delta function)
    - If cell j is not observed: b_{t+1}^e(j, x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * 
      b_t^e(j, x_j) * Π_{l∈N(j)} b_t^e(l, x_l)
    """

    height: int
    width: int
    # Dictionary mapping (lat, lon) -> P(E_j = 1) for binary mode (backward compatibility)
    # For 3-state mode: use state_probs instead
    probabilities: dict[tuple[float, float], float] = field(init=False, default_factory=dict)
    # Dictionary mapping (lat, lon, state) -> P(state) for 3-state mode
    state_probs: dict[tuple[float, float, int], float] = field(init=False, default_factory=dict)
    # Set of (lat, lon) tuples that have valid beliefs
    valid_locations: set[tuple[float, float]] = field(init=False, default_factory=set)
    # Prior probability for unobserved cells (for binary mode)
    prior_probability: float = 0.5
    # Number of states (M) - 2 for binary, 3 for kernel mode
    num_states: int = 2

    def __post_init__(self):
        """Initialize belief dictionary."""
        self.probabilities = {}
        self.state_probs = {}
        self.valid_locations = set()
    
    def get_probability(self, lat: float, lon: float, state: Optional[int] = None) -> float:
        """
        Get probability for a specific (lat, lon) location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            state: State index (0, 1, or 2). If None, returns P(state=1) for backward compatibility.
            
        Returns:
            Probability P(state) for 3-state mode, or P(E_j = 1) for binary mode.
            Returns prior_probability/num_states if location not in belief.
        """
        key = (lat, lon)
        if self.num_states > 2:
            if state is None:
                # Backward compatibility: return P(state=1) for binary compatibility
                return self.state_probs.get((lat, lon, 1), 1.0 / self.num_states)
            else:
                return self.state_probs.get((lat, lon, state), 1.0 / self.num_states)
        else:
            # Binary mode: state parameter ignored
            return self.probabilities.get(key, self.prior_probability)
    
    def set_probability(self, lat: float, lon: float, prob: float, state: Optional[int] = None) -> None:
        """
        Set probability for a specific (lat, lon) location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            prob: Probability, must be in [0, 1]
            state: State index (0, 1, or 2). If None, sets P(state=1) for backward compatibility.
        """
        key = (lat, lon)
        prob = float(np.clip(prob, 0.0, 1.0))
        
        if self.num_states > 2:
            if state is None:
                # Backward compatibility: set P(state=1) and infer others
                # For n-state mode, this is an approximation
                self.state_probs[(lat, lon, 1)] = prob
                # Distribute remaining probability uniformly among other states
                remaining_prob = 1.0 - prob
                for s in range(self.num_states):
                    if s != 1:
                        self.state_probs[(lat, lon, s)] = remaining_prob / (self.num_states - 1)
            else:
                self.state_probs[(lat, lon, state)] = prob
                # Normalize probabilities to sum to 1
                total = sum(self.state_probs.get((lat, lon, s), 0.0) for s in range(self.num_states))
                if total > 1e-10:
                    for s in range(self.num_states):
                        if (lat, lon, s) in self.state_probs:
                            self.state_probs[(lat, lon, s)] /= total
                else:
                    # Uniform prior if all zero
                    for s in range(self.num_states):
                        self.state_probs[(lat, lon, s)] = 1.0 / self.num_states
        else:
            # Binary mode: state parameter ignored
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
                - Integer (H, W) array of observed states (0, 1, or 2 for kernel mode), OR
                - Dictionary mapping (lat, lon) -> observed state (0, 1, or 2) for cells in observation_mask
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
        """
        observation_mask = np.asarray(observation_mask, dtype=bool)
        assert observation_mask.shape == (self.height, self.width), "observation_mask must be (H,W)"
        
        if isinstance(observations, dict):
            # Update from dictionary: (lat, lon) -> state
            for (lat, lon), observed_state in observations.items():
                observed_state = int(observed_state)
                if self.num_states > 2:
                    # Multi-state mode: set delta function for observed state
                    for s in range(self.num_states):
                        self.state_probs[(lat, lon, s)] = 1.0 if s == observed_state else 0.0
                    # Do NOT update probabilities dict in multi-state mode
                else:
                    # Binary mode: P(E_j = 1) = observed_state (0 or 1)
                    self.set_probability(lat, lon, float(observed_state))
                self.valid_locations.add((lat, lon))
        else:
            # Update from array: find (lat, lon) for each observed cell
            observations = np.asarray(observations, dtype=int)
            assert observations.shape == (self.height, self.width), "observations must be (H,W)"
            
            for y in range(self.height):
                for x in range(self.width):
                    if observation_mask[y, x]:
                        lat, lon = lat_grid[y, x], lon_grid[y, x]
                        observed_state = int(observations[y, x])
                        if self.num_states > 2:
                            # Multi-state mode: set delta function for observed state
                            for s in range(self.num_states):
                                self.state_probs[(lat, lon, s)] = 1.0 if s == observed_state else 0.0
                            # Do NOT update probabilities dict in multi-state mode
                        else:
                            # Binary mode: P(E_j = 1) = observed_state (0 or 1)
                            self.set_probability(lat, lon, float(observed_state))
                        self.valid_locations.add((lat, lon))

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
        
        For binary state: H(b) = -p*log2(p) - (1-p)*log2(1-p) where p is P(state=1).
        For 3-state mode: H(b) = -Σ_s p_s*log2(p_s) where p_s is P(state=s).
        
        Args:
            cell_mask: Optional (H, W) mask indicating which cells to include.
                      If None, computes entropy over all valid locations.
            lat_grid: (H, W) array of latitude values (required if cell_mask is provided)
            lon_grid: (H, W) array of longitude values (required if cell_mask is provided)
        
        Returns:
            Total entropy (sum over cells)
        """
        total_entropy = 0.0
        
        if cell_mask is None:
            # Compute entropy over all valid locations
            locations = list(self.valid_locations)
        else:
            # Compute entropy over masked region
            if lat_grid is None or lon_grid is None:
                raise ValueError("lat_grid and lon_grid must be provided when cell_mask is specified")
            cell_mask = np.asarray(cell_mask, dtype=bool)
            locations = []
            for y in range(self.height):
                for x in range(self.width):
                    if cell_mask[y, x]:
                        lat, lon = lat_grid[y, x], lon_grid[y, x]
                        locations.append((lat, lon))
        
        if len(locations) == 0:
            return 0.0
        
        # Compute entropy for each location
        for lat, lon in locations:
            if self.num_states == 2:
                # Binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)
                p = self.get_probability(lat, lon)
                eps = 1e-12
                p = min(max(p, eps), 1.0 - eps)  # Clamp to [eps, 1-eps]
                if p == 0.0 or p == 1.0:
                    entropy_val = 0.0
                else:
                    entropy_val = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
            else:
                # Multi-state entropy: H(b) = -Σ_s p_s*log2(p_s)
                entropy_val = 0.0
                for s in range(self.num_states):
                    p_s = self.get_probability(lat, lon, state=s)
                    eps = 1e-12
                    p_s = min(max(p_s, eps), 1.0 - eps)  # Clamp to [eps, 1-eps]
                    if p_s > eps and p_s < 1.0 - eps:
                        entropy_val -= p_s * np.log2(p_s)
            
            total_entropy += entropy_val
        
        return float(total_entropy)

    def reset(self) -> None:
        """Reset belief to empty state."""
        self.probabilities.clear()
        self.state_probs.clear()
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
        
        # Determine environment mode and num_states
        mode_map = {"dbn2": 0, "rsp": 1, "viirs_table": 2, "kernel": 3}
        mode = mode_map.get(env.mode, 1)  # Default to RSP if unknown
        
        # Update num_states based on environment mode
        if env.mode == "kernel":
            self.num_states = 3
        
        # Create a copy to update - maintain ALL locations in belief
        new_probs = {}
        new_state_probs = {}
        
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
                if self.num_states > 2:
                    for s in range(self.num_states):
                        new_state_probs[(lat, lon, s)] = self.state_probs.get((lat, lon, s), 1.0 / self.num_states)
                    # Do NOT update new_probs in multi-state mode
                else:
                    new_probs[(lat, lon)] = self.probabilities.get((lat, lon), self.prior_probability)
                continue
            
            y, x = grid_coords
            
            # Skip observed cells (they are updated separately via update_from_observation)
            if observation_mask[y, x]:
                # Keep observed cells at their current value (will be updated separately)
                if self.num_states > 2:
                    for s in range(self.num_states):
                        new_state_probs[(lat, lon, s)] = self.state_probs.get((lat, lon, s), 1.0 / self.num_states)
                    # Do NOT update new_probs in multi-state mode
                else:
                    new_probs[(lat, lon)] = self.probabilities.get((lat, lon), self.prior_probability)
                continue
            
            # Get neighbors for this cell
            neighbors = list(neighbor_getter(y, x))
            num_neighbors = len(neighbors)
            
            if num_neighbors == 0:
                # No neighbors - keep current belief (or use uniform prior)
                if self.num_states > 2:
                    for s in range(self.num_states):
                        new_state_probs[(lat, lon, s)] = self.state_probs.get((lat, lon, s), 1.0 / self.num_states)
                    # Do NOT update new_probs in multi-state mode
                else:
                    new_probs[(lat, lon)] = self.probabilities.get((lat, lon), self.prior_probability)
                continue
            
            # Prepare current cell belief [num_states]
            current_belief = np.zeros(self.num_states, dtype=np.float64)
            if self.num_states > 2:
                for s in range(self.num_states):
                    current_belief[s] = self.state_probs.get((lat, lon, s), 1.0 / self.num_states)
            else:
                p_current = self.get_probability(lat, lon)
                current_belief[0] = 1.0 - p_current  # P(E=0)
                current_belief[1] = p_current  # P(E=1)
            
            # Prepare neighbor beliefs [num_neighbors, num_states]
            neighbor_beliefs = np.zeros((num_neighbors, self.num_states), dtype=np.float64)
            for i, (ny, nx) in enumerate(neighbors):
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    n_lat, n_lon = lat_grid[ny, nx], lon_grid[ny, nx]
                    if self.has_location(n_lat, n_lon):
                        if self.num_states > 2:
                            for s in range(self.num_states):
                                neighbor_beliefs[i, s] = self.state_probs.get((n_lat, n_lon, s), 1.0 / self.num_states)
                        else:
                            p_neighbor = self.get_probability(n_lat, n_lon)
                            neighbor_beliefs[i, 0] = 1.0 - p_neighbor  # P(E=0)
                            neighbor_beliefs[i, 1] = p_neighbor  # P(E=1)
                    else:
                        # Neighbor not in belief - use uniform prior
                        if self.num_states > 2:
                            for s in range(self.num_states):
                                neighbor_beliefs[i, s] = 1.0 / self.num_states
                        else:
                            neighbor_beliefs[i, 0] = 1.0 - self.prior_probability
                            neighbor_beliefs[i, 1] = self.prior_probability
                else:
                    # Out of bounds neighbor - assume unburned (state 0)
                    neighbor_beliefs[i, 0] = 1.0
                    for s in range(1, self.num_states):
                        neighbor_beliefs[i, s] = 0.0
            
            # Handle kernel mode separately (needs kernel_learner)
            if env.mode == "kernel":
                # Kernel mode: use kernel_learner for transition probabilities
                # Note: Kernel mode is specifically 3 states, so we use range(3) here
                if not hasattr(env, 'kernel_learner') or env.kernel_learner is None:
                    # Fallback: keep current belief
                    for s in range(self.num_states):
                        new_state_probs[(lat, lon, s)] = current_belief[s]
                    # Do NOT update new_probs in multi-state mode
                    continue
                
                # Count burning neighbors (state 1) for kernel
                num_burning_neighbors = 0
                for i in range(num_neighbors):
                    num_burning_neighbors += neighbor_beliefs[i, 1]
                num_burning_neighbors = int(np.round(num_burning_neighbors))
                
                # Get material for this cell
                cell_key = (y, x)
                if not hasattr(env, 'material_map') or cell_key not in env.material_map:
                    # Fallback: keep current belief
                    for s in range(self.num_states):
                        new_state_probs[(lat, lon, s)] = current_belief[s]
                    # Do NOT update new_probs in multi-state mode
                    continue
                
                material = env.material_map[cell_key]
                
                # Compute new belief using kernel transition probabilities
                new_belief = np.zeros(3, dtype=np.float64)
                for next_state in range(3):
                    prob = 0.0
                    for current_state in range(3):
                        # Get transition probability from kernel
                        transition_prob = env.kernel_learner.get_transition_probability(
                            current_state, material, num_burning_neighbors, next_state
                        )
                        prob += transition_prob * current_belief[current_state]
                    new_belief[next_state] = prob
                
                # Normalize
                total = np.sum(new_belief)
                if total > 1e-10:
                    new_belief = new_belief / total
                else:
                    new_belief[:] = 1.0 / 3.0
                
                # Store 3-state probabilities (kernel mode is always 3 states)
                # States: 0=unburned, 1=burning, 2=burned
                for s in range(3):
                    new_state_probs[(lat, lon, s)] = float(new_belief[s])
                # Do NOT update probabilities dict in multi-state mode
                continue
            
            # Get environment parameters for this cell (for non-kernel modes)
            # Handle ReplayEnvironment which doesn't have these attributes
            # Use hasattr to safely check for attributes
            if env.mode == "dbn2":
                birth_rate = getattr(env, 'birth_rate', 0.01)
                death_rate = getattr(env, 'death_rate', 0.1)
                neighbor_influence = getattr(env, 'neighbor_influence', 0.05)
                # RSP params not used for DBN-2
                lam = beta0 = alpha = delta = 0.0
            elif env.mode == "rsp":
                # Get per-cell RSP parameters (use hasattr for ReplayEnvironment compatibility)
                lam_map = getattr(env, 'ignition_map', None)
                beta0_map = getattr(env, 'beta0_map', None)
                alpha_map = getattr(env, 'alpha_map', None)
                persistence_map = getattr(env, 'persistence_map', None)
                
                # Get RSP object if available, otherwise use defaults
                rsp_obj = getattr(env, 'rsp', None)
                if rsp_obj is None:
                    # Default RSP parameters if rsp object not available (e.g., ReplayEnvironment)
                    lam_default = 0.01
                    beta0_default = 0.1
                    alpha_default = 1.0
                    delta_default = 0.9
                else:
                    lam_default = rsp_obj.lam
                    beta0_default = rsp_obj.beta0
                    alpha_default = rsp_obj.alpha
                    delta_default = rsp_obj.delta
                
                lam = float(lam_map[y, x]) if lam_map is not None else lam_default
                beta0 = float(beta0_map[y, x]) if beta0_map is not None else beta0_default
                alpha = float(alpha_map[y, x]) if alpha_map is not None else alpha_default
                delta = float(persistence_map[y, x]) if persistence_map is not None else delta_default
                
                # DBN-2 params not used for RSP
                birth_rate = death_rate = neighbor_influence = 0.0
            else:  # viirs_table or unknown
                # Not supported - keep current belief
                if self.num_states > 2:
                    for s in range(self.num_states):
                        new_state_probs[(lat, lon, s)] = current_belief[s]
                    # Do NOT update new_probs in multi-state mode
                else:
                    new_probs[(lat, lon)] = self.probabilities.get((lat, lon), self.prior_probability)
                continue
            
            # Check if cell is blocked (can't evolve)
            blocked_mask = getattr(env, 'blocked_mask', None)
            if blocked_mask is not None and blocked_mask[y, x]:
                # Blocked cells stay at unburned (state 0)
                if self.num_states > 2:
                    new_state_probs[(lat, lon, 0)] = 1.0
                    for s in range(1, self.num_states):
                        new_state_probs[(lat, lon, s)] = 0.0
                    # Do NOT update new_probs in multi-state mode
                else:
                    new_probs[(lat, lon)] = 0.0
                continue
            
            # Use Numba-optimized belief evolution if available (only for binary modes)
            if NUMBA_AVAILABLE and self.num_states == 2:
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
                    # Extract P(E=1) from new belief (binary mode only)
                    if self.num_states == 2:
                        new_probs[(lat, lon)] = float(np.clip(new_belief[1], 0.0, 1.0))
                    continue
                except Exception as e:
                    # Fall back to Python implementation if Numba fails
                    import warnings
                    warnings.warn(f"Numba belief evolution failed, falling back to Python: {e}")
            
            # Fallback Python implementation (slower but correct)
            # This implements the full mean field formula without Numba
            new_belief = np.zeros(self.num_states, dtype=np.float64)
            num_combinations = self.num_states ** num_neighbors
            
            # For each next state x'
            for next_state in range(self.num_states):
                prob = 0.0
                
                # Sum over all current states x_j
                for current_state in range(self.num_states):
                    p_current_state = current_belief[current_state]
                    
                    # Sum over all neighbor configurations
                    for combo_idx in range(num_combinations):
                        # Generate neighbor state combination
                        combination = []
                        temp_idx = combo_idx
                        for _ in range(num_neighbors):
                            combination.append(temp_idx % self.num_states)
                            temp_idx //= self.num_states
                        
                        # Count active neighbors (state 1) for binary, or use combination for 3-state
                        if self.num_states == 2:
                            active_neighbors = sum(1 for s in combination if s == 1)
                            # Compute transition probability using env method
                            transition_prob = env.transition_probability(y, x, current_state, active_neighbors)
                            # env.transition_probability returns P(x'=1), so adjust for next_state
                            if next_state == 0:
                                transition_prob = 1.0 - transition_prob
                        else:
                            # For 3-state, would need kernel transition probabilities
                            # This shouldn't happen here (kernel mode handled above)
                            transition_prob = 1.0 / self.num_states  # Fallback uniform
                        
                        # Compute product of neighbor beliefs for this configuration
                        prob_neighbor_config = 1.0
                        for i, neighbor_state in enumerate(combination):
                            prob_neighbor_config *= neighbor_beliefs[i, neighbor_state]
                        
                        # Add to sum
                        prob += transition_prob * p_current_state * prob_neighbor_config
                
                new_belief[next_state] = prob
            
            # Normalize
            total = np.sum(new_belief)
            if total > 1e-10:
                new_belief = new_belief / total
            else:
                # Uniform if total too small
                new_belief[:] = 1.0 / self.num_states
            
            # Store probabilities
            if self.num_states > 2:
                for s in range(self.num_states):
                    new_state_probs[(lat, lon, s)] = float(new_belief[s])
                # Do NOT update new_probs in multi-state mode
            else:
                new_probs[(lat, lon)] = float(new_belief[1])
        
        # Update probabilities - maintain all locations
        # Only update probabilities dict for binary mode
        if self.num_states == 2:
            self.probabilities = new_probs
        if self.num_states > 2:
            self.state_probs = new_state_probs

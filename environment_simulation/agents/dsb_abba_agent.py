"""Decentralized SB-ABBA agent for multi-agent wildfire detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set
from collections import defaultdict
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from .sharing_monte_carlo_agent import SharingMonteCarloAgent
from .ground_station import GroundStation
from .belief import Belief


# ============================================================================
# NUMBA-accelerated kernels for hot loops
# ============================================================================

@njit(cache=True)
def entropy_binary(p: float) -> float:
    """
    Compute binary entropy H(p) = -[p*log2(p) + (1-p)*log2(1-p)].
    
    Args:
        p: Probability P(X=1)
    
    Returns:
        Entropy in bits
    """
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)  # Clamp to [eps, 1-eps]
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


@njit(cache=True)
def info_gain_for_action_numba(
    prob_grid: np.ndarray,  # Dense [H, W] probability grid
    action_y: np.ndarray,   # [K] array of y indices
    action_x: np.ndarray,   # [K] array of x indices
) -> float:
    """
    Compute information gain for an action using NUMBA.
    
    Info gain = sum of entropies over observed cells (post-entropy is 0 for perfect obs).
    
    Args:
        prob_grid: Dense probability grid [height, width]
        action_y: Array of y (row) indices [num_cells]
        action_x: Array of x (col) indices [num_cells]
    
    Returns:
        Information gain (sum of entropies)
    """
    total_entropy = 0.0
    H, W = prob_grid.shape
    for k in range(action_y.shape[0]):
        y = action_y[k]
        x = action_x[k]
        if 0 <= y < H and 0 <= x < W:
            p = prob_grid[y, x]
            total_entropy += entropy_binary(p)
    return total_entropy


@njit(cache=True)
def expected_value_for_action_numba(
    prob_grid: np.ndarray,  # Dense [H, W] probability grid
    action_y: np.ndarray,   # [K] array of y indices
    action_x: np.ndarray,   # [K] array of x indices
    utility_0: float,
    utility_1: float,
    utility_2: float = 0.0,  # Utility for state 2 (burned) - default 0 for backward compatibility
) -> float:
    """
    Compute expected event value for an action using NUMBA.
    
    For binary states: E[value] = sum over cells of [P(0)*u0 + P(1)*u1].
    For 3 states: E[value] = sum over cells of [P(0)*u0 + P(1)*u1 + P(2)*u2].
    
    Note: For binary mode, prob_grid contains P(state=1), so:
      - P(0) = 1 - p
      - P(1) = p
      - P(2) = 0
    
    For 3-state mode, we approximate by treating prob_grid as P(state=1), and:
      - P(0) = 1 - p (if p < 0.5, assume unburned)
      - P(1) = p (if p >= 0.5, assume burning)
      - P(2) = 0 (burned cells would have p=0, but we can't distinguish from unburned)
    
    This is a simplification - full 3-state belief would require separate probability grids.
    
    Args:
        prob_grid: Dense probability grid [height, width] (P(state=1) for binary, approximate for 3-state)
        action_y: Array of y (row) indices [num_cells]
        action_x: Array of x (col) indices [num_cells]
        utility_0: Utility for state 0 (unburned)
        utility_1: Utility for state 1 (burning)
        utility_2: Utility for state 2 (burned) - only used if > 0 (indicates 3-state mode)
    
    Returns:
        Expected value
    """
    total_value = 0.0
    H, W = prob_grid.shape
    is_3_state = utility_2 != 0.0  # If utility_2 is non-zero, assume 3-state mode
    
    for k in range(action_y.shape[0]):
        y = action_y[k]
        x = action_x[k]
        if 0 <= y < H and 0 <= x < W:
            p = prob_grid[y, x]
            if is_3_state:
                # For 3-state mode, approximate probabilities:
                # If p is very low (< 0.1), likely unburned
                # If p is high (> 0.9), likely burning
                # If p is medium (0.1-0.9), split between unburned and burning
                # We can't distinguish burned from unburned with binary belief, so assume burned has p=0
                if p < 0.1:
                    # Mostly unburned
                    p0, p1, p2 = 1.0 - p, p, 0.0
                elif p > 0.9:
                    # Mostly burning
                    p0, p1, p2 = 0.0, p, 1.0 - p
                else:
                    # Mixed: assume unburned and burning, no burned
                    p0, p1, p2 = 1.0 - p, p, 0.0
                total_value += p0 * utility_0 + p1 * utility_1 + p2 * utility_2
            else:
                # Binary mode: E[value] = P(0)*u0 + P(1)*u1 = (1-p)*u0 + p*u1
                total_value += (1.0 - p) * utility_0 + p * utility_1
    return total_value


@njit(cache=True)
def apply_perfect_obs_numba(
    prob_grid: np.ndarray,  # Dense [H, W] probability grid (modified in-place)
    action_y: np.ndarray,   # [K] array of y indices
    action_x: np.ndarray,   # [K] array of x indices
    obs_values: np.ndarray, # [K] array of observed values (0 or 1)
) -> None:
    """
    Apply perfect observations to a dense probability grid using NUMBA.
    
    Sets P = obs_value for each observed cell (0 or 1).
    
    Args:
        prob_grid: Dense probability grid [height, width] (modified in-place)
        action_y: Array of y (row) indices [num_cells]
        action_x: Array of x (col) indices [num_cells]
        obs_values: Array of observed values [num_cells]
    """
    H, W = prob_grid.shape
    for k in range(action_y.shape[0]):
        y = action_y[k]
        x = action_x[k]
        if 0 <= y < H and 0 <= x < W:
            obs_val = obs_values[k]
            prob_grid[y, x] = float(obs_val)


@dataclass
class ObservationRecord:
    """Record of an observation with source agent and timestamp."""
    timestep: int
    source_agent: str
    cell: tuple[float, float]
    value: int


@dataclass
class DSBABBAAgent(SharingMonteCarloAgent):
    """
    Decentralized SB-ABBA agent.
    
    Implements the three-phase SB-ABBA algorithm:
    1. Build belief point set
    2. PBVI-style backups
    3. Extract open-loop plan
    
    Uses sync maps (σ_i) and clean times (t_clean_i) to track information.
    """
    
    # SB-ABBA parameters (increased for accuracy)
    n_seed: int = 50  # Number of belief point seeds (increased for better coverage)
    n_roll: int = 20  # Number of rollouts per Q-value estimate (increased for better estimates)
    n_sweep: int = 5  # Number of PBVI backup sweeps (increased for convergence)
    horizon: int = 5  # Planning horizon (can be adaptive, matches config)
    gamma: float = 0.95  # Discount factor
    
    # Sync map tracking
    _sigma: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _received_obs: List[ObservationRecord] = field(default_factory=list, init=False, repr=False)
    _t_sync: int = field(default=0, init=False, repr=False)
    _need_replan: bool = field(default=True, init=False, repr=False)
    _current_plan: List[Any] = field(default_factory=list, init=False, repr=False)
    _plan_idx: int = field(default=0, init=False, repr=False)
    _plan_start_time: int = field(default=0, init=False, repr=False)
    
    # Belief point set storage
    _belief_points: List[Tuple[int, Dict, Any]] = field(default_factory=list, init=False, repr=False)
    _value_function: Dict[Any, float] = field(default_factory=dict, init=False, repr=False)
    _q_function: Dict[Tuple[Any, Any], float] = field(default_factory=dict, init=False, repr=False)
    
    # Peer belief tracking (B_j for each peer agent j)
    # Updated when we receive observations from peers
    _peer_beliefs: Dict[str, Belief] = field(default_factory=dict, init=False, repr=False)
    
    # Peer belief tracking (B_j for each peer agent j)
    _peer_beliefs: Dict[str, Belief] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize sync maps."""
        # Initialize sigma for all agents (will be updated when we know agent names)
        # Note: Parent classes don't have __post_init__, so we don't call super()
        if not hasattr(self, '_sigma'):
            self._sigma = {}
    
    def _update_sync_map(self, all_agent_names: List[str], current_time: int) -> None:
        """
        Update sync map σ_i(j) for all agents j.
        
        σ_i(j) = max{τ ≤ t : agent i has integrated data originating from j up to time τ}
        
        Uses _last_processed_timestep for peers (which tracks what we've actually integrated)
        and own observation history for self.
        """
        # Initialize sigma for all agents if not present
        for agent_name in all_agent_names:
            if agent_name not in self._sigma:
                self._sigma[agent_name] = -1
        
        # Update sigma for peers: use _last_processed_timestep (what we've actually integrated)
        for agent_name in all_agent_names:
            if agent_name != self.name:
                # σ_i(j) = latest timestep we've processed from agent j
                self._sigma[agent_name] = self._last_processed_timestep.get(agent_name, -1)
        
        # Update sigma for self: use own observation history
        if self._own_observation_history:
            self._sigma[self.name] = max(self._own_observation_history.keys())
        else:
            self._sigma[self.name] = -1
    
    def _get_t_clean(self) -> int:
        """
        Compute local clean time: t_clean_i = min_j σ_i(j).
        
        Returns:
            Clean time, or -1 if no observations yet
        """
        if not self._sigma:
            return -1
        return min(self._sigma.values()) if self._sigma.values() else -1
    
    def _get_aoi(self, current_time: int, agent_name: str) -> int:
        """
        Compute Age of Information: Δ_i(j) = t - σ_i(j).
        
        Args:
            current_time: Current time step
            agent_name: Name of agent j
        
        Returns:
            AoI, or large value if unknown
        """
        if agent_name not in self._sigma or self._sigma[agent_name] < 0:
            return 10000  # Large value for unknown
        return current_time - self._sigma[agent_name]
    
    def _station_contact_happens(
        self,
        t: int,
        gs: GroundStation,
        all_agents: List[Any],
        min_satellites_per_station: int = 2,
    ) -> bool:
        """
        Check if a ground station contact event happens at time t.
        
        A contact happens if >= min_satellites_per_station agents are in range of the station.
        
        Args:
            t: Time step to check
            gs: Ground station to check
            all_agents: List of all agents
            min_satellites_per_station: Minimum number of satellites required for contact
        
        Returns:
            True if contact happens at time t
        """
        count = 0
        for ag in all_agents:
            if ag._is_in_contact_with_ground_station(t, gs):
                count += 1
        return count >= min_satellites_per_station
    
    def _next_contact_time(
        self,
        current_time: int,
        ground_stations: List[GroundStation],
        all_agents: Optional[List[Any]] = None,
        min_satellites_per_station: int = 2,
        max_horizon: int = 50,
    ) -> Optional[int]:
        """
        Find next time when a ground station contact event happens.
        
        Args:
            current_time: Current time step
            ground_stations: List of ground stations
            all_agents: List of all agents (required for correct contact check)
            min_satellites_per_station: Minimum number of satellites required for contact
            max_horizon: Maximum steps to look ahead
        
        Returns:
            Next contact time, or None if not found within horizon
        """
        if all_agents is None:
            # Fallback: use old behavior (check only this agent)
            for t in range(current_time + 1, current_time + max_horizon + 1):
                if self._is_in_contact_with_ground_station(t, ground_stations[0]):
                    return t
            return None
        
        # Check for contact events (>= min_satellites_per_station agents in range)
        for t in range(current_time + 1, current_time + max_horizon + 1):
            for gs in ground_stations:
                if self._station_contact_happens(t, gs, all_agents, min_satellites_per_station):
                    return t
        return None
    
    def _belief_at_time(
        self,
        env: "Environment",  # type: ignore
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        target_time: int,
    ) -> Belief:
        """
        Reconstruct belief at a specific time by replaying observations.
        
        Args:
            env: Environment object
            lat_grid: Latitude grid
            lon_grid: Longitude grid
            target_time: Target time step
        
        Returns:
            Belief at target_time
        """
        # Start from prior belief
        from copy import deepcopy
        belief = Belief(height=lat_grid.shape[0], width=lat_grid.shape[1], num_states=self.belief.num_states)
        
        # Get all observations up to target_time
        all_obs_by_time: Dict[int, Dict[tuple[float, float], int]] = {}
        
        # Add own observations
        for t, obs_dict in self._own_observation_history.items():
            if t <= target_time:
                if t not in all_obs_by_time:
                    all_obs_by_time[t] = {}
                all_obs_by_time[t].update(obs_dict)
        
        # Add received observations
        for obs_record in self._received_obs:
            if obs_record.timestep <= target_time:
                if obs_record.timestep not in all_obs_by_time:
                    all_obs_by_time[obs_record.timestep] = {}
                all_obs_by_time[obs_record.timestep][obs_record.cell] = obs_record.value
        
        # Replay from t=0 to target_time
        # The belief should represent the state at the START of each timestep
        # So we evolve first, then apply observations
        for t in range(target_time):
            # Evolve belief from t to t+1
            def neighbor_getter(y: int, x: int) -> list[tuple[int, int]]:
                return list(env._iter_neighbors(y, x))
            
            # Create empty observation mask (all cells unobserved during evolution)
            observation_mask = np.zeros((lat_grid.shape[0], lat_grid.shape[1]), dtype=bool)
            
            belief.evolve_with_transition_kernel(
                env, observation_mask, lat_grid, lon_grid, neighbor_getter
            )
            
            # Apply observations at time t+1 (after evolution)
            if t + 1 in all_obs_by_time:
                observation_mask = self.points_to_mask(
                    list(all_obs_by_time[t + 1].keys()),
                    lat_grid, lon_grid,
                    lat_grid.shape[0], lat_grid.shape[1]
                )
                belief.update_from_observation(
                    observation_mask,
                    all_obs_by_time[t + 1],
                    lat_grid, lon_grid
                )
        
        # If target_time is 0, we might have observations at t=0
        if target_time == 0 and 0 in all_obs_by_time:
            observation_mask = self.points_to_mask(
                list(all_obs_by_time[0].keys()),
                lat_grid, lon_grid,
                lat_grid.shape[0], lat_grid.shape[1]
            )
            belief.update_from_observation(
                observation_mask,
                all_obs_by_time[0],
                lat_grid, lon_grid
            )
        
        return belief
    
    def _reconstruct_peer_belief(
        self,
        env: "Environment",  # type: ignore
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        peer_name: str,
        target_time: int,
        received_lookup: Optional[Dict[str, Dict[int, Dict[tuple[float, float], int]]]] = None,
    ) -> Belief:
        """
        Reconstruct a peer agent's belief at a specific time.
        
        Uses only observations we've received from this peer (tau <= sigma_i(j)).
        This gives us the peer's belief state based on what they would know.
        
        Args:
            env: Environment object
            lat_grid: Latitude grid
            lon_grid: Longitude grid
            peer_name: Name of the peer agent
            target_time: Target time step
            received_lookup: Pre-built lookup: received_lookup[j][tau] -> obs_dict
        
        Returns:
            Peer's belief at target_time
        """
        # Start from prior belief
        belief = Belief(height=lat_grid.shape[0], width=lat_grid.shape[1], num_states=self.belief.num_states)
        
        # Build received lookup if not provided
        if received_lookup is None:
            received_lookup = defaultdict(lambda: defaultdict(dict))
            for obs_record in self._received_obs:
                if obs_record.source_agent == peer_name:
                    received_lookup[peer_name][obs_record.timestep][obs_record.cell] = obs_record.value
        
        # Get all observations from this peer up to target_time
        all_obs_by_time: Dict[int, Dict[tuple[float, float], int]] = {}
        for t, obs_dict in received_lookup.get(peer_name, {}).items():
            if t <= target_time:
                all_obs_by_time[t] = obs_dict
        
        # Replay from t=0 to target_time
        for t in range(target_time):
            # Evolve belief from t to t+1
            def neighbor_getter(y: int, x: int) -> list[tuple[int, int]]:
                return list(env._iter_neighbors(y, x))
            
            observation_mask = np.zeros((lat_grid.shape[0], lat_grid.shape[1]), dtype=bool)
            belief.evolve_with_transition_kernel(
                env, observation_mask, lat_grid, lon_grid, neighbor_getter
            )
            
            # Apply observations at time t+1 (after evolution)
            if t + 1 in all_obs_by_time:
                observation_mask = self.points_to_mask(
                    list(all_obs_by_time[t + 1].keys()),
                    lat_grid, lon_grid,
                    lat_grid.shape[0], lat_grid.shape[1]
                )
                belief.update_from_observation(
                    observation_mask,
                    all_obs_by_time[t + 1],
                    lat_grid, lon_grid
                )
        
        # If target_time is 0, we might have observations at t=0
        if target_time == 0 and 0 in all_obs_by_time:
            observation_mask = self.points_to_mask(
                list(all_obs_by_time[0].keys()),
                lat_grid, lon_grid,
                lat_grid.shape[0], lat_grid.shape[1]
            )
            belief.update_from_observation(
                observation_mask,
                all_obs_by_time[0],
                lat_grid, lon_grid
            )
        
        return belief
    
    def _sample_system_state(
        self,
        env: "Environment",  # type: ignore
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        t_clean: int,
        t_sync: int,
        all_agents: List[Any],
        rng: np.random.Generator,
        received_lookup: Optional[Dict[str, Dict[int, Dict[tuple[float, float], int]]]] = None,
        ground_stations: Optional[List[GroundStation]] = None,
    ) -> Tuple[Dict[str, int], Belief, Dict[str, Dict[int, Dict[tuple[float, float], int]]]]:
        """
        Sample a possible fully-informed belief at t_sync.
        
        Implements SB-ABBA's delayed information model:
        - B_i: agent i's belief (only includes own obs + already-received peer obs)
        - pending[j]: sampled peer observations that haven't been delivered yet
        - Only fuse pending at contact events
        
        Args:
            env: Environment object
            lat_grid: Latitude grid
            lon_grid: Longitude grid
            t_clean: Clean time
            t_sync: Sync time
            all_agents: List of all agents (for peer models)
            rng: Random number generator
            received_lookup: Pre-built lookup: received_lookup[j][tau] -> obs_dict
            ground_stations: List of ground stations (for contact events)
        
        Returns:
            Tuple of (sampled_sigma, sampled_belief)
        """
        from copy import deepcopy
        
        # Start from belief at t_clean (B_i: only what agent i knows)
        belief = self._belief_at_time(env, lat_grid, lon_grid, t_clean)
        
        # Sample sigma (for now, keep it fixed - can extend later)
        sampled_sigma = deepcopy(self._sigma)
        
        # Pending messages: pending[j][tau] -> obs_dict (not yet fused into B_i)
        pending: Dict[str, Dict[int, Dict[tuple[float, float], int]]] = defaultdict(lambda: defaultdict(dict))
        
        # Build received lookup if not provided
        if received_lookup is None:
            received_lookup = defaultdict(lambda: defaultdict(dict))
            for obs_record in self._received_obs:
                received_lookup[obs_record.source_agent][obs_record.timestep][obs_record.cell] = obs_record.value
        
        # Initialize peer beliefs (B_j) at t_clean from received observations
        peer_beliefs: Dict[str, Belief] = {}
        for agent in all_agents:
            if agent.name != self.name:
                peer_beliefs[agent.name] = self._reconstruct_peer_belief(
                    env, lat_grid, lon_grid, agent.name, t_clean, received_lookup
                )
        
        # Roll forward from t_clean to t_sync
        for t in range(t_clean, t_sync):
            # Contact fusion is now handled separately before storing belief points
            # (see _fuse_pending_if_contact call in Phase 1)
            
            # Predict belief (B_i evolves)
            def neighbor_getter(y: int, x: int) -> list[tuple[int, int]]:
                return list(env._iter_neighbors(y, x))
            
            observation_mask = np.zeros((lat_grid.shape[0], lat_grid.shape[1]), dtype=bool)
            belief.evolve_with_transition_kernel(
                env, observation_mask, lat_grid, lon_grid, neighbor_getter
            )
            
            # Evolve peer beliefs (B_j) forward
            for peer_name, peer_belief in peer_beliefs.items():
                observation_mask = np.zeros((lat_grid.shape[0], lat_grid.shape[1]), dtype=bool)
                peer_belief.evolve_with_transition_kernel(
                    env, observation_mask, lat_grid, lon_grid, neighbor_getter
                )
            
            # For each agent, sample observations
            tau = t + 1  # Observation timestep
            for agent in all_agents:
                if agent.name == self.name:
                    # Use own actual observations if available (always fuse into B_i)
                    if tau in self._own_observation_history:
                        obs_dict = self._own_observation_history[tau]
                        observation_mask = self.points_to_mask(
                            list(obs_dict.keys()),
                            lat_grid, lon_grid,
                            lat_grid.shape[0], lat_grid.shape[1]
                        )
                        belief.update_from_observation(
                            observation_mask, obs_dict, lat_grid, lon_grid
                        )
                else:
                    # Determine known vs unknown using sigma
                    # Known if tau <= sigma_i(j)
                    sigma_j = sampled_sigma.get(agent.name, -1)
                    
                    if tau <= sigma_j:
                        # Known: use received observation (if any) - fuse into B_i immediately
                        known_obs = received_lookup.get(agent.name, {}).get(tau, {})
                        if known_obs:
                            observation_mask = self.points_to_mask(
                                list(known_obs.keys()),
                                lat_grid, lon_grid,
                                lat_grid.shape[0], lat_grid.shape[1]
                            )
                            belief.update_from_observation(
                                observation_mask, known_obs, lat_grid, lon_grid
                            )
                            # Also update peer belief (they saw this observation)
                            if agent.name in peer_beliefs:
                                peer_beliefs[agent.name].update_from_observation(
                                    observation_mask, known_obs, lat_grid, lon_grid
                                )
                        # If no observation at this timestep, treat as "no observation sent"
                    else:
                        # Unknown: sample peer action and observation using peer's belief (B_j)
                        # Get or initialize peer belief
                        if agent.name not in peer_beliefs:
                            peer_beliefs[agent.name] = Belief(height=lat_grid.shape[0], width=lat_grid.shape[1])
                        peer_belief = peer_beliefs[agent.name]
                        
                        # Store in pending (do NOT fuse into B_i yet)
                        for_mask = agent.coverage_mask(lat_grid, lon_grid, t)
                        fov_candidates = self._get_all_contiguous_subsets(
                            for_mask, lat_grid, lon_grid,
                            max_ratio=agent.max_observation_ratio if hasattr(agent, 'max_observation_ratio') else 0.2,
                            rng=rng
                        )
                        
                        if fov_candidates:
                            # Pick best FOV using PEER'S BELIEF (B_j), not our belief (B_i)
                            best_fov = None
                            best_score = -np.inf
                            for fov in fov_candidates[:5]:  # Limit candidates
                                # Compute score using PEER'S BELIEF (B_j)
                                info_gain = self._information_gain_from_belief(
                                    peer_belief, fov, lat_grid, lon_grid
                                )
                                event_value = self._expected_event_value_from_belief(
                                    peer_belief, fov, agent.event_utility if hasattr(agent, 'event_utility') else {0: 0.0, 1: 1.0},
                                    lat_grid=lat_grid, lon_grid=lon_grid
                                )
                                w_h = agent.w_h if hasattr(agent, 'w_h') else 1.0
                                w_v = agent.w_v if hasattr(agent, 'w_v') else 1.0
                                score = w_h * info_gain + w_v * event_value
                                if score > best_score:
                                    best_score = score
                                    best_fov = fov
                            
                            if best_fov:
                                # Sample observation from PEER'S BELIEF (B_j)
                                obs_dict = {}
                                for cell in best_fov:
                                    if peer_belief.has_location(cell[0], cell[1]):
                                        prob = peer_belief.get_probability(cell[0], cell[1])
                                    else:
                                        prob = peer_belief.prior_probability
                                    obs_dict[cell] = rng.binomial(1, np.clip(prob, 0.0, 1.0))
                                
                                # Update peer belief with this observation (they saw it)
                                observation_mask = self.points_to_mask(
                                    list(obs_dict.keys()),
                                    lat_grid, lon_grid,
                                    lat_grid.shape[0], lat_grid.shape[1]
                                )
                                peer_belief.update_from_observation(
                                    observation_mask, obs_dict, lat_grid, lon_grid
                                )
                                
                                # Store in pending (do NOT fuse into B_i)
                                pending[agent.name][tau] = obs_dict
        
        # DO NOT fuse pending at t_sync unless there's an actual contact
        # Pending remains as-is until the next contact event
        return sampled_sigma, belief, pending
    
    def _simulate_one_step(
        self,
        env: "Environment",  # type: ignore
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        t: int,
        belief: Belief,
        action: Set[tuple[float, float]],
        all_agents: List[Any],
        rng: np.random.Generator,
        sigma: Optional[Dict[str, int]] = None,
        pending: Optional[Dict[str, Dict[int, Dict[tuple[float, float], int]]]] = None,
        ground_stations: Optional[List[GroundStation]] = None,
        received_lookup: Optional[Dict[str, Dict[int, Dict[tuple[float, float], int]]]] = None,
    ) -> Tuple[float, Belief, Dict[str, Dict[int, Dict[tuple[float, float], int]]]]:
        """
        Simulate one step forward in belief space with delayed information model.
        
        Implements SB-ABBA's delayed information:
        - B_i: agent i's belief (only own obs + already-received peer obs)
        - pending: sampled peer observations not yet delivered
        - Only fuse pending at contact events
        
        Args:
            env: Environment object
            lat_grid: Latitude grid
            lon_grid: Longitude grid
            t: Current time step
            belief: Current belief (B_i)
            action: Action (set of cells to observe)
            all_agents: List of all agents
            rng: Random number generator
            sigma: Sync map (for determining known vs unknown)
            pending: Pending messages buffer (will be updated)
            ground_stations: List of ground stations (for contact events)
        
        Returns:
            Tuple of (reward, next_belief, updated_pending)
        """
        from copy import deepcopy
        next_belief = deepcopy(belief)
        
        # Initialize pending if not provided
        if pending is None:
            pending = defaultdict(lambda: defaultdict(dict))
        if sigma is None:
            sigma = {}
        
        # Initialize peer beliefs if not provided (reconstruct from sigma)
        # In rollouts, we need to track peer beliefs separately
        if received_lookup is None:
            received_lookup = defaultdict(lambda: defaultdict(dict))
            for obs_record in self._received_obs:
                received_lookup[obs_record.source_agent][obs_record.timestep][obs_record.cell] = obs_record.value
        
        # Reconstruct peer beliefs at time t from received observations
        peer_beliefs: Dict[str, Belief] = {}
        for agent in all_agents:
            if agent.name != self.name:
                # Reconstruct peer belief up to sigma_j (what they would know)
                sigma_j = sigma.get(agent.name, -1)
                if sigma_j >= 0:
                    peer_beliefs[agent.name] = self._reconstruct_peer_belief(
                        env, lat_grid, lon_grid, agent.name, sigma_j, received_lookup
                    )
                else:
                    # No observations yet, start from prior
                    peer_beliefs[agent.name] = Belief(height=lat_grid.shape[0], width=lat_grid.shape[1])
        
        # Contact fusion is now handled separately before storing belief points
        # (see _fuse_pending_if_contact call in Phase 2 rollouts)
        
        # Predict belief (B_i evolves)
        def neighbor_getter(y: int, x: int) -> list[tuple[int, int]]:
            return list(env._iter_neighbors(y, x))
        
        observation_mask = np.zeros((lat_grid.shape[0], lat_grid.shape[1]), dtype=bool)
        next_belief.evolve_with_transition_kernel(
            env, observation_mask, lat_grid, lon_grid, neighbor_getter
        )
        
        # Evolve peer beliefs (B_j) forward
        for peer_name, peer_belief in peer_beliefs.items():
            observation_mask = np.zeros((lat_grid.shape[0], lat_grid.shape[1]), dtype=bool)
            peer_belief.evolve_with_transition_kernel(
                env, observation_mask, lat_grid, lon_grid, neighbor_getter
            )
        
        # Own observation (always fuse into B_i immediately)
        own_observations: Dict[tuple[float, float], int] = {}
        for cell in action:
            if next_belief.has_location(cell[0], cell[1]):
                prob = next_belief.get_probability(cell[0], cell[1])
            else:
                prob = next_belief.prior_probability
            own_observations[cell] = rng.binomial(1, np.clip(prob, 0.0, 1.0))
        
        # Apply own observation to B_i
        observation_mask = self.points_to_mask(
            list(own_observations.keys()),
            lat_grid, lon_grid,
            lat_grid.shape[0], lat_grid.shape[1]
        )
        next_belief.update_from_observation(
            observation_mask, own_observations, lat_grid, lon_grid
        )
        
        # Peer observations: sample and store in pending (do NOT fuse into B_i)
        tau = t + 1  # Observation timestep
        for agent in all_agents:
            if agent.name == self.name:
                continue
            
            # Determine known vs unknown using sigma
            sigma_j = sigma.get(agent.name, -1)
            
            if tau <= sigma_j:
                # Known: would have been fused already, skip
                # But update peer belief if we have the observation
                known_obs = received_lookup.get(agent.name, {}).get(tau, {})
                if known_obs and agent.name in peer_beliefs:
                    observation_mask = self.points_to_mask(
                        list(known_obs.keys()),
                        lat_grid, lon_grid,
                        lat_grid.shape[0], lat_grid.shape[1]
                    )
                    peer_beliefs[agent.name].update_from_observation(
                        observation_mask, known_obs, lat_grid, lon_grid
                    )
                continue
            else:
                # Unknown: sample peer action and observation using peer's belief (B_j)
                if agent.name not in peer_beliefs:
                    peer_beliefs[agent.name] = Belief(height=lat_grid.shape[0], width=lat_grid.shape[1])
                peer_belief = peer_beliefs[agent.name]
                
                for_mask = agent.coverage_mask(lat_grid, lon_grid, t)
                fov_candidates = self._get_all_contiguous_subsets(
                    for_mask, lat_grid, lon_grid,
                    max_ratio=agent.max_observation_ratio if hasattr(agent, 'max_observation_ratio') else 0.2,
                    rng=rng
                )
                
                if fov_candidates:
                    # Pick best FOV using PEER'S BELIEF (B_j), not our belief (B_i)
                    best_fov = None
                    best_score = -np.inf
                    for fov in fov_candidates[:5]:  # Limit candidates
                        # Score using PEER'S BELIEF (B_j)
                        info_gain = self._information_gain_from_belief(
                            peer_belief, fov, lat_grid, lon_grid
                        )
                        event_value = self._expected_event_value_from_belief(
                            peer_belief, fov, agent.event_utility if hasattr(agent, 'event_utility') else {0: 0.0, 1: 1.0},
                            lat_grid=lat_grid, lon_grid=lon_grid
                        )
                        w_h = agent.w_h if hasattr(agent, 'w_h') else 1.0
                        w_v = agent.w_v if hasattr(agent, 'w_v') else 1.0
                        score = w_h * info_gain + w_v * event_value
                        if score > best_score:
                            best_score = score
                            best_fov = fov
                    
                    if best_fov:
                        # Sample observation from PEER'S BELIEF (B_j)
                        obs_dict = {}
                        for cell in best_fov:
                            if peer_belief.has_location(cell[0], cell[1]):
                                prob = peer_belief.get_probability(cell[0], cell[1])
                            else:
                                prob = peer_belief.prior_probability
                            obs_dict[cell] = rng.binomial(1, np.clip(prob, 0.0, 1.0))
                        
                        # Update peer belief with this observation (they saw it)
                        observation_mask = self.points_to_mask(
                            list(obs_dict.keys()),
                            lat_grid, lon_grid,
                            lat_grid.shape[0], lat_grid.shape[1]
                        )
                        peer_belief.update_from_observation(
                            observation_mask, obs_dict, lat_grid, lon_grid
                        )
                        
                        # Store in pending (do NOT fuse into B_i)
                        pending[agent.name][tau] = obs_dict
        
        # Compute reward from simulated beliefs (B4 fix, NUMBA-accelerated)
        # Reward = w_h * entropy_reduction + w_v * expected_detection_value
        # Entropy reduction: H(belief) - H(next_belief) for observed cells
        
        # Convert to dense grids and index arrays for NUMBA
        prob_grid = self._belief_to_dense_grid(belief, lat_grid, lon_grid)
        action_y, action_x = self._action_to_indices(action, lat_grid, lon_grid)
        
        # Prior entropy for observed cells (NUMBA-accelerated)
        entropy_reduction = 0.0
        if action_y.shape[0] > 0:
            entropy_reduction = float(info_gain_for_action_numba(prob_grid, action_y, action_x))
        
        # Expected detection value from next_belief (NUMBA-accelerated)
        next_prob_grid = self._belief_to_dense_grid(next_belief, lat_grid, lon_grid)
        expected_detection_value = 0.0
        if action_y.shape[0] > 0:
            utility_0 = self.event_utility.get(0, 0.0)
            utility_1 = self.event_utility.get(1, 0.0)
            utility_2 = self.event_utility.get(2, 0.0)  # For 3-state kernel mode
            expected_detection_value = float(expected_value_for_action_numba(
                next_prob_grid, action_y, action_x, utility_0, utility_1, utility_2
            ))
        
        reward = self.w_h * entropy_reduction + self.w_v * expected_detection_value
        
        return reward, next_belief, pending
    
    def _build_grid_cache(self, lat_grid: np.ndarray, lon_grid: np.ndarray) -> None:
        """
        Build and cache mapping from (lat, lon) to (y, x) indices.
        
        Called once per grid to avoid rebuilding the mapping on every call.
        
        Args:
            lat_grid: Latitude grid [H, W]
            lon_grid: Longitude grid [H, W]
        """
        H, W = lat_grid.shape
        self._H, self._W = H, W
        self._cell_to_idx = {
            (float(lat_grid[y, x]), float(lon_grid[y, x])): (y, x)
            for y in range(H) for x in range(W)
        }
    
    def _action_to_indices(
        self,
        action: Set[tuple[float, float]],
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert action from (lat, lon) set to (y, x) index arrays for NUMBA.
        Uses cached mapping for O(1) lookups.
        
        Args:
            action: Set of (lat, lon) tuples
            lat_grid: Latitude grid [H, W]
            lon_grid: Longitude grid [H, W]
        
        Returns:
            Tuple of (action_y, action_x) arrays of shape [K]
        """
        if not action:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        
        # Build cache if missing / shape changed
        if (not hasattr(self, "_cell_to_idx") or 
            not hasattr(self, "_H") or not hasattr(self, "_W") or
            self._H != lat_grid.shape[0] or self._W != lat_grid.shape[1]):
            self._build_grid_cache(lat_grid, lon_grid)
        
        # Convert action points to indices using cached mapping
        ys = []
        xs = []
        for lat, lon in action:
            idx = self._cell_to_idx.get((float(lat), float(lon)))
            if idx is not None:
                y, x = idx
                ys.append(y)
                xs.append(x)
        
        if not ys:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        
        return np.array(ys, dtype=np.int32), np.array(xs, dtype=np.int32)
    
    def _belief_to_dense_grid(
        self,
        belief: Belief,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Extract dense probability grid from Belief for NUMBA acceleration.
        Uses cached mapping for O(1) lookups.
        
        Creates a [H, W] array where prob_grid[y, x] = P(1) for that cell.
        Uses prior_probability for cells not in belief.valid_locations.
        
        Args:
            belief: Belief object
            lat_grid: Latitude grid [H, W]
            lon_grid: Longitude grid [H, W]
        
        Returns:
            Dense probability grid [H, W]
        """
        # Build cache if missing / shape changed
        if (not hasattr(self, "_cell_to_idx") or 
            not hasattr(self, "_H") or not hasattr(self, "_W") or
            self._H != lat_grid.shape[0] or self._W != lat_grid.shape[1]):
            self._build_grid_cache(lat_grid, lon_grid)
        
        # Use appropriate prior based on num_states
        prior = belief.prior_probability if belief.num_states == 2 else (1.0 / belief.num_states)
        prob_grid = np.full((self._H, self._W), prior, dtype=np.float32)
        
        # Fill in probabilities for valid locations using cached mapping
        for lat, lon in belief.valid_locations:
            idx = self._cell_to_idx.get((float(lat), float(lon)))
            if idx is not None:
                y, x = idx
                # Get probability for state=1 (burning) for backward compatibility
                # For 3-state mode, this returns P(state=1) which is what we want for utilities
                prob_grid[y, x] = belief.get_probability(lat, lon, state=1)
        
        return prob_grid
    
    def _information_gain_from_belief(
        self,
        belief: Belief,
        observation_points: Set[tuple[float, float]],
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
    ) -> float:
        """
        Compute information gain from observing cells using a given belief (NUMBA-accelerated).
        
        Information gain = H_prior - H_post. For perfect observations, H_post = 0.
        
        Args:
            belief: Belief object to compute from
            observation_points: Set of (lat, lon) cells to observe
            lat_grid: Latitude grid
            lon_grid: Longitude grid
        
        Returns:
            Information gain (entropy reduction) in bits
        """
        # Convert to dense grid and index arrays for NUMBA
        prob_grid = self._belief_to_dense_grid(belief, lat_grid, lon_grid)
        action_y, action_x = self._action_to_indices(observation_points, lat_grid, lon_grid)
        
        # Use NUMBA-accelerated kernel
        if action_y.shape[0] > 0:
            return float(info_gain_for_action_numba(prob_grid, action_y, action_x))
        return 0.0
    
    def _expected_event_value_from_belief(
        self,
        belief: Belief,
        observation_points: Set[tuple[float, float]],
        event_utility: Dict[int, float],
        lat_grid: Optional[np.ndarray] = None,
        lon_grid: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute expected event value from a given belief (NUMBA-accelerated).
        
        E[value] = Σ_{cell in observation_points} [P(1) * f(1) + P(0) * f(0)]
        
        Args:
            belief: Belief object to compute from
            observation_points: Set of (lat, lon) cells
            event_utility: Dictionary mapping state to utility
            lat_grid: Latitude grid (required for NUMBA path)
            lon_grid: Longitude grid (required for NUMBA path)
        
        Returns:
            Expected event value
        """
        utility_0 = event_utility.get(0, 0.0)
        utility_1 = event_utility.get(1, 0.0)
        utility_2 = event_utility.get(2, 0.0)  # For 3-state kernel mode
        
        # Use NUMBA-accelerated path if grids provided
        if lat_grid is not None and lon_grid is not None:
            prob_grid = self._belief_to_dense_grid(belief, lat_grid, lon_grid)
            action_y, action_x = self._action_to_indices(observation_points, lat_grid, lon_grid)
            if action_y.shape[0] > 0:
                return float(expected_value_for_action_numba(prob_grid, action_y, action_x, utility_0, utility_1, utility_2))
            return 0.0
        
        # Fallback to Python loop if grids not provided (for backward compatibility)
        expected_value = 0.0
        is_3_state = utility_2 != 0.0
        for lat, lon in observation_points:
            if belief.has_location(lat, lon):
                prob = belief.get_probability(lat, lon)
            else:
                prob = belief.prior_probability
            # E[value] = P(0) * f(0) + P(1) * f(1) [+ P(2) * f(2) for 3-state]
            if is_3_state:
                # Approximate 3-state probabilities (same logic as NUMBA function)
                if prob < 0.1:
                    p0, p1, p2 = 1.0 - prob, prob, 0.0
                elif prob > 0.9:
                    p0, p1, p2 = 0.0, prob, 1.0 - prob
                else:
                    p0, p1, p2 = 1.0 - prob, prob, 0.0
                cell_value = p0 * utility_0 + p1 * utility_1 + p2 * utility_2
            else:
                cell_value = (1.0 - prob) * utility_0 + prob * utility_1
            expected_value += cell_value
        
        return float(expected_value)
    
    def _fuse_pending_if_contact(
        self,
        t: int,
        belief: Belief,
        sigma: Dict[str, int],
        pending: Dict[str, Dict[int, Dict[tuple[float, float], int]]],
        all_agents: List[Any],
        ground_stations: Optional[List[GroundStation]],
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        min_sats: int = 2,
    ) -> None:
        """
        Fuse pending messages at time t if a contact event happens.
        
        This ensures consistent timing: belief points are stored post-contact,
        matching what act() sees after communication handling.
        
        Args:
            t: Time step to check for contact
            belief: Belief to update (modified in-place)
            sigma: Sync map to update (modified in-place)
            pending: Pending messages buffer (modified in-place)
            all_agents: List of all agents
            ground_stations: List of ground stations
            lat_grid: Latitude grid
            lon_grid: Longitude grid
            min_sats: Minimum satellites required for contact
        """
        if not ground_stations or not all_agents:
            return
        
        for gs in ground_stations:
            if not self._station_contact_happens(t, gs, all_agents, min_satellites_per_station=min_sats):
                continue
            
            # Fuse tau <= (t-1), keep the rest
            # Contact at time t delivers observations up to t-1 (matches act() timing)
            cutoff = t - 1
            for agent_name in list(pending.keys()):
                by_tau = pending[agent_name]
                for tau in list(by_tau.keys()):
                    if tau <= cutoff:
                        obs_dict = by_tau[tau]
                        observation_mask = self.points_to_mask(
                            list(obs_dict.keys()),
                            lat_grid, lon_grid, lat_grid.shape[0], lat_grid.shape[1]
                        )
                        belief.update_from_observation(observation_mask, obs_dict, lat_grid, lon_grid)
                        sigma[agent_name] = max(sigma.get(agent_name, -1), tau)
                        del by_tau[tau]
                if len(by_tau) == 0:
                    del pending[agent_name]
    
    def _sigma_signature(self, sigma: Dict[str, int], all_agent_names: List[str]) -> tuple:
        """Create hashable signature from sigma map."""
        return tuple(sigma.get(name, -1) for name in sorted(all_agent_names))
    
    def _pending_signature(self, pending: Dict[str, Dict[int, Dict[tuple[float, float], int]]], 
                           all_agent_names: List[str]) -> tuple:
        """
        Create hashable signature from pending messages.
        
        Returns tuple of (latest_tau, total_count) per agent.
        """
        sig = []
        for name in sorted(all_agent_names):
            by_tau = pending.get(name, {})
            if not by_tau:
                sig.append((-1, 0))
            else:
                latest_tau = max(by_tau.keys())
                count = 0
                for tau, obs_dict in by_tau.items():
                    count += len(obs_dict)
                sig.append((latest_tau, count))
        return tuple(sig)
    
    def _info_state_key(self, belief: Belief, sigma: Dict[str, int],
                       pending: Dict[str, Dict[int, Dict[tuple[float, float], int]]],
                       all_agent_names: List[str]) -> tuple:
        """Create info-state key combining belief, sigma, and pending."""
        return (self._belief_key(belief),
                self._sigma_signature(sigma, all_agent_names),
                self._pending_signature(pending, all_agent_names))
    
    def _belief_key(self, belief: Belief) -> int:
        """
        Create a hashable key for a belief (for value function lookup).
        
        Uses a fixed-size hash based on downsampled probabilities for performance (C2 fix).
        """
        # Use a hash based on downsampled grid vector
        locations = sorted(belief.valid_locations)
        if not locations:
            return hash(0)
        
        # Sample up to 50 locations (or all if fewer) and round probabilities
        sample_size = min(50, len(locations))
        step = max(1, len(locations) // sample_size)
        sampled_locations = locations[::step][:sample_size]
        
        probs = [belief.get_probability(loc[0], loc[1]) for loc in sampled_locations]
        # Round to 3 decimal places for matching
        probs_rounded = tuple(round(p, 3) for p in probs)
        
        # Also include entropy as a summary statistic
        entropy_val = round(belief.entropy(), 3)
        
        return hash((tuple(sampled_locations), probs_rounded, entropy_val))
    
    def _action_key(self, action: Set[tuple[float, float]]) -> tuple:
        """Convert an action (set) to a hashable key."""
        # Convert set to sorted tuple of tuples for hashing
        return tuple(sorted(action))
    
    def _action_from_key(self, action_key: tuple) -> Set[tuple[float, float]]:
        """Convert an action key back to a set."""
        return set(action_key)
    
    def _match_state(
        self,
        target_belief: Belief,
        target_sigma: Dict[str, int],
        target_pending: Dict[str, Dict[int, Dict[tuple[float, float], int]]],
        belief_points: List[Tuple[int, Dict, Belief, Dict]],
        t: int,
        all_agent_names: List[str],
    ) -> Optional[tuple]:
        """
        Match an info-state (belief + sigma + pending) to the nearest belief point.
        
        Args:
            target_belief: Belief to match
            target_sigma: Sync map to match
            target_pending: Pending messages to match
            belief_points: List of (time, sigma, belief, pending) tuples
            t: Current time step
            all_agent_names: List of all agent names
        
        Returns:
            Info-state key of matched belief point, or None
        """
        # Filter candidates at the same time and same sigma/pending signatures
        tgt_sig = self._sigma_signature(target_sigma, all_agent_names)
        tgt_psig = self._pending_signature(target_pending, all_agent_names)
        
        candidates = []
        for t_b, sigma_b, b_b, pending_b in belief_points:
            if t_b != t:
                continue
            if self._sigma_signature(sigma_b, all_agent_names) != tgt_sig:
                continue
            if self._pending_signature(pending_b, all_agent_names) != tgt_psig:
                continue
            skey = self._info_state_key(b_b, sigma_b, pending_b, all_agent_names)
            candidates.append((skey, b_b))
        
        if not candidates:
            # Fallback: match only by time (still returns a valid key)
            for t_b, sigma_b, b_b, pending_b in belief_points:
                if t_b == t:
                    return self._info_state_key(b_b, sigma_b, pending_b, all_agent_names)
            return None
        
        # Choose nearest by L1 over common locations
        target_locs = set(target_belief.valid_locations)
        best_key = candidates[0][0]
        best_dist = 1e30
        
        for skey, b in candidates:
            common = target_locs.intersection(b.valid_locations)
            if not common:
                continue
            dist = 0.0
            for lat, lon in common:
                dist += abs(target_belief.get_probability(lat, lon) - b.get_probability(lat, lon))
            if dist < best_dist:
                best_dist = dist
                best_key = skey
        
        return best_key
    
    def plan_dsb_abba(
        self,
        env: "Environment",  # type: ignore
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        current_time: int,
        all_agents: List[Any],
        ground_stations: Optional[List[GroundStation]] = None,
        transition_env: Optional["Environment"] = None,  # Original env for transition kernel (for ReplayEnvironment)
    ) -> List[Set[tuple[float, float]]]:
        """
        Plan using decentralized SB-ABBA algorithm.
        
        Args:
            env: Environment object
            lat_grid: Latitude grid
            lon_grid: Longitude grid
            current_time: Current time step
            all_agents: List of all agents
            ground_stations: List of ground stations
            transition_env: Original environment for transition kernel (for ReplayEnvironment)
        
        Returns:
            Open-loop plan (list of actions)
        """
        # Validate inputs
        if not all_agents or len(all_agents) == 0:
            # Return empty plan if no agents (shouldn't happen, but be defensive)
            print(f"Warning: [{self.name}] plan_dsb_abba called with no agents, returning empty plan")
            return []
        
        # Update sync map
        all_agent_names = [a.name for a in all_agents]
        self._update_sync_map(all_agent_names, current_time)
        
        # Get clean time and sync time
        t_clean = self._get_t_clean()
        t_sync = current_time
        
        if t_clean < 0:
            # No observations yet, use prior
            t_clean = 0
        
        # Ensure t_clean <= t_sync
        if t_clean > t_sync:
            t_clean = t_sync
        
        # Determine horizon
        if ground_stations:
            next_contact = self._next_contact_time(current_time, ground_stations, all_agents=all_agents, max_horizon=self.horizon * 2)
            if next_contact:
                H = min(self.horizon, next_contact - current_time)
            else:
                H = self.horizon
        else:
            H = self.horizon
        
        H = max(1, H)  # At least 1 step
        
        rng = np.random.default_rng()
        
        # Build received lookup structure once (B2 fix)
        from collections import defaultdict
        received_lookup = defaultdict(lambda: defaultdict(dict))
        for obs_record in self._received_obs:
            received_lookup[obs_record.source_agent][obs_record.timestep][obs_record.cell] = obs_record.value
        
        # Phase 1: Build belief point set
        belief_points: List[Tuple[int, Dict, Belief, Dict]] = []
        
        # Use transition_env for belief evolution (original env when using ReplayEnvironment)
        # This ensures we have access to transition kernel attributes (ignition_map, rsp, kernel_learner, etc.)
        transition_env_used = transition_env if transition_env is not None else env
        
        # If no belief points can be generated, return greedy plan
        try:
            for seed in range(self.n_seed):
                # Sample system state (pass received_lookup and ground_stations)
                # Use transition_env_used for belief evolution
                sigma, belief, pending = self._sample_system_state(
                    transition_env_used, lat_grid, lon_grid, t_clean, t_sync,
                    all_agents, rng, received_lookup=received_lookup,
                    ground_stations=ground_stations
                )
                
                # Roll forward H steps with random actions
                current_belief = belief
                current_pending = pending  # Start from seed pending
                for h in range(H):
                    t = t_sync + h
                    
                    # Make belief point at time t be post-contact (matches act())
                    # Fuse pending at time t if contact happens
                    self._fuse_pending_if_contact(
                        t, current_belief, sigma, current_pending,
                        all_agents, ground_stations, lat_grid, lon_grid, min_sats=2
                    )
                    
                    # Store belief point with deep copy (post-contact state)
                    from copy import deepcopy
                    belief_copy = deepcopy(current_belief)
                    sigma_copy = deepcopy(sigma)
                    pending_copy = deepcopy(current_pending)
                    belief_points.append((t, sigma_copy, belief_copy, pending_copy))
                    
                    # Random action
                    for_mask = self.coverage_mask(lat_grid, lon_grid, t)
                    fov_candidates = self._get_all_contiguous_subsets(
                        for_mask, lat_grid, lon_grid,
                        max_ratio=self.max_observation_ratio,
                        rng=rng
                    )
                    
                    if fov_candidates:
                        # Randomly select an action
                        if len(fov_candidates) > 1:
                            action_idx = rng.integers(0, len(fov_candidates))
                            action = fov_candidates[action_idx]
                        else:
                            action = fov_candidates[0]
                        reward, current_belief, current_pending = self._simulate_one_step(
                            transition_env_used, lat_grid, lon_grid, t,
                            current_belief, action, all_agents, rng,
                            sigma=sigma, pending=current_pending,
                            ground_stations=ground_stations,
                            received_lookup=received_lookup
                        )
                    else:
                        # No valid actions, use empty set
                        action = set()
                        reward = 0.0
        except Exception as e:
            # If belief point generation fails, return greedy plan
            print(f"Warning: Belief point generation failed for {self.name}: {e}")
            import traceback
            traceback.print_exc()
            plan = []
            for h in range(H):
                t = t_sync + h
                for_mask = self.coverage_mask(lat_grid, lon_grid, t)
                fov_candidates = self._get_all_contiguous_subsets(
                    for_mask, lat_grid, lon_grid,
                    max_ratio=self.max_observation_ratio,
                    rng=rng
                )
                plan.append(fov_candidates[0] if fov_candidates else set())
            return plan
        
        self._belief_points = belief_points
        
        # If no belief points were generated, return greedy plan
        if not belief_points:
            plan = []
            for h in range(H):
                t = t_sync + h
                for_mask = self.coverage_mask(lat_grid, lon_grid, t)
                fov_candidates = self._get_all_contiguous_subsets(
                    for_mask, lat_grid, lon_grid,
                    max_ratio=self.max_observation_ratio,
                    rng=rng
                )
                plan.append(fov_candidates[0] if fov_candidates else set())
            return plan
        
        # Phase 2: PBVI backups
        # Initialize value function (using info-state keys)
        value_func: Dict[tuple, float] = {}
        q_func: Dict[Tuple[tuple, Any], float] = {}
        
        for t, sigma, belief, pending in belief_points:
            skey = self._info_state_key(belief, sigma, pending, all_agent_names)
            value_func[skey] = 0.0
        
        # Backup sweeps
        max_actions_per_belief = 3  # Limit actions to prevent explosion
        for sweep in range(self.n_sweep):
            for t, sigma, belief, pending in belief_points:
                key = self._info_state_key(belief, sigma, pending, all_agent_names)
                
                # Get feasible actions
                for_mask = self.coverage_mask(lat_grid, lon_grid, t)
                fov_candidates = self._get_all_contiguous_subsets(
                    for_mask, lat_grid, lon_grid,
                    max_ratio=self.max_observation_ratio,
                    rng=rng
                )
                
                if not fov_candidates:
                    continue
                
                # Evaluate each action (limit to prevent explosion)
                best_q = -np.inf
                num_actions_to_eval = min(max_actions_per_belief, len(fov_candidates))
                for action in fov_candidates[:num_actions_to_eval]:
                    # Estimate Q-value with rollouts
                    q_sum = 0.0
                    
                    for rollout in range(self.n_roll):
                        # Start each rollout from belief point's pending and sigma (deepcopy for independence)
                        from copy import deepcopy
                        rollout_pending = deepcopy(pending)
                        rollout_sigma = deepcopy(sigma)
                        
                        reward, next_belief, rollout_pending = self._simulate_one_step(
                            env, lat_grid, lon_grid, t,
                            belief, action, all_agents, rng,
                            sigma=rollout_sigma, pending=rollout_pending,
                            ground_stations=ground_stations,
                            received_lookup=received_lookup
                        )
                        
                        # Fuse pending at t+1 if contact happens (matches Phase 1 timing)
                        self._fuse_pending_if_contact(
                            t + 1, next_belief, rollout_sigma, rollout_pending,
                            all_agents, ground_stations, lat_grid, lon_grid, min_sats=2
                        )
                        
                        # Match using belief + sigma + pending
                        next_key = self._match_state(
                            next_belief, rollout_sigma, rollout_pending,
                            belief_points, t + 1, all_agent_names
                        )
                        
                        next_value = value_func.get(next_key, 0.0) if next_key is not None else 0.0
                        q_sum += reward + self.gamma * next_value
                    
                    q_avg = q_sum / self.n_roll
                    # Convert action to hashable key (use frozenset)
                    action_key = frozenset(action)
                    q_func[(key, action_key)] = q_avg
                    
                    if q_avg > best_q:
                        best_q = q_avg
                
                # Update value function
                value_func[key] = best_q
        
        self._value_function = value_func
        self._q_function = q_func
        
        # Phase 3: Extract open-loop plan (B6 fix: group by sigma signature)
        plan: List[Set[tuple[float, float]]] = []
        
        # Build sigma signature function
        def sigma_signature(sigma: Dict[str, int], all_agent_names: List[str]) -> tuple:
            """Create hashable signature from sigma map."""
            return tuple(sigma.get(name, -1) for name in sorted(all_agent_names))
        
        for h in range(H):
            t = t_sync + h
            
            # Group belief points by (t, sigma_signature) (B6 fix)
            belief_points_by_sigma: Dict[tuple, List[Tuple[int, Belief, frozenset, float]]] = defaultdict(list)
            
            for (key, action_key), q_val in q_func.items():
                # Find matching belief points at time t (key is now info-state key)
                for t_b, sigma_b, b_b, pending_b in belief_points:
                    if t_b == t and self._info_state_key(b_b, sigma_b, pending_b, all_agent_names) == key:
                        sig = sigma_signature(sigma_b, all_agent_names)
                        action = set(action_key)
                        belief_points_by_sigma[sig].append((key, b_b, frozenset(action), q_val))
                        break
            
            if not belief_points_by_sigma:
                # Fallback: use greedy
                for_mask = self.coverage_mask(lat_grid, lon_grid, t)
                fov_candidates = self._get_all_contiguous_subsets(
                    for_mask, lat_grid, lon_grid,
                    max_ratio=self.max_observation_ratio,
                    rng=rng
                )
                if fov_candidates:
                    plan.append(fov_candidates[0])
                else:
                    plan.append(set())
                continue
            
            # Aggregate Q-values by action within each sigma group, then across groups
            action_scores: Dict[frozenset, List[float]] = defaultdict(list)
            for sig, points in belief_points_by_sigma.items():
                # Aggregate within this sigma group
                group_action_scores: Dict[frozenset, List[float]] = defaultdict(list)
                for _, _, action_frozen, q_val in points:
                    group_action_scores[action_frozen].append(q_val)
                
                # Add best action from this group to global scores
                for action_frozen, scores in group_action_scores.items():
                    avg_score = np.mean(scores)
                    action_scores[action_frozen].append(avg_score)
            
            # Choose action with highest average Q-value across all sigma groups
            best_action = None
            best_avg_score = -np.inf
            
            for action_frozen, scores in action_scores.items():
                avg_score = np.mean(scores)
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_action = set(action_frozen)  # Convert back to set
            
            if best_action and len(best_action) > 0:
                plan.append(best_action)
            else:
                # Fallback: use greedy
                for_mask = self.coverage_mask(lat_grid, lon_grid, t)
                fov_candidates = self._get_all_contiguous_subsets(
                    for_mask, lat_grid, lon_grid,
                    max_ratio=self.max_observation_ratio,
                    rng=rng
                )
                if fov_candidates:
                    plan.append(fov_candidates[0])
                else:
                    plan.append(set())  # Empty action if no candidates
        
        # Ensure plan has exactly H actions
        while len(plan) < H:
            # Fill remaining with greedy fallback
            t = t_sync + len(plan)
            for_mask = self.coverage_mask(lat_grid, lon_grid, t)
            fov_candidates = self._get_all_contiguous_subsets(
                for_mask, lat_grid, lon_grid,
                max_ratio=self.max_observation_ratio,
                rng=rng
            )
            if fov_candidates:
                plan.append(fov_candidates[0])
            else:
                plan.append(set())
        
        # Trim to H if somehow we got more
        plan = plan[:H]
        
        return plan
    
    def act(
        self,
        step: int,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        env: "Environment",  # type: ignore
        horizon: Optional[int] = None,
        ground_stations: Optional[List[GroundStation]] = None,
        transition_env: Optional["Environment"] = None,  # Original env for transition kernel (for ReplayEnvironment)
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Select action using d-SB-ABBA planning.
        
        Args:
            step: Current time step
            lat_grid: Latitude grid
            lon_grid: Longitude grid
            env: Environment object
            horizon: Planning horizon (optional)
            ground_stations: List of ground stations
            transition_env: Original environment for transition kernel (for ReplayEnvironment)
            **kwargs: Additional arguments
        
        Returns:
            Action dictionary
        """
        # Get all agents from kwargs (needed for communication handling and planning)
        all_agents = kwargs.get('all_agents', [])
        
        # Handle communication first (from parent class)
        # Use same contact model as planning (>= min_satellites_per_station)
        if ground_stations:
            for gs in ground_stations:
                if all_agents and self._station_contact_happens(step, gs, all_agents, min_satellites_per_station=2):
                    # FIX 1: Cache last_processed BEFORE downloading (since download updates it)
                    last_processed_before = dict(self._last_processed_timestep)
                    
                    # Download observations (this updates _last_processed_timestep)
                    new_obs = self._download_new_observations_from_ground_station(gs)
                    
                    # FIX 1: Build _received_obs from NEW observations only
                    # Use gs.observations to preserve agent_id information (new_obs loses it)
                    # Dedup by (agent_id, timestep, cell) to avoid duplicates
                    existing_keys = {(obs.source_agent, obs.timestep, obs.cell) for obs in self._received_obs}
                    
                    for (agent_id, t), obs_dict in gs.observations.items():
                        if agent_id != self.name:  # Skip own observations
                            # Only add if this is NEW (wasn't processed before download)
                            last_before_download = last_processed_before.get(agent_id, -1)
                            if t > last_before_download:
                                for cell, value in obs_dict.items():
                                    key = (agent_id, t, cell)
                                    if key not in existing_keys:
                                        self._received_obs.append(
                                            ObservationRecord(t, agent_id, cell, value)
                                        )
                                        existing_keys.add(key)
                    
                    # FIX 2: Update sync map BEFORE purging (so t_clean is correct)
                    if all_agents and len(all_agents) > 0:
                        all_agent_names = [a.name for a in all_agents]
                        self._update_sync_map(all_agent_names, step)
                    
                    # FIX 2: Purge old observations AFTER updating sync map
                    t_clean = self._get_t_clean()
                    if t_clean >= 0:
                        # Keep observations from t_clean onwards (plus some buffer)
                        self._received_obs = [
                            obs for obs in self._received_obs
                            if obs.timestep >= max(0, t_clean - 1)
                        ]
                    
                    # Upload own observations
                    if step in self._own_observation_history:
                        gs.upload_observations(
                            self.name, step, self._own_observation_history[step]
                        )
                    
                    # Trigger replan
                    self._need_replan = True
        
        # FIX 2: Ensure sigma is updated before planning (for accurate t_clean)
        # Update sync map if we have all_agents (needed for planning, even without contact)
        if all_agents and len(all_agents) > 0:
            all_agent_names = [a.name for a in all_agents]
            self._update_sync_map(all_agent_names, step)
        
        # Replan if needed
        if self._need_replan or self._plan_idx >= len(self._current_plan):
            if all_agents and len(all_agents) > 0:
                try:
                    # Print progress for first few planning steps
                    if not hasattr(self, '_planning_step_count'):
                        self._planning_step_count = 0
                    self._planning_step_count += 1
                    
                    if self._planning_step_count <= 3:
                        print(f"[{self.name}] Planning at step {step} (n_seed={self.n_seed}, n_roll={self.n_roll}, n_sweep={self.n_sweep})...", end='', flush=True)
                    
                    # Use transition_env for planning (original env when using ReplayEnvironment)
                    transition_env_used = transition_env if transition_env is not None else env
                    self._current_plan = self.plan_dsb_abba(
                        env, lat_grid, lon_grid, step, all_agents, ground_stations,
                        transition_env=transition_env_used
                    )
                    
                    if self._planning_step_count <= 3:
                        print(f" done (plan length: {len(self._current_plan)})")
                    
                    if len(self._current_plan) == 0:
                        print(f"\nWarning: [{self.name}] DSB-ABBA returned empty plan at step {step}, using greedy fallback")
                    
                    self._plan_idx = 0
                    self._plan_start_time = step
                    self._need_replan = False
                except Exception as e:
                    # If planning fails, use greedy fallback
                    print(f"\nWarning: DSB-ABBA planning failed for {self.name} at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    self._current_plan = []
                    self._plan_idx = 0
            else:
                # Fallback if all_agents not available
                if not all_agents:
                    print(f"\nWarning: [{self.name}] all_agents is None or empty at step {step}, using greedy fallback")
                self._current_plan = []
                self._plan_idx = 0
        
        # Execute next action from plan
        if (self._plan_idx < len(self._current_plan) and 
            len(self._current_plan) > 0 and
            isinstance(self._current_plan[self._plan_idx], set)):
            action = self._current_plan[self._plan_idx]
            self._plan_idx += 1
        else:
            # Fallback: greedy
            for_mask = self.coverage_mask(lat_grid, lon_grid, step)
            fov_candidates = self._get_all_contiguous_subsets(
                for_mask, lat_grid, lon_grid,
                max_ratio=self.max_observation_ratio
            )
            action = fov_candidates[0] if fov_candidates else set()
        
        # Convert action to observation points
        observation_points = action
        
        # FIX 5: Compute expected reward using same formula as planning
        # Use belief-based reward (consistent with planning rewards)
        observation_mask = self.points_to_mask(
            list(observation_points), lat_grid, lon_grid,
            lat_grid.shape[0], lat_grid.shape[1]
        )
        # Prior entropy for observed cells
        prior_entropy = self.belief.entropy(observation_mask, lat_grid, lon_grid)
        entropy_reduction = prior_entropy  # After perfect observation, entropy is 0
        
        # Expected detection value from current belief (NUMBA-accelerated)
        expected_detection_value = self._expected_event_value_from_belief(
            self.belief, observation_points, self.event_utility,
            lat_grid=lat_grid, lon_grid=lon_grid
        )
        
        expected_reward = self.w_h * entropy_reduction + self.w_v * expected_detection_value
        
        # Record own observation (will be updated after actual observation)
        # This is a placeholder - actual observation will be recorded in simulation step
        # But we need to mark that we took this action
        
        return {
            "observation_points": observation_points,
            "expected_reward": expected_reward,
        }


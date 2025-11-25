"""Belief representation for graph-based agents over the environment state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Callable
import itertools

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

if TYPE_CHECKING:
    from environment_simulation.graph_environment import GraphEnvironment


# Numba-optimized helper: generate neighbor state combination index
@njit(cache=True)
def _get_combination_from_index(idx: int, num_neighbors: int, num_states: int) -> np.ndarray:
    """Convert linear index to neighbor state combination (base num_states representation)."""
    combination = np.zeros(num_neighbors, dtype=np.int64)
    for i in range(num_neighbors):
        combination[i] = idx % num_states
        idx = idx // num_states
    return combination

# Numba-optimized RSP transition probability (same as in graph_environment but accessible here)
@njit(cache=True)
def _rsp_transition_prob_numba(
    next_state: int,
    current_state: int,
    active_neighbors: int,
    num_neighbors: int,
    lam: float,
    beta0: float,
    alpha: float,
    delta: float,
) -> float:
    """Numba-optimized RSP transition probability computation."""
    norm_active = active_neighbors / num_neighbors if num_neighbors > 0 else 0.0
    contagion = 1.0 - np.exp(-alpha * norm_active)
    
    if current_state == 0:  # NO_EVENT → {NO_EVENT, EVENT}
        p_event = 1.0 - np.exp(-(beta0 + lam + contagion))
        return p_event if next_state == 1 else (1.0 - p_event)
    else:  # current_state == 1: EVENT → {NO_EVENT, EVENT}
        mu = 1.0 - delta
        return delta if next_state == 1 else mu

# Numba-optimized belief evolution for a single node - ENTIRE computation in Numba
@njit(cache=True, parallel=False)
def _evolve_node_belief_numba_full(
    current_belief: np.ndarray,  # Current belief for all states of this node [num_states]
    neighbor_beliefs: np.ndarray,  # Beliefs for neighbors [num_neighbors, num_states]
    num_states: int,
    num_neighbors: int,
    lam: float,
    beta0: float,
    alpha: float,
    delta: float,
) -> np.ndarray:
    """
    Fully Numba-optimized single node belief evolution.
    Computes everything inside Numba, including generating combinations and computing transitions.
    
    Computes: b_{t+1}(j, x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * b_t(j, x_j) * Π_{l∈N(j)} b_t(l, x_l)
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
                transition_prob = _rsp_transition_prob_numba(
                    next_state, current_state, active_neighbors, num_neighbors,
                    lam, beta0, alpha, delta
                )
                
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


# Numba-optimized entropy computation for a single node
@njit(cache=True)
def _entropy_numba(prob: float) -> float:
    """Compute Shannon entropy for a binary state: H = -p*log2(p) - (1-p)*log2(1-p)"""
    if prob <= 1e-10 or prob >= 1.0 - 1e-10:
        return 0.0
    return -prob * np.log2(prob) - (1.0 - prob) * np.log2(1.0 - prob)

# Numba-optimized expected event value computation
@njit(cache=True)
def _expected_event_value_numba(prob: float, utility_0: float, utility_1: float) -> float:
    """Compute expected event value: E[value] = P(0) * f(0) + P(1) * f(1)"""
    return (1.0 - prob) * utility_0 + prob * utility_1

# Numba-optimized reward computation
@njit(cache=True)
def _compute_reward_numba(
    node_prob: float,
    w_h: float,
    w_v: float,
    utility_0: float,
    utility_1: float,
) -> float:
    """Compute reward: R = w_h * H(b_k) + w_v * E[value]"""
    info_gain = _entropy_numba(node_prob)  # Information gain = entropy (perfect observation)
    event_value = _expected_event_value_numba(node_prob, utility_0, utility_1)
    return w_h * info_gain + w_v * event_value

# Numba-optimized belief evolution for ALL nodes at once
@njit(cache=True, parallel=False)
def _evolve_all_nodes_belief_numba(
    belief_array: np.ndarray,  # [num_nodes, num_states] - will be modified in place
    adjacency_list_flat: np.ndarray,  # [num_nodes, max_neighbors] flattened (use -1 for padding)
    adjacency_counts: np.ndarray,  # [num_nodes] number of neighbors per node
    rsp_params_per_node: np.ndarray,  # [num_nodes, 4] (lam, beta0, alpha, delta) per node
    observed_nodes: np.ndarray,  # [num_observed] array of observed node indices (-1 for padding)
    num_observed: int,  # Actual number of observed nodes
    num_nodes: int,
    num_states: int,
    max_neighbors: int,
) -> None:
    """
    Evolve belief for all unobserved nodes in one pass (fully Numba-optimized).
    Modifies belief_array in place.
    """
    # Create a mask for observed nodes (for fast lookup)
    observed_mask = np.zeros(num_nodes, dtype=np.int8)
    for i in range(num_observed):
        if observed_nodes[i] >= 0:
            observed_mask[observed_nodes[i]] = 1
    
    # Evolve all nodes
    for node in range(num_nodes):
        # Skip observed nodes
        if observed_mask[node] == 1:
            continue
        
        num_neighbors = int(adjacency_counts[node])
        if num_neighbors == 0:
            continue
        
        # Get neighbor beliefs
        neighbor_beliefs = np.zeros((num_neighbors, num_states), dtype=np.float64)
        for i in range(num_neighbors):
            neighbor = int(adjacency_list_flat[node * max_neighbors + i])
            if neighbor >= 0:  # Valid neighbor (not padding)
                neighbor_beliefs[i, :] = belief_array[neighbor, :]
        
        # Get RSP parameters for this node
        lam = rsp_params_per_node[node, 0]
        beta0 = rsp_params_per_node[node, 1]
        alpha = rsp_params_per_node[node, 2]
        delta = rsp_params_per_node[node, 3]
        
        # Evolve belief for this node
        current_belief = belief_array[node, :].copy()
        new_belief = _evolve_node_belief_numba_full(
            current_belief,
            neighbor_beliefs,
            num_states,
            num_neighbors,
            lam, beta0, alpha, delta,
        )
        belief_array[node, :] = new_belief

# Numba-optimized single rollout step
@njit(cache=True)
def _single_rollout_step_numba(
    belief_array: np.ndarray,  # [num_nodes, num_states] belief array
    adjacency_list_flat: np.ndarray,  # Flattened adjacency list [num_nodes * max_neighbors]
    adjacency_counts: np.ndarray,  # [num_nodes] number of neighbors per node
    node: int,
    num_nodes: int,
    num_states: int,
    max_neighbors: int,
    rsp_params_per_node: np.ndarray,  # [num_nodes, 4] (lam, beta0, alpha, delta) per node
    w_h: float,
    w_v: float,
    utility_0: float,
    utility_1: float,
    observation_random: float,  # Pre-generated random number [0, 1) for observation sampling
) -> tuple[float, np.ndarray]:
    """
    Single rollout step: move to node, observe, evolve belief, compute reward.
    All loops inside Numba for maximum speed.
    
    Returns:
        (reward, updated_belief_array)
    """
    # Compute reward BEFORE observation (uses prior entropy)
    node_prob = belief_array[node, 1]  # P(E=1)
    reward = _compute_reward_numba(node_prob, w_h, w_v, utility_0, utility_1)
    
    # Sample observation from belief using pre-generated random number
    observation = 1 if observation_random < node_prob else 0
    
    # Update belief: delta function for observed node
    belief_array = belief_array.copy()  # Don't modify in place
    belief_array[node, observation] = 1.0
    belief_array[node, 1 - observation] = 0.0
    
    # Evolve belief for all unobserved nodes (using optimized function)
    observed_nodes = np.array([node], dtype=np.int32)
    _evolve_all_nodes_belief_numba(
        belief_array,
        adjacency_list_flat,
        adjacency_counts,
        rsp_params_per_node,
        observed_nodes,
        1,  # num_observed
        num_nodes,
        num_states,
        max_neighbors,
    )
    
    return reward, belief_array

# Numba-optimized full rollout for a single action (multiple steps)
# Note: Uses pre-generated random numbers passed in to avoid RNG issues in Numba
@njit(cache=True)
def _rollout_numba(
    belief_array: np.ndarray,  # [num_nodes, num_states] initial belief
    adjacency_list_flat: np.ndarray,  # [num_nodes * max_neighbors] flattened
    adjacency_counts: np.ndarray,  # [num_nodes] number of neighbors per node
    target_node: int,  # Initial target node (first step)
    horizon: int,
    num_nodes: int,
    num_states: int,
    max_neighbors: int,
    rsp_params_per_node: np.ndarray,  # [num_nodes, 4] (lam, beta0, alpha, delta)
    w_h: float,
    w_v: float,
    utility_0: float,
    utility_1: float,
    discount_factor: float,
    random_samples: np.ndarray,  # Pre-generated random numbers for sampling [horizon * 2] (observation, action choice)
) -> float:
    """
    Full rollout computation in Numba - all loops optimized.
    Returns cumulative discounted reward over horizon steps.
    
    Args:
        random_samples: Pre-generated random numbers [horizon * 2] where:
                       - random_samples[h*2] = random for observation sampling at step h
                       - random_samples[h*2+1] = random for action choice at step h
    """
    # Copy belief to avoid modifying original
    belief = belief_array.copy()
    current_node = target_node
    total_reward = 0.0
    random_idx = 0
    
    # Step 0: immediate reward and observation
    observation_random = random_samples[0]
    immediate_reward, belief = _single_rollout_step_numba(
        belief,
        adjacency_list_flat,
        adjacency_counts,
        current_node,
        num_nodes,
        num_states,
        max_neighbors,
        rsp_params_per_node,
        w_h,
        w_v,
        utility_0,
        utility_1,
        observation_random,
    )
    total_reward += immediate_reward
    random_idx = 1  # Start from index 1 (0 was used for observation)
    
    # Future steps (1 to horizon-1)
    for h in range(1, horizon):
        # Get available nodes (neighbors + current)
        num_current_neighbors = int(adjacency_counts[current_node])
        future_available = np.zeros(max_neighbors + 1, dtype=np.int32)  # +1 for current node
        future_available[0] = current_node  # Can stay
        count = 1
        
        for i in range(num_current_neighbors):
            neighbor = int(adjacency_list_flat[current_node * max_neighbors + i])
            if neighbor >= 0:
                future_available[count] = neighbor
                count += 1
        
        if count == 0:
            # No neighbors, must stay - still evolve belief
            observed_nodes = np.empty(0, dtype=np.int32)
            _evolve_all_nodes_belief_numba(
                belief,
                adjacency_list_flat,
                adjacency_counts,
                rsp_params_per_node,
                observed_nodes,
                0,
                num_nodes,
                num_states,
                max_neighbors,
            )
            continue
        
        # Choose randomly from available nodes using pre-generated random number
        action_random = random_samples[random_idx]
        random_idx += 1
        chosen_idx = int(action_random * count)
        chosen_idx = max(0, min(chosen_idx, count - 1))  # Clamp to valid range
        future_target_node = int(future_available[chosen_idx])
        current_node = future_target_node
        
        # Get observation random number
        observation_random = random_samples[random_idx]
        random_idx += 1
        
        # Compute reward and update belief
        future_reward, belief = _single_rollout_step_numba(
            belief,
            adjacency_list_flat,
            adjacency_counts,
            future_target_node,
            num_nodes,
            num_states,
            max_neighbors,
            rsp_params_per_node,
            w_h,
            w_v,
            utility_0,
            utility_1,
            observation_random,
        )
        total_reward += (discount_factor ** h) * future_reward
    
    return total_reward

# Numba-optimized multiple rollouts for a single action
@njit(cache=True, parallel=True)
def _run_rollouts_numba(
    belief_array: np.ndarray,  # [num_nodes, num_states] initial belief
    adjacency_list_flat: np.ndarray,
    adjacency_counts: np.ndarray,
    target_node: int,
    horizon: int,
    num_rollouts: int,
    num_nodes: int,
    num_states: int,
    max_neighbors: int,
    rsp_params_per_node: np.ndarray,
    w_h: float,
    w_v: float,
    utility_0: float,
    utility_1: float,
    discount_factor: float,
    all_random_samples: np.ndarray,  # [num_rollouts, horizon * 2] pre-generated random numbers
) -> float:
    """
    Run multiple rollouts in parallel using Numba.
    Returns average reward across all rollouts.
    """
    total = 0.0
    for rollout_idx in prange(num_rollouts):
        rollout_reward = _rollout_numba(
            belief_array,
            adjacency_list_flat,
            adjacency_counts,
            target_node,
            horizon,
            num_nodes,
            num_states,
            max_neighbors,
            rsp_params_per_node,
            w_h,
            w_v,
            utility_0,
            utility_1,
            discount_factor,
            all_random_samples[rollout_idx, :],
        )
        total += rollout_reward
    return total / num_rollouts if num_rollouts > 0 else 0.0


@dataclass
class GraphBelief:
    """
    Belief over the graph environment state.
    
    Represents the agent's belief about the probability of events at each node.
    
    The belief is stored as a dictionary mapping node_id -> P(E_j(t) = 1), where:
    - Keys: node indices (integers)
    - Values: P(E_j(t) = 1) = probability of event at that node
    - For binary states {0, 1}: P(E_j(t) = 0) = 1 - P(E_j(t) = 1)
    
    The belief is updated according to:
    - If node j is observed: b_{t+1}^e(j, x') = δ[x' = o_j] (delta function)
    - If node j is not observed: b_{t+1}^e(j, x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * 
      b_t^e(j, x_j) * Π_{l∈N(j)} b_t^e(l, x_l)
    """

    num_nodes: int
    # Dictionary mapping (node_id, state) -> P(node = state) for M states
    # For binary states (M=2), we also support backward-compatible access via node_id -> P(E_j = 1)
    # Only contains nodes that have valid beliefs (observed or reachable)
    probabilities: dict[tuple[int, int], float] = field(init=False, default_factory=dict)
    # Binary compatibility: node_id -> P(E_j = 1) for backward compatibility
    _binary_probs: dict[int, float] = field(init=False, default_factory=dict)
    # Set of node IDs that have valid beliefs
    valid_nodes: set[int] = field(init=False, default_factory=set)
    # Prior probability for unobserved nodes (for binary states)
    prior_probability: float = 0.5
    # Number of states (M)
    num_states: int = 2

    def __post_init__(self):
        """Initialize belief dictionary for ALL nodes in the environment."""
        self.probabilities = {}
        self._binary_probs = {}
        self.valid_nodes = set()
        
        # Initialize belief for ALL nodes in the environment with uniform prior
        # For binary states: P(E=1) = prior_probability for all nodes
        # For M states: P(node=s) = 1/M for all nodes and all states
        for node in range(self.num_nodes):
            # For binary backward compatibility
            self._binary_probs[node] = self.prior_probability
            # For M states: uniform prior over all states
            for state in range(self.num_states):
                uniform_prob = 1.0 / self.num_states
                self.probabilities[(node, state)] = uniform_prob
            # Also set binary representation
            self.probabilities[(node, 1)] = self.prior_probability
            self.probabilities[(node, 0)] = 1.0 - self.prior_probability
            self.valid_nodes.add(node)
    
    def get_probability(self, node: int, state: Optional[int] = None) -> float:
        """
        Get probability P(node = state) for a specific node and state.
        
        For backward compatibility, if state is None, returns P(E_j = 1) for binary states.
        
        Args:
            node: Node index (must be in range 0 to num_nodes-1)
            state: State index (0 to M-1). If None, returns P(E_j = 1) for binary (backward compat)
            
        Returns:
            Probability P(node = state). Returns uniform prior if node not explicitly initialized.
        """
        # Validate node index
        if node < 0 or node >= self.num_nodes:
            raise ValueError(f"Node index {node} out of range [0, {self.num_nodes-1}]")
        
        if state is None:
            # Backward compatibility: return P(E_j = 1)
            return self._binary_probs.get(node, self.prior_probability)
        else:
            # M-state: return P(node = state)
            # If node not in probabilities dict, return uniform prior
            return self.probabilities.get((node, state), 1.0 / self.num_states)
    
    def set_probability(self, node: int, prob: float, state: Optional[int] = None) -> None:
        """
        Set probability P(node = state) for a specific node and state.
        
        For backward compatibility, if state is None, sets P(E_j = 1) for binary states.
        
        Args:
            node: Node index
            prob: Probability P(node = state), must be in [0, 1]
            state: State index (0 to M-1). If None, sets P(E_j = 1) for binary (backward compat)
        """
        prob = float(np.clip(prob, 0.0, 1.0))
        if state is None:
            # Backward compatibility: set P(E_j = 1)
            self._binary_probs[node] = prob
            # Also update M-state representation for binary
            self.probabilities[(node, 1)] = prob
            self.probabilities[(node, 0)] = 1.0 - prob
        else:
            # M-state: set P(node = state)
            self.probabilities[(node, state)] = prob
            # Normalize to ensure probabilities sum to 1 (optional, for safety)
            # For binary backward compat, update _binary_probs
            if state == 1 and self.num_states == 2:
                self._binary_probs[node] = prob
        self.valid_nodes.add(node)
    
    def has_node(self, node: int) -> bool:
        """Check if node is in the belief. Since belief is over all nodes, always returns True for valid indices."""
        return 0 <= node < self.num_nodes

    def update_from_observation(
        self, 
        observed_nodes: set[int],
        observations: dict[int, int],
    ) -> None:
        """
        Update belief with observations (perfect observation model).
        
        For observed nodes, set belief to delta function: b_{t+1}(j, x') = δ[x' = o_j]
        
        Args:
            observed_nodes: Set of node indices that were observed
            observations: Dictionary mapping node_id -> observed state (0 or 1)
        """
        for node in observed_nodes:
            observed_state = observations.get(node, 0)
            # Set to delta function: P(E_j = 1) = observed_state (0 or 1)
            self.set_probability(node, float(observed_state))

    def entropy(self, nodes: Optional[set[int]] = None) -> float:
        """
        Compute Shannon entropy of the belief.
        
        For a binary state, entropy is: H(b) = -p*log2(p) - (1-p)*log2(1-p)
        where p is the probability of event=1.
        
        Args:
            nodes: Optional set of node indices to include.
                  If None, computes entropy over all nodes in the environment.
        
        Returns:
            Total entropy (sum over nodes)
        """
        if nodes is None:
            nodes = set(range(self.num_nodes))
        
        if len(nodes) == 0:
            return 0.0
        
        probs = np.array([self.get_probability(node) for node in nodes])
        
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
        entropy = -probs * np.log2(probs) - (1.0 - probs) * np.log2(1.0 - probs)
        return float(np.sum(entropy))

    def reset(self) -> None:
        """Reset belief to uniform prior for all nodes."""
        self.probabilities.clear()
        self._binary_probs.clear()
        self.valid_nodes.clear()
        
        # Re-initialize all nodes with uniform prior
        for node in range(self.num_nodes):
            self._binary_probs[node] = self.prior_probability
            for state in range(self.num_states):
                uniform_prob = 1.0 / self.num_states
                self.probabilities[(node, state)] = uniform_prob
            # Also set binary representation
            self.probabilities[(node, 1)] = self.prior_probability
            self.probabilities[(node, 0)] = 1.0 - self.prior_probability
            self.valid_nodes.add(node)

    def _poisson_binomial_pmf(self, probs: list[float], k: int) -> float:
        """
        Compute Poisson binomial PMF: P(exactly k successes) for independent Bernoulli trials.
        
        Uses dynamic programming to compute: P(k successes) = sum over all subsets S of size k of
        [Π_{i in S} p_i * Π_{i not in S} (1 - p_i)]
        
        For M states, this generalizes to: P(k neighbors in state s) for any state s.
        
        Args:
            probs: List of success probabilities for each trial (for binary: P(E=1) for each neighbor)
            k: Number of successes
            
        Returns:
            Probability of exactly k successes
        """
        n = len(probs)
        if k < 0 or k > n:
            return 0.0
        if n == 0:
            return 1.0 if k == 0 else 0.0
        
        # Dynamic programming: dp[j] = P(exactly j successes so far)
        dp = np.zeros(n + 1, dtype=float)
        dp[0] = 1.0  # P(0 successes initially) = 1
        
        # Process each trial
        for p in probs:
            # Update backwards to avoid overwriting
            new_dp = np.zeros(n + 1, dtype=float)
            for j in range(n + 1):
                # j successes can come from:
                # - j-1 successes + current success (with prob p)
                # - j successes + current failure (with prob 1-p)
                if j > 0:
                    new_dp[j] += dp[j - 1] * p  # Add success
                new_dp[j] += dp[j] * (1.0 - p)  # Add failure
            dp = new_dp
        
        return float(dp[k])
    
    def evolve_with_transition_kernel(
        self,
        env: "GraphEnvironment",
        observed_nodes: set[int],
        num_states: int = 2,
    ) -> None:
        """
        Evolve belief using the environment's transition kernel (mean field update).
        
        Implements the mean field update formula:
        b_{t+1}(j, x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * b_t(j, x_j) * Π_{l∈N(j)} b_t(l, x_l)
        
        Where:
        - x' is the next state (0 to M-1)
        - x_j is the current state of node j (0 to M-1)
        - x_N(j) is the configuration of neighbor states
        - φ_j(x'; x_j, x_N(j)) is the transition probability
        - b_t(j, x_j) is the current belief P(node j = x_j)
        - b_t(l, x_l) is the current belief P(neighbor l = x_l)
        
        Note: The current transition_probability only returns P(x'=1 | current_state, active_neighbors)
        where active_neighbors is the count of neighbors in state 1. For M states, we count active neighbors
        from the neighbor configuration.
        
        Args:
            env: GraphEnvironment object with transition kernel
            observed_nodes: Set of node indices that were observed (these are not evolved)
            num_states: Number of possible states (M)
        """
        # CRITICAL: Evolve ALL nodes in the environment, not just valid_nodes
        # Belief is defined over the entire environment
        self.num_states = num_states
        
        # Evolve all nodes from 0 to num_nodes-1
        for node in range(self.num_nodes):
            # Skip observed nodes (they are updated separately via update_from_observation)
            if node in observed_nodes:
                continue
            
            neighbors = list(env.get_neighbors(node))
            
            # Fast path: Use Numba-optimized computation if available
            num_neighbors = len(neighbors)
            if num_neighbors == 0:
                # No neighbors - simplified case
                new_probabilities = {}
                for next_state in range(num_states):
                    prob = 0.0
                    for current_state in range(num_states):
                        transition_prob = env.transition_probability_full(
                            node, next_state, current_state, (), num_states
                        )
                        prob_current = self.get_probability(node, current_state)
                        prob += transition_prob * prob_current
                    new_probabilities[next_state] = prob
            elif NUMBA_AVAILABLE and num_states == 2:
                # FAST PATH: Use fully Numba-optimized computation
                # Handle both GraphEnvironment and GraphReplayEnvironment
                try:
                    # Check if it's a replay environment first
                    if hasattr(env, 'original_env') and env.original_env is not None:
                        # GraphReplayEnvironment - get params from original_env
                        orig_env = env.original_env
                        if hasattr(orig_env, 'mode') and orig_env.mode == "rsp":
                            lam = float(orig_env.ignition_map[node]) if orig_env.ignition_map is not None else orig_env.rsp.lam
                            beta0 = float(orig_env.beta0_map[node]) if orig_env.beta0_map is not None else orig_env.rsp.beta0
                            alpha = float(orig_env.alpha_map[node]) if orig_env.alpha_map is not None else orig_env.rsp.alpha
                            delta = float(orig_env.persistence_map[node]) if orig_env.persistence_map is not None else orig_env.rsp.delta
                        else:
                            # Not RSP mode, fall through to slow path
                            raise AttributeError("Not RSP mode")
                    elif hasattr(env, 'mode') and env.mode == "rsp" and hasattr(env, 'rsp'):
                        # GraphEnvironment - direct access
                        lam = float(env.ignition_map[node]) if env.ignition_map is not None else env.rsp.lam
                        beta0 = float(env.beta0_map[node]) if env.beta0_map is not None else env.rsp.beta0
                        alpha = float(env.alpha_map[node]) if env.alpha_map is not None else env.rsp.alpha
                        delta = float(env.persistence_map[node]) if env.persistence_map is not None else env.rsp.delta
                    else:
                        # Not RSP mode, fall through to slow path
                        raise AttributeError("Not RSP mode")
                    
                    # Successfully got RSP parameters - use fast path
                    # Convert beliefs to numpy arrays (only if Numba available)
                    current_belief_array = np.zeros(num_states, dtype=np.float64)
                    for state in range(num_states):
                        current_belief_array[state] = self.get_probability(node, state)
                    
                    neighbor_beliefs_array = np.zeros((num_neighbors, num_states), dtype=np.float64)
                    for i, neighbor in enumerate(neighbors):
                        for state in range(num_states):
                            neighbor_beliefs_array[i, state] = self.get_probability(neighbor, state)
                    
                    # Call fully Numba-optimized function (generates combinations inside Numba)
                    new_belief_array = _evolve_node_belief_numba_full(
                        current_belief_array,
                        neighbor_beliefs_array,
                        num_states,
                        num_neighbors,
                        lam, beta0, alpha, delta,
                    )
                    new_probabilities = {s: float(new_belief_array[s]) for s in range(num_states)}
                    # Skip to normalization below
                except (AttributeError, KeyError):
                    # Fall through to slow path if we can't get RSP parameters
                    new_probabilities = {}
                    for next_state in range(num_states):
                        prob = 0.0
                        for current_state in range(num_states):
                            for neighbor_states_tuple in itertools.product(range(num_states), repeat=num_neighbors):
                                transition_prob = env.transition_probability_full(
                                    node, next_state, current_state, neighbor_states_tuple, num_states
                                )
                                prob_current = self.get_probability(node, current_state)
                                prob_neighbor_config = 1.0
                                for i, neighbor_state in enumerate(neighbor_states_tuple):
                                    prob_neighbor_config *= self.get_probability(neighbors[i], neighbor_state)
                                prob += transition_prob * prob_current * prob_neighbor_config
                        new_probabilities[next_state] = prob
            else:
                # SLOW PATH: Pure Python fallback (only if Numba not available or not RSP mode)
                new_probabilities = {}
                for next_state in range(num_states):
                    prob = 0.0
                    for current_state in range(num_states):
                        for neighbor_states_tuple in itertools.product(range(num_states), repeat=num_neighbors):
                            transition_prob = env.transition_probability_full(
                                node, next_state, current_state, neighbor_states_tuple, num_states
                            )
                            prob_current = self.get_probability(node, current_state)
                            prob_neighbor_config = 1.0
                            for i, neighbor_state in enumerate(neighbor_states_tuple):
                                prob_neighbor_config *= self.get_probability(neighbors[i], neighbor_state)
                            prob += transition_prob * prob_current * prob_neighbor_config
                    new_probabilities[next_state] = prob
            
            # Normalize probabilities for this node to ensure they sum to 1 (matching Julia: normalize_belief_distributions)
            total = sum(new_probabilities.values())
            if total > 1e-10:
                for next_state, prob in new_probabilities.items():
                    normalized_prob = prob / total
                    self.set_probability(node, normalized_prob, state=next_state)
            else:
                # If total is too small, use uniform distribution
                uniform_prob = 1.0 / num_states
                for next_state in range(num_states):
                    self.set_probability(node, uniform_prob, state=next_state)
    
    def initialize_nodes(self, nodes: set[int]) -> None:
        """
        Initialize belief for a set of nodes with prior probability.
        
        Note: This method is kept for backward compatibility, but since
        belief is now over all nodes, all nodes are already initialized.
        This method ensures nodes have the correct prior if they were modified.
        """
        for node in nodes:
            if 0 <= node < self.num_nodes:
                # Set uniform prior for M states
                for state in range(self.num_states):
                    uniform_prob = 1.0 / self.num_states
                    self.probabilities[(node, state)] = uniform_prob
                # Set binary representation
                self._binary_probs[node] = self.prior_probability
                self.probabilities[(node, 1)] = self.prior_probability
                self.probabilities[(node, 0)] = 1.0 - self.prior_probability
                self.valid_nodes.add(node)
    
    def to_numpy_array(self, num_states: Optional[int] = None) -> np.ndarray:
        """
        Convert belief to numpy array format for Numba optimization.
        
        Args:
            num_states: Number of states (defaults to self.num_states)
            
        Returns:
            Array of shape [num_nodes, num_states] where array[node, state] = P(node=state)
        """
        if num_states is None:
            num_states = self.num_states
        
        belief_array = np.zeros((self.num_nodes, num_states), dtype=np.float64)
        for node in range(self.num_nodes):
            for state in range(num_states):
                belief_array[node, state] = self.get_probability(node, state)
        return belief_array
    
    def from_numpy_array(self, belief_array: np.ndarray, num_states: Optional[int] = None) -> None:
        """
        Update belief from numpy array format (from Numba computation).
        
        Args:
            belief_array: Array of shape [num_nodes, num_states] with probabilities
            num_states: Number of states (defaults to self.num_states)
        """
        if num_states is None:
            num_states = self.num_states
        
        for node in range(min(self.num_nodes, belief_array.shape[0])):
            for state in range(min(num_states, belief_array.shape[1])):
                self.set_probability(node, float(belief_array[node, state]), state=state)


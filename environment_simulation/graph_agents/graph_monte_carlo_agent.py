"""Monte Carlo agent for graph-based planning with individual beliefs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import time
from .graph_agent import GraphAgent
from .graph_belief import GraphBelief, NUMBA_AVAILABLE
from typing import Optional
if NUMBA_AVAILABLE:
    from .graph_belief import _run_rollouts_numba
from environment_simulation.graph_replay_environment import GraphReplayEnvironment


@dataclass
class GraphMonteCarloAgent(GraphAgent):
    """
    Monte Carlo agent that plans using individual beliefs and no inter-agent communication.
    
    Uses Monte Carlo rollouts to estimate expected cumulative reward and selects movement actions
    (moving to neighbor nodes) that maximize this reward.
    """

    # Monte Carlo planning parameters
    num_rollouts: int = 50
    w_h: float = 1.0  # Weight for information gain term
    w_v: float = 1.0  # Weight for event detection value term
    event_utility: dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 1.0})
    discount_factor: float = 0.95
    epsilon: float = 0.0  # Epsilon-greedy exploration: probability of taking random action
    
    def compute_reward(
        self,
        env: "GraphEnvironment",
        target_node: Optional[int] = None,
        w_h: Optional[float] = None,
        w_v: Optional[float] = None,
        event_utility: Optional[dict[int, float]] = None,
    ) -> float:
        """
        Compute reward for moving to and observing a target node.
        
        Implements the reward function:
        R(τ,b,a) = w_h * G(b_k) + w_v * Σ_x b_k(x) f(x)
        
        where:
        - k is the target node to observe
        - G(b_k) = H_prior(b_k) - H_post(b_k) (information gain)
        - H_post(b_k) = 0 for perfect observations
        - f(x) is the normalized utility of observing state x
        
        Args:
            env: GraphEnvironment object
            target_node: Target node to move to and observe. If None, uses current node.
            w_h: Weight for information gain term. If None, uses self.w_h
            w_v: Weight for event detection value term. If None, uses self.w_v
            event_utility: Dictionary mapping state (0 or 1) to utility value.
                          If None, uses self.event_utility
        
        Returns:
            Reward value
        """
        if target_node is None:
            target_node = self.current_node
        
        if w_h is None:
            w_h = self.w_h
        if w_v is None:
            w_v = self.w_v
        if event_utility is None:
            event_utility = self.event_utility
        
        # Compute information gain for observed node
        info_gain = self.information_gain({target_node})
        
        # Compute expected event value for observed node
        event_value = self.expected_event_value({target_node}, event_utility)
        
        # Total reward
        reward = w_h * info_gain + w_v * event_value
        return float(reward)
    
    def _sample_observation_from_belief(
        self, 
        node: int,
        rng: np.random.Generator
    ) -> int:
        """
        Sample a simulated observation from the current belief.
        
        For Monte Carlo rollouts, we simulate what the agent might observe by sampling
        from the belief distribution b_k(x) for node k.
        
        Sample x ~ Bernoulli(b_k(1)), where b_k(1) = P(E_k = 1)
        
        Args:
            node: Node index to sample observation for
            rng: Random number generator
            
        Returns:
            Sampled observation (0 or 1)
        """
        # Get probability from belief (or use prior if not in belief)
        prob = self.belief.get_probability(node)
        
        # Sample observation: x ~ Bernoulli(prob)
        sampled = rng.binomial(1, np.clip(prob, 0.0, 1.0))
        return int(sampled)
    
    def _rollout_trajectory(
        self,
        env: "GraphEnvironment",
        start_node: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> float:
        """
        Perform a single Monte Carlo rollout to estimate cumulative reward.
        
        Simulates forward steps from the current belief state:
        1. Move to a node (or stay)
        2. Observe the node's state (sampled from belief)
        3. Update belief with observation
        4. Compute reward
        5. Evolve belief using transition kernel
        6. Repeat for horizon steps
        
        Args:
            env: GraphEnvironment object (for belief evolution)
            start_node: Starting node for the rollout
            horizon: Number of steps to look ahead
            rng: Random number generator
            
        Returns:
            Cumulative discounted reward
        """
        from environment_simulation.graph_environment import GraphEnvironment
        from copy import deepcopy
        
        # Create a copy of the belief for this rollout
        rollout_belief = deepcopy(self.belief)
        original_belief = self.belief
        self.belief = rollout_belief
        
        current_node = start_node
        total_reward = 0.0
        
        try:
            for h in range(horizon):
                # For rollout, use a simple heuristic: move to neighbor with highest belief probability
                # or randomly choose from neighbors
                neighbors = env.get_neighbors(current_node)
                available_nodes = list(neighbors) + [current_node]  # Can stay or move
                
                # Simple rollout policy: choose node with highest uncertainty (highest entropy)
                # Or randomly choose
                if len(available_nodes) > 1 and h > 0:
                    # Choose randomly from available nodes (simple rollout policy)
                    target_node = available_nodes[rng.integers(len(available_nodes))]
                else:
                    target_node = current_node
                
                # Move to target node
                current_node = target_node
                
                # Compute reward BEFORE updating belief (uses prior entropy)
                step_reward = self.compute_reward(
                    env,
                    target_node=target_node,
                )
                
                total_reward += (self.discount_factor ** h) * step_reward
                
                # Simulate observation: sample from belief distribution
                observation = self._sample_observation_from_belief(current_node, rng)
                
                # Update belief with observation: b_{t+1}(j, x') = δ[x' = o_j] for observed node
                self.belief.update_from_observation(
                    {current_node},
                    {current_node: observation}
                )
                
                # Evolve belief for unobserved nodes using transition kernel
                self.belief.evolve_with_transition_kernel(env, {current_node})
        finally:
            # Restore original belief
            self.belief = original_belief
        
        return total_reward

    def compute_policy(self, env: "GraphEnvironment", horizon: int) -> None:
        """
        Compute policy using Monte Carlo rollouts.
        
        Performs multiple rollouts from the current belief state to estimate
        expected cumulative reward. Stores the policy as rollout statistics.
        
        Args:
            env: GraphEnvironment object
            horizon: Planning horizon (number of steps ahead)
        """
        # For Monte Carlo planning, we estimate expected cumulative reward
        # Since the agent moves between nodes, the "policy" is implicit
        # We store the expected reward estimate for decision-making
        self._policy = {
            "type": "monte_carlo",
            "horizon": horizon,
            "computed": True,
        }
    
    def _extract_graph_data_for_numba(
        self,
        env: "GraphEnvironment",
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
        """
        Extract graph structure and RSP parameters as numpy arrays for Numba optimization.
        
        Returns:
            Tuple of (adjacency_list_flat, adjacency_counts, rsp_params_per_node, max_neighbors, num_states)
            or None if extraction fails or not RSP mode
        """
        if not NUMBA_AVAILABLE:
            return None
        
        # Handle both GraphEnvironment and GraphReplayEnvironment
        actual_env = env
        if isinstance(env, GraphReplayEnvironment):
            if env.original_env is None:
                return None
            actual_env = env.original_env
        
        # Check if RSP mode
        if not (hasattr(actual_env, 'mode') and actual_env.mode == "rsp"):
            return None
        
        num_nodes = actual_env.num_nodes
        num_states = 2  # Binary states
        
        # Extract adjacency list
        max_neighbors = 0
        for node in range(num_nodes):
            neighbors = actual_env.get_neighbors(node)
            max_neighbors = max(max_neighbors, len(neighbors))
        
        if max_neighbors == 0:
            return None
        
        # Flatten adjacency list: [num_nodes * max_neighbors] with -1 for padding
        adjacency_list_flat = np.full(num_nodes * max_neighbors, -1, dtype=np.int32)
        adjacency_counts = np.zeros(num_nodes, dtype=np.int32)
        
        for node in range(num_nodes):
            neighbors = actual_env.get_neighbors(node)
            adjacency_counts[node] = len(neighbors)
            for i, neighbor in enumerate(neighbors):
                if i < max_neighbors:
                    adjacency_list_flat[node * max_neighbors + i] = neighbor
        
        # Extract RSP parameters per node: [num_nodes, 4] (lam, beta0, alpha, delta)
        rsp_params_per_node = np.zeros((num_nodes, 4), dtype=np.float64)
        
        for node in range(num_nodes):
            # Get RSP parameters (from maps if available, otherwise from baseline)
            lam = float(actual_env.ignition_map[node]) if actual_env.ignition_map is not None else actual_env.rsp.lam
            beta0 = float(actual_env.beta0_map[node]) if actual_env.beta0_map is not None else actual_env.rsp.beta0
            alpha = float(actual_env.alpha_map[node]) if actual_env.alpha_map is not None else actual_env.rsp.alpha
            delta = float(actual_env.persistence_map[node]) if actual_env.persistence_map is not None else actual_env.rsp.delta
            
            rsp_params_per_node[node, 0] = lam
            rsp_params_per_node[node, 1] = beta0
            rsp_params_per_node[node, 2] = alpha
            rsp_params_per_node[node, 3] = delta
        
        return (adjacency_list_flat, adjacency_counts, rsp_params_per_node, max_neighbors, num_states)

    def act(self, env: "GraphEnvironment") -> dict[str, Any]:
        """
        Choose an action based on Monte Carlo planning with epsilon-greedy exploration.
        
        With probability epsilon, takes a random action (exploration).
        Otherwise, evaluates different movement actions (moving to neighbor nodes or staying)
        using Monte Carlo rollouts and selects the one with the highest expected reward (exploitation).
        
        This method acts as a wrapper that tries to use Numba-optimized rollouts when available,
        falling back to pure Python implementation otherwise. The interface and return types
        are identical in both cases.
        
        Args:
            env: GraphEnvironment object (required for belief evolution in rollouts)
            
        Returns:
            Dictionary containing action information:
            - target_node: Node index to move to (or current_node to stay)
            - expected_reward: Expected cumulative reward for selected action
            - all_rewards: Dictionary mapping node indices to expected rewards (for debugging)
        """
        from environment_simulation.graph_environment import GraphEnvironment
        from copy import deepcopy
        
        # Get available actions: neighbors of current node + current node (stay)
        available_nodes = list(env.get_neighbors(self.current_node)) + [self.current_node]
        
        if len(available_nodes) == 0:
            # No neighbors, must stay
            return {
                "target_node": self.current_node,
                "expected_reward": 0.0,
            }
        
        # Epsilon-greedy: with probability epsilon, take a random action
        rng = np.random.default_rng()
        if self.epsilon > 0.0 and rng.random() < self.epsilon:
            # Exploration: select random action
            random_node = available_nodes[rng.integers(len(available_nodes))]
            # Build all_rewards dictionary with 0 rewards for all actions (since we didn't compute them)
            all_rewards = {node: 0.0 for node in available_nodes}
            return {
                "target_node": random_node,
                "expected_reward": 0.0,  # Unknown reward for random action
                "all_rewards": all_rewards,
            }
        
        # Exploitation: use Monte Carlo planning to select optimal action
        # Default horizon if not set in policy
        horizon = self._policy.get("horizon", 5) if self._policy else 5
        
        # Try Numba-optimized path first
        graph_data = self._extract_graph_data_for_numba(env)
        if graph_data is not None and NUMBA_AVAILABLE:
            try:
                adjacency_list_flat, adjacency_counts, rsp_params_per_node, max_neighbors, num_states = graph_data
                return self._act_numba(
                    env, available_nodes, horizon, adjacency_list_flat, adjacency_counts,
                    rsp_params_per_node, max_neighbors, num_states
                )
            except Exception as e:
                # Fallback to Python if Numba fails
                import traceback
                print(f"Warning: Numba rollout failed, falling back to Python: {e}")
                traceback.print_exc()
        
        # Python fallback path (original implementation)
        return self._act_python(env, available_nodes, horizon)
    
    def _act_numba(
        self,
        env: "GraphEnvironment",
        available_nodes: list[int],
        horizon: int,
        adjacency_list_flat: np.ndarray,
        adjacency_counts: np.ndarray,
        rsp_params_per_node: np.ndarray,
        max_neighbors: int,
        num_states: int,
    ) -> dict[str, Any]:
        """
        Numba-optimized implementation of act method.
        Maintains exact same interface and return types as _act_python.
        """
        rng = np.random.default_rng()
        action_rewards = []
        node_to_idx = {node: i for i, node in enumerate(available_nodes)}
        
        original_belief = self.belief
        num_nodes = self.num_nodes
        
        # Convert belief to numpy array
        belief_array = original_belief.to_numpy_array(num_states=num_states)
        
        # Extract utility values
        utility_0 = self.event_utility.get(0, 0.0)
        utility_1 = self.event_utility.get(1, 1.0)
        
        for target_node in available_nodes:
            # Pre-generate random numbers for all rollouts
            # _rollout_numba uses: 1 for step 0 observation + (horizon - 1) steps * 2 each (action choice + observation)
            # Total: 1 + (horizon - 1) * 2 = 2 * horizon - 1 random numbers per rollout
            # But ensure at least 1 even if horizon is 0 or 1
            num_random_per_rollout = max(1, 2 * horizon - 1) if horizon > 0 else 1
            all_random_samples = rng.random((self.num_rollouts, num_random_per_rollout))
            
            # Run all rollouts for this action using Numba
            expected_reward = _run_rollouts_numba(
                belief_array,
                adjacency_list_flat,
                adjacency_counts,
                target_node,
                horizon,
                self.num_rollouts,
                num_nodes,
                num_states,
                max_neighbors,
                rsp_params_per_node,
                self.w_h,
                self.w_v,
                utility_0,
                utility_1,
                self.discount_factor,
                all_random_samples,
            )
            action_rewards.append(float(expected_reward))
        
        # Select action with highest expected reward
        if action_rewards:
            best_idx = int(np.argmax(action_rewards))
            best_node = available_nodes[best_idx]
            best_reward = action_rewards[best_idx]
        else:
            # Fallback: stay at current node
            best_node = self.current_node
            best_reward = 0.0
        
        # Build all_rewards dictionary
        all_rewards = {node: action_rewards[node_to_idx[node]] for node in available_nodes}
        
        return {
            "target_node": best_node,
            "expected_reward": best_reward,
            "all_rewards": all_rewards,
        }
    
    def _act_python(
        self,
        env: "GraphEnvironment",
        available_nodes: list[int],
        horizon: int,
    ) -> dict[str, Any]:
        """
        Pure Python implementation of act method (original implementation).
        This is the fallback when Numba is not available or fails.
        """
        from copy import deepcopy
        
        # Evaluate each action using Monte Carlo rollouts
        rng = np.random.default_rng()
        action_rewards = []
        node_to_idx = {node: i for i, node in enumerate(available_nodes)}
        
        original_belief = self.belief
        original_node = self.current_node
        
        for target_node in available_nodes:
            # Temporarily simulate moving to this node
            rollout_rewards = []
            
            for rollout_idx in range(self.num_rollouts):
                # Create a copy of belief for this evaluation
                eval_belief = deepcopy(original_belief)
                self.belief = eval_belief
                
                try:
                    # Compute immediate reward BEFORE updating belief (uses prior entropy)
                    immediate_reward = self.compute_reward(
                        env,
                        target_node=target_node,
                    )
                    
                    # Simulate moving to target node and observing
                    current_node = target_node
                    
                    # Sample observation from belief distribution
                    observation = self._sample_observation_from_belief(current_node, rng)
                    
                    # Update belief with observation: b_{t+1}(j, x') = δ[x' = o_j] for observed node
                    self.belief.update_from_observation(
                        {current_node},
                        {current_node: observation}
                    )
                    
                    # CRITICAL: Evolve belief after immediate observation to get belief at step t+1
                    # This evolves unobserved nodes using the transition kernel
                    self.belief.evolve_with_transition_kernel(env, {current_node})
                    
                    # Rollout future steps using a rollout policy
                    future_reward = 0.0
                    for h in range(1, horizon):
                        # Get neighbors of current position
                        neighbors = env.get_neighbors(current_node)
                        future_available = list(neighbors) + [current_node]
                        
                        if len(future_available) == 0:
                            # No neighbors, must stay - still evolve belief
                            self.belief.evolve_with_transition_kernel(env, set())
                            continue
                        
                        # Rollout policy: choose node with highest uncertainty or randomly
                        # Simple heuristic: choose randomly
                        future_target_node = future_available[rng.integers(len(future_available))]
                        current_node = future_target_node
                        
                        # Compute reward for the selected action (before observation)
                        future_r = self.compute_reward(
                            env,
                            target_node=future_target_node,
                        )
                        future_reward += (self.discount_factor ** h) * future_r
                        
                        # Sample observation from belief distribution
                        future_obs = self._sample_observation_from_belief(future_target_node, rng)
                        
                        # Update belief: b_{t+1}(j, x') = δ[x' = o_j] for observed node
                        self.belief.update_from_observation(
                            {future_target_node},
                            {future_target_node: future_obs}
                        )
                        
                        # CRITICAL: Evolve belief for next step (for unobserved nodes)
                        self.belief.evolve_with_transition_kernel(env, {future_target_node})
                    
                    total_reward = immediate_reward + future_reward
                    rollout_rewards.append(total_reward)
                except Exception as e:
                    # If rollout fails, use a default low reward
                    import traceback
                    if len(rollout_rewards) == 0:  # Only print for first failure
                        print(f"Warning: Rollout failed for node {target_node}: {e}")
                        traceback.print_exc()
                    rollout_rewards.append(0.0)
            
            expected_reward = float(np.mean(rollout_rewards)) if rollout_rewards else 0.0
            action_rewards.append(expected_reward)
        
        # Restore original belief and node
        self.belief = original_belief
        self.current_node = original_node
        
        # Select action with highest expected reward
        if action_rewards:
            best_idx = int(np.argmax(action_rewards))
            best_node = available_nodes[best_idx]
            best_reward = action_rewards[best_idx]
        else:
            # Fallback: stay at current node
            best_node = self.current_node
            best_reward = 0.0
        
        # Build all_rewards dictionary
        all_rewards = {node: action_rewards[node_to_idx[node]] for node in available_nodes}
        
        return {
            "target_node": best_node,
            "expected_reward": best_reward,
            "all_rewards": all_rewards,
        }


"""
Kernel Learning for Fire Physics Simulation.

Learns a local transition kernel P(s' | s, m, k') from fire simulation trajectories,
where:
- s: current state (0=unburned, 1=burning, 2=burned)
- m: material type (WOOD, GRASS, WATER, etc.)
- k': clipped count of burning neighbors
- s': next state

The learned kernel can then be used for belief prediction in planning algorithms.
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple
from collections import defaultdict
import numpy as np

# Import Material enum
import sys
import os
FIRE_SIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'fire-simulation')
if FIRE_SIM_PATH not in sys.path:
    sys.path.insert(0, FIRE_SIM_PATH)

# Mock src.window_option to avoid FileNotFoundError for favicon when running from environment_simulation
import types
if 'src.window_option' not in sys.modules:
    window_option_mock = types.ModuleType('src.window_option')
    window_option_mock.CELL_WIDTH = 10
    window_option_mock.CELL_HEIGHT = 10
    sys.modules['src.window_option'] = window_option_mock

from src.material import Material


class TransitionKernelLearner:
    """
    Learns a local transition kernel from fire simulation trajectories.
    
    Collects transitions (s, m, k', s') and estimates P(s' | s, m, k')
    using maximum-likelihood with Laplace smoothing.
    """
    
    def __init__(self, max_burning_neighbors: int = 8, laplace_alpha: float = 1.0):
        """
        Initialize the kernel learner.
        
        Args:
            max_burning_neighbors: Maximum value for k' (clipped neighbor count)
            laplace_alpha: Smoothing constant for Laplace smoothing
        """
        self.max_burning_neighbors = max_burning_neighbors
        self.laplace_alpha = laplace_alpha
        
        # Count tensor: C(s, m, k', s')
        # Structure: count_dict[(current_state, material, k_prime, next_state)] = count
        self.count_dict: Dict[Tuple[int, str, int, int], int] = defaultdict(int)
        
        # Learned transition probabilities: P(s' | s, m, k')
        # Structure: transition_probs[(current_state, material, k_prime)] = {next_state: prob}
        self.transition_probs: Dict[Tuple[int, str, int], Dict[int, float]] = {}
        
        # Total number of transitions collected
        self.total_transitions = 0
    
    def collect_transition(
        self,
        current_state: int,
        material: Material,
        num_burning_neighbors: int,
        next_state: int,
    ) -> None:
        """
        Record a single transition from the simulation.
        
        Args:
            current_state: X_{j,t} ∈ {0, 1, 2}
            material: Material type M_j
            num_burning_neighbors: Raw count of burning neighbors (will be clipped)
            next_state: X_{j,t+1} ∈ {0, 1, 2}
        """
        # Clip k to max_burning_neighbors
        k_prime = min(num_burning_neighbors, self.max_burning_neighbors)
        
        # Get material name as string
        material_name = material.name
        
        # Record transition: (s, m, k', s')
        key = (current_state, material_name, k_prime, next_state)
        self.count_dict[key] += 1
        self.total_transitions += 1
    
    def collect_from_environment_step(
        self,
        env: "GraphEnvironmentFire",
        state_before: np.ndarray,
        state_after: np.ndarray,
    ) -> None:
        """
        Collect all transitions from a single environment step.
        
        Args:
            env: GraphEnvironmentFire instance
            state_before: State array at time t
            state_after: State array at time t+1
        """
        for node in range(env.num_nodes):
            current_state = int(state_before[node])
            next_state = int(state_after[node])
            material = env.material_map[node]
            
            # Count burning neighbors
            num_burning_neighbors = 0
            neighbors = env.get_neighbors(node)
            for neighbor in neighbors:
                if state_before[neighbor] == 1:  # Neighbor is burning
                    num_burning_neighbors += 1
            
            # Record transition
            self.collect_transition(
                current_state=current_state,
                material=material,
                num_burning_neighbors=num_burning_neighbors,
                next_state=next_state,
            )
    
    def estimate_kernel(self) -> Dict[Tuple[int, str, int], Dict[int, float]]:
        """
        Estimate transition probabilities using maximum-likelihood with Laplace smoothing.
        
        Returns:
            Dictionary mapping (s, m, k') -> {s': probability}
        """
        num_states = 3  # 0, 1, 2
        
        # Initialize transition_probs dictionary
        self.transition_probs = {}
        
        # Iterate over all observed (s, m, k') combinations
        observed_parent_configs = set()
        for (s, m, k_prime, s_prime), count in self.count_dict.items():
            observed_parent_configs.add((s, m, k_prime))
        
        # Estimate probabilities for each parent configuration
        for (s, m, k_prime) in observed_parent_configs:
            # Count transitions for this parent configuration
            counts_by_next_state = {}
            total_count = 0
            
            for s_prime in range(num_states):
                key = (s, m, k_prime, s_prime)
                count = self.count_dict.get(key, 0)
                counts_by_next_state[s_prime] = count
                total_count += count
            
            # Compute probabilities with Laplace smoothing
            # Formula: P̂(s' | s,m,k) = (C(s,m,k,s') + α) / (Σ_{u∈S} C(s,m,k,u) + α|S|)
            # Smoothing is over ALL states in S (all 3 states), as per specification
            probs = {}
            denominator = total_count + self.laplace_alpha * num_states
            
            for s_prime in range(num_states):
                numerator = counts_by_next_state[s_prime] + self.laplace_alpha
                probs[s_prime] = numerator / denominator
            
            # Enforce physical constraints: set invalid transitions to 0 and renormalize
            # Physical constraints:
            # - UNBURNED (0) can only go to UNBURNED (0) or BURNING (1), never BURNED (2)
            # - BURNING (1) can go to BURNING (1) or BURNED (2), never UNBURNED (0) directly
            # - BURNED (2) can go to BURNED (2) or UNBURNED (0) via recovery, never BURNING (1)
            valid_next_states = []
            if s == 0:  # UNBURNED
                valid_next_states = [0, 1]
                probs[2] = 0.0  # Cannot go directly to BURNED
            elif s == 1:  # BURNING
                valid_next_states = [1, 2]
                probs[0] = 0.0  # Cannot go directly to UNBURNED
            else:  # s == 2: BURNED
                valid_next_states = [0, 2]
                probs[1] = 0.0  # Cannot go to BURNING
            
            # Renormalize over valid states
            total_prob = sum(probs[s_prime] for s_prime in valid_next_states)
            if total_prob > 0:
                for s_prime in valid_next_states:
                    probs[s_prime] /= total_prob
            
            # Store probabilities
            self.transition_probs[(s, m, k_prime)] = probs
        
        return self.transition_probs
    
    def get_transition_probability(
        self,
        current_state: int,
        material: Material,
        num_burning_neighbors: int,
        next_state: int,
    ) -> float:
        """
        Get learned transition probability P(s' | s, m, k').
        
        Uses interpolation if exact configuration not observed.
        
        Args:
            current_state: Current state s ∈ {0, 1, 2}
            material: Material type
            num_burning_neighbors: Number of burning neighbors (will be clipped)
            next_state: Next state s' ∈ {0, 1, 2}
        
        Returns:
            Probability P(s' | s, m, k')
        """
        # Clip k
        k_prime = min(num_burning_neighbors, self.max_burning_neighbors)
        material_name = material.name
        
        key = (current_state, material_name, k_prime)
        
        # Exact match
        if key in self.transition_probs:
            return self.transition_probs[key].get(next_state, 0.0)
        
        # Try to find similar configurations (same state, material, different k')
        # and interpolate
        similar_keys = [
            (s, m, k) for (s, m, k) in self.transition_probs.keys()
            if s == current_state and m == material_name
        ]
        
        if similar_keys:
            # Find nearest k' values
            k_values = sorted(set(k for _, _, k in similar_keys))
            if k_prime <= k_values[0]:
                # Use smallest available k
                nearest_key = (current_state, material_name, k_values[0])
            elif k_prime >= k_values[-1]:
                # Use largest available k
                nearest_key = (current_state, material_name, k_values[-1])
            else:
                # Interpolate between two nearest k values
                for i in range(len(k_values) - 1):
                    if k_values[i] <= k_prime <= k_values[i + 1]:
                        k_low, k_high = k_values[i], k_values[i + 1]
                        key_low = (current_state, material_name, k_low)
                        key_high = (current_state, material_name, k_high)
                        
                        p_low = self.transition_probs[key_low].get(next_state, 0.0)
                        p_high = self.transition_probs[key_high].get(next_state, 0.0)
                        
                        # Linear interpolation
                        if k_high > k_low:
                            alpha = (k_prime - k_low) / (k_high - k_low)
                            return (1 - alpha) * p_low + alpha * p_high
                        else:
                            return p_low
                # Fallback (shouldn't reach here)
                nearest_key = (current_state, material_name, k_values[0])
            
            return self.transition_probs[nearest_key].get(next_state, 0.0)
        
        # No similar configuration found: use domain knowledge
        # Based on physical constraints of fire dynamics
        # Enforce physical constraints: 0→{0,1}, 1→{1,2}, 2→{0,2}
        if current_state == 2:  # BURNED
            # Burned cells mostly stay burned (unless recovery happens)
            if next_state == 2:
                return 0.95
            elif next_state == 0:  # Recovery
                return 0.05
            else:  # next_state == 1 (BURNING) - physically impossible
                return 0.0
        elif current_state == 1:  # BURNING
            # Burning cells likely transition to burned, sometimes stay burning
            if next_state == 2:
                return 0.7
            elif next_state == 1:
                return 0.3
            else:  # next_state == 0 (UNBURNED) - physically impossible
                return 0.0
        else:  # current_state == 0: UNBURNED
            # Unburned cells mostly stay unburned, but can ignite
            # Higher probability of ignition with more burning neighbors
            p_ignite = min(0.5, 0.02 + 0.08 * k_prime)  # Increased sensitivity to neighbors
            if next_state == 0:
                return 1.0 - p_ignite
            elif next_state == 1:
                return p_ignite
            else:  # next_state == 2 (BURNED) - physically impossible
                return 0.0
    
    def predict_belief(
        self,
        current_belief: Dict[int, float],
        material: Material,
        num_burning_neighbors: int,
    ) -> Dict[int, float]:
        """
        Predict next belief using learned kernel.
        
        According to specification:
        b_{j,t+1}^{pred}(s') = Σ_{s ∈ S} b_{j,t}(s) * P̂(s' | s, M_j, k')
        
        where k' = min(round(K_{j,t}), K_max) and K_{j,t} = Σ_{l ∈ N(j)} b_{l,t}(1)
        
        Args:
            current_belief: Dict mapping state -> probability {0: p0, 1: p1, 2: p2}
            material: Material type M_j
            num_burning_neighbors: Estimated burning neighbor count K_{j,t} (will be rounded and clipped)
        
        Returns:
            Predicted belief for next timestep {0: p0, 1: p1, 2: p2}
        """
        # Round and clip: k'_{j,t} = min(round(K_{j,t}), K_max)
        k_prime = min(int(round(num_burning_neighbors)), self.max_burning_neighbors)
        material_name = material.name
        
        predicted_belief = {0: 0.0, 1: 0.0, 2: 0.0}
        
        # Sum over current states
        for current_state in range(3):
            prob_current = current_belief.get(current_state, 0.0)
            
            if prob_current > 0:
                # Get transition probabilities for this (s, m, k')
                key = (current_state, material_name, k_prime)
                
                if key in self.transition_probs:
                    transition_probs = self.transition_probs[key]
                else:
                    # Not observed: use get_transition_probability which handles interpolation/fallback
                    transition_probs = {
                        s_prime: self.get_transition_probability(
                            current_state, material, num_burning_neighbors, s_prime
                        )
                        for s_prime in range(3)
                    }
                
                # Accumulate probability for each next state
                for next_state in range(3):
                    predicted_belief[next_state] += prob_current * transition_probs.get(next_state, 0.0)
        
        return predicted_belief
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the learned kernel.
        
        Returns:
            Dictionary with statistics about collected transitions and learned kernel
        """
        num_parent_configs = len(self.transition_probs)
        
        # Count how many transitions per parent configuration
        transitions_per_config = []
        for key, probs in self.transition_probs.items():
            # Count total transitions for this config
            total = sum(
                self.count_dict.get((key[0], key[1], key[2], s_prime), 0)
                for s_prime in range(3)
            )
            transitions_per_config.append(total)
        
        return {
            "total_transitions": self.total_transitions,
            "num_parent_configurations": num_parent_configs,
            "max_transitions_per_config": max(transitions_per_config) if transitions_per_config else 0,
            "min_transitions_per_config": min(transitions_per_config) if transitions_per_config else 0,
            "avg_transitions_per_config": np.mean(transitions_per_config) if transitions_per_config else 0,
        }
    
    def save_kernel(self, filepath: str) -> None:
        """
        Save learned kernel to file (numpy format).
        
        Args:
            filepath: Path to save the kernel
        """
        # Convert to numpy arrays for saving
        # Create a structured array or dictionary format
        import pickle
        
        kernel_data = {
            "transition_probs": self.transition_probs,
            "count_dict": dict(self.count_dict),
            "max_burning_neighbors": self.max_burning_neighbors,
            "laplace_alpha": self.laplace_alpha,
            "total_transitions": self.total_transitions,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(kernel_data, f)
    
    def load_kernel(self, filepath: str) -> None:
        """
        Load a saved kernel from a pickle file.
        
        Args:
            filepath: Path to the kernel file (can be relative or absolute)
                     If relative, it's resolved relative to the script's directory
        """
        import os
        import pickle
        
        # Convert to absolute path if relative (relative to the script's directory)
        if not os.path.isabs(filepath):
            # Get the directory of this file (kernel_learning.py)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filepath)
        
        with open(filepath, 'rb') as f:
            kernel_data = pickle.load(f)
        
        self.transition_probs = kernel_data["transition_probs"]
        self.count_dict = defaultdict(int, kernel_data["count_dict"])
        self.max_burning_neighbors = kernel_data.get("max_burning_neighbors", 8)
        self.laplace_alpha = kernel_data.get("laplace_alpha", 1.0)
        self.total_transitions = kernel_data.get("total_transitions", 0)
    
    def compute_neighbor_burning_count_from_beliefs(
        self,
        neighbor_beliefs: list[Dict[int, float]],
    ) -> int:
        """
        Compute estimated neighbor burning count from neighbor beliefs.
        
        According to specification:
        K_{j,t} = Σ_{l ∈ N(j)} b_{l,t}(1)
        
        Then: k'_{j,t} = min(round(K_{j,t}), K_max)
        
        Args:
            neighbor_beliefs: List of belief dictionaries, one per neighbor
                            Each dict maps state -> probability {0: p0, 1: p1, 2: p2}
        
        Returns:
            Rounded and clipped neighbor burning count k'
        """
        # Sum over neighbors: K_{j,t} = Σ_{l ∈ N(j)} b_{l,t}(1)
        K_jt = sum(belief.get(1, 0.0) for belief in neighbor_beliefs)
        
        # Round and clip: k'_{j,t} = min(round(K_{j,t}), K_max)
        k_prime = min(int(round(K_jt)), self.max_burning_neighbors)
        
        return k_prime


def learn_kernel_from_simulation(
    env: "GraphEnvironmentFire",
    num_steps: int = 1000,
    collect_every: int = 1,
    max_burning_neighbors: int = 8,
    laplace_alpha: float = 1.0,
    num_episodes: int = 1,
    verbose: bool = True,
) -> TransitionKernelLearner:
    """
    Learn transition kernel by running fire simulation and collecting transitions.
    
    Can run multiple episodes to collect more diverse transitions, especially
    for rare events like ignitions.
    
    Args:
        env: GraphEnvironmentFire instance
        num_steps: Number of simulation steps per episode
        collect_every: Collect transitions every N steps (1 = every step)
        max_burning_neighbors: Maximum value for k' (clipped neighbor count)
        laplace_alpha: Smoothing constant for Laplace smoothing (smaller = less smoothing)
        num_episodes: Number of episodes to run (each episode resets environment)
        verbose: Print progress messages
    
    Returns:
        TransitionKernelLearner with learned kernel
    """
    learner = TransitionKernelLearner(
        max_burning_neighbors=max_burning_neighbors,
        laplace_alpha=laplace_alpha,
    )
    
    if verbose:
        print(f"Learning kernel from {num_episodes} episode(s), {num_steps} steps each...")
    
    # Save environment configuration for recreating episodes
    saved_edges = getattr(env, 'edges', None)
    saved_width = getattr(env, 'width', None)
    saved_height = getattr(env, 'height', None)
    saved_blocked_mask = getattr(env, 'blocked_mask', None)
    saved_material_map = getattr(env, 'material_map', None) if hasattr(env, 'material_map') else None
    saved_num_nodes = env.num_nodes
    
    for episode in range(num_episodes):
        # Reset environment for new episode (if multiple episodes)
        if episode > 0:
            # Recreate environment with fresh state but same structure
            from test_fire_physics_graph import GraphEnvironmentFire
            from main_graph_agents import create_graph_environment
            
            # Use different seed for each episode to see different scenarios
            episode_seed = 42 + episode * 1000
            
            # Recreate base environment to get graph structure for this episode
            base_env_ep = create_graph_environment(
                num_nodes=saved_num_nodes,
                mode="rsp",
                seed=episode_seed,
            )
            
            # Create NEW random clusters for this episode (different locations)
            rng_ep = np.random.default_rng(episode_seed)
            num_clusters = 5
            initial_burning_nodes_ep = []
            
            for cluster_idx in range(num_clusters):
                available_nodes = [n for n in range(saved_num_nodes) if n not in initial_burning_nodes_ep]
                if not available_nodes:
                    break
                start_node = rng_ep.choice(available_nodes)
                neighbors = list(base_env_ep.get_neighbors(start_node))
                neighbors = [n for n in neighbors if n not in initial_burning_nodes_ep]
                cluster_size = min(3, len(neighbors) + 1)
                cluster_nodes = [start_node] + neighbors[:cluster_size-1]
                initial_burning_nodes_ep.extend(cluster_nodes)
            
            # Handle blocked_mask
            blocked_mask = saved_blocked_mask
            if blocked_mask is not None and blocked_mask.ndim == 2:
                blocked_mask = blocked_mask.flatten()[:saved_num_nodes]
            
            # Recreate fire environment with same structure but NEW random initial burning nodes
            env = GraphEnvironmentFire(
                num_nodes=saved_num_nodes,
                mode="fire",
                edges=saved_edges if saved_edges else base_env_ep.edges,
                width=saved_width if saved_width else base_env_ep.width,
                height=saved_height if saved_height else base_env_ep.height,
                blocked_mask=blocked_mask,
                seed=episode_seed,
                initial_burning_nodes=initial_burning_nodes_ep,
                material_map=saved_material_map.copy() if saved_material_map else None,
            )
        
        # Get initial state
        state_before = env.get_state().copy()
        initial_burning = np.sum(state_before == 1)
        
        # Track fire spread during this episode
        max_burning = initial_burning
        total_ignitions_observed = 0
        max_burned = 0
        
        for step in range(num_steps):
            # Step environment
            env.step()
            state_after = env.get_state().copy()
            
            # Track ignitions (0→1 transitions)
            ignitions_this_step = np.sum((state_before == 0) & (state_after == 1))
            total_ignitions_observed += ignitions_this_step
            
            # Track max burning and burned
            burning_count = np.sum(state_after == 1)
            burned_count = np.sum(state_after == 2)
            max_burning = max(max_burning, burning_count)
            max_burned = max(max_burned, burned_count)
            
            # Collect transitions
            if step % collect_every == 0:
                learner.collect_from_environment_step(env, state_before, state_after)
            
            state_before = state_after.copy()
            
            # Progress update
            if verbose and (step + 1) % 100 == 0:
                burning = np.sum(state_after == 1)
                burned = np.sum(state_after == 2)
                print(f"  Episode {episode+1}, Step {step + 1}/{num_steps}: "
                      f"Burning={burning}, Burned={burned}, "
                      f"Transitions={learner.total_transitions}")
        
        if verbose:
            print(f"  Episode {episode+1} complete: "
                  f"Initial={initial_burning} burning → Max={max_burning} burning, "
                  f"Total ignitions={total_ignitions_observed}, "
                  f"Max burned={max_burned}, "
                  f"Transitions collected={learner.total_transitions}")
            
            if total_ignitions_observed == 0 and max_burning == initial_burning:
                print(f"    WARNING: No fire spread observed in this episode!")
                print(f"    The kernel will not learn ignition probabilities from this episode.")
    
    # Estimate kernel
    if verbose:
        print("\nEstimating transition probabilities...")
    learner.estimate_kernel()
    
    # Print statistics
    stats = learner.get_statistics()
    if verbose:
        print(f"\nKernel learning complete!")
        print(f"  Total transitions: {stats['total_transitions']}")
        print(f"  Parent configurations: {stats['num_parent_configurations']}")
        print(f"  Avg transitions per config: {stats['avg_transitions_per_config']:.1f}")
        
        # Check ignition probabilities and raw counts
        print(f"\nChecking learned ignition probabilities:")
        from src.material import Material
        found_any_ignition = False
        for material_name in ["WOOD", "GRASS"]:  # Only WOOD and GRASS (no GASOLINE)
            for k in range(9):
                key = (0, material_name, k)
                if key in learner.transition_probs:
                    probs = learner.transition_probs[key]
                    p_ignite = probs.get(1, 0.0)
                    total_trans = sum(
                        learner.count_dict.get((0, material_name, k, s_next), 0)
                        for s_next in range(3)
                    )
                    ignited_count = learner.count_dict.get((0, material_name, k, 1), 0)
                    stayed_count = learner.count_dict.get((0, material_name, k, 0), 0)
                    
                    if p_ignite > 0.001 or ignited_count > 0:  # Show if probability > 0.1% or we observed any ignitions
                        print(f"  {material_name}, k'={k}: P(0→1)={p_ignite:.6f} "
                              f"(from {total_trans} transitions: {ignited_count} ignited, {stayed_count} stayed)")
                        found_any_ignition = True
        
        if not found_any_ignition:
            print("  WARNING: No significant ignition probabilities found!")
            print("  This means the kernel will not learn how fires spread.")
            print("  Possible causes:")
            print("    1. Fire did not spread during training (check if max_burning increased)")
            print("    2. Not enough transitions collected (increase --num_steps or --num_episodes)")
            print("    3. Fire parameters too low (increase heat_transfer_coefficient in fire model)")
            print("    4. Laplace smoothing too strong (try smaller --laplace_alpha, e.g. 0.1)")
    
    return learner


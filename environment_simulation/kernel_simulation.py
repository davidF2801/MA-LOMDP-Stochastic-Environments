"""
Kernel-based simulation using learned transition probabilities.

Simulates fire spread using the learned kernel instead of physical model.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

# Import Material enum
import sys
import os
FIRE_SIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'fire-simulation')
if FIRE_SIM_PATH not in sys.path:
    sys.path.insert(0, FIRE_SIM_PATH)

from src.material import Material


class KernelBasedEnvironment:
    """
    Environment that evolves using learned transition kernel instead of physics.
    
    Uses the same graph structure and materials as the physical model,
    but transitions are sampled from learned probabilities.
    """
    
    def __init__(
        self,
        num_nodes: int,
        kernel_learner: "TransitionKernelLearner",
        material_map: Dict[int, Material],
        adjacency_list: Dict[int, list[int]],
        initial_state: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize kernel-based environment.
        
        Args:
            num_nodes: Number of nodes in the graph
            kernel_learner: TransitionKernelLearner with learned kernel
            material_map: Dict mapping node -> Material
            adjacency_list: Dict mapping node -> list of neighbor nodes
            initial_state: Initial state array (0=unburned, 1=burning, 2=burned). If None, all unburned.
            rng: Random number generator
        """
        self.num_nodes = num_nodes
        self.kernel = kernel_learner
        self.material_map = material_map
        self.adjacency_list = adjacency_list
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Initialize state
        if initial_state is None:
            self.state = np.zeros(num_nodes, dtype=np.int8)
        else:
            self.state = initial_state.copy()
        
        self.time = 0
    
    def get_state(self) -> np.ndarray:
        """Get current state array."""
        return self.state.copy()
    
    def get_neighbors(self, node: int) -> list[int]:
        """Get list of neighbor nodes."""
        return self.adjacency_list.get(node, [])
    
    def step(self) -> np.ndarray:
        """
        Evolve environment one step using kernel probabilities.
        
        For each node, samples next state based on:
        - Current state s
        - Material m
        - Number of burning neighbors k' (counted from current state)
        - Learned transition probabilities P(s' | s, m, k')
        
        IMPORTANT: All nodes are updated synchronously - we use state_before
        for all neighbor counts, then update all states at once.
        """
        state_before = self.state.copy()
        state_after = np.zeros(self.num_nodes, dtype=np.int8)
        
        # Debug: track how many cells are changing state
        num_ignitions = 0
        num_burnouts = 0
        num_recoveries = 0
        
        # Compute next state for each node
        for node in range(self.num_nodes):
            current_state = int(state_before[node])
            material = self.material_map[node]
            
            # Count burning neighbors from current state
            num_burning_neighbors = 0
            for neighbor in self.get_neighbors(node):
                if state_before[neighbor] == 1:  # Neighbor is burning
                    num_burning_neighbors += 1
            
            # Get transition probabilities from kernel
            k_prime = min(num_burning_neighbors, self.kernel.max_burning_neighbors)
            material_name = material.name
            
            key = (current_state, material_name, k_prime)
            
            # Get transition probabilities
            if key in self.kernel.transition_probs:
                transition_probs = self.kernel.transition_probs[key]
            else:
                # Fallback: use get_transition_probability (handles interpolation/domain knowledge)
                # Import Material here to avoid circular imports
                from src.material import Material as MaterialEnum
                transition_probs = {
                    s_prime: self.kernel.get_transition_probability(
                        current_state, material, num_burning_neighbors, s_prime
                    )
                    for s_prime in range(3)
                }
            
            # ENFORCE PHYSICAL CONSTRAINTS: Set impossible transitions to 0
            # - UNBURNED (0) → BURNED (2): Must be 0
            # - BURNING (1) → UNBURNED (0): Must be 0
            # - BURNED (2) → BURNING (1): Must be 0
            if current_state == 0:  # UNBURNED
                transition_probs[2] = 0.0  # Cannot go directly to BURNED
            elif current_state == 1:  # BURNING
                transition_probs[0] = 0.0  # Cannot go directly to UNBURNED
            elif current_state == 2:  # BURNED
                transition_probs[1] = 0.0  # Cannot go to BURNING
            
            # Determine valid next states based on current state
            if current_state == 0:  # UNBURNED → {0, 1}
                valid_next_states = [0, 1]
            elif current_state == 1:  # BURNING → {1, 2}
                valid_next_states = [1, 2]
            else:  # BURNED → {0, 2}
                valid_next_states = [0, 2]
            
            # Sample next state from the distribution
            # Only use valid next states
            probs = [transition_probs.get(s, 0.0) for s in valid_next_states]
            
            # Normalize over valid states only
            probs = np.array(probs, dtype=np.float64)
            prob_sum = probs.sum()
            
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                # Fallback: uniform distribution over valid states if all probabilities are 0
                probs = np.ones(len(valid_next_states)) / len(valid_next_states)
            
            # Ensure probabilities are valid (non-negative, sum to 1)
            probs = np.clip(probs, 0.0, 1.0)
            probs = probs / probs.sum()
            
            # Sample from valid states only
            next_state = self.rng.choice(valid_next_states, p=probs)
            state_after[node] = next_state
            
            # Safety check: verify transition is valid (should never trigger due to constraints above)
            if (current_state == 0 and next_state == 2) or \
               (current_state == 1 and next_state == 0) or \
               (current_state == 2 and next_state == 1):
                print(f"WARNING: Impossible transition detected! {current_state} → {next_state} at node {node}")
                # Force valid transition: stay in current state if burning, go to burned if burning
                if current_state == 0:
                    state_after[node] = 0  # Stay unburned
                elif current_state == 1:
                    state_after[node] = 1  # Stay burning (safer than going to burned)
                elif current_state == 2:
                    state_after[node] = 2  # Stay burned
            
            # Track state changes for debugging
            if current_state == 0 and next_state == 1:
                num_ignitions += 1
            elif current_state == 1 and next_state == 2:
                num_burnouts += 1
            elif current_state == 2 and next_state == 0:
                num_recoveries += 1
        
        # Debug output every 10 steps (or always for first 5 steps)
        if (self.time <= 5 or self.time % 10 == 0) and (num_ignitions > 0 or num_burnouts > 0 or num_recoveries > 0):
            print(f"Kernel step {self.time}: {num_ignitions} ignitions, {num_burnouts} burnouts, {num_recoveries} recoveries")
        
        # Debug: check if we have any burning cells and their probabilities
        if self.time == 1:
            burning_nodes = np.where(state_before == 1)[0]
            if len(burning_nodes) > 0:
                node = burning_nodes[0]
                material = self.material_map[node]
                key = (1, material.name, 0)
                if key in self.kernel.transition_probs:
                    probs = self.kernel.transition_probs[key]
                    print(f"  Burning node {node} ({material.name}): transition probs = {probs}")
        
        self.state = state_after
        self.time += 1
        
        return self.state.copy()


def create_comparison_environment(
    base_env: "GraphEnvironmentFire",
    kernel_learner: "TransitionKernelLearner",
    rng: Optional[np.random.Generator] = None,
) -> KernelBasedEnvironment:
    """
    Create a kernel-based environment with the same structure as the physical one.
    
    Args:
        base_env: GraphEnvironmentFire instance (for structure and initial conditions)
        kernel_learner: TransitionKernelLearner with learned kernel
        rng: Random number generator (optional, uses same seed for reproducibility)
    
    Returns:
        KernelBasedEnvironment with same materials, graph structure, and initial state
    """
    # Get initial state from base environment
    initial_state = base_env.get_state().copy()
    
    # Build adjacency list
    adjacency_list = {}
    for node in range(base_env.num_nodes):
        adjacency_list[node] = base_env.get_neighbors(node)
    
    # Create kernel-based environment
    kernel_env = KernelBasedEnvironment(
        num_nodes=base_env.num_nodes,
        kernel_learner=kernel_learner,
        material_map=base_env.material_map,
        adjacency_list=adjacency_list,
        initial_state=initial_state,
        rng=rng,
    )
    
    return kernel_env


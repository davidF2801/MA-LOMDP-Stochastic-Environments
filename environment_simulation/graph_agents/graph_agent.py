"""Abstract graph agent class for graph-based planning framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .graph_belief import GraphBelief


@dataclass
class GraphAgent(ABC):
    """
    Abstract graph agent class for graph-based planner testing framework.
    
    Agents maintain a belief over the graph environment state and can act based on their policy.
    Agents move between nodes and observe the exact state of the node they're currently at.
    """

    name: str
    current_node: int  # Current node the agent is at
    belief: GraphBelief
    num_nodes: int  # Total number of nodes in the graph
    color: str = "#ffdd57"
    # Policy storage (subclasses can use this to store policy data)
    _policy: Optional[Any] = field(default=None, init=False, repr=False)

    def position(self) -> int:
        """Return the current node the agent is at."""
        return self.current_node
    
    def get_reachable_nodes(self, env: "GraphEnvironment") -> set[int]:
        """
        Get set of nodes the agent can move to from current position.
        
        Args:
            env: GraphEnvironment object
            
        Returns:
            Set of node indices that the agent can move to (neighbors of current node + current node for staying)
        """
        neighbors = set(env.get_neighbors(self.current_node))
        neighbors.add(self.current_node)  # Can always stay at current node
        return neighbors
    
    def move_to_node(self, node: int, env: "GraphEnvironment") -> bool:
        """
        Move agent to a new node.
        
        Args:
            node: Target node index
            env: GraphEnvironment object (to check if move is valid)
            
        Returns:
            True if move was successful, False otherwise
        """
        if node == self.current_node:
            return True  # Staying is always valid
        
        neighbors = env.get_neighbors(self.current_node)
        if node in neighbors:
            self.current_node = node
            return True
        return False
    
    def observe_current_node(self, env: "GraphEnvironment") -> int:
        """
        Observe the exact state of the current node.
        
        Args:
            env: GraphEnvironment object
            
        Returns:
            Observed state (0 or 1) of the current node
        """
        return env.get_node_state(self.current_node)
    
    def update_belief_from_observation(
        self,
        env: "GraphEnvironment",
        observed_node: int,
        observed_state: int,
    ) -> None:
        """
        Update belief based on observation.
        
        For observed node: b_{t+1}(j, x') = δ[x' = o_j]
        
        Args:
            env: GraphEnvironment object
            observed_node: Node index that was observed
            observed_state: Observed state (0 or 1)
        """
        observed_nodes = {observed_node}
        observations = {observed_node: observed_state}
        self.belief.update_from_observation(observed_nodes, observations)
    
    def evolve_belief_with_environment(
        self,
        env: "GraphEnvironment",
        observed_nodes: set[int],
    ) -> None:
        """
        Evolve belief using the environment's transition kernel.
        
        This implements the belief update formula:
        - For observed nodes: b_{t+1}(j, x') = δ[x' = o_j] (handled separately)
        - For unobserved nodes: b_{t+1}(j, x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * 
          b_t(j, x_j) * Π_{l∈N(j)} b_t(l, x_l)
        
        Args:
            env: GraphEnvironment object with transition kernel
            observed_nodes: Set of node indices that were observed
        """
        # Initialize belief for nodes that might be reachable from observed nodes
        # (neighbors of observed nodes, and their neighbors)
        nodes_to_initialize = set(observed_nodes)
        for node in observed_nodes:
            nodes_to_initialize.update(env.get_neighbors(node))
            for neighbor in env.get_neighbors(node):
                nodes_to_initialize.update(env.get_neighbors(neighbor))
        
        self.belief.initialize_nodes(nodes_to_initialize)
        
        # Evolve beliefs for unobserved nodes using transition kernel
        self.belief.evolve_with_transition_kernel(env, observed_nodes)
    
    def information_gain(
        self,
        nodes: Optional[set[int]] = None,
    ) -> float:
        """
        Compute information gain from observing nodes.
        
        Information gain is: G(b) = H_prior(b) - H_post(b)
        Since we assume perfect observations, H_post = 0 for observed nodes.
        
        Args:
            nodes: Optional set of node indices to observe.
                  If None, uses current node.
        
        Returns:
            Information gain (entropy reduction) in bits
        """
        if nodes is None:
            nodes = {self.current_node}
        
        # Prior entropy for nodes that will be observed
        prior_entropy = self.belief.entropy(nodes)
        
        # After perfect observation, entropy is 0 for observed nodes
        # So information gain equals prior entropy
        return prior_entropy
    
    def expected_event_value(
        self,
        nodes: Optional[set[int]] = None,
        event_utility: Optional[dict[int, float]] = None,
    ) -> float:
        """
        Compute expected value of detecting events in nodes.
        
        E[value] = Σ_{x in X} b(x) * f(x)
        where f(x) is the utility of observing state x.
        
        Args:
            nodes: Optional set of node indices to consider.
                  If None, uses current node.
            event_utility: Dictionary mapping state (0 or 1) to utility value.
                          If None, uses {0: 0.0, 1: 1.0}.
        
        Returns:
            Expected event value
        """
        if nodes is None:
            nodes = {self.current_node}
        
        if event_utility is None:
            event_utility = {0: 0.0, 1: 1.0}
        
        utility_0 = event_utility.get(0, 0.0)
        utility_1 = event_utility.get(1, 0.0)
        
        # Compute expected value for each node
        expected_value = 0.0
        for node in nodes:
            # Get probability from belief (or use prior if not in belief)
            prob = self.belief.get_probability(node)
            # E[value] = P(0) * f(0) + P(1) * f(1)
            node_value = (1.0 - prob) * utility_0 + prob * utility_1
            expected_value += node_value
        
        return float(expected_value)

    @abstractmethod
    def compute_reward(
        self,
        env: "GraphEnvironment",
        action: Optional[int] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute reward for taking an action.
        
        Subclasses should implement their own reward function.
        
        Args:
            env: GraphEnvironment object
            action: Optional target node to move to (or None for current node observation)
            **kwargs: Additional arguments for reward computation (subclass-specific)
        
        Returns:
            Reward value
        """
        pass

    @abstractmethod
    def act(self, env: "GraphEnvironment") -> dict[str, Any]:
        """
        Choose an action based on the current belief and policy.
        
        Action is selecting which neighbor node to move to (or stay at current node).
        
        Args:
            env: GraphEnvironment object
            
        Returns:
            Dictionary containing action information:
            - target_node: Node index to move to (or current_node to stay)
            - expected_reward: Expected reward for this action (if computed)
        """
        pass

    @abstractmethod
    def compute_policy(self, env: "GraphEnvironment", horizon: int) -> None:
        """
        Compute or update the agent's policy.
        
        Args:
            env: GraphEnvironment object
            horizon: Planning horizon (number of steps ahead)
        """
        pass

    def save_policy(self, filepath: str) -> None:
        """
        Save the current policy to disk.
        
        Args:
            filepath: Path to save the policy
        """
        if self._policy is None:
            raise ValueError("No policy to save. Call compute_policy() first.")
        # Default implementation: save as numpy array if policy is numpy-compatible
        # Subclasses should override for custom policy formats
        try:
            np.save(filepath, self._policy)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot save policy to {filepath}: {e}. Override save_policy() for custom formats.")

    def load_policy(self, filepath: str) -> None:
        """
        Load a policy from disk.
        
        Args:
            filepath: Path to load the policy from
        """
        try:
            self._policy = np.load(filepath, allow_pickle=True)
        except Exception as e:
            raise ValueError(f"Cannot load policy from {filepath}: {e}")


"""Random agent implementation for graph-based environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .graph_agent import GraphAgent
from .graph_belief import GraphBelief


@dataclass
class GraphRandomAgent(GraphAgent):
    """
    Random agent that selects random actions (movement to neighbor nodes).
    
    This agent serves as a baseline for comparison with planning agents.
    It randomly selects a neighbor node to move to at each time step.
    """

    seed: Optional[int] = None  # Optional seed for reproducibility
    
    def __post_init__(self):
        """Initialize random number generator."""
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)
        else:
            self._rng = np.random.default_rng()
    
    def act(self, env: "GraphEnvironment") -> dict[str, Any]:
        """
        Choose a random action (move to random neighbor node or stay).
        
        Args:
            env: GraphEnvironment object
            
        Returns:
            Dictionary containing action information:
            - target_node: Node index to move to (or current_node to stay)
            - expected_reward: Always 0.0 for random agent
        """
        # Get available nodes (neighbors + current node)
        available_nodes = list(self.get_reachable_nodes(env))
        
        if not available_nodes:
            # No neighbors, must stay
            return {
                "target_node": self.current_node,
                "expected_reward": 0.0,
            }
        
        # Randomly select a node
        target_node = available_nodes[self._rng.integers(len(available_nodes))]
        
        return {
            "target_node": target_node,
            "expected_reward": 0.0,  # Random agent doesn't compute expected reward
        }
    
    def compute_reward(
        self,
        env: "GraphEnvironment",
        action: Optional[int] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute reward for random agent.
        
        Random agent doesn't compute reward, so this returns 0.0.
        
        Args:
            env: GraphEnvironment object
            action: Optional target node (not used)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Reward value (always 0.0)
        """
        return 0.0
    
    def compute_policy(self, env: "GraphEnvironment", horizon: int) -> None:
        """
        Compute or update the agent's policy.
        
        Random agent doesn't compute a policy, so this is a no-op.
        
        Args:
            env: GraphEnvironment object
            horizon: Planning horizon (not used)
        """
        pass


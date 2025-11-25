"""Greedy agent implementation for graph-based environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .graph_agent import GraphAgent
from .graph_belief import GraphBelief


@dataclass
class GraphGreedyAgent(GraphAgent):
    """
    Greedy agent that selects the action with the highest immediate reward.
    
    Unlike Monte Carlo agents, this agent does not perform lookahead planning.
    It evaluates all available neighbor nodes and selects the one with the highest
    immediate reward (information gain + event value).
    """

    w_h: float = 1.0  # Weight for information gain term
    w_v: float = 1.0  # Weight for event detection value term
    event_utility: dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 1.0})
    
    def act(self, env: "GraphEnvironment") -> dict[str, Any]:
        """
        Choose the action with the highest immediate reward.
        
        Evaluates all available neighbor nodes and selects the one with the highest
        immediate reward (information gain + event value).
        
        Args:
            env: GraphEnvironment object
            
        Returns:
            Dictionary containing action information:
            - target_node: Node index to move to (or current_node to stay)
            - expected_reward: Immediate reward for selected action
        """
        # Get available nodes (neighbors + current node)
        available_nodes = list(self.get_reachable_nodes(env))
        
        if not available_nodes:
            # No neighbors, must stay
            return {
                "target_node": self.current_node,
                "expected_reward": 0.0,
            }
        
        # Evaluate each available node and select the one with highest immediate reward
        best_node = None
        best_reward = float('-inf')
        
        for target_node in available_nodes:
            # Compute immediate reward for moving to this node
            reward = self.compute_reward(env, action=target_node)
            
            if reward > best_reward:
                best_reward = reward
                best_node = target_node
        
        # Fallback: if no valid node found, stay at current
        if best_node is None:
            best_node = self.current_node
            best_reward = self.compute_reward(env, action=best_node)
        
        return {
            "target_node": best_node,
            "expected_reward": best_reward,
        }
    
    def compute_reward(
        self,
        env: "GraphEnvironment",
        action: Optional[int] = None,
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
        - G(b_k) = H_prior(b_k) (information gain, entropy reduction)
        - f(x) is the normalized utility of observing state x
        
        Args:
            env: GraphEnvironment object
            action: Target node to move to and observe. If None, uses current node.
            w_h: Weight for information gain term. If None, uses self.w_h
            w_v: Weight for event detection value term. If None, uses self.w_v
            event_utility: Dictionary mapping state (0 or 1) to utility value.
                          If None, uses self.event_utility
        
        Returns:
            Reward value
        """
        if action is None:
            action = self.current_node
        
        if w_h is None:
            w_h = self.w_h
        if w_v is None:
            w_v = self.w_v
        if event_utility is None:
            event_utility = self.event_utility
        
        # Compute information gain for observed node
        info_gain = self.information_gain({action})
        
        # Compute expected event value for observed node
        event_value = self.expected_event_value({action}, event_utility)
        
        # Total reward
        reward = w_h * info_gain + w_v * event_value
        return float(reward)
    
    def compute_policy(self, env: "GraphEnvironment", horizon: int) -> None:
        """
        Compute or update the agent's policy.
        
        Greedy agent doesn't compute a policy ahead of time, so this is a no-op.
        
        Args:
            env: GraphEnvironment object
            horizon: Planning horizon (not used)
        """
        pass


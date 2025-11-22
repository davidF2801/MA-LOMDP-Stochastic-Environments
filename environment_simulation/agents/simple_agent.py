"""Simple concrete agent implementation for testing and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .agent import Agent
from .belief import Belief
from .trajectory import Trajectory


@dataclass
class SimpleAgent(Agent):
    """
    Simple concrete agent implementation.
    
    This agent can be used for visualization and testing when no specific
    planning policy is required. It implements abstract methods with stub
    implementations.
    """

    def act(self, step: int, lat_grid: np.ndarray, lon_grid: np.ndarray) -> dict[str, Any]:
        """
        Choose an action based on the current belief and policy.
        
        For SimpleAgent, this is a no-op that returns an empty action dict.
        
        Args:
            step: Current time step
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            
        Returns:
            Empty action dictionary
        """
        return {}

    def compute_policy(self, horizon: int, lat_grid: np.ndarray, lon_grid: np.ndarray) -> None:
        """
        Compute or update the agent's policy.
        
        For SimpleAgent, this is a no-op.
        
        Args:
            horizon: Planning horizon (number of steps ahead)
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
        """
        pass

    def compute_reward(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        step: int,
        observation_mask: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute reward for SimpleAgent.
        
        Returns 0.0 as SimpleAgent does not perform planning.
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            observation_mask: Optional (H, W) mask of cells to observe
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Reward value (always 0.0)
        """
        return 0.0


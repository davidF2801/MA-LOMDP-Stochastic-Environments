"""Replay environment for deterministic playback of recorded environment evolution."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .environment import Environment, EventState2


class ReplayEnvironment:
    """
    Replay environment that plays back recorded states from a stochastic environment.
    
    This allows multiple agents to experience the exact same sequence of environmental
    changes for fair comparison. The environment evolution is recorded once and then
    replayed deterministically.
    
    Attributes:
        width: Grid width
        height: Grid height
        mode: Environment mode ('dbn2' or 'rsp')
        topology: Topology ('plane' or 'sphere')
        state_history: List of state arrays, one per time step
        time: Current time step (index into state_history)
        max_time: Maximum recorded time step
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        mode: str,
        topology: str,
        state_history: list[np.ndarray],
    ):
        """
        Initialize replay environment from recorded state history.
        
        Args:
            width: Grid width
            height: Grid height
            mode: Environment mode ('dbn2' or 'rsp')
            topology: Topology ('plane' or 'sphere')
            state_history: List of (H, W) state arrays, one per time step.
                          state_history[0] is the initial state (time=0),
                          state_history[t] is the state at time t.
        """
        self.width = width
        self.height = height
        self.mode = mode
        self.topology = topology
        
        # Validate state history
        if not state_history:
            raise ValueError("state_history must be non-empty")
        
        for i, state in enumerate(state_history):
            state = np.asarray(state, dtype=np.int8)
            if state.shape != (height, width):
                raise ValueError(
                    f"state_history[{i}] has shape {state.shape}, "
                    f"expected ({height}, {width})"
                )
            state_history[i] = state
        
        self.state_history = state_history
        self.max_time = len(state_history) - 1
        self.time = 0
        self.state = self.state_history[0].copy()
    
    @classmethod
    def record_from(cls, env: Environment, num_steps: int) -> ReplayEnvironment:
        """
        Record environment evolution and create a replay environment.
        
        Records the state at each step by running env.step() for num_steps iterations.
        The original environment is modified in place.
        
        Args:
            env: Environment to record from
            num_steps: Number of steps to record (not including initial state)
        
        Returns:
            ReplayEnvironment with recorded state history
        """
        # Record initial state
        state_history = [env.state.copy()]
        
        # Record state after each step
        for _ in range(num_steps):
            env.step()
            state_history.append(env.state.copy())
        
        return cls(
            width=env.width,
            height=env.height,
            mode=env.mode,
            topology=env.topology,
            state_history=state_history,
        )
    
    def reset(self, initial_events: Optional[int] = None) -> np.ndarray:
        """
        Reset to initial state (time=0).
        
        Args:
            initial_events: Ignored (for compatibility with Environment interface)
        
        Returns:
            Copy of initial state
        """
        self.time = 0
        self.state = self.state_history[0].copy()
        return self.state.copy()
    
    def step(self) -> np.ndarray:
        """
        Advance to next recorded state.
        
        Returns:
            Copy of current state after stepping
        
        Raises:
            IndexError: If already at maximum recorded time
        """
        if self.time >= self.max_time:
            raise IndexError(
                f"Cannot step beyond recorded time {self.max_time}. "
                f"Current time: {self.time}"
            )
        
        self.time += 1
        self.state = self.state_history[self.time].copy()
        return self.state.copy()
    
    def get_state(self) -> np.ndarray:
        """
        Get current state.
        
        Returns:
            Copy of current state
        """
        return self.state.copy()
    
    def count_active(self) -> int:
        """
        Count active events in current state.
        
        Returns:
            Number of active events (cells with value == EVENT_PRESENT)
        """
        return int(self.state.sum())
    
    def seed(self, seed: int) -> None:
        """
        No-op for replay environment (states are deterministic).
        
        Args:
            seed: Ignored (for compatibility with Environment interface)
        """
        pass
    
    def transition_probability(
        self, y: int, x: int, current_state: int, active_neighbors: int
    ) -> float:
        """
        Compute transition probability (for belief updates).
        
        This method is not meaningful for replay environments, but is provided
        for compatibility. In practice, agents should use the original environment's
        transition_probability method.
        
        Args:
            y: Cell y coordinate
            x: Cell x coordinate
            current_state: Current state (0 or 1)
            active_neighbors: Number of active neighbors
        
        Returns:
            Transition probability (always 0.0 for replay - not meaningful)
        """
        # This is not meaningful for replay environments
        # Agents should use the original environment's transition kernel
        return 0.0
    
    def _iter_neighbors(self, y: int, x: int):
        """
        Iterate over neighbors (for topology handling).
        
        This method matches Environment's interface but is not used for replay.
        Agents should use the original environment's _iter_neighbors method
        for belief updates.
        
        Args:
            y: Cell y coordinate
            x: Cell x coordinate
        
        Yields:
            (ny, nx) neighbor coordinates
        """
        # Delegate to the same logic as Environment for topology handling
        H, W = self.height, self.width
        seen = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                
                ny = y + dy
                nx = x + dx
                
                if self.topology == "sphere":
                    # Wrap longitude
                    if nx < 0:
                        nx = W - 1
                    elif nx >= W:
                        nx = 0
                    # Wrap latitude (pole crossing)
                    if ny < 0:
                        ny = 0
                        nx = (W // 2 + nx) % W  # Flip to opposite side
                    elif ny >= H:
                        ny = H - 1
                        nx = (W // 2 + nx) % W  # Flip to opposite side
                
                if 0 <= ny < H and 0 <= nx < W:
                    key = (ny, nx)
                    if key not in seen:
                        seen.add(key)
                        yield key


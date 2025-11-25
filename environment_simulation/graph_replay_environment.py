"""Replay environment for graph-based environments."""

from __future__ import annotations

from typing import Optional

import numpy as np


class GraphReplayEnvironment:
    """
    Replay environment that plays back recorded states from a stochastic graph environment.
    
    This allows multiple agents to experience the exact same sequence of environmental
    changes for fair comparison. The environment evolution is recorded once and then
    replayed deterministically.
    
    Attributes:
        num_nodes: Number of nodes in the graph
        adjacency_list: Adjacency list (dict) of the graph
        edges: List of edges in the graph
        mode: Environment mode ('dbn2' or 'rsp')
        state_history: List of state arrays, one per time step
        time: Current time step (index into state_history)
        max_time: Maximum recorded time step
    """
    
    def __init__(
        self,
        num_nodes: int,
        adjacency_list: dict[int, list[int]],
        edges: list[tuple[int, int]],
        mode: str,
        state_history: list[np.ndarray],
        width: Optional[int] = None,
        height: Optional[int] = None,
        node_positions: Optional[dict[int, tuple[int, int]]] = None,
        original_env: Optional["GraphEnvironment"] = None,
    ):
        """
        Initialize replay environment from recorded state history.
        
        Args:
            num_nodes: Number of nodes in the graph
            adjacency_list: Adjacency list (dict mapping node -> list of neighbors)
            edges: List of edges in the graph
            mode: Environment mode ('dbn2' or 'rsp')
            state_history: List of (num_nodes,) state arrays, one per time step.
                          state_history[0] is the initial state (time=0),
                          state_history[t] is the state at time t.
            width: Grid width (optional, for visualization)
            height: Grid height (optional, for visualization)
            node_positions: Optional dict mapping node index to (y, x) grid coordinates
        """
        self.num_nodes = num_nodes
        self.adjacency_list = {k: v.copy() for k, v in adjacency_list.items()}
        self.edges = edges.copy()
        self.mode = mode
        
        # Grid dimensions (for visualization compatibility)
        if width is None or height is None:
            # Infer from num_nodes (assume square grid)
            grid_size = int(np.sqrt(num_nodes))
            self.width = grid_size
            self.height = grid_size
        else:
            self.width = width
            self.height = height
        
        # Node positions for visualization
        if node_positions is None:
            # Generate default positions (grid layout)
            self.node_positions: dict[int, tuple[int, int]] = {}
            for i in range(num_nodes):
                y = i // self.width
                x = i % self.width
                self.node_positions[i] = (y, x)
        else:
            self.node_positions = node_positions.copy()
        
        # Validate state history
        if not state_history:
            raise ValueError("state_history must be non-empty")
        
        for i, state in enumerate(state_history):
            state = np.asarray(state, dtype=np.int8)
            if state.shape != (num_nodes,):
                raise ValueError(
                    f"state_history[{i}] has shape {state.shape}, "
                    f"expected ({num_nodes},)"
                )
            state_history[i] = state
        
        self.state_history = state_history
        self.max_time = len(state_history) - 1
        self.time = 0
        self.state = self.state_history[0].copy()
        
        # Store reference to original environment for transition kernel queries
        # This allows the replay environment to delegate transition probability calls
        self.original_env = original_env
    
    @classmethod
    def record_from(cls, env: "GraphEnvironment", num_steps: int, original_env: Optional["GraphEnvironment"] = None) -> "GraphReplayEnvironment":
        """
        Record environment evolution and create a replay environment.
        
        Records the state at each step by running env.step() for num_steps iterations.
        The original environment is modified in place.
        
        Args:
            env: GraphEnvironment to record from (will be modified)
            num_steps: Number of steps to record (not including initial state)
            original_env: Optional reference to original environment for transition kernel queries.
                         If None, a new environment with same parameters will need to be provided
                         separately for belief updates.
        
        Returns:
            GraphReplayEnvironment with recorded state history
        """
        # Record initial state
        state_history = [env.get_state().copy()]
        
        # Record state after each step
        for _ in range(num_steps):
            env.step()
            state_history.append(env.get_state().copy())
        
        return cls(
            num_nodes=env.num_nodes,
            adjacency_list=env.adjacency_list,
            edges=env.edges,
            mode=env.mode,
            state_history=state_history,
            width=env.width,
            height=env.height,
            node_positions=env.node_positions if hasattr(env, 'node_positions') else None,
            original_env=original_env,
        )
    
    def reset(self, initial_events: Optional[int] = None) -> np.ndarray:
        """
        Reset to initial state (time=0).
        
        Args:
            initial_events: Ignored (for compatibility with GraphEnvironment interface)
        
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
            Number of active events (nodes with value == 1)
        """
        return int(self.state.sum())
    
    def get_neighbors(self, node_idx: int) -> list[int]:
        """
        Get neighbors of a node (for compatibility with GraphEnvironment interface).
        
        Args:
            node_idx: Node index
            
        Returns:
            List of neighbor node indices
        """
        return self.adjacency_list.get(node_idx, []).copy()
    
    def get_node_state(self, node: int) -> int:
        """
        Get the state of a specific node.
        
        Args:
            node: Node index
            
        Returns:
            State value (0 or 1) of the node
        """
        return int(self.state[node])
    
    def seed(self, seed: int) -> None:
        """
        No-op for replay environment (states are deterministic).
        
        Args:
            seed: Ignored (for compatibility with GraphEnvironment interface)
        """
        pass
    
    def transition_probability(
        self, node_idx: int, current_state: int, active_neighbors: int
    ) -> float:
        """
        Compute transition probability (for belief updates).
        
        Delegates to the original environment if available. If no original environment
        is stored, raises an error indicating that transition_env should be used instead.
        
        Args:
            node_idx: Node index
            current_state: Current state (0 or 1)
            active_neighbors: Number of active neighbors
            
        Returns:
            Transition probability from the original environment
            
        Raises:
            AttributeError: If no original environment is stored
        """
        if self.original_env is not None:
            return self.original_env.transition_probability(node_idx, current_state, active_neighbors)
        else:
            raise AttributeError(
                "GraphReplayEnvironment.transition_probability requires an original environment. "
                "Either provide original_env when creating the replay environment, "
                "or use transition_env parameter in agent belief updates."
            )
    
    def transition_probability_full(
        self, 
        node: int, 
        next_state: int, 
        current_state: int, 
        neighbor_states: tuple[int, ...],
        num_states: int = 2
    ) -> float:
        """
        Compute transition probability with full neighbor configuration (for belief updates).
        
        Delegates to the original environment if available. If no original environment
        is stored, raises an error indicating that transition_env should be used instead.
        
        Args:
            node: Node index
            next_state: Next state (0 to M-1)
            current_state: Current state of node (0 to M-1)
            neighbor_states: Tuple of neighbor states (0 to M-1 for each neighbor)
            num_states: Number of possible states (M)
            
        Returns:
            Transition probability from the original environment
            
        Raises:
            AttributeError: If no original environment is stored
        """
        if self.original_env is not None:
            return self.original_env.transition_probability_full(
                node, next_state, current_state, neighbor_states, num_states
            )
        else:
            raise AttributeError(
                "GraphReplayEnvironment.transition_probability_full requires an original environment. "
                "Either provide original_env when creating the replay environment, "
                "or use transition_env parameter in agent belief updates."
            )


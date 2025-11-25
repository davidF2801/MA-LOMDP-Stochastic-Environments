# graph_environment.py
# 2D graph-based wildfire-like environment (ground truth) with local kernel contagion.
# Nodes represent cells, edges represent connectivity.
# Only depends on numpy.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
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


# Numba-optimized transition probability computation for RSP
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
    num_states: int,
) -> float:
    """
    Numba-optimized RSP transition probability computation.
    
    Args:
        next_state: Next state (0 or 1)
        current_state: Current state (0 or 1)
        active_neighbors: Number of active neighbors (state 1)
        num_neighbors: Total number of neighbors
        lam: Ignition intensity
        beta0: Spontaneous ignition rate
        alpha: Contagion strength
        delta: Persistence probability
        num_states: Number of states (should be 2 for binary)
        
    Returns:
        Transition probability P(x'=next_state | current_state, active_neighbors)
    """
    if num_states == 2:
        # Normalize active neighbors by total neighbors (0 to 1)
        norm_active = active_neighbors / num_neighbors if num_neighbors > 0 else 0.0
        
        # Contagion term: exponential form
        contagion = 1.0 - np.exp(-alpha * norm_active)
        
        if current_state == 0:  # NO_EVENT → {NO_EVENT, EVENT}
            # Ignition probability: exponential form
            p_event = 1.0 - np.exp(-(beta0 + lam + contagion))
            return p_event if next_state == 1 else (1.0 - p_event)
        else:  # current_state == 1: EVENT → {NO_EVENT, EVENT}
            # Persistence: P(EVENT→EVENT) = δ, P(EVENT→NO_EVENT) = 1-δ
            mu = 1.0 - delta  # extinction
            return delta if next_state == 1 else mu
    else:
        # For M > 2, simplified version (not fully implemented)
        return 0.0


# Numba-optimized transition probability computation for DBN-2
@njit(cache=True)
def _dbn2_transition_prob_numba(
    next_state: int,
    current_state: int,
    active_neighbors: int,
    birth_rate: float,
    death_rate: float,
    neighbor_influence: float,
) -> float:
    """
    Numba-optimized DBN-2 transition probability computation.
    """
    if next_state == 1:
        if current_state == 1:
            return max(0.0, min(1.0, 1.0 - death_rate))
        else:
            p_event = birth_rate + neighbor_influence * active_neighbors
            return max(0.0, min(1.0, p_event))
    else:  # next_state == 0
        if current_state == 1:
            return max(0.0, min(1.0, death_rate))
        else:
            p_event = birth_rate + neighbor_influence * active_neighbors
            return max(0.0, min(1.0, 1.0 - p_event))


class EventState2:
    NO_EVENT = 0
    EVENT_PRESENT = 1


@dataclass
class RSPParams:
    lam: float = 0.05     # local ignition intensity baseline
    beta0: float = 0.00   # spontaneous/background birth when no neighbors
    alpha: float = 0.05   # contagion per active neighbor
    delta: float = 0.90   # persistence if active (P(1→1))


class GraphEnvironment:
    """
    2D graph-based binary event environment.
    
    Nodes represent cells (like grid cells but with graph connectivity).
    Edges represent connections between cells.
    Uses the same local kernel contagion dynamics as the grid-based environment.
    
    Args:
        num_nodes: Number of nodes (cells) in the graph
        edges: List of (node1, node2) tuples representing undirected edges, or None to auto-generate
        mode: 'dbn2' or 'rsp'
        birth_rate, death_rate, neighbor_influence: DBN-2 params
        rsp_params: RSPParams
        ignition_map: optional (num_nodes,) float array (λ-map) for RSP; overrides lam baseline
        beta0_map, alpha_map, persistence_map: optional per-node overrides for RSP parameters
        blocked_mask: optional boolean (num_nodes,) mask of nodes that can never ignite
        seed: RNG seed
        transition_table: optional (512,) transition probability table for viirs_table mode
    """
    def __init__(
        self,
        num_nodes: int,
        edges: Optional[list[tuple[int, int]]] = None,
        mode: str = "dbn2",
        birth_rate: float = 0.02,
        death_rate: float = 0.05,
        neighbor_influence: float = 0.02,
        rsp_params: Optional[RSPParams] = None,
        ignition_map: Optional[np.ndarray] = None,
        beta0_map: Optional[np.ndarray] = None,
        alpha_map: Optional[np.ndarray] = None,
        persistence_map: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
        seed: int = 0,
        transition_table: Optional[np.ndarray] = None,
        # Grid-like generation parameters
        width: Optional[int] = None,
        height: Optional[int] = None,
        connect_8_neighbors: bool = True,  # If True, creates 8-connected grid-like graph
    ):
        assert mode in ("dbn2", "rsp", "viirs_table")
        self.num_nodes = num_nodes
        self.mode = mode

        # DBN-2 params
        self.birth_rate = float(birth_rate)
        self.death_rate = float(death_rate)
        self.neighbor_influence = float(neighbor_influence)

        # RSP params
        self.rsp = rsp_params if rsp_params is not None else RSPParams()
        
        # Grid dimensions (optional, for visualization and graph generation)
        if width is None or height is None:
            # Try to infer from num_nodes (assume square grid)
            grid_size = int(np.sqrt(num_nodes))
            self.width = grid_size
            self.height = grid_size
        else:
            self.width = width
            self.height = height
        
        # Generate or validate edges
        if edges is None:
            edges = self._generate_grid_graph(connect_8_neighbors)
        else:
            # Validate edges
            for u, v in edges:
                assert 0 <= u < num_nodes and 0 <= v < num_nodes, f"Invalid edge: ({u}, {v})"
        
        # Build adjacency structure
        self.adjacency_list: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        for u, v in edges:
            if v not in self.adjacency_list[u]:
                self.adjacency_list[u].append(v)
            if u not in self.adjacency_list[v]:
                self.adjacency_list[v].append(u)
        
        # Store edges for reference
        self.edges = edges
        
        # Node positions (for visualization) - stored as (y, x) grid coordinates
        self.node_positions: dict[int, tuple[int, int]] = {}
        for i in range(num_nodes):
            y = i // self.width
            x = i % self.width
            self.node_positions[i] = (y, x)
        
        # Parameter maps
        def _validate_map(name: str, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                assert arr.shape[0] == num_nodes, f"{name} must have length {num_nodes}"
            else:
                # Assume 2D grid format (height, width)
                assert arr.shape == (self.height, self.width), f"{name} must be (H,W) or ({num_nodes},)"
                # Flatten to 1D node-indexed array
                arr = arr.flatten()
            return arr

        self.ignition_map = _validate_map("ignition_map", ignition_map)
        self.beta0_map = _validate_map("beta0_map", beta0_map)
        self.alpha_map = _validate_map("alpha_map", alpha_map)
        self.persistence_map = _validate_map("persistence_map", persistence_map)

        if blocked_mask is not None:
            blocked_mask = np.asarray(blocked_mask, dtype=bool)
            if blocked_mask.ndim == 1:
                assert blocked_mask.shape[0] == num_nodes, "blocked_mask must have length num_nodes"
            else:
                # Assume 2D grid format
                assert blocked_mask.shape == (self.height, self.width), "blocked_mask must be (H,W) or (num_nodes,)"
                blocked_mask = blocked_mask.flatten()
        self.blocked_mask = blocked_mask

        self.rng = np.random.default_rng(seed)
        self.time = 0
        self.state = np.zeros(num_nodes, dtype=np.int8)  # ground truth: 0/1
        
        # For viirs_table mode: 512-pattern transition probability table
        self.transition_table = transition_table
        if mode == "viirs_table":
            if transition_table is None:
                raise ValueError("transition_table must be provided for mode='viirs_table'")
            if transition_table.shape != (512,):
                raise ValueError(f"transition_table must have shape (512,), got {transition_table.shape}")
    
    def _generate_grid_graph(self, connect_8_neighbors: bool) -> list[tuple[int, int]]:
        """Generate a grid-like graph structure."""
        edges = []
        for i in range(self.num_nodes):
            y = i // self.width
            x = i % self.width
            
            # Add edges to neighbors
            if connect_8_neighbors:
                # 8-connected neighbors
                neighbors = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1),
                ]
            else:
                # 4-connected neighbors
                neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    neighbor_idx = ny * self.width + nx
                    if neighbor_idx < self.num_nodes:
                        # Add edge (undirected, so only add once)
                        if i < neighbor_idx:
                            edges.append((i, neighbor_idx))
        
        return edges
    
    def get_neighbors(self, node: int) -> list[int]:
        """Get list of neighbor nodes."""
        return self.adjacency_list.get(node, [])
    
    def reset(self, initial_events: int = 1) -> np.ndarray:
        """Clear state and initialize `initial_events` random active nodes."""
        self.state.fill(EventState2.NO_EVENT)
        available = None
        if self.blocked_mask is not None:
            available = np.argwhere(~self.blocked_mask).flatten()
            if available.size == 0:
                return self.state.copy()

        for _ in range(int(initial_events)):
            if available is not None:
                idx = self.rng.integers(0, len(available))
                node = available[idx]
            else:
                node = self.rng.integers(0, self.num_nodes)
            if self.blocked_mask is not None and self.blocked_mask[node]:
                continue
            self.state[node] = EventState2.EVENT_PRESENT
        self.time = 0
        return self.state.copy()

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def get_state(self) -> np.ndarray:
        return self.state.copy()
    
    def get_node_state(self, node: int) -> int:
        """Get state of a specific node."""
        return int(self.state[node])
    
    def count_active(self) -> int:
        return int(self.state.sum())
    
    def count_active_neighbors(self, node: int) -> int:
        """Count number of active neighbors for a node."""
        count = 0
        for neighbor in self.get_neighbors(node):
            count += int(self.state[neighbor] == EventState2.EVENT_PRESENT)
        return count
    
    def step(self) -> np.ndarray:
        """Update environment state by one time step."""
        if self.mode == "rsp":
            self._step_rsp()
        elif self.mode == "viirs_table":
            self._step_viirs_table()
        else:
            self._step_dbn2()
        self.time += 1
        return self.get_state()
    
    def _step_dbn2(self):
        """Update using DBN-2 dynamics."""
        old = self.state.copy()
        new = old.copy()

        for node in range(self.num_nodes):
            current = int(old[node])

            if self.blocked_mask is not None and self.blocked_mask[node]:
                new[node] = EventState2.NO_EVENT
                continue

            # Count active neighbors
            active = self.count_active_neighbors(node)

            if current == EventState2.EVENT_PRESENT:
                # survive with prob 1 - death_rate
                p_event = max(0.0, min(1.0, 1.0 - self.death_rate))
            else:
                # birth from background + neighbor influence
                p_event = self.birth_rate + self.neighbor_influence * active
                p_event = max(0.0, min(1.0, p_event))

            new[node] = EventState2.EVENT_PRESENT if self.rng.random() < p_event else EventState2.NO_EVENT

        self.state = new
    
    def _step_rsp(self):
        """Update using RSP dynamics."""
        old = self.state.copy()
        new = old.copy()

        for node in range(self.num_nodes):
            cur = int(old[node])

            if self.blocked_mask is not None and self.blocked_mask[node]:
                new[node] = 0
                continue

            # Count active neighbors
            active = self.count_active_neighbors(node)

            if cur == 1:
                delta = float(self.persistence_map[node]) if self.persistence_map is not None else self.rsp.delta
                p_event = max(0.0, min(1.0, delta))  # persistence
            else:
                lam = float(self.ignition_map[node]) if self.ignition_map is not None else self.rsp.lam
                beta0 = float(self.beta0_map[node]) if self.beta0_map is not None else self.rsp.beta0
                alpha = float(self.alpha_map[node]) if self.alpha_map is not None else self.rsp.alpha
                p_event = beta0 + lam + alpha * active
                p_event = max(0.0, min(1.0, p_event))

            new[node] = 1 if self.rng.random() < p_event else 0

        self.state = new
    
    def _step_viirs_table(self):
        """Update using VIIRS table mode (simplified for graph)."""
        # For graph mode, we approximate using neighbor count
        # This is a simplified version - full 9-bit pattern would require more structure
        old = self.state.copy()
        new = old.copy()

        for node in range(self.num_nodes):
            if self.blocked_mask is not None and self.blocked_mask[node]:
                new[node] = 0
                continue

            center = int(old[node])
            active_neighbors = self.count_active_neighbors(node)
            
            # Approximate pattern: use center bit + neighbor count
            # This is a simplification - ideally we'd encode full neighbor pattern
            # For now, we'll use a simplified transition based on center + neighbor count
            max_neighbors = len(self.get_neighbors(node))
            if max_neighbors == 0:
                # Isolated node - use average transition probability
                if self.transition_table is not None:
                    p_event = float(np.mean(self.transition_table))
                else:
                    p_event = 0.5
            else:
                # Approximate: find patterns with same center and similar neighbor count
                patterns_matching = []
                for pattern_idx in range(512):
                    center_bit = (pattern_idx >> 8) & 1
                    neighbor_count = sum((pattern_idx >> i) & 1 for i in range(8))
                    if center_bit == center and neighbor_count == active_neighbors:
                        patterns_matching.append(pattern_idx)
                
                if patterns_matching and self.transition_table is not None:
                    p_event = float(np.mean([self.transition_table[p] for p in patterns_matching]))
                else:
                    # Fallback
                    p_event = 0.5
            
            p_event = max(0.0, min(1.0, p_event))
            new[node] = 1 if self.rng.random() < p_event else 0

        self.state = new
    
    def transition_probability(self, node: int, current_state: int, active_neighbors: int) -> float:
        """
        Compute transition probability P(x'=1 | x_j, x_N(j)) for belief updates.
        
        Args:
            node: Node index
            current_state: Current state of node (0 or 1)
            active_neighbors: Number of active neighbors
            
        Returns:
            Probability P(x'=1 | current_state, active_neighbors)
        """
        if self.blocked_mask is not None and self.blocked_mask[node]:
            return 0.0
        
        if self.mode == "dbn2":
            if current_state == EventState2.EVENT_PRESENT:
                # survive with prob 1 - death_rate
                return max(0.0, min(1.0, 1.0 - self.death_rate))
            else:
                # birth from background + neighbor influence
                p_event = self.birth_rate + self.neighbor_influence * active_neighbors
                return max(0.0, min(1.0, p_event))
        elif self.mode == "rsp":
            if current_state == 1:
                delta = float(self.persistence_map[node]) if self.persistence_map is not None else self.rsp.delta
                return max(0.0, min(1.0, delta))
            else:
                lam = float(self.ignition_map[node]) if self.ignition_map is not None else self.rsp.lam
                beta0 = float(self.beta0_map[node]) if self.beta0_map is not None else self.rsp.beta0
                alpha = float(self.alpha_map[node]) if self.alpha_map is not None else self.rsp.alpha
                p_event = beta0 + lam + alpha * active_neighbors
                return max(0.0, min(1.0, p_event))
        else:  # viirs_table mode - simplified
            if self.transition_table is None:
                return 0.0
            # Approximation using neighbor count
            patterns_matching = []
            for pattern_idx in range(512):
                center_bit = (pattern_idx >> 8) & 1
                neighbor_count = sum((pattern_idx >> i) & 1 for i in range(8))
                if center_bit == current_state and neighbor_count == active_neighbors:
                    patterns_matching.append(pattern_idx)
            
            if patterns_matching:
                return float(np.clip(np.mean([self.transition_table[p] for p in patterns_matching]), 0.0, 1.0))
            else:
                return float(np.clip(np.mean(self.transition_table), 0.0, 1.0))
    
    def transition_probability_full(
        self, 
        node: int, 
        next_state: int, 
        current_state: int, 
        neighbor_states: tuple[int, ...],
        num_states: int = 2
    ) -> float:
        """
        Compute transition probability P(x'=s' | x_j, x_N(j)) for belief updates with full neighbor configuration.
        
        This is a more general version that accepts the full neighbor state configuration,
        matching the Julia implementation interface.
        
        Args:
            node: Node index
            next_state: Next state (0 to M-1)
            current_state: Current state of node (0 to M-1)
            neighbor_states: Tuple of neighbor states (0 to M-1 for each neighbor)
            num_states: Number of possible states (M)
            
        Returns:
            Probability P(x'=next_state | current_state, neighbor_states)
        """
        if self.blocked_mask is not None and self.blocked_mask[node]:
            return 1.0 if next_state == 0 else 0.0
        
        # Count active neighbors (state 1) from neighbor configuration
        # For RSP/DBN-2, only state 1 is considered "active"
        active_neighbors = sum(1 for ns in neighbor_states if ns == 1)
        
        if self.mode == "dbn2":
            # DBN-2: binary states only
            if NUMBA_AVAILABLE:
                return _dbn2_transition_prob_numba(
                    next_state, current_state, active_neighbors,
                    self.birth_rate, self.death_rate, self.neighbor_influence
                )
            else:
                # Fallback to pure Python
                if next_state == 1:
                    if current_state == EventState2.EVENT_PRESENT:
                        return max(0.0, min(1.0, 1.0 - self.death_rate))
                    else:
                        p_event = self.birth_rate + self.neighbor_influence * active_neighbors
                        return max(0.0, min(1.0, p_event))
                else:  # next_state == 0
                    if current_state == EventState2.EVENT_PRESENT:
                        return max(0.0, min(1.0, self.death_rate))
                    else:
                        p_event = self.birth_rate + self.neighbor_influence * active_neighbors
                        return max(0.0, min(1.0, 1.0 - p_event))
        
        elif self.mode == "rsp":
            # RSP: binary states only for now
            # Matches Julia implementation: get_transition_probability_rsp
            if num_states == 2:
                # Get cell-specific RSP parameters
                lam = float(self.ignition_map[node]) if self.ignition_map is not None else self.rsp.lam
                beta0 = float(self.beta0_map[node]) if self.beta0_map is not None else self.rsp.beta0
                alpha = float(self.alpha_map[node]) if self.alpha_map is not None else self.rsp.alpha
                delta = float(self.persistence_map[node]) if self.persistence_map is not None else self.rsp.delta
                
                # Use numba-optimized version if available
                num_neighbors = len(neighbor_states)
                if NUMBA_AVAILABLE:
                    return _rsp_transition_prob_numba(
                        next_state, current_state, active_neighbors, num_neighbors,
                        lam, beta0, alpha, delta, num_states
                    )
                else:
                    # Fallback to pure Python
                    norm_active = active_neighbors / num_neighbors if num_neighbors > 0 else 0.0
                    contagion = 1.0 - np.exp(-alpha * norm_active)
                    
                    if current_state == 0:  # NO_EVENT → {NO_EVENT, EVENT}
                        p_event = 1.0 - np.exp(-(beta0 + lam + contagion))
                        return float(p_event) if next_state == 1 else float(1.0 - p_event)
                    else:  # current_state == 1: EVENT → {NO_EVENT, EVENT}
                        mu = 1.0 - delta
                        return float(delta) if next_state == 1 else float(mu)
            else:
                # For M > 2, would need more general model
                # For now, assume only state 1 transitions
                if next_state == 1:
                    if current_state == 1:
                        delta = float(self.persistence_map[node]) if self.persistence_map is not None else self.rsp.delta
                        return max(0.0, min(1.0, delta))
                    else:
                        lam = float(self.ignition_map[node]) if self.ignition_map is not None else self.rsp.lam
                        beta0 = float(self.beta0_map[node]) if self.beta0_map is not None else self.rsp.beta0
                        alpha = float(self.alpha_map[node]) if self.alpha_map is not None else self.rsp.alpha
                        p_event = beta0 + lam + alpha * active_neighbors
                        return max(0.0, min(1.0, p_event))
                elif next_state == current_state:
                    # Persistence
                    if current_state == 1:
                        delta = float(self.persistence_map[node]) if self.persistence_map is not None else self.rsp.delta
                        return max(0.0, min(1.0, delta))
                    else:
                        lam = float(self.ignition_map[node]) if self.ignition_map is not None else self.rsp.lam
                        beta0 = float(self.beta0_map[node]) if self.beta0_map is not None else self.rsp.beta0
                        alpha = float(self.alpha_map[node]) if self.alpha_map is not None else self.rsp.alpha
                        p_event = beta0 + lam + alpha * active_neighbors
                        return max(0.0, min(1.0, 1.0 - p_event))
                else:
                    return 0.0
        
        else:  # viirs_table mode
            if self.transition_table is None:
                return 1.0 / num_states  # Uniform if no table
            # Use neighbor count approximation
            patterns_matching = []
            for pattern_idx in range(512):
                center_bit = (pattern_idx >> 8) & 1
                neighbor_count = sum((pattern_idx >> i) & 1 for i in range(8))
                if center_bit == current_state and neighbor_count == active_neighbors:
                    patterns_matching.append(pattern_idx)
            
            if patterns_matching:
                avg_prob = float(np.clip(np.mean([self.transition_table[p] for p in patterns_matching]), 0.0, 1.0))
                return avg_prob if next_state == 1 else (1.0 - avg_prob)
            else:
                return 1.0 / num_states  # Uniform if no match
    
    def to_grid_state(self) -> np.ndarray:
        """Convert node-based state to 2D grid for visualization."""
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        for node in range(self.num_nodes):
            y, x = self.node_positions[node]
            grid[y, x] = self.state[node]
        return grid


if __name__ == "__main__":
    # Test graph environment
    num_nodes = 100  # 10x10 grid
    env = GraphEnvironment(
        num_nodes=num_nodes,
        mode="rsp",
        rsp_params=RSPParams(lam=0.02, beta0=0.001, alpha=0.03, delta=0.92),
        seed=42,
        width=10,
        height=10,
    )
    env.reset(initial_events=5)
    print(f"t={env.time:02d}, active={env.count_active()}")
    for _ in range(5):
        env.step()
        print(f"t={env.time:02d}, active={env.count_active()}")


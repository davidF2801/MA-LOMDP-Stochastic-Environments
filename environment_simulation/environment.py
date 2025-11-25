# environment.py
# 2D wildfire-like environment (ground truth) with DBN-2 and RSP update rules.
# Only depends on numpy.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


class EventState2:
    NO_EVENT = 0
    EVENT_PRESENT = 1


@dataclass
class RSPParams:
    lam: float = 0.05     # local ignition intensity baseline
    beta0: float = 0.00   # spontaneous/background birth when no neighbors
    alpha: float = 0.05   # contagion per active neighbor
    delta: float = 0.90   # persistence if active (P(1→1))


class Environment:
    """
    2D binary event map updated by:
      - mode='dbn2': birth/death + neighbor influence
      - mode='rsp' : Díaz–Avalos random spread process

    Args:
        width, height: grid size (latitude × longitude)
        mode: 'dbn2' or 'rsp'
        topology: 'plane' for bounded grid, 'sphere' for wrap-around Earth-like surface
        birth_rate, death_rate, neighbor_influence: DBN-2 params
        rsp_params: RSPParams
        ignition_map: optional (H,W) float array (λ-map) for RSP; overrides lam baseline
        beta0_map, alpha_map, persistence_map: optional per-cell overrides for RSP parameters
        blocked_mask: optional boolean (H,W) mask of cells that can never ignite
        seed: RNG seed
    """
    def __init__(
        self,
        width: int,
        height: int,
        mode: str = "dbn2",
        topology: str = "plane",
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
    ):
        assert mode in ("dbn2", "rsp", "viirs_table")
        assert topology in ("plane", "sphere"), "topology must be 'plane' or 'sphere'"
        self.width = width
        self.height = height
        self.mode = mode
        self.topology = topology

        # DBN-2 params
        self.birth_rate = float(birth_rate)
        self.death_rate = float(death_rate)
        self.neighbor_influence = float(neighbor_influence)

        # RSP params
        self.rsp = rsp_params if rsp_params is not None else RSPParams()
        def _validate_map(name: str, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=float)
            assert arr.shape == (height, width), f"{name} must be (H,W)"
            return arr

        self.ignition_map = _validate_map("ignition_map", ignition_map)
        self.beta0_map = _validate_map("beta0_map", beta0_map)
        self.alpha_map = _validate_map("alpha_map", alpha_map)
        self.persistence_map = _validate_map("persistence_map", persistence_map)

        if blocked_mask is not None:
            blocked_mask = np.asarray(blocked_mask, dtype=bool)
            assert blocked_mask.shape == (height, width), "blocked_mask must be (H,W)"
        self.blocked_mask = blocked_mask

        self.rng = np.random.default_rng(seed)
        self.time = 0
        self.state = np.zeros((height, width), dtype=np.int8)  # ground truth: 0/1
        
        # For viirs_table mode: 512-pattern transition probability table
        self.transition_table = transition_table
        if mode == "viirs_table":
            if transition_table is None:
                raise ValueError("transition_table must be provided for mode='viirs_table'")
            if transition_table.shape != (512,):
                raise ValueError(f"transition_table must have shape (512,), got {transition_table.shape}")
            # Define neighbor offsets in the same order as p_table_compute_from_data.py
            # Order: (-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)
            self.neighbor_offsets_8 = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]

    # ---------- utilities ----------
    def reset(self, initial_events: int = 1) -> np.ndarray:
        """Clear grid and drop `initial_events` random active cells."""
        self.state.fill(EventState2.NO_EVENT)
        available = None
        if self.blocked_mask is not None:
            available = np.argwhere(~self.blocked_mask)
            if available.size == 0:
                return self.state.copy()

        for _ in range(int(initial_events)):
            if available is not None:
                idx = self.rng.integers(0, len(available))
                y, x = available[idx]
            else:
                y = self.rng.integers(0, self.height)
                x = self.rng.integers(0, self.width)
            if self.blocked_mask is not None and self.blocked_mask[y, x]:
                continue
            self.state[y, x] = EventState2.EVENT_PRESENT
        self.time = 0
        return self.state.copy()

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def count_active(self) -> int:
        return int(self.state.sum())

    # ---------- topology helpers ----------
    def _iter_neighbors(self, y: int, x: int):
        H, W = self.state.shape
        seen = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny = y + dy
                nx = x + dx

                if self.topology == "sphere":
                    # Longitudinal wrap
                    nx %= W
                    # Crossing a pole flips longitude by 180 degrees
                    if ny < 0:
                        ny = 0
                        nx = (nx + W // 2) % W
                    elif ny >= H:
                        ny = H - 1
                        nx = (nx + W // 2) % W
                else:
                    if not (0 <= ny < H and 0 <= nx < W):
                        continue

                key = (ny, nx)
                if key not in seen:
                    seen.add(key)
                    yield key

    # ---------- core update ----------
    def step(self) -> np.ndarray:
        if self.mode == "rsp":
            self._step_rsp()
        elif self.mode == "viirs_table":
            self._step_viirs_table()
        else:
            self._step_dbn2()
        self.time += 1
        return self.get_state()

    # ---------- DBN-2 ----------
    def _step_dbn2(self):
        H, W = self.state.shape
        old = self.state
        new = old.copy()

        for y in range(H):
            for x in range(W):
                current = int(old[y, x])

                if self.blocked_mask is not None and self.blocked_mask[y, x]:
                    new[y, x] = EventState2.NO_EVENT
                    continue

                # count active neighbors according to topology
                active = 0
                for ny, nx in self._iter_neighbors(y, x):
                    active += int(old[ny, nx] == EventState2.EVENT_PRESENT)

                if current == EventState2.EVENT_PRESENT:
                    # survive with prob 1 - death_rate
                    p_event = max(0.0, min(1.0, 1.0 - self.death_rate))
                else:
                    # birth from background + neighbor influence
                    p_event = self.birth_rate + self.neighbor_influence * active
                    p_event = max(0.0, min(1.0, p_event))

                new[y, x] = EventState2.EVENT_PRESENT if self.rng.random() < p_event else EventState2.NO_EVENT

        self.state = new

    # ---------- transition kernel for belief updates ----------
    def transition_probability(self, y: int, x: int, current_state: int, active_neighbors: int) -> float:
        """
        Compute transition probability P(x'=1 | x_j, x_N(j)) for belief updates.
        
        Args:
            y, x: Cell coordinates
            current_state: Current state of cell (0 or 1)
            active_neighbors: Number of active neighbors (0-8)
            
        Returns:
            Probability P(x'=1 | current_state, active_neighbors)
        """
        if self.blocked_mask is not None and self.blocked_mask[y, x]:
            return 0.0
        
        if self.mode == "dbn2":
            if current_state == EventState2.EVENT_PRESENT:
                # survive with prob 1 - death_rate
                return max(0.0, min(1.0, 1.0 - self.death_rate))
            else:
                # birth from background + neighbor influence
                p_event = self.birth_rate + self.neighbor_influence * active_neighbors
                return max(0.0, min(1.0, p_event))
        else:  # RSP mode
            lam_map = self.ignition_map if self.ignition_map is not None else None
            if current_state == 1:
                delta = float(self.persistence_map[y, x]) if self.persistence_map is not None else self.rsp.delta
                return max(0.0, min(1.0, delta))
            else:
                lam = float(lam_map[y, x]) if lam_map is not None else self.rsp.lam
                beta0 = float(self.beta0_map[y, x]) if self.beta0_map is not None else self.rsp.beta0
                alpha = float(self.alpha_map[y, x]) if self.alpha_map is not None else self.rsp.alpha
                p_event = beta0 + lam + alpha * active_neighbors
                return max(0.0, min(1.0, p_event))

    # ---------- RSP ----------
    def _step_rsp(self):
        H, W = self.state.shape
        old = self.state
        new = old.copy()

        lam_map = self.ignition_map if self.ignition_map is not None else None

        for y in range(H):
            for x in range(W):
                cur = int(old[y, x])

                if self.blocked_mask is not None and self.blocked_mask[y, x]:
                    new[y, x] = 0
                    continue

                # count active neighbors
                active = 0
                for ny, nx in self._iter_neighbors(y, x):
                    active += int(old[ny, nx] == 1)

                if cur == 1:
                    delta = float(self.persistence_map[y, x]) if self.persistence_map is not None else self.rsp.delta
                    p_event = max(0.0, min(1.0, delta))  # persistence
                else:
                    lam = float(lam_map[y, x]) if lam_map is not None else self.rsp.lam
                    beta0 = float(self.beta0_map[y, x]) if self.beta0_map is not None else self.rsp.beta0
                    alpha = float(self.alpha_map[y, x]) if self.alpha_map is not None else self.rsp.alpha
                    p_event = beta0 + lam + alpha * active
                    p_event = max(0.0, min(1.0, p_event))

                new[y, x] = 1 if self.rng.random() < p_event else 0

        self.state = new

    # ---------- VIIRS table mode ----------
    def _encode_local_pattern(self, center: int, neighbors: list[int]) -> int:
        """Encode a 9-bit pattern (center + 8 neighbors) into an integer in [0, 511]."""
        bits = [int(center)] + [int(b) for b in neighbors]
        r = 0
        for b in bits:
            r = (r << 1) | b
        return r

    def _step_viirs_table(self):
        """Step using the learned 512-pattern transition probability table."""
        H, W = self.state.shape
        old = self.state
        new = old.copy()

        for y in range(H):
            for x in range(W):
                if self.blocked_mask is not None and self.blocked_mask[y, x]:
                    new[y, x] = 0
                    continue

                center = int(old[y, x])
                
                # Get neighbors in the same order as p_table_compute_from_data.py
                neighbors = []
                for dy, dx in self.neighbor_offsets_8:
                    ny = y + dy
                    nx = x + dx
                    
                    if self.topology == "sphere":
                        # Handle wrap-around for sphere topology
                        nx %= W
                        if ny < 0:
                            ny = 0
                            nx = (nx + W // 2) % W
                        elif ny >= H:
                            ny = H - 1
                            nx = (nx + W // 2) % W
                        
                        if 0 <= ny < H and 0 <= nx < W:
                            neighbors.append(int(old[ny, nx]))
                        else:
                            neighbors.append(0)
                    else:
                        # Plane topology: outside boundaries = 0
                        if 0 <= ny < H and 0 <= nx < W:
                            neighbors.append(int(old[ny, nx]))
                        else:
                            neighbors.append(0)
                
                # Encode pattern and look up transition probability
                pattern = self._encode_local_pattern(center, neighbors)
                p_event = float(self.transition_table[pattern])
                p_event = max(0.0, min(1.0, p_event))  # Clip to [0, 1]
                
                new[y, x] = 1 if self.rng.random() < p_event else 0

        self.state = new

    # ---------- transition kernel for belief updates ----------
    def transition_probability(self, y: int, x: int, current_state: int, active_neighbors: int) -> float:
        """
        Compute transition probability P(x'=1 | x_j, x_N(j)) for belief updates.
        
        Args:
            y, x: Cell coordinates
            current_state: Current state of cell (0 or 1)
            active_neighbors: Number of active neighbors (0-8)
            
        Returns:
            Probability P(x'=1 | current_state, active_neighbors)
        
        Note: For viirs_table mode, this is an approximation using only neighbor count.
        For exact probabilities, the full 9-bit pattern would be needed.
        """
        if self.blocked_mask is not None and self.blocked_mask[y, x]:
            return 0.0
        
        if self.mode == "viirs_table":
            # Approximation: use average probability over patterns with given center state and neighbor count
            # This is a simplified version - ideally we'd need the full pattern
            if self.transition_table is None:
                return 0.0
            
            # For now, approximate by averaging over all patterns with same center and similar neighbor count
            # This is not perfect but provides a reasonable approximation for belief updates
            patterns_with_state = []
            for pattern_idx in range(512):
                # Extract center bit (most significant bit)
                center_bit = (pattern_idx >> 8) & 1
                if center_bit == current_state:
                    # Count active neighbors
                    neighbor_count = sum((pattern_idx >> i) & 1 for i in range(8))
                    if neighbor_count == active_neighbors:
                        patterns_with_state.append(pattern_idx)
            
            if patterns_with_state:
                avg_prob = np.mean([self.transition_table[p] for p in patterns_with_state])
                return float(np.clip(avg_prob, 0.0, 1.0))
            else:
                # Fallback: use global average
                return float(np.clip(np.mean(self.transition_table), 0.0, 1.0))
        
        if self.mode == "dbn2":
            if current_state == EventState2.EVENT_PRESENT:
                # survive with prob 1 - death_rate
                return max(0.0, min(1.0, 1.0 - self.death_rate))
            else:
                # birth from background + neighbor influence
                p_event = self.birth_rate + self.neighbor_influence * active_neighbors
                return max(0.0, min(1.0, p_event))
        else:  # RSP mode
            lam_map = self.ignition_map if self.ignition_map is not None else None
            if current_state == 1:
                delta = float(self.persistence_map[y, x]) if self.persistence_map is not None else self.rsp.delta
                return max(0.0, min(1.0, delta))
            else:
                lam = float(lam_map[y, x]) if lam_map is not None else self.rsp.lam
                beta0 = float(self.beta0_map[y, x]) if self.beta0_map is not None else self.rsp.beta0
                alpha = float(self.alpha_map[y, x]) if self.alpha_map is not None else self.rsp.alpha
                p_event = beta0 + lam + alpha * active_neighbors
                return max(0.0, min(1.0, p_event))


if __name__ == "__main__":
    # DBN-2 example
    env = Environment(20, 12, mode="dbn2",
                      birth_rate=0.02, death_rate=0.05, neighbor_influence=0.03,
                      seed=42)
    env.reset(initial_events=10)
    for _ in range(5):
        env.step()
        print(f"t={env.time:02d}, active={env.count_active()}")

    # RSP example on spherical grid with heterogeneous parameters
    lam_map = np.linspace(0.01, 0.03, 12).reshape(-1, 1) * np.ones((12, 20))
    alpha_map = 0.02 + 0.01 * np.random.default_rng(0).random((12, 20))
    env2 = Environment(20, 12, mode="rsp", topology="sphere",
                       rsp_params=RSPParams(lam=0.02, beta0=0.001, alpha=0.03, delta=0.92),
                       ignition_map=lam_map,
                       alpha_map=alpha_map,
                       seed=0)
    env2.reset(initial_events=5)
    for _ in range(5):
        env2.step()
        print(f"[RSP] t={env2.time:02d}, active={env2.count_active()}")

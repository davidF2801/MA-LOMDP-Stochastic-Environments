"""Monte Carlo agent for planning with individual beliefs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .agent import Agent
from .belief import Belief
from .trajectory import Trajectory


@dataclass
class MonteCarloAgent(Agent):
    """
    Monte Carlo agent that plans using individual beliefs and no inter-agent communication.
    
    Uses Monte Carlo rollouts to estimate expected cumulative reward and selects actions
    that maximize this reward.
    """

    # Monte Carlo planning parameters
    num_rollouts: int = 50
    w_h: float = 1.0  # Weight for information gain term
    w_v: float = 1.0  # Weight for event detection value term
    event_utility: dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 1.0})
    discount_factor: float = 0.95
    max_observation_ratio: float = 0.2  # Maximum fraction of field of regard to observe (default 20%)
    
    def compute_reward(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        step: int,
        observation_points: Optional[set[tuple[float, float]]] = None,
        w_h: Optional[float] = None,
        w_v: Optional[float] = None,
        event_utility: Optional[dict[int, float]] = None,
    ) -> float:
        """
        Compute reward for observing cells at the current step.
        
        Implements the reward function:
        R(τ,b,a) = Σ_{k∈U_t} [w_h * G(b_k) + w_v * Σ_x b_k(x) f(x)]
        
        where:
        - U_t is the set of cells observed at time t
        - G(b_k) = H_prior(b_k) - H_post(b_k) (information gain)
        - H_post(b_k) = 0 for perfect observations
        - f(x) is the normalized utility of observing state x
        
        Args:
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            step: Current time step
            observation_points: Optional set of (lat, lon) tuples to observe.
                              If None, uses field of regard at current step.
            w_h: Weight for information gain term. If None, uses self.w_h
            w_v: Weight for event detection value term. If None, uses self.w_v
            event_utility: Dictionary mapping state (0 or 1) to utility value.
                          If None, uses self.event_utility
        
        Returns:
            Reward value
        """
        if observation_points is None:
            # Convert FOR mask to set of (lat, lon) points
            for_mask = self.coverage_mask(lat_grid, lon_grid, step)
            observation_points = self.mask_to_points(for_mask, lat_grid, lon_grid)
        else:
            observation_points = set(observation_points)
        
        if w_h is None:
            w_h = self.w_h
        if w_v is None:
            w_v = self.w_v
        if event_utility is None:
            event_utility = self.event_utility
        
        # Compute information gain for observed cells
        info_gain = self.information_gain(lat_grid, lon_grid, step, observation_points)
        
        # Compute expected event value for observed cells
        event_value = self.expected_event_value(
            lat_grid, lon_grid, step, event_utility, observation_points
        )
        
        # Total reward
        reward = w_h * info_gain + w_v * event_value
        return float(reward)
    
    def _sample_observation_from_belief(
        self, 
        observation_points: set[tuple[float, float]],
        rng: np.random.Generator
    ) -> dict[tuple[float, float], int]:
        """
        Sample a simulated observation from the current belief.
        
        For Monte Carlo rollouts, we simulate what the agent might observe by sampling
        from the belief distributions b_k(x) ONLY for cells k in the observation points
        (the field of view/action taken).
        
        For each (lat, lon) cell k in the observation points:
        - Sample x ~ Bernoulli(b_k(1)), where b_k(1) = P(E_k = 1)
        - This gives us a simulated observation o_k ∈ {0, 1}
        
        Args:
            observation_points: Set of (lat, lon) tuples to observe (field of view/action)
            rng: Random number generator
            
        Returns:
            Dictionary mapping (lat, lon) -> observed state (0 or 1) for cells in observation_points
        """
        observations_dict: dict[tuple[float, float], int] = {}
        
        # Sample from belief distribution only for cells in the observation points (field of view)
        for lat, lon in observation_points:
            # Get probability from belief (or use prior if not in belief)
            if self.belief.has_location(lat, lon):
                prob = self.belief.get_probability(lat, lon)
            else:
                prob = self.belief.prior_probability
            
            # Sample observation: x ~ Bernoulli(prob)
            sampled = rng.binomial(1, np.clip(prob, 0.0, 1.0))
            observations_dict[(lat, lon)] = int(sampled)
        
        return observations_dict

    def _rollout_trajectory(
        self,
        env: "Environment",  # type: ignore
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        start_step: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> float:
        """
        Perform a single Monte Carlo rollout to estimate cumulative reward.
        
        Simulates forward steps from the current belief state:
        1. Observe cells in field of regard (sampled from belief)
        2. Update belief with observations
        3. Compute reward
        4. Evolve belief using transition kernel
        5. Repeat for horizon steps
        
        Args:
            env: Environment object (for belief evolution)
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            start_step: Starting time step
            horizon: Number of steps to look ahead
            rng: Random number generator
            
        Returns:
            Cumulative discounted reward
        """
        from environment_simulation.environment import Environment
        from copy import deepcopy
        
        # Create a copy of the belief for this rollout
        rollout_belief = deepcopy(self.belief)
        original_belief = self.belief
        self.belief = rollout_belief
        
        total_reward = 0.0
        
        try:
            for h in range(horizon):
                step = start_step + h
                
                # Get observation mask (field of regard)
                observation_mask = self.coverage_mask(lat_grid, lon_grid, step)
                
                # Convert FOR mask to set of (lat, lon) points
                observation_points = set()
                for y in range(lat_grid.shape[0]):
                    for x in range(lon_grid.shape[1]):
                        if observation_mask[y, x]:
                            lat, lon = lat_grid[y, x], lon_grid[y, x]
                            observation_points.add((float(lat), float(lon)))
                
                # Compute reward BEFORE updating belief (uses prior entropy)
                step_reward = self.compute_reward(
                    lat_grid,
                    lon_grid,
                    step,
                    observation_points=observation_points,
                )
                
                total_reward += (self.discount_factor ** h) * step_reward
                
                # Simulate observation: sample from belief distributions b_k(x) for each cell k in FOR
                # For each cell k: sample x ~ Bernoulli(b_k(1)) to get simulated observation o_k
                observation = self._sample_observation_from_belief(observation_points, rng)
                
                # Update belief with observation: b_{t+1}(j, x') = δ[x' = o_j] for observed cells j
                rollout_belief.update_from_observation(observation_mask, observation, lat_grid, lon_grid)
                
                # Evolve belief for unobserved cells using transition kernel:
                # b_{t+1}^e(j,x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * b_t^e(j,x_j) * Π_{l∈N(j)} b_t^e(l,x_l)
                self.evolve_belief_with_environment(
                    env, lat_grid, lon_grid, step, observation_mask=observation_mask
                )
        finally:
            # Restore original belief
            self.belief = original_belief
        
        return total_reward

    def compute_policy(self, horizon: int, lat_grid: np.ndarray, lon_grid: np.ndarray) -> None:
        """
        Compute policy using Monte Carlo rollouts.
        
        Performs multiple rollouts from the current belief state to estimate
        expected cumulative reward. Stores the policy as rollout statistics.
        
        Args:
            horizon: Planning horizon (number of steps ahead)
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
        """
        from environment_simulation.environment import Environment
        
        # For Monte Carlo planning, we estimate expected cumulative reward
        # Since the agent has a fixed trajectory, the "policy" is implicit
        # We store the expected reward estimate for decision-making
        
        # Note: The agent needs access to the environment for belief evolution
        # This will be provided via act() or a separate method
        # For now, we'll just mark that policy computation is done
        self._policy = {
            "type": "monte_carlo",
            "horizon": horizon,
            "computed": True,
        }

    def _get_all_contiguous_subsets(
        self,
        field_mask: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        max_cells: Optional[int] = None,
        max_ratio: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> list[set[tuple[float, float]]]:
        """
        Get multiple contiguous subsets from the field of regard.
        
        Since the FOR is typically a single connected circular region, we generate multiple
        different contiguous subsets by starting from different seed points and growing
        connected regions up to max_cells.
        
        Returns a list of sets of (lat, lon) tuples, each representing a contiguous subset.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Calculate max cells based on ratio if not specified
        total_field_cells = np.sum(field_mask)
        if max_cells is None:
            if max_ratio is None:
                max_ratio = self.max_observation_ratio
            max_cells = max(1, int(total_field_cells * max_ratio))
        
        # Get all cells in the field of regard
        for_cells = []
        for y in range(field_mask.shape[0]):
            for x in range(field_mask.shape[1]):
                if field_mask[y, x]:
                    lat, lon = lat_grid[y, x], lon_grid[y, x]
                    for_cells.append((y, x, float(lat), float(lon)))
        
        if not for_cells:
            return []
        
        # If FOR is smaller than max_cells, return the entire FOR as a single subset
        if len(for_cells) <= max_cells:
            subset = set((lat, lon) for _, _, lat, lon in for_cells)
            return [subset]
        
        # Generate multiple contiguous subsets by starting from different seed points
        # Use BFS to grow connected regions from each seed
        H, W = field_mask.shape
        subsets = []
        num_seeds = min(10, len(for_cells))  # Try up to 10 different seed points
        
        # Sample seed points (can be random or spaced)
        seed_indices = rng.choice(len(for_cells), size=num_seeds, replace=False)
        
        def grow_region_from_seed(start_y: int, start_x: int, max_size: int) -> set[tuple[float, float]]:
            """Grow a connected region from a seed point using BFS."""
            if not field_mask[start_y, start_x]:
                return set()
            
            visited = np.zeros((H, W), dtype=bool)
            queue = [(start_y, start_x)]
            visited[start_y, start_x] = True
            region = set()
            
            # 8-connected neighbors
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            
            while queue and len(region) < max_size:
                cy, cx = queue.pop(0)
                lat, lon = lat_grid[cy, cx], lon_grid[cy, cx]
                region.add((float(lat), float(lon)))
                
                # Add neighbors that are in the FOR and not yet visited
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < H and 0 <= nx < W and 
                        field_mask[ny, nx] and 
                        not visited[ny, nx] and
                        len(region) < max_size):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            
            return region
        
        # Generate subsets from different seeds
        seen_subsets = set()  # Use frozenset to track unique subsets
        for seed_idx in seed_indices:
            seed_y, seed_x, seed_lat, seed_lon = for_cells[seed_idx]
            subset = grow_region_from_seed(seed_y, seed_x, max_cells)
            
            if subset and len(subset) >= 1:  # At least 1 cell
                # Convert to frozenset for comparison (order doesn't matter)
                subset_frozen = frozenset(subset)
                if subset_frozen not in seen_subsets:
                    seen_subsets.add(subset_frozen)
                    subsets.append(subset)
        
        # If we didn't get enough subsets, try some additional strategies
        if len(subsets) < 3:
            # Strategy 1: Try seeds from different regions (spatial distribution)
            # Divide FOR into a grid and pick one seed from each grid cell
            if len(for_cells) > 4:
                # Simple spatial clustering: pick seeds from corners/center
                corner_indices = [
                    0,  # First cell
                    len(for_cells) // 4,  # ~25%
                    len(for_cells) // 2,  # ~50%
                    3 * len(for_cells) // 4,  # ~75%
                    len(for_cells) - 1,  # Last cell
                ]
                for idx in corner_indices:
                    if idx < len(for_cells):
                        seed_y, seed_x, seed_lat, seed_lon = for_cells[idx]
                        subset = grow_region_from_seed(seed_y, seed_x, max_cells)
                        if subset:
                            subset_frozen = frozenset(subset)
                            if subset_frozen not in seen_subsets:
                                seen_subsets.add(subset_frozen)
                                subsets.append(subset)
                                if len(subsets) >= 5:  # Enough subsets
                                    break
        
        # Ensure we have at least one subset
        if not subsets:
            # Fallback: return a single subset from the center
            center_idx = len(for_cells) // 2
            seed_y, seed_x, seed_lat, seed_lon = for_cells[center_idx]
            subset = grow_region_from_seed(seed_y, seed_x, max_cells)
            if subset:
                subsets.append(subset)
            else:
                # Last resort: return a small subset from the first cell
                if for_cells:
                    seed_y, seed_x, seed_lat, seed_lon = for_cells[0]
                    subset = {(seed_lat, seed_lon)}
                    subsets.append(subset)
        
        return subsets

    def _select_contiguous_subset(
        self,
        field_mask: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        max_cells: Optional[int] = None,
        max_ratio: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> set[tuple[float, float]]:
        """
        Select a contiguous subset of cells from the field of regard.
        
        Uses a simple flood-fill approach to find connected components and
        selects one that fits within the maximum observation size.
        
        Args:
            field_mask: (H, W) boolean mask of cells in field of regard
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            max_cells: Maximum number of cells to observe. If None, uses max_ratio.
            max_ratio: Maximum fraction of field of regard to observe. If None, uses self.max_observation_ratio.
            rng: Random number generator for selection
            
        Returns:
            Set of (lat, lon) tuples representing selected contiguous cells to observe
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Calculate max cells based on ratio if not specified
        total_field_cells = np.sum(field_mask)
        if max_cells is None:
            if max_ratio is None:
                max_ratio = self.max_observation_ratio
            max_cells = max(1, int(total_field_cells * max_ratio))
        
        try:
            from scipy.ndimage import label
            use_scipy = True
        except ImportError:
            use_scipy = False
        
        if use_scipy:
            # Find connected components in the field of regard using scipy
            labeled, num_features = label(field_mask, structure=np.ones((3, 3), dtype=bool))
            
            if num_features == 0:
                return set()
            
            # Get sizes of each component
            components = []
            for i in range(1, num_features + 1):
                component_mask = (labeled == i)
                # Convert mask to set of (lat, lon) points
                component_points = set()
                for y in range(field_mask.shape[0]):
                    for x in range(field_mask.shape[1]):
                        if component_mask[y, x]:
                            lat, lon = lat_grid[y, x], lon_grid[y, x]
                            component_points.add((float(lat), float(lon)))
                size = len(component_points)
                components.append((component_points, size))
            
            # Filter by max_cells if specified
            if max_cells is not None:
                filtered = [points for points, s in components if s <= max_cells]
                if filtered:
                    # Return largest component that fits
                    return max(filtered, key=len)
                else:
                    # If none fit, return smallest component
                    smallest = min(components, key=lambda x: x[1])
                    return smallest[0]
            else:
                # Return largest component
                largest = max(components, key=lambda x: x[1])
                return largest[0]
        else:
            # Fallback: simple connected component using flood fill
            H, W = field_mask.shape
            visited = np.zeros_like(field_mask, dtype=bool)
            components = []
            
            def flood_fill(start_y, start_x):
                """Flood fill to find connected component."""
                if not field_mask[start_y, start_x] or visited[start_y, start_x]:
                    return []
                stack = [(start_y, start_x)]
                component = []
                while stack:
                    cy, cx = stack.pop()
                    if (not (0 <= cy < H and 0 <= cx < W) or 
                        not field_mask[cy, cx] or 
                        visited[cy, cx]):
                        continue
                    visited[cy, cx] = True
                    component.append((cy, cx))
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        stack.append((cy + dy, cx + dx))
                return component
            
            # Find all components
            for y in range(H):
                for x in range(W):
                    if field_mask[y, x] and not visited[y, x]:
                        component = flood_fill(y, x)
                        if component:
                            # Convert component to set of (lat, lon) points
                            component_points = set()
                            for cy, cx in component:
                                lat, lon = lat_grid[cy, cx], lon_grid[cy, cx]
                                component_points.add((float(lat), float(lon)))
                            components.append(component_points)
            
            if not components:
                return set()
            
            # Filter by max_cells and select largest
            if max_cells is not None:
                filtered = [c for c in components if len(c) <= max_cells]
                if filtered:
                    return max(filtered, key=len)
                else:
                    # If none fit, return smallest
                    return min(components, key=len)
            else:
                # Return largest component
                return max(components, key=len)

    def act(
        self,
        step: int,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        env: Optional["Environment"] = None,  # type: ignore
        horizon: Optional[int] = None,
        max_observation_cells: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Choose an action based on Monte Carlo planning.
        
        Evaluates different contiguous subsets within the field of regard using Monte Carlo
        rollouts and selects the one with the highest expected reward.
        
        Args:
            step: Current time step
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            env: Environment object (required for belief evolution in rollouts)
            horizon: Planning horizon for Monte Carlo rollouts
            max_observation_cells: Maximum number of cells to observe in a contiguous subset.
                                  If None, uses max_observation_ratio.
            
        Returns:
            Dictionary containing action information:
            - observation_points: Set of (lat, lon) tuples representing cells to observe
            - expected_reward: Expected cumulative reward for selected action
            - all_rewards: Dictionary mapping action indices to expected rewards (for debugging)
        """
        from environment_simulation.environment import Environment
        from copy import deepcopy
        
        # Get field of regard
        field_mask = self.coverage_mask(lat_grid, lon_grid, step)
        
        # Check if field of regard has any cells
        if not np.any(field_mask):
            # No cells in field of regard, return empty action
            return {
                "observation_points": set(),
                "expected_reward": 0.0,
            }
        
        # Get all possible contiguous subsets
        all_subsets = self._get_all_contiguous_subsets(
            field_mask, lat_grid, lon_grid, max_cells=max_observation_cells
        )
        
        if not all_subsets:
            # Fallback: if no subsets found (shouldn't happen after fix, but safety check)
            # Return the entire field of regard as a single action
            for_points = self.mask_to_points(field_mask, lat_grid, lon_grid)
            return {
                "observation_points": for_points,
                "expected_reward": 0.0,
            }
        
        # If no environment/horizon provided, just select largest subset
        if env is None or horizon is None:
            largest_subset = max(all_subsets, key=len)
            return {
                "observation_points": largest_subset,
                "expected_reward": 0.0,
            }
        
        # Evaluate each subset using Monte Carlo rollouts
        rng = np.random.default_rng()
        action_rewards = []
        
        original_belief = self.belief
        
        # Evaluate more subsets for better action selection
        # But still limit to avoid excessive computation
        max_subsets_to_evaluate = 10
        if len(all_subsets) > max_subsets_to_evaluate:
            # Use a mix: largest ones and some randomly sampled ones
            all_subsets_sorted = sorted(all_subsets, key=len, reverse=True)
            # Take top 5 largest
            selected = all_subsets_sorted[:5]
            # Randomly sample 5 more from the rest
            if len(all_subsets_sorted) > 5:
                remaining = all_subsets_sorted[5:]
                num_additional = min(5, len(remaining))
                additional = rng.choice(remaining, size=num_additional, replace=False).tolist()
                selected.extend(additional)
            all_subsets = selected
        
        for subset_points in all_subsets:
            # Temporarily set this as the observation points for rollout evaluation
            # We'll simulate taking this action and see the expected reward
            rollout_rewards = []
            
            for rollout_idx in range(self.num_rollouts):
                # Create a copy of belief for this evaluation
                eval_belief = deepcopy(original_belief)
                self.belief = eval_belief
                
                try:
                    # Compute immediate reward BEFORE updating belief (uses prior entropy)
                    immediate_reward = self.compute_reward(
                        lat_grid, lon_grid, step, observation_points=subset_points
                    )
                    
                    # Simulate taking this action: sample observations from belief distributions
                    # For each cell k in subset_points, sample x ~ Bernoulli(b_k(1))
                    observation = self._sample_observation_from_belief(subset_points, rng)
                    
                    # Convert observation_points to mask for update_from_observation
                    observation_mask = self.points_to_mask(
                        subset_points, lat_grid, lon_grid, self.belief.height, self.belief.width
                    )
                    
                    # Update belief with observations: b_{t+1}(j, x') = δ[x' = o_j] for observed cells
                    self.belief.update_from_observation(observation_mask, observation, lat_grid, lon_grid)
                    
                    # CRITICAL: Evolve belief after immediate observation to get belief at step t+1
                    # This evolves unobserved cells using the transition kernel
                    self.evolve_belief_with_environment(
                        env, lat_grid, lon_grid, step, observation_mask=observation_mask
                    )
                    
                    # Rollout future steps using a rollout policy
                    # At each future step, we select an action (contiguous subset) from FOR
                    future_reward = 0.0
                    for h in range(1, horizon):
                        future_step = step + h
                        future_for_mask = self.coverage_mask(lat_grid, lon_grid, future_step)
                        
                        # Skip if no field of regard
                        if not np.any(future_for_mask):
                            # Still need to evolve belief even if no FOR
                            self.evolve_belief_with_environment(
                                env, lat_grid, lon_grid, future_step, observation_mask=None
                            )
                            continue
                        
                        # Rollout policy: select a contiguous subset from FOR
                        # This is a simple heuristic - we could use a more sophisticated policy
                        future_action_points = self._select_contiguous_subset(
                            future_for_mask, lat_grid, lon_grid, max_cells=max_observation_cells, rng=rng
                        )
                        
                        # Compute reward for the selected action (before observation)
                        future_r = self.compute_reward(
                            lat_grid, lon_grid, future_step, observation_points=future_action_points
                        )
                        future_reward += (self.discount_factor ** h) * future_r
                        
                        # Sample observations from belief distributions for the selected action
                        # For each cell k in future_action_points: sample x ~ Bernoulli(b_k(1))
                        future_obs = self._sample_observation_from_belief(future_action_points, rng)
                        
                        # Convert future_action_points to mask for update_from_observation
                        future_obs_mask = self.points_to_mask(
                            future_action_points, lat_grid, lon_grid, self.belief.height, self.belief.width
                        )
                        
                        # Update belief: b_{t+1}(j, x') = δ[x' = o_j] for observed cells
                        self.belief.update_from_observation(future_obs_mask, future_obs, lat_grid, lon_grid)
                        
                        # CRITICAL: Evolve belief for next step (for unobserved cells)
                        # This moves the belief forward in time to step (future_step + 1)
                        self.evolve_belief_with_environment(
                            env, lat_grid, lon_grid, future_step, observation_mask=future_obs_mask
                        )
                    
                    total_reward = immediate_reward + future_reward
                    rollout_rewards.append(total_reward)
                except Exception as e:
                    # If rollout fails, use a default low reward
                    # Print error for debugging but continue
                    import traceback
                    if len(rollout_rewards) == 0:  # Only print for first failure to avoid spam
                        print(f"Warning: Rollout failed for subset {len(subset_points)} cells: {e}")
                        traceback.print_exc()
                    rollout_rewards.append(0.0)
            
            expected_reward = float(np.mean(rollout_rewards)) if rollout_rewards else 0.0
            action_rewards.append(expected_reward)
        
        # Restore original belief
        self.belief = original_belief
        
        # Select action with highest expected reward
        if action_rewards:
            best_idx = int(np.argmax(action_rewards))
            best_action = all_subsets[best_idx]
            best_reward = action_rewards[best_idx]
        else:
            # Fallback: select largest subset
            best_action = max(all_subsets, key=len) if all_subsets else set()
            best_reward = 0.0
        
        return {
            "observation_points": best_action,
            "expected_reward": best_reward,
            "all_rewards": {i: r for i, r in enumerate(action_rewards)},
        }


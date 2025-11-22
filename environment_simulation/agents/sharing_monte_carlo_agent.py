"""Monte Carlo agent with ground station data sharing capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .ground_station import GroundStation
from .monte_carlo_agent import MonteCarloAgent


@dataclass
class SharingMonteCarloAgent(MonteCarloAgent):
    """
    Monte Carlo agent that shares observations with ground stations.
    
    When the agent's field of regard overlaps with a ground station:
    1. Downloads NEW observations from the ground station (from other agents)
       - Only downloads observations newer than last processed timestep for each agent
    2. Uploads its own observations to the ground station
    3. Reconstructs belief temporally: goes back to earliest new observation timestep,
       then simulates belief forward in time, applying all observations (downloaded and own)
       in correct temporal order
    4. Then proceeds with normal Monte Carlo planning and action selection
    
    This agent is identical to MonteCarloAgent except for the data sharing step.
    """
    
    # Communication range for ground station contact (degrees)
    # Agent can communicate with GS if GS is within this angular distance
    communication_range_deg: float = 5.0  # Default: 5 degrees
    
    # Track last processed timestep for each other agent
    # This ensures we only download new observations, not ones we've already processed
    _last_processed_timestep: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    
    # Store our own observation history: timestep -> observation dict
    # This allows us to reconstruct belief with correct temporal ordering
    _own_observation_history: dict[int, dict[tuple[float, float], int]] = field(
        default_factory=dict, init=False, repr=False
    )
    
    # Store belief snapshots: timestep -> Belief snapshot
    # Used for temporal belief reconstruction when we need to go back in time
    # Only save snapshots periodically to save memory (e.g., every 10 steps)
    _belief_snapshots: dict[int, "Belief"] = field(default_factory=dict, init=False, repr=False)
    _snapshot_interval: int = 10  # Save snapshot every N steps
    
    def _is_in_contact_with_ground_station(
        self,
        step: int,
        ground_station: GroundStation,
    ) -> bool:
        """
        Check if the agent is in contact with a ground station at the current step.
        
        An agent is in contact if the ground station is within its communication range.
        Note: We use communication range (not FOR) for ground station contact.
        
        Args:
            step: Current time step
            ground_station: Ground station to check contact with
        
        Returns:
            True if agent is in contact with the ground station
        """
        # Get agent's current position
        agent_lat, agent_lon = self.position(step)
        
        # Compute angular distance to ground station using haversine formula
        from .agent import _angular_distance_deg
        
        # Create single-point arrays for distance calculation
        gs_lat_array = np.array([[ground_station.latitude]])
        gs_lon_array = np.array([[ground_station.longitude]])
        
        distance_array = _angular_distance_deg(agent_lat, agent_lon, gs_lat_array, gs_lon_array)
        distance = float(distance_array[0, 0])
        
        # Check if GS is within communication range (not FOR)
        return distance <= self.communication_range_deg
    
    def _download_new_observations_from_ground_station(
        self,
        ground_station: GroundStation,
    ) -> dict[int, dict[tuple[float, float], int]]:
        """
        Download NEW observations from the ground station that we haven't processed yet.
        
        Only downloads observations from other agents with timesteps newer than
        the last processed timestep for each agent.
        
        Args:
            ground_station: Ground station to download from
        
        Returns:
            Dictionary mapping timestep -> observation dictionary {(lat, lon): value}
            Only includes NEW observations we haven't processed yet
        """
        new_observations_by_timestep: dict[int, dict[tuple[float, float], int]] = {}
        
        # Get all observations from the ground station
        for (agent_id, timestep), observation in ground_station.observations.items():
            # Skip observations from this agent (we track our own separately)
            if agent_id == self.name:
                continue
            
            # Check if this is a new observation (timestep > last processed for this agent)
            last_processed = self._last_processed_timestep.get(agent_id, -1)
            if timestep > last_processed:
                # Merge observations at this timestep
                # If multiple agents observed at the same timestep, merge their observations
                if timestep not in new_observations_by_timestep:
                    new_observations_by_timestep[timestep] = {}
                new_observations_by_timestep[timestep].update(observation)
                
                # Update last processed timestep for this agent
                self._last_processed_timestep[agent_id] = timestep
        
        return new_observations_by_timestep
    
    def _save_belief_snapshot(self, timestep: int) -> None:
        """
        Save a belief snapshot at the given timestep.
        
        This is called BEFORE observations are applied at this timestep,
        so the snapshot represents the belief state at the START of timestep t.
        Used for temporal belief reconstruction.
        
        Args:
            timestep: Time step to save snapshot for
        """
        # Save snapshot periodically to save memory
        if timestep % self._snapshot_interval == 0 or timestep == 0:
            from copy import deepcopy
            # Save a snapshot of the current belief state (before observation at this timestep)
            self._belief_snapshots[timestep] = deepcopy(self.belief)
    
    def _record_own_observation(
        self,
        timestep: int,
        observation: dict[tuple[float, float], int],
    ) -> None:
        """
        Record an observation made by this agent at a specific timestep.
        
        This is called after the agent makes an observation, to maintain
        observation history for temporal belief reconstruction.
        
        Args:
            timestep: Time step when observation was made
            observation: Dictionary mapping (lat, lon) -> observed state (0 or 1)
        """
        if observation:
            # Store observation at this timestep (overwrite if exists)
            self._own_observation_history[timestep] = observation.copy()
    
    def _get_own_observations_in_range(
        self,
        start_timestep: int,
        end_timestep: int,
    ) -> dict[int, dict[tuple[float, float], int]]:
        """
        Get this agent's observations within a timestep range.
        
        Args:
            start_timestep: Start timestep (inclusive)
            end_timestep: End timestep (inclusive)
        
        Returns:
            Dictionary mapping timestep -> observation dictionary
        """
        own_obs_in_range = {}
        for t in range(start_timestep, end_timestep + 1):
            if t in self._own_observation_history:
                own_obs_in_range[t] = self._own_observation_history[t]
        return own_obs_in_range
    
    def _reconstruct_belief_temporally(
        self,
        env: "Environment",  # type: ignore
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        start_timestep: int,
        current_timestep: int,
        downloaded_observations_by_timestep: dict[int, dict[tuple[float, float], int]],
        own_observations_by_timestep: dict[int, dict[tuple[float, float], int]],
    ) -> None:
        """
        Reconstruct belief by simulating forward from start_timestep to current_timestep,
        applying all observations in correct temporal order.
        
        For each timestep t from start_timestep to current_timestep:
        1. If there are observations at t (downloaded or own), update belief with them
        2. Evolve belief forward using transition kernel to get to t+1
        
        Args:
            env: Environment object (for transition kernel)
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            start_timestep: Starting timestep to reconstruct from
            current_timestep: Current timestep (end of reconstruction)
            downloaded_observations_by_timestep: Observations from other agents by timestep
            own_observations_by_timestep: This agent's observations by timestep
        """
        from copy import deepcopy
        
        # We need to reconstruct belief from start_timestep
        # Since we don't have a belief snapshot at start_timestep, we'll need to
        # go back further or use a prior belief
        
        # For now, we'll assume we can reconstruct from the current belief state
        # by undoing updates, OR we store belief snapshots (simpler but memory intensive)
        # OR we assume start_timestep is recent enough that we can re-simulate from known prior
        
        # Actually, the user said to "go back to that timestep you start"
        # This means we need to have saved a belief snapshot at earlier timesteps
        # OR reconstruct by working backwards
        
        # Simplest approach: Save belief snapshot when we first make an observation
        # For now, let's assume we save a snapshot at some early timestep
        # OR we can reconstruct from prior (uniform) belief if start_timestep is early enough
        
        # Actually, let's use a different approach:
        # We'll simulate from start_timestep forward, starting with the belief state
        # we have (which may be at current_timestep). But this is wrong because
        # the belief at current_timestep already includes evolution from start_timestep.
        
        # The correct approach is to have a saved belief snapshot at start_timestep
        # OR reconstruct it by working backwards from current belief
        
        # For this implementation, let's assume we can reconstruct from a prior belief
        # at start_timestep. If start_timestep is too far back, we'll use uniform prior.
        
        # Better: Save belief snapshots periodically
        # For now, we'll use the current belief and re-simulate from start_timestep
        # by creating a temporary belief copy
        
        # Create a copy of belief to reconstruct
        # We'll reset it to prior for cells that will be re-observed, but keep
        # existing beliefs for cells that won't be re-observed
        
        # Actually, the simplest correct approach:
        # 1. Save a belief snapshot at start_timestep (if available)
        # 2. If not available, use uniform prior (or saved earlier snapshot)
        # 3. Simulate forward from that snapshot
        
        # For now, let's implement it assuming we have the belief at start_timestep
        # We'll need to add belief snapshots storage
        
        # Get belief snapshot at or before start_timestep
        # Find the closest snapshot <= start_timestep
        snapshot_timestep = None
        for saved_t in sorted(self._belief_snapshots.keys(), reverse=True):
            if saved_t <= start_timestep:
                snapshot_timestep = saved_t
                break
        
        # If we have a snapshot, restore it; otherwise start from uniform prior
        if snapshot_timestep is not None:
            # Restore belief from snapshot
            from copy import deepcopy
            restored_belief = deepcopy(self._belief_snapshots[snapshot_timestep])
            self.belief = restored_belief
            
            # If snapshot is before start_timestep, evolve belief forward to start_timestep
            # without applying observations (just transition)
            for t in range(snapshot_timestep, start_timestep):
                # Evolve belief forward (no observations at these timesteps)
                observation_mask = self.coverage_mask(lat_grid, lon_grid, t)
                self.evolve_belief_with_environment(
                    env, lat_grid, lon_grid, t, observation_mask=observation_mask
                )
        else:
            # No snapshot available - start from uniform prior
            # Reset belief to prior probability for all FOR locations
            for lat, lon in list(self.belief.valid_locations):
                self.belief.set_probability(lat, lon, self.belief.prior_probability)
        
        # Now simulate forward from start_timestep to current_timestep
        # applying all observations in correct temporal order
        # For each timestep from start_timestep to current_timestep
        for t in range(start_timestep, current_timestep + 1):
            # Merge observations at this timestep (from downloaded and own)
            observations_at_t = {}
            
            # Add downloaded observations at this timestep
            if t in downloaded_observations_by_timestep:
                observations_at_t.update(downloaded_observations_by_timestep[t])
            
            # Add own observations at this timestep
            if t in own_observations_by_timestep:
                observations_at_t.update(own_observations_by_timestep[t])
            
            # Update belief with observations at this timestep (if any)
            if observations_at_t:
                # Create observation mask
                height, width = lat_grid.shape
                observation_mask = np.zeros((height, width), dtype=bool)
                
                # Create mapping from (lat, lon) to grid coordinates
                latlon_to_grid = {}
                for y in range(height):
                    for x in range(width):
                        lat, lon = float(lat_grid[y, x]), float(lon_grid[y, x])
                        latlon_to_grid[(lat, lon)] = (y, x)
                
                # Mark cells that were observed
                for (lat, lon) in observations_at_t.keys():
                    lat_key = float(lat)
                    lon_key = float(lon)
                    
                    # Try exact match first
                    grid_coords = latlon_to_grid.get((lat_key, lon_key))
                    
                    # If not found, try tolerance-based search
                    if grid_coords is None:
                        for (g_lat, g_lon), (gy, gx) in latlon_to_grid.items():
                            if abs(g_lat - lat_key) < 1e-6 and abs(g_lon - lon_key) < 1e-6:
                                grid_coords = (gy, gx)
                                break
                    
                    if grid_coords is not None:
                        y, x = grid_coords
                        observation_mask[y, x] = True
                
                # Update belief with observations: b_{t+1}(j, x') = δ[x' = o_j] for observed cells
                self.belief.update_from_observation(
                    observation_mask=observation_mask,
                    observations=observations_at_t,
                    lat_grid=lat_grid,
                    lon_grid=lon_grid,
                )
            
            # Evolve belief forward using transition kernel (for unobserved cells)
            # This moves belief from timestep t to t+1
            # Only evolve if not at the last timestep (current_timestep)
            if t < current_timestep:
                # Create observation mask for evolution (if observations exist at t)
                observation_mask_for_evolution = None
                if observations_at_t:
                    # Use the mask we created above
                    observation_mask_for_evolution = observation_mask
                else:
                    # Use field of regard at timestep t
                    observation_mask_for_evolution = self.coverage_mask(lat_grid, lon_grid, t)
                
                # Evolve belief: b_{t+1}^e(j,x') = Σ_{x_j} Σ_{x_N(j)} φ_j(x'; x_j, x_N(j)) * ...
                self.evolve_belief_with_environment(
                    env, lat_grid, lon_grid, t, observation_mask=observation_mask_for_evolution
                )
    
    def act(
        self,
        step: int,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        env: Optional["Environment"] = None,  # type: ignore
        horizon: Optional[int] = None,
        max_observation_cells: Optional[int] = None,
        ground_stations: Optional[list[GroundStation]] = None,
    ) -> dict[str, Any]:
        """
        Choose an action based on Monte Carlo planning with ground station data sharing.
        
        Before planning:
        1. Check if agent is in contact with any ground station
        2. If in contact:
           a. Download NEW observations from GS (only those newer than last processed)
           b. Upload own observations to GS
           c. Reconstruct belief temporally: go back to earliest new observation timestep,
              then simulate forward, applying all observations in correct temporal order
        3. Then proceed with normal Monte Carlo planning
        
        Args:
            step: Current time step
            lat_grid: (H, W) array of latitude values in degrees
            lon_grid: (H, W) array of longitude values in degrees
            env: Environment object (required for belief evolution in rollouts)
            horizon: Planning horizon for Monte Carlo rollouts
            max_observation_cells: Maximum number of cells to observe in a contiguous subset.
                                  If None, uses max_observation_ratio.
            ground_stations: Optional list of ground stations for data sharing
        
        Returns:
            Dictionary containing action information:
            - observation_points: Set of (lat, lon) tuples representing cells to observe
            - expected_reward: Expected cumulative reward for selected action
            - all_rewards: Dictionary mapping action indices to expected rewards (for debugging)
        """
        from environment_simulation.environment import Environment
        from copy import deepcopy
        
        # Step 1: Check for ground station contact and exchange data
        if ground_stations and env is not None:
            # DEBUG: Check distance to all ground stations for this agent (only when close or first few steps)
            agent_lat, agent_lon = self.position(step)
            for gs in ground_stations:
                from .agent import _angular_distance_deg
                gs_lat_array = np.array([[gs.latitude]])
                gs_lon_array = np.array([[gs.longitude]])
                distance_array = _angular_distance_deg(agent_lat, agent_lon, gs_lat_array, gs_lon_array)
                distance = float(distance_array[0, 0])
                # Use communication range for ground station contact (not FOR)
                in_range = distance <= self.communication_range_deg
                # Print when in range, or when close (within 30°), or first 5 steps
                if in_range or (distance <= 30.0 and step < 50) or step < 5:
                    print(f"[GS Check] Step {step} | {self.name} -> {gs.name}: "
                          f"distance={distance:.2f}° (comm_range={self.communication_range_deg:.1f}°) "
                          f"{'✓ IN RANGE' if in_range else '✗ out of range'}")
            
            for ground_station in ground_stations:
                if self._is_in_contact_with_ground_station(step, ground_station):
                    # DEBUG: Print ground station contact info
                    print(f"\n[GS Contact] Step {step} | {self.name} in contact with {ground_station.name} "
                          f"(lat={ground_station.latitude:.2f}°, lon={ground_station.longitude:.2f}°)")
                    
                    # DEBUG: Show what agents have observations in this ground station
                    agents_in_gs = set()
                    timesteps_by_agent = {}
                    for (agent_id, timestep), obs in ground_station.observations.items():
                        agents_in_gs.add(agent_id)
                        if agent_id not in timesteps_by_agent:
                            timesteps_by_agent[agent_id] = []
                        timesteps_by_agent[agent_id].append(timestep)
                    print(f"[GS State] Ground station has observations from agents: {sorted(agents_in_gs)}")
                    for agent_id, timesteps in sorted(timesteps_by_agent.items()):
                        print(f"  - {agent_id}: {len(timesteps)} observations at timesteps {sorted(timesteps)[:10]}{'...' if len(timesteps) > 10 else ''}")
                    
                    # Download NEW observations from ground station (from other agents)
                    # Returns dict mapping timestep -> observation dict
                    downloaded_observations_by_timestep = self._download_new_observations_from_ground_station(
                        ground_station
                    )
                    
                    # DEBUG: Show what was downloaded
                    if downloaded_observations_by_timestep:
                        print(f"[GS Download] {self.name} downloaded {len(downloaded_observations_by_timestep)} timesteps of observations "
                              f"from other agents")
                        for t, obs in sorted(downloaded_observations_by_timestep.items()):
                            print(f"  Timestep {t}: {len(obs)} cells")
                    else:
                        print(f"[GS Download] {self.name} found NO new observations from other agents")
                        # DEBUG: Show why - check what's actually in the GS
                        other_agents_in_gs = [aid for aid in agents_in_gs if aid != self.name]
                        if other_agents_in_gs:
                            print(f"  [DEBUG] But GS has data from: {other_agents_in_gs}")
                            print(f"  [DEBUG] Last processed timesteps: {self._last_processed_timestep}")
                            # Show what timesteps are available from other agents
                            for other_agent in other_agents_in_gs:
                                other_timesteps = timesteps_by_agent.get(other_agent, [])
                                last_processed = self._last_processed_timestep.get(other_agent, -1)
                                new_timesteps = [t for t in other_timesteps if t > last_processed]
                                print(f"  [DEBUG] {other_agent}: has {len(other_timesteps)} timesteps, "
                                      f"last processed={last_processed}, new={sorted(new_timesteps)[:5]}")
                        else:
                            print(f"  [DEBUG] GS has no data from other agents (only from {sorted(agents_in_gs)})")
                    
                    # Find earliest timestep with new downloaded observations
                    earliest_downloaded_timestep = (
                        min(downloaded_observations_by_timestep.keys())
                        if downloaded_observations_by_timestep
                        else step + 1  # No new observations
                    )
                    
                    # If we have new downloaded observations, reconstruct belief temporally
                    if downloaded_observations_by_timestep:
                        # Get our own observations from earliest downloaded timestep to current step
                        own_observations_by_timestep = self._get_own_observations_in_range(
                            earliest_downloaded_timestep, step
                        )
                        
                        # Reconstruct belief by simulating forward from earliest new timestep
                        # This applies all observations (downloaded and own) in correct temporal order
                        self._reconstruct_belief_temporally(
                            env=env,
                            lat_grid=lat_grid,
                            lon_grid=lon_grid,
                            start_timestep=earliest_downloaded_timestep,
                            current_timestep=step,
                            downloaded_observations_by_timestep=downloaded_observations_by_timestep,
                            own_observations_by_timestep=own_observations_by_timestep,
                        )
                    
                    # Upload our observations to ground station (for other agents to download)
                    # Upload all observations we have that other agents don't have
                    # Get observations from the earliest new downloaded timestep (or step 0 if no downloads)
                    upload_start_timestep = (
                        earliest_downloaded_timestep if downloaded_observations_by_timestep else 0
                    )
                    own_obs_to_upload = self._get_own_observations_in_range(upload_start_timestep, step)
                    
                    # DEBUG: Show what we're trying to upload
                    print(f"[GS Upload] {self.name} has {len(own_obs_to_upload)} timesteps of own observations to upload "
                          f"(from timestep {upload_start_timestep} to {step})")
                    
                    # Upload our observations to the ground station
                    # Each agent uploads its own observations separately - they're stored by (agent_id, timestep)
                    total_uploaded = 0
                    for upload_timestep, observation in own_obs_to_upload.items():
                        # Check if we've already uploaded observations at this timestep
                        # (to avoid re-uploading if we've visited the GS before)
                        already_uploaded_key = (self.name, upload_timestep)
                        if already_uploaded_key in ground_station.observations:
                            # We've already uploaded at this timestep, skip to avoid overwriting
                            continue
                        
                        # Upload all our observations at this timestep
                        # Each agent's observations are stored separately in the ground station
                        ground_station.record_observation(
                            agent_id=self.name,
                            timestep=upload_timestep,
                            observation=observation,
                        )
                        total_uploaded += len(observation)
                    
                    # DEBUG: Show upload summary
                    if total_uploaded > 0:
                        print(f"[GS Upload] {self.name} uploaded {total_uploaded} new observation cells to ground station")
                    else:
                        print(f"[GS Upload] {self.name} had no new observations to upload (all cells already in GS from other agents)")
        
        # Step 2: Proceed with normal Monte Carlo planning
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
            # Fallback: if no subsets found, return entire field of regard
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
                # NOTE: This copy already includes any downloaded observations from GS
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

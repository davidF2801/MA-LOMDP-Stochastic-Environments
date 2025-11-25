"""Monte Carlo agent for graph-based planning with peer-to-peer data sharing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from copy import deepcopy

from .graph_agent import GraphAgent
from .graph_monte_carlo_agent import GraphMonteCarloAgent
from .graph_belief import GraphBelief


@dataclass
class GraphSharingMonteCarloAgent(GraphMonteCarloAgent):
    """
    Monte Carlo agent that shares observations with other agents via peer-to-peer communication.
    
    When agents are at the same node or connected nodes (within one step distance):
    1. They exchange observations
    2. Each agent receives NEW observations from the other agent
    3. Observations are stored as (timestep, node, state) tuples
    4. Agents integrate received observations into their belief
    5. Information propagates through the network (Agent C can receive observations
       from Agent A that were relayed through Agent B)
    
    This agent extends GraphMonteCarloAgent with peer-to-peer communication capabilities.
    """
    
    # Store received observations: (timestep, node, state) tuples from other agents
    # Structure: {agent_name: [(timestep, node, state), ...]}
    _received_observations: dict[str, list[tuple[int, int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )
    
    # Track last processed timestep for each other agent
    # This ensures we only receive new observations, not ones we've already processed
    _last_processed_timestep: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    
    # Store our own observation history: timestep -> (node, state)
    # This allows us to share our observations with other agents
    _own_observation_history: list[tuple[int, int, int]] = field(
        default_factory=list, init=False, repr=False
    )
    
    # Store belief snapshots: timestep -> GraphBelief snapshot
    # Used for temporal belief reconstruction when we need to go back in time
    _belief_snapshots: dict[int, GraphBelief] = field(
        default_factory=dict, init=False, repr=False
    )
    _snapshot_interval: int = 10  # Save snapshot every N steps
    
    def _can_communicate_with(self, other_agent: "GraphAgent", env: "GraphEnvironment") -> bool:
        """
        Check if this agent can communicate with another agent.
        
        Agents can communicate if:
        - They are at the same node, OR
        - Their nodes are connected (within one step distance in the graph)
        
        Args:
            other_agent: Another GraphAgent to check communication with
            env: GraphEnvironment object
            
        Returns:
            True if agents can communicate
        """
        # Check if at same node
        if self.current_node == other_agent.current_node:
            return True
        
        # Check if nodes are connected (within one step distance)
        self_neighbors = set(env.get_neighbors(self.current_node))
        self_neighbors.add(self.current_node)  # Include current node
        
        other_neighbors = set(env.get_neighbors(other_agent.current_node))
        other_neighbors.add(other_agent.current_node)  # Include current node
        
        # Can communicate if other agent's node is in our neighbors or vice versa
        return other_agent.current_node in self_neighbors or self.current_node in other_neighbors
    
    def _get_new_observations_from_agent(
        self,
        other_agent: "GraphSharingMonteCarloAgent",
    ) -> list[tuple[int, int, int]]:
        """
        Get NEW observations from another agent that we haven't received yet.
        
        Returns observations from the other agent's observation history that are newer
        than the last processed timestep for that agent. Also includes observations
        the other agent received from other agents (propagated information).
        
        Args:
            other_agent: Another GraphSharingMonteCarloAgent to get observations from
            
        Returns:
            List of (timestep, node, state) tuples that are new for this agent
        """
        new_observations = []
        
        # Get last processed timestep for this agent
        last_processed = self._last_processed_timestep.get(other_agent.name, -1)
        
        # Get all observations from the other agent (own + received)
        # Start with other agent's own observations
        all_agent_observations = []
        
        # Add other agent's own observations
        for timestep, node, state in other_agent._own_observation_history:
            if timestep > last_processed:
                all_agent_observations.append((timestep, node, state))
        
        # Add observations the other agent received from other agents (propagated info)
        for source_agent_name, received_obs_list in other_agent._received_observations.items():
            for timestep, node, state in received_obs_list:
                if timestep > last_processed:
                    all_agent_observations.append((timestep, node, state))
        
        # Filter to only new observations (avoid duplicates)
        # Track what we've already seen to avoid duplicates
        seen_observations = set()
        
        # First, add observations we already have from this agent
        if other_agent.name in self._received_observations:
            for timestep, node, state in self._received_observations[other_agent.name]:
                seen_observations.add((timestep, node))
        
        # Also check our own observations
        for timestep, node, state in self._own_observation_history:
            seen_observations.add((timestep, node))
        
        # Filter new observations
        for timestep, node, state in all_agent_observations:
            if (timestep, node) not in seen_observations:
                new_observations.append((timestep, node, state))
                seen_observations.add((timestep, node))
        
        # Update last processed timestep for this agent
        if new_observations:
            max_timestep = max(timestep for timestep, _, _ in new_observations)
            self._last_processed_timestep[other_agent.name] = max_timestep
        
        return new_observations
    
    def _store_received_observations(
        self,
        source_agent_name: str,
        observations: list[tuple[int, int, int]],
    ) -> None:
        """
        Store observations received from another agent.
        
        Args:
            source_agent_name: Name of the agent these observations came from
            observations: List of (timestep, node, state) tuples to store
        """
        if not observations:
            return
        
        # Store received observations
        if source_agent_name not in self._received_observations:
            self._received_observations[source_agent_name] = []
        
        # Add only truly new observations (avoid duplicates)
        existing = set((t, n) for t, n, _ in self._received_observations[source_agent_name])
        for timestep, node, state in observations:
            if (timestep, node) not in existing:
                self._received_observations[source_agent_name].append((timestep, node, state))
    
    def _record_own_observation(
        self,
        timestep: int,
        node: int,
        state: int,
    ) -> None:
        """
        Record an observation made by this agent.
        
        Args:
            timestep: Time step when observation was made
            node: Node index that was observed
            state: Observed state (0 or 1)
        """
        self._own_observation_history.append((timestep, node, state))
    
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
            self._belief_snapshots[timestep] = deepcopy(self.belief)
    
    def _reconstruct_belief_temporally(
        self,
        env: "GraphEnvironment",
        start_timestep: int,
        current_timestep: int,
        received_observations: list[tuple[int, int, int]],
    ) -> None:
        """
        Reconstruct belief by simulating forward from start_timestep to current_timestep,
        applying all observations in correct temporal order.
        
        For each timestep t from start_timestep to current_timestep:
        1. If there are observations at t (received or own), update belief with them
        2. Evolve belief forward using transition kernel to get to t+1
        
        Args:
            env: GraphEnvironment object (for transition kernel)
            start_timestep: Starting timestep to reconstruct from
            current_timestep: Current timestep (end of reconstruction)
            received_observations: List of (timestep, node, state) tuples from other agents
        """
        # Group observations by timestep
        observations_by_timestep: dict[int, dict[int, int]] = {}
        
        # Add received observations
        for timestep, node, state in received_observations:
            if start_timestep <= timestep <= current_timestep:
                if timestep not in observations_by_timestep:
                    observations_by_timestep[timestep] = {}
                observations_by_timestep[timestep][node] = state
        
        # Add own observations in the time range
        for timestep, node, state in self._own_observation_history:
            if start_timestep <= timestep <= current_timestep:
                if timestep not in observations_by_timestep:
                    observations_by_timestep[timestep] = {}
                # Own observations take precedence if there's a conflict
                observations_by_timestep[timestep][node] = state
        
        # Get belief snapshot at or before start_timestep
        snapshot_timestep = None
        for saved_t in sorted(self._belief_snapshots.keys(), reverse=True):
            if saved_t <= start_timestep:
                snapshot_timestep = saved_t
                break
        
        # Restore belief from snapshot or start from prior
        if snapshot_timestep is not None:
            self.belief = deepcopy(self._belief_snapshots[snapshot_timestep])
            
            # If snapshot is before start_timestep, evolve belief forward to start_timestep
            # without applying observations (just transition)
            for t in range(snapshot_timestep, start_timestep):
                # Evolve belief forward (no observations at these timesteps)
                self.belief.evolve_with_transition_kernel(env, set())
        else:
            # No snapshot available - we'll initialize nodes as we encounter them through observations
            # Just proceed with current belief state
            pass
        
        # Simulate forward from start_timestep to current_timestep
        # applying all observations in correct temporal order
        for t in range(start_timestep, current_timestep + 1):
            # Get observations at this timestep
            observations_at_t = observations_by_timestep.get(t, {})
            observed_nodes = set(observations_at_t.keys())
            
            # Update belief with observations at this timestep (if any)
            if observations_at_t:
                # Update belief: b_{t+1}(j, x') = δ[x' = o_j] for observed nodes
                self.belief.update_from_observation(
                    observed_nodes, observations_at_t
                )
            
            # Evolve belief forward using transition kernel (for unobserved nodes)
            # This moves belief from timestep t to t+1
            # Only evolve if not at the last timestep (current_timestep)
            if t < current_timestep:
                self.belief.evolve_with_transition_kernel(env, observed_nodes)
    
    def act(self, env: "GraphEnvironment") -> dict[str, Any]:
        """
        Choose an action based on Monte Carlo planning with peer-to-peer data sharing.
        
        Note: Communication is handled separately by the simulation runner before calling act().
        This method just performs the Monte Carlo planning using the updated belief.
        
        Args:
            env: GraphEnvironment object
            
        Returns:
            Dictionary containing action information:
            - target_node: Node index to move to
            - expected_reward: Expected cumulative reward for selected action
        """
        # Get current timestep from environment
        current_timestep = env.time
        
        # Save belief snapshot before planning
        self._save_belief_snapshot(current_timestep)
        
        # Proceed with normal Monte Carlo planning (from parent class)
        return super().act(env)
    
    def communicate_with_agents(
        self,
        other_agents: list["GraphSharingMonteCarloAgent"],
        env: "GraphEnvironment",
        current_timestep: int,
    ) -> None:
        """
        Check for communication with other agents and exchange observations.
        
        This method is called by the simulation runner for each agent at each step.
        It handles the peer-to-peer communication and belief reconstruction.
        
        Args:
            other_agents: List of other GraphSharingMonteCarloAgent instances
            env: GraphEnvironment object
            current_timestep: Current time step
        """
        # Collect all new observations from all communicating agents
        all_new_observations = []
        communicated_agents = []
        
        for other_agent in other_agents:
            if other_agent.name == self.name:
                continue  # Skip self
            
            # Check if we can communicate with this agent
            if self._can_communicate_with(other_agent, env):
                communicated_agents.append(other_agent.name)
                
                # Get new observations from this agent (includes propagated observations)
                # This also updates _last_processed_timestep
                new_obs = self._get_new_observations_from_agent(other_agent)
                
                if new_obs:
                    all_new_observations.extend(new_obs)
                    # Store received observations
                    self._store_received_observations(other_agent.name, new_obs)
        
        # If we received new observations, reconstruct belief temporally
        if all_new_observations:
            # Find earliest timestep with new observations
            earliest_timestep = min(timestep for timestep, _, _ in all_new_observations)
            
            # Reconstruct belief by simulating forward from earliest timestep
            self._reconstruct_belief_temporally(
                env=env,
                start_timestep=earliest_timestep,
                current_timestep=current_timestep,
                received_observations=all_new_observations,
            )
            
            if communicated_agents:
                print(f"[Communication] Step {current_timestep} | {self.name} communicated with {communicated_agents} "
                      f"(received {len(all_new_observations)} new observations)")


"""Training loop for CTDE agents."""

from __future__ import annotations

from typing import Any, Optional
import copy
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .graph_ctde_agent import GraphCTDEAgent
from .graph_ctde_trainer import CTDETrainer, TrajectoryBuffer, compute_gae_advantages
from .graph_belief import GraphBelief
from .graph_ctde_networks import ActorNetwork, CriticNetwork

if not TORCH_AVAILABLE:
    raise ImportError("PyTorch is required for CTDE training. Install with: pip install torch")


def create_graph_ctde_agents(
    num_nodes: int,
    num_agents: int = 3,
    initial_nodes: Optional[list[int]] = None,
    actor_network: Optional[ActorNetwork] = None,
    critic_network: Optional[CriticNetwork] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[GraphCTDEAgent]:
    """
    Create CTDE agents with shared actor-critic networks.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_agents: Number of agents
        initial_nodes: Optional list of initial node positions for each agent.
                      If None, randomly initializes positions.
        actor_network: Optional shared actor network. If None, creates a new one.
        critic_network: Optional critic network. If None, creates a new one.
        rng: Random number generator
        
    Returns:
        List of GraphCTDEAgent instances with shared networks
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Create shared networks if not provided
    if actor_network is None:
        actor_network = ActorNetwork(
            num_nodes=num_nodes,
            num_states=2,
            num_agents=num_agents,
            hidden_dim=128,
            num_layers=3,
        )
    
    if critic_network is None:
        critic_network = CriticNetwork(
            num_nodes=num_nodes,
            num_states=2,
            num_agents=num_agents,
            hidden_dim=128,
            num_layers=3,
        )
    
    # Initialize agent positions
    if initial_nodes is None:
        initial_nodes = rng.choice(num_nodes, size=num_agents, replace=False).tolist()
    
    # Create agents
    agents = []
    for i in range(num_agents):
        agent_name = f"CTDE_Agent_{i}"
        belief = GraphBelief(num_nodes=num_nodes, num_states=2)
        agent = GraphCTDEAgent(
            name=agent_name,
            current_node=initial_nodes[i],
            belief=belief,
            num_nodes=num_nodes,
        )
        agent.set_networks(actor_network, critic_network)
        agents.append(agent)
    
    return agents


def build_ideal_centralized_belief(
    env: "GraphEnvironment",
    agents: list[GraphCTDEAgent],
    observations_history: list[dict[str, dict[int, int]]],  # [{agent_name: {node: state}} for each timestep]
    current_timestep: int,
) -> "GraphBelief":
    """
    Build ideal centralized belief by combining ALL observations from ALL agents.
    
    The ideal centralized belief is computed by:
    1. Starting from uniform prior
    2. For each timestep t from 0 to current_timestep:
       - Collect all observations from all agents at timestep t
       - Update belief with these observations (delta function for observed nodes)
       - Evolve belief using transition kernel (mean field update)
    
    This represents what the belief would be if all agents perfectly shared
    their observations immediately and we applied the belief update model correctly.
    
    Args:
        env: GraphEnvironment object (for transition kernel)
        agents: List of CTDE agents
        observations_history: List of observation dictionaries, one per timestep.
                            Each dict maps agent_name -> {node: observed_state}
                            observations_history[t] contains observations at timestep t
        current_timestep: Current timestep (0-indexed)
        
    Returns:
        GraphBelief object representing the ideal centralized belief at current_timestep
    """
    from .graph_belief import GraphBelief
    
    # Create fresh centralized belief starting from uniform prior
    ideal_belief = GraphBelief(num_nodes=env.num_nodes, num_states=2)
    ideal_belief.reset()  # Initialize to uniform prior for all nodes
    
    # Reconstruct belief by replaying all observations in chronological order
    for t in range(min(current_timestep + 1, len(observations_history))):
        # Collect all observations from all agents at timestep t
        all_observations_at_t = {}  # {node: observed_state}
        
        if t < len(observations_history):
            obs_dict = observations_history[t]
            for agent_name, agent_observations in obs_dict.items():
                # Merge observations (if multiple agents observe same node, use one)
                # In practice, all agents at same node should observe same state
                for node, observed_state in agent_observations.items():
                    if node not in all_observations_at_t:
                        all_observations_at_t[node] = observed_state
                    # If conflict, prefer state=1 (event present) as it's more informative
                    elif observed_state == 1:
                        all_observations_at_t[node] = 1
        
        # Update belief with observations at timestep t
        if all_observations_at_t:
            observed_nodes = set(all_observations_at_t.keys())
            # Belief is already over all nodes (initialized in __post_init__), so just update with observations
            ideal_belief.update_from_observation(observed_nodes, all_observations_at_t)
        
        # Evolve belief using transition kernel (unless this is the last timestep)
        if t < current_timestep:
            # Evolve all unobserved nodes (empty set means no nodes are currently observed for evolution)
            ideal_belief.evolve_with_transition_kernel(env, set(), num_states=2)
    
    return ideal_belief


def build_centralized_state(
    env: "GraphEnvironment",
    agents: list[GraphCTDEAgent],
    observations_history: list[dict[str, dict[int, int]]],
    current_timestep: int,
) -> tuple[torch.Tensor, torch.Tensor, "GraphBelief"]:
    """
    Build centralized state for critic using ideal centralized belief.
    
    S_t = (B_ideal(t), p_1(t), ..., p_N(t))
    
    This function rebuilds the ideal belief from observations_history (useful for bootstrap value).
    For trajectory collection, the ideal belief is maintained incrementally in collect_trajectory.
    
    Args:
        env: GraphEnvironment object
        agents: List of CTDE agents
        observations_history: List of observations for ideal belief computation.
                            Format: [{agent_name: {node: state}} for each timestep]
        current_timestep: Current timestep (0-indexed)
        
    Returns:
        Tuple of (state_or_belief_tensor, positions_tensor, centralized_belief)
        - state_or_belief: [num_nodes, num_states] tensor
        - positions: [num_agents] tensor with agent positions
        - centralized_belief: GraphBelief object (for reward computation)
    """
    # Build ideal centralized belief from joint observations
    centralized_belief = build_ideal_centralized_belief(
        env, agents, observations_history, current_timestep
    )
    
    # Convert to tensor
    num_nodes = env.num_nodes
    belief_array = np.zeros((num_nodes, 2), dtype=np.float32)
    for node in range(num_nodes):
        prob = centralized_belief.get_probability(node, 1)  # P(E=1)
        belief_array[node, 0] = 1.0 - prob
        belief_array[node, 1] = prob
    state_tensor = torch.from_numpy(belief_array)
    
    # Agent positions
    positions = torch.tensor([agent.current_node for agent in agents], dtype=torch.long)
    
    return state_tensor, positions, centralized_belief


def compute_centralized_entropy(belief: "GraphBelief") -> float:
    """
    Compute average entropy of the centralized belief over all nodes.
    
    H(b) = (1/K) * Σ_k h_k where h_k = -p_k*log(p_k) - (1-p_k)*log(1-p_k)
    
    Args:
        belief: GraphBelief object
        
    Returns:
        Average entropy (scalar)
    """
    num_nodes = belief.num_nodes
    if num_nodes == 0:
        return 0.0
    
    total_entropy = belief.entropy()  # Sum over all nodes
    return total_entropy / num_nodes  # Average


def compute_detection_reward(
    env_state_before: np.ndarray,
    env_state_after: np.ndarray,
    observations_at_step: dict[str, dict[int, int]],
    events: dict[tuple[int, int], dict],
    current_step: int,
    env: "GraphEnvironment",
    event_utility: Optional[dict[int, float]] = None,
) -> tuple[float, dict[tuple[int, int], dict]]:
    """
    Compute detection reward based on events first detected at this timestep.
    
    R_det(t) = Σ_{events detected at t} u_e * (1 - NDD_e)
    
    where NDD_e = min((det_time - start_time) / E[L_k], 1.0)
    and E[L_k] = 1/(1 - delta_k) for geometric lifetime.
    
    Args:
        env_state_before: Environment state before env.step() [num_nodes]
        env_state_after: Environment state after env.step() [num_nodes]
        observations_at_step: Dict {agent_name: {node: observed_state}}
        events: Dict {(node_id, start_step): {start_time, det_time, detected, utility, ...}}
        current_step: Current timestep (after env.step(), so t+1)
        env: GraphEnvironment object
        event_utility: Optional utility mapping {state: utility}
        
    Returns:
        Tuple of (detection_reward, updated_events_dict)
    """
    if event_utility is None:
        event_utility = {0: 0.0, 1: 1.0}
    
    utility_default = event_utility.get(1, 1.0)
    
    # Detect new events: state transition 0→1
    for node in range(len(env_state_before)):
        if env_state_before[node] == 0 and env_state_after[node] == 1:
            # New event started
            event_id = (node, current_step)
            if event_id not in events:
                # Get persistence probability for expected lifetime
                if env.mode == "rsp":
                    delta = float(env.persistence_map[node]) if env.persistence_map is not None else env.rsp.delta
                else:  # DBN-2: persistence = 1 - death_rate
                    delta = 1.0 - env.death_rate
                
                E_L = 1.0 / (1.0 - delta) if delta < 1.0 else 1000.0  # Avoid division by zero
                
                events[event_id] = {
                    "node_id": node,
                    "start_time": current_step,
                    "det_time": None,
                    "detected": False,
                    "utility": utility_default,
                    "expected_lifetime": E_L,
                }
    
    # Detect events that ended: state transition 1→0
    for event_id, event_info in list(events.items()):
        node = event_info["node_id"]
        if env_state_before[node] == 1 and env_state_after[node] == 0:
            # Event ended (if not already ended)
            if event_info.get("end_time") is None:
                event_info["end_time"] = current_step
    
    # Check which events are first detected at this timestep
    detected_nodes = set()
    for agent_name, agent_observations in observations_at_step.items():
        for node, observed_state in agent_observations.items():
            if observed_state == 1:
                detected_nodes.add(node)
    
    detection_reward = 0.0
    for node in detected_nodes:
        # Find active event at this node
        active_event_id = None
        for event_id, event_info in events.items():
            if (event_info["node_id"] == node and 
                event_info["start_time"] <= current_step and
                not event_info["detected"] and
                (event_info.get("end_time") is None or event_info["end_time"] >= current_step)):
                active_event_id = event_id
                break
        
        if active_event_id is not None:
            event_info = events[active_event_id]
            
            # First detection: compute NDD and reward
            if not event_info["detected"]:
                delay = current_step - event_info["start_time"]
                E_L = event_info["expected_lifetime"]
                ndd = min(delay / E_L, 1.0) if E_L > 0 else 1.0
                
                reward_contribution = event_info["utility"] * (1.0 - ndd)
                detection_reward += reward_contribution
                
                # Mark as detected
                event_info["detected"] = True
                event_info["det_time"] = current_step
    
    return detection_reward, events


def compute_team_reward(
    centralized_belief_before: "GraphBelief",
    centralized_belief_after: "GraphBelief",
    env_state_before: np.ndarray,
    env_state_after: np.ndarray,
    observations_at_step: dict[str, dict[int, int]],
    events: dict[tuple[int, int], dict],
    current_step: int,
    env: "GraphEnvironment",
    w_h: float = 1.0,
    w_v: float = 1.0,
    event_utility: Optional[dict[int, float]] = None,
) -> tuple[float, dict[tuple[int, int], dict]]:
    """
    Compute team reward: R_t = w_h * R_entropy(t) + w_v * R_det(t)
    
    where:
    - R_entropy(t) = H(b_{t-1}) - H(b_t) (entropy reduction)
    - R_det(t) = Σ_{events detected at t} u_e * (1 - NDD_e)
    
    Args:
        centralized_belief_before: Belief before observations at this step
        centralized_belief_after: Belief after observations at this step
        env_state_before: Environment state before env.step()
        env_state_after: Environment state after env.step()
        observations_at_step: Observations made at this step
        events: Event tracking dictionary
        current_step: Current timestep
        env: GraphEnvironment object
        w_h: Weight for entropy reduction
        w_v: Weight for detection
        event_utility: Optional utility mapping
        
    Returns:
        Tuple of (team_reward, updated_events_dict)
    """
    # Compute entropy reduction reward
    H_before = compute_centralized_entropy(centralized_belief_before)
    H_after = compute_centralized_entropy(centralized_belief_after)
    R_entropy = H_before - H_after
    
    # Compute detection reward
    R_det, updated_events = compute_detection_reward(
        env_state_before, env_state_after, observations_at_step,
        events, current_step, env, event_utility
    )
    
    # Total reward
    R_t = w_h * R_entropy + w_v * R_det
    return float(R_t), updated_events


def collect_trajectory(
    env: "GraphEnvironment",
    agents: list[GraphCTDEAgent],
    num_steps: int,
    w_h: float = 1.0,
    w_v: float = 1.0,
    event_utility: Optional[dict[int, float]] = None,
    gamma: float = 0.99,
) -> tuple[
    TrajectoryBuffer,
    list[dict[str, dict[int, int]]],
    "GraphBelief",
    torch.Tensor,
]:
    """
    Collect a trajectory using current policy.
    
    Implements the trajectory rollout from Section 4.1:
    1. Update beliefs
    2. Build local states
    3. Sample actions
    4. Execute actions
    5. Environment evolves
    6. Compute rewards
    
    The critic always uses the ideal centralized belief (joint observations from all agents).
    
    Args:
        env: GraphEnvironment object
        agents: List of CTDE agents
        num_steps: Number of steps in trajectory
        w_h: Weight for information gain
        w_v: Weight for event value
        event_utility: Event utility mapping
        gamma: Discount factor
        
    Returns:
        Tuple of:
        - TrajectoryBuffer with collected trajectory
        - observations_history: List of observation dicts, one per timestep.
                              Each dict maps agent_name -> {node: observed_state}
        - ideal_belief: Final ideal centralized belief (after all evolutions)
        - final_positions: Final agent positions tensor [num_agents]
    """
    from .graph_belief import GraphBelief
    
    buffer = TrajectoryBuffer(max_size=num_steps * 2)  # Allow some overflow
    observations_history = []  # Track observations for ideal belief computation
    
    # Initialize ideal centralized belief (always maintained for critic)
    ideal_belief = GraphBelief(num_nodes=env.num_nodes, num_states=2)
    ideal_belief.reset()  # Uniform prior
    
    # Event tracking: {(node_id, start_step): {start_time, det_time, detected, utility, expected_lifetime}}
    events: dict[tuple[int, int], dict] = {}
    
    # Reset environment and agents if needed
    # (Assuming env and agents are already reset)
    
    # Get initial environment state
    env_state_before = env.get_state().copy()
    
    for step in range(num_steps):
        # STEP 1: Save belief before acting (for reward computation and critic state)
        # Create a copy of the belief by manually copying its state
        ideal_belief_before = GraphBelief(num_nodes=env.num_nodes, num_states=ideal_belief.num_states)
        ideal_belief_before.probabilities = copy.deepcopy(ideal_belief.probabilities)
        ideal_belief_before._binary_probs = copy.deepcopy(ideal_belief._binary_probs)
        ideal_belief_before.valid_nodes = copy.deepcopy(ideal_belief.valid_nodes)
        
        # STEP 2: Update communication info (check if agents can communicate)
        for agent in agents:
            other_agents = [a for a in agents if a.name != agent.name]
            agent.update_communication_info(other_agents, env, step)
        
        # STEP 3: Build local states for all agents (based on current beliefs, positions, AoI)
        local_states = {}
        for agent in agents:
            other_agents = [a for a in agents if a.name != agent.name]
            local_states[agent.name] = agent.build_local_state(other_agents, step)
        
        # STEP 4: Sample actions from current policy (based on local states)
        joint_action = {}
        action_log_probs = {}
        for agent in agents:
            other_agents = [a for a in agents if a.name != agent.name]
            action_info = agent.act(env, other_agents, step, return_log_prob=True)
            joint_action[agent.name] = action_info['action']
            action_log_probs[agent.name] = action_info['log_prob']
        
        # STEP 5: Build centralized state/belief for critic (using belief BEFORE acting)
        num_nodes = env.num_nodes
        belief_array = np.zeros((num_nodes, 2), dtype=np.float32)
        for node in range(num_nodes):
            prob = ideal_belief_before.get_probability(node, 1)  # P(E=1)
            belief_array[node, 0] = 1.0 - prob
            belief_array[node, 1] = prob
        state_tensor = torch.from_numpy(belief_array)
        
        # Agent positions at time t (before action execution)
        positions = torch.tensor([agent.current_node for agent in agents], dtype=torch.long)
        
        # STEP 6: Execute actions (move agents to target nodes)
        for agent in agents:
            target_node = joint_action[agent.name]
            agent.move_to_node(target_node, env)
        
        # STEP 7: Environment evolves
        env.step()
        env_state_after = env.get_state().copy()
        
        # STEP 8: Observe new nodes (agents observe where they moved to)
        observations_at_step = {}
        for agent in agents:
            observed_state = agent.observe_current_node(env)
            observed_node = agent.current_node
            observations_at_step[agent.name] = {observed_node: observed_state}
        
        # Store observations for this timestep (for bootstrap value computation)
        observations_history.append(observations_at_step)
        
        # STEP 9: Update individual agent beliefs with observations
        # (This handles both own observations and received data from communication)
        for agent in agents:
            observed_state = observations_at_step[agent.name][agent.current_node]
            agent.update_belief_from_observation(env, agent.current_node, observed_state)
        
        # STEP 10: Update ideal centralized belief with joint observations
        # Collect all observations from all agents at this timestep
        all_observations_at_step = {}
        for agent_name, agent_observations in observations_at_step.items():
            for node, observed_state in agent_observations.items():
                if node not in all_observations_at_step:
                    all_observations_at_step[node] = observed_state
                elif observed_state == 1:  # Prefer state=1 (event present)
                    all_observations_at_step[node] = 1
        
        # Update ideal belief with observations
        if all_observations_at_step:
            observed_nodes = set(all_observations_at_step.keys())
            ideal_belief.update_from_observation(observed_nodes, all_observations_at_step)
        
        # ideal_belief is now the "after" belief
        
        # STEP 11: Compute reward using before/after beliefs and detection
        # current_step = step + 1 (since we're after env.step())
        team_reward, events = compute_team_reward(
            centralized_belief_before=ideal_belief_before,
            centralized_belief_after=ideal_belief,
            env_state_before=env_state_before,
            env_state_after=env_state_after,
            observations_at_step=observations_at_step,
            events=events,
            current_step=step + 1,  # After env.step(), so time is step + 1
            env=env,
            w_h=w_h,
            w_v=w_v,
            event_utility=event_utility,
        )
        
        # STEP 12: Get value estimate from critic (using "before" belief state)
        shared_critic = agents[0].critic_network
        if shared_critic is not None:
            shared_critic.eval()
            with torch.no_grad():
                value_tensor = shared_critic(
                    state_tensor.unsqueeze(0),  # Add batch dim
                    positions.unsqueeze(0),  # Add batch dim
                )
                value_estimate = value_tensor.item()
        else:
            value_estimate = 0.0
        
        # STEP 13: Get action masks for all agents (for proper masking during updates)
        action_masks = {}
        for agent in agents:
            mask = agent.get_action_mask(env)
            action_masks[agent.name] = mask
        
        # STEP 14: Store transition in buffer
        # Mark last step as done for proper episode termination in GAE
        done_flag = (step == num_steps - 1)
        buffer.add(
            centralized_state=state_tensor.cpu().numpy(),  # Store as numpy
            local_states=local_states,
            joint_action=joint_action,
            reward=team_reward,
            done=done_flag,  # True only for the last step of the episode
            value=value_estimate,
            action_log_probs=action_log_probs,
            action_masks=action_masks,
            positions=positions.cpu().numpy(),  # Store positions at this timestep
        )
        
        # STEP 15: Evolve beliefs with transition model (after observation and reward computation)
        for agent in agents:
            agent.evolve_belief_with_environment(env, {agent.current_node})
        
        # Evolve ideal centralized belief with transition model
        # Evolve all nodes (no nodes are currently observed during evolution)
        ideal_belief.evolve_with_transition_kernel(env, set(), num_states=2)
        
        # Prepare for next iteration
        env_state_before = env_state_after.copy()
    
    # Return buffer, observations_history, and final ideal_belief/positions for bootstrap
    # Final positions after last action execution
    final_positions = torch.tensor([agent.current_node for agent in agents], dtype=torch.long)
    
    return buffer, observations_history, ideal_belief, final_positions


def train_ctde_agents(
    env: "GraphEnvironment",
    agents: list[GraphCTDEAgent],
    trainer: CTDETrainer,
    num_episodes: int = 100,
    steps_per_episode: int = 20,
    update_frequency: int = 5,  # Update every N episodes
    num_updates: int = 5,  # Number of update iterations per update frequency
    w_h: float = 1.0,
    w_v: float = 1.0,
    event_utility: Optional[dict[int, float]] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Train CTDE agents using MAPPO-style updates.
    
    The critic always uses the ideal centralized belief (joint observations from all agents).
    
    Training loop:
    1. Collect trajectories using current policy
    2. Compute advantages using GAE
    3. Update critic
    4. Save old actor
    5. Update actor (PPO)
    6. Repeat
    
    Args:
        env: GraphEnvironment object
        agents: List of CTDE agents
        trainer: CTDETrainer instance
        num_episodes: Total number of training episodes
        steps_per_episode: Number of steps per episode
        update_frequency: Update every N episodes
        num_updates: Number of update iterations per update frequency
        w_h: Weight for information gain
        w_v: Weight for event value
        event_utility: Event utility mapping
        verbose: If True, print training progress
        
    Returns:
        Dictionary with training statistics
    """
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'value_losses': [],
        'policy_losses': [],
        'entropies': [],
    }
    
    # Accumulate trajectories across multiple episodes before updating
    accumulated_buffer = TrajectoryBuffer(max_size=num_episodes * steps_per_episode * 2)
    
    for episode in range(num_episodes):
        # Reset environment state (preserves environment parameters: RSP params, RNG seed, etc.)
        # The environment remains stochastic but uses the same parameters across all episodes
        env.reset()
        for agent in agents:
            agent.belief.reset()
            # Reset agent positions (could be randomized)
            # For now, keep initial positions
        
        # Collect trajectory
        buffer, observations_history, final_ideal_belief, final_positions = collect_trajectory(
            env,
            agents,
            steps_per_episode,
            w_h,
            w_v,
            event_utility,
            gamma=trainer.gamma,
        )
        
        # Accumulate trajectory data
        for t in range(buffer.length()):
            accumulated_buffer.add(
                centralized_state=buffer.centralized_states[t],
                local_states={agent_name: buffer.local_states[agent_name][t] 
                             for agent_name in buffer.local_states.keys()},
                joint_action=buffer.joint_actions[t],
                reward=buffer.rewards[t],
                done=buffer.dones[t],
                value=buffer.values[t],
                action_log_probs={agent_name: buffer.action_log_probs[agent_name][t]
                                 for agent_name in buffer.action_log_probs.keys()},
                action_masks={agent_name: buffer.action_masks[agent_name][t]
                             for agent_name in buffer.action_masks.keys()},
                positions=buffer.positions[t] if len(buffer.positions) > t else None,
            )
        
        # Compute episode statistics
        episode_reward = sum(buffer.rewards)
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(buffer.length())
        
        if verbose and (episode + 1) % update_frequency != 0:
            print(f"Episode {episode+1}/{num_episodes}: Reward={episode_reward:.2f}, "
                  f"Length={buffer.length()}")
        
        # Update networks periodically (using accumulated buffer from multiple episodes)
        if (episode + 1) % update_frequency == 0:
            # Use final ideal belief and positions from last episode for bootstrap
            # This corresponds to state at time T (after last step's evolution)
            if accumulated_buffer.length() > 0 and final_ideal_belief is not None:
                # Convert final ideal belief to tensor
                num_nodes = env.num_nodes
                belief_array = np.zeros((num_nodes, 2), dtype=np.float32)
                for node in range(num_nodes):
                    prob = final_ideal_belief.get_probability(node, 1)
                    belief_array[node, 0] = 1.0 - prob
                    belief_array[node, 1] = prob
                last_centralized_state = torch.from_numpy(belief_array)
                last_positions = final_positions
                
                # All agents share the same critic network (centralized/shared)
                shared_critic = agents[0].critic_network
                shared_critic.eval()
                with torch.no_grad():
                    next_value = shared_critic(
                        last_centralized_state.unsqueeze(0),
                        last_positions.unsqueeze(0),
                    ).item()
            else:
                # Truncated episodes: no bootstrap (valid for finite-horizon)
                next_value = 0.0
            
            # Compute advantages using accumulated buffer
            advantages, returns = compute_gae_advantages(
                accumulated_buffer.rewards,
                accumulated_buffer.values,
                accumulated_buffer.dones,
                gamma=trainer.gamma,
                lambda_gae=trainer.lambda_gae,
                next_value=next_value,
            )
            
            # Convert states and positions to tensors for updates (from accumulated buffer)
            centralized_states_list = [
                torch.from_numpy(state) if isinstance(state, np.ndarray) else state
                for state in accumulated_buffer.centralized_states
            ]
            # Use stored positions from buffer (not current agent positions!)
            positions_list = [
                torch.from_numpy(pos) if isinstance(pos, np.ndarray) else torch.tensor(pos, dtype=torch.long)
                for pos in accumulated_buffer.positions
            ]
            positions_batch = torch.stack(positions_list)  # [T, num_agents]
            
            # Convert local_states from {agent_name: [s_i(0), s_i(1), ...]} format
            # to [{agent_name: s_i(t)} for each t] format expected by update_actor
            local_states_list = []
            if accumulated_buffer.length() > 0:
                agent_names = list(accumulated_buffer.local_states.keys())
                for t in range(accumulated_buffer.length()):
                    timestep_states = {}
                    for agent_name in agent_names:
                        if t < len(accumulated_buffer.local_states[agent_name]):
                            timestep_states[agent_name] = accumulated_buffer.local_states[agent_name][t]
                    local_states_list.append(timestep_states)
            
            # Multiple update iterations
            for update_iter in range(num_updates):
                # Update critic
                critic_stats = trainer.update_critic(
                    centralized_states_list,
                    returns,
                    positions_batch,
                )
                
                # Save old actor
                trainer.save_old_actor()
                
                # Update actor
                actor_stats = trainer.update_actor(
                    agents,
                    env,
                    local_states_list,
                    accumulated_buffer.joint_actions,
                    advantages,
                    accumulated_buffer.action_log_probs,
                    action_masks_old=accumulated_buffer.action_masks if hasattr(accumulated_buffer, 'action_masks') else None,
                )
                
                # Track losses for statistics
                training_stats['value_losses'].append(critic_stats['value_loss'])
                training_stats['policy_losses'].append(actor_stats['policy_loss'])
                training_stats['entropies'].append(actor_stats['entropy'])
                
                if verbose:
                    print(f"  Update {update_iter+1}/{num_updates}: "
                          f"Value Loss={critic_stats['value_loss']:.4f}, "
                          f"Policy Loss={actor_stats['policy_loss']:.4f}, "
                          f"Entropy={actor_stats['entropy']:.4f}, "
                          f"Return Mean={critic_stats['return_mean']:.2f}, "
                          f"Value Mean={critic_stats['value_mean']:.2f}, "
                          f"Adv Mean={actor_stats.get('advantage_mean_raw', 0):.4f}, "
                          f"Adv Std={actor_stats.get('advantage_std_raw', 0):.4f}")
            
            # Clear accumulated buffer after updates
            accumulated_buffer.clear()
    
    return training_stats


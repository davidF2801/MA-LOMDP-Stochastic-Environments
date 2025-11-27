"""Training loop for CTDE agents."""

from __future__ import annotations

from typing import Any, Optional
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


def compute_team_reward(
    centralized_belief: "GraphBelief",
    env: "GraphEnvironment",
    joint_action: dict[str, int],
    w_h: float = 1.0,
    w_v: float = 1.0,
    event_utility: Optional[dict[int, float]] = None,
) -> float:
    """
    Compute team reward over joint action using centralized/ideal belief.
    
    This is the reward that the critic sees during training.
    
    r_t = w_h * Σ_{a in joint_action} G(B_ideal, {a}) + w_v * Σ_{a in joint_action} E[value | B_ideal, {a}]
    
    where:
    - B_ideal is the ideal centralized belief (joint observations from all agents)
    - joint_action is the set of nodes that will be observed by the team
    
    Args:
        centralized_belief: GraphBelief object representing the centralized/ideal belief
                           (or true state converted to belief format)
        env: GraphEnvironment object
        joint_action: Dictionary {agent_name: action (target node)}
        w_h: Weight for information gain
        w_v: Weight for event value
        event_utility: Event utility mapping {state: utility}
        
    Returns:
        Team reward computed over joint action and centralized belief
    """
    if event_utility is None:
        event_utility = {0: 0.0, 1: 1.0}
    
    utility_0 = event_utility.get(0, 0.0)
    utility_1 = event_utility.get(1, 0.0)
    
    # Collect all nodes that will be observed by the joint action
    nodes_to_observe = set(joint_action.values())
    
    # Compute information gain: entropy reduction from observing these nodes
    # Information gain = prior entropy (after perfect observation, entropy = 0)
    info_gain = centralized_belief.entropy(nodes_to_observe)
    
    # Compute expected event value: Σ_{node in nodes_to_observe} Σ_x B_ideal(node, x) * f(x)
    event_value = 0.0
    for node in nodes_to_observe:
        prob = centralized_belief.get_probability(node, 1)  # P(E=1)
        # E[value] = P(0) * f(0) + P(1) * f(1)
        node_value = (1.0 - prob) * utility_0 + prob * utility_1
        event_value += node_value
    
    # Total reward
    reward = w_h * info_gain + w_v * event_value
    return float(reward)


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
    
    # Reset environment and agents if needed
    # (Assuming env and agents are already reset)
    
    for step in range(num_steps):
        # STEP 1: Observe current nodes (agents observe where they are)
        observations_at_step = {}
        for agent in agents:
            observed_state = agent.observe_current_node(env)
            observed_node = agent.current_node
            observations_at_step[agent.name] = {observed_node: observed_state}
        
        # STEP 2: Update communication info (check if agents can communicate)
        for agent in agents:
            other_agents = [a for a in agents if a.name != agent.name]
            agent.update_communication_info(other_agents, env, step)
        
        # STEP 3: Update individual agent beliefs with observations
        # (This handles both own observations and received data from communication)
        for agent in agents:
            observed_state = observations_at_step[agent.name][agent.current_node]
            agent.update_belief_from_observation(env, agent.current_node, observed_state)
        
        # STEP 4: Update ideal centralized belief with joint observations
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
        
        # Store observations for this timestep (for bootstrap value computation)
        observations_history.append(observations_at_step)
        
        # STEP 5: Build local states for all agents (based on updated beliefs, positions, AoI)
        local_states = {}
        for agent in agents:
            other_agents = [a for a in agents if a.name != agent.name]
            local_states[agent.name] = agent.build_local_state(other_agents, step)
        
        # STEP 6: Sample actions from current policy (based on local states)
        joint_action = {}
        action_log_probs = {}
        for agent in agents:
            other_agents = [a for a in agents if a.name != agent.name]
            action_info = agent.act(env, other_agents, step, return_log_prob=True)
            joint_action[agent.name] = action_info['action']
            action_log_probs[agent.name] = action_info['log_prob']
        
        # STEP 7: Build centralized state/belief for critic (after observations and belief updates)
        # Always use ideal centralized belief (maintained incrementally)
        num_nodes = env.num_nodes
        belief_array = np.zeros((num_nodes, 2), dtype=np.float32)
        for node in range(num_nodes):
            prob = ideal_belief.get_probability(node, 1)  # P(E=1)
            belief_array[node, 0] = 1.0 - prob
            belief_array[node, 1] = prob
        state_tensor = torch.from_numpy(belief_array)
        centralized_belief = ideal_belief  # Use the maintained belief
        
        # Agent positions
        positions = torch.tensor([agent.current_node for agent in agents], dtype=torch.long)
        
        # STEP 8: Compute team reward over joint action using centralized belief
        # (same belief that the critic sees)
        team_reward = compute_team_reward(
            centralized_belief, env, joint_action, w_h, w_v, event_utility
        )
        
        # Get value estimate from critic
        # Note: All agents share the same critic network instance (it's centralized/shared)
        # We can access it from any agent (agent[0] is just convenient)
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
        
        # Get action masks for all agents (for proper masking during updates)
        action_masks = {}
        for agent in agents:
            mask = agent.get_action_mask(env)
            action_masks[agent.name] = mask
        
        # Store in buffer (store positions at this timestep)
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
        
        # STEP 9: Execute actions (move agents to target nodes)
        for agent in agents:
            target_node = joint_action[agent.name]
            agent.move_to_node(target_node, env)
        
        # STEP 10: Evolve beliefs with transition model (after action execution)
        for agent in agents:
            agent.evolve_belief_with_environment(env, {agent.current_node})
        
        # Evolve ideal centralized belief with transition model
        # Evolve all nodes (no nodes are currently observed during evolution)
        ideal_belief.evolve_with_transition_kernel(env, set(), num_states=2)
        
        # STEP 11: Environment evolves
        env.step()
    
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


module PolicyTreePlanner

using POMDPs
using POMDPTools
using Random

# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..EventState2, ..NO_EVENT_2, ..EVENT_PRESENT_2
import ..SensingAction, ..GridObservation, ..Agent
import ..CircularTrajectory, ..LinearTrajectory, ..RangeLimitedSensor
import ..get_position_at_time
import ..EventDynamicsModule

export best_policy_tree, enumerate_policy_tree

# A depth-C policy tree can be stored as a Dict{Vector{GridObservation}, Vector{SensingAction}}
# key = history (length ≤ k)  |  value = action to execute at node
const PolicyTree = Dict{Vector{GridObservation}, SensingAction}

"""
enumerate_policy_tree(actions::Vector{SensingAction}, observations::Vector{GridObservation}, C::Int)
    recursive generator, returns Vector{PolicyTree}
"""
function enumerate_policy_tree(actions::Vector{SensingAction}, observations::Vector{GridObservation}, C::Int)
    if C == 0
        return PolicyTree[]
    end
    
    trees = PolicyTree[]
    
    # For each possible action at the root
    for action in actions
        # Generate all possible observation sequences of length C-1
        obs_sequences = generate_observation_sequences(observations, C-1)
        
        # For each observation sequence, create a subtree
        for obs_seq in obs_sequences
            tree = PolicyTree()
            
            # Root action
            tree[GridObservation[]] = action
            
            # Build subtree recursively
            build_subtree!(tree, action, obs_seq, actions, observations, 1, C)
            
            push!(trees, tree)
        end
    end
    
    return trees
end

"""
Build subtree recursively
"""
function build_subtree!(tree::PolicyTree, parent_action::SensingAction, obs_sequence::Vector{GridObservation}, 
                       actions::Vector{SensingAction}, observations::Vector{GridObservation}, depth::Int, max_depth::Int)
    if depth >= max_depth
        return
    end
    
    # For each possible action after this observation
    for action in actions
        # Create history path
        history = [parent_action.agent_id => obs_sequence[1:depth]]
        
        # Add action to tree
        tree[history] = action
        
        # Recursively build deeper subtrees
        if depth + 1 < max_depth
            remaining_obs = obs_sequence[depth+1:end]
            build_subtree!(tree, action, remaining_obs, actions, observations, depth + 1, max_depth)
        end
    end
end

"""
Generate all possible observation sequences of length C
"""
function generate_observation_sequences(observations::Vector{GridObservation}, C::Int)
    if C == 0
        return Vector{GridObservation}[]
    elseif C == 1
        return [[obs] for obs in observations]
    else
        sequences = Vector{GridObservation}[]
        shorter_sequences = generate_observation_sequences(observations, C - 1)
        
        for obs in observations
            for shorter_seq in shorter_sequences
                new_seq = [obs; shorter_seq]
                push!(sequences, new_seq)
            end
        end
        
        return sequences
    end
end

"""
best_policy_tree(env, belief, agent, C, other_policies)
  – Enumerate all policy trees generated above.
  – For each tree:
        • Monte-Carlo sample N=50 rollouts  (exact enumeration explosive)
        • During rollout draw obs ∼ P(O|a, s′)
        • Update belief with true drawn obs
        • For other agents: sample their tree branch using same obs history
  – Average discounted reward; keep best.
"""
function best_policy_tree(env, belief, agent, C::Int, other_policies; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Get available actions and observations
    available_actions = get_available_actions(agent, env)
    available_observations = get_available_observations(agent, env)
    
    if isempty(available_actions)
        return PolicyTree()
    end
    
    # Enumerate all policy trees
    policy_trees = enumerate_policy_tree(available_actions, available_observations, C)
    
    best_tree = nothing
    best_value = -Inf
    
    println("🌳 Evaluating $(length(policy_trees)) policy trees for agent $(agent.id)")
    
    for (i, tree) in enumerate(policy_trees)
        # Evaluate this policy tree using Monte Carlo rollouts
        value = evaluate_policy_tree(env, belief, agent, tree, other_policies, C, rng)
        
        if value > best_value
            best_value = value
            best_tree = tree
        end
        
        if i % 10 == 0
            println("  Evaluated $(i)/$(length(policy_trees)) trees, best value: $(round(best_value, digits=3))")
        end
    end
    
    println("✅ Best policy tree found with value: $(round(best_value, digits=3))")
    return best_tree
end

"""
Evaluate a policy tree using Monte Carlo rollouts
"""
function evaluate_policy_tree(env, belief, agent, tree::PolicyTree, other_policies, C::Int, rng::AbstractRNG)
    num_rollouts = 50  # Number of Monte Carlo samples
    total_value = 0.0
    
    for rollout in 1:num_rollouts
        # Copy belief for this rollout
        current_belief = copy(belief)
        current_state = sample_from_belief(current_belief, rng)
        
        rollout_value = 0.0
        discount_factor = 1.0
        
        # Initialize observation history
        obs_history = GridObservation[]
        
        for k in 1:C
            # Get action from policy tree based on observation history
            action = get_action_from_tree(tree, obs_history)
            
            if action === nothing
                # Default wait action if no policy found
                action = SensingAction(agent.id, Tuple{Int, Int}[], false)
            end
            
            # Get other agents' actions for this step
            other_actions = get_other_actions_from_policies(other_policies, obs_history, k)
            
            # Simulate environment transition
            joint_action = [action; other_actions]
            next_state = simulate_transition(env, current_state, joint_action, rng)
            
            # Calculate reward (information gain - cost)
            reward = calculate_reward(env, current_state, action, current_belief)
            
            # Sample observation from P(O|a, s')
            observation = sample_observation(env, next_state, action, rng)
            
            # Update belief with true observation
            update_belief_with_observation!(current_belief, action, observation, env)
            
            # Update observation history
            push!(obs_history, observation)
            
            # Accumulate discounted reward
            rollout_value += discount_factor * reward
            
            # Update state and discount factor
            current_state = next_state
            discount_factor *= 0.95  # gamma discount
        end
        
        total_value += rollout_value
    end
    
    return total_value / num_rollouts
end

"""
Get action from policy tree based on observation history
"""
function get_action_from_tree(tree::PolicyTree, obs_history::Vector{GridObservation})
    # Try to find exact match
    if haskey(tree, obs_history)
        return tree[obs_history]
    end
    
    # Try to find partial match (use longest matching prefix)
    for i in length(obs_history)-1:-1:0
        prefix = obs_history[1:i]
        if haskey(tree, prefix)
            return tree[prefix]
        end
    end
    
    return nothing
end

"""
Get other agents' actions from their policies
"""
function get_other_actions_from_policies(other_policies, obs_history::Vector{GridObservation}, step::Int)
    actions = SensingAction[]
    
    for policy in other_policies
        action = get_action_from_tree(policy, obs_history)
        if action === nothing
            # Default wait action
            push!(actions, SensingAction(0, Tuple{Int, Int}[], false))
        else
            push!(actions, action)
        end
    end
    
    return actions
end

"""
Sample observation from P(O|a, s')
"""
function sample_observation(env, state, action::SensingAction, rng::AbstractRNG)
    # For now, return a simple observation
    # In a full implementation, this would sample from P(O|a, s')
    return GridObservation(action.agent_id, action.target_cells, EventState[], [])
end

"""
Get available actions for an agent
"""
function get_available_actions(agent, env)
    # Get agent's current position
    current_pos = get_agent_position(agent, env)
    
    # Get cells within sensor range (Field of View)
    fov_cells = get_field_of_view(agent, current_pos, env)
    
    # Generate sensing actions for different subsets of FOV cells
    actions = SensingAction[]
    
    # Add wait action
    push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))
    
    # Add single-cell sensing actions
    for cell in fov_cells
        push!(actions, SensingAction(agent.id, [cell], false))
    end
    
    # Add multi-cell sensing actions (up to max_sensing_targets)
    if length(fov_cells) > 1 && env.max_sensing_targets > 1
        for subset_size in 2:min(env.max_sensing_targets, length(fov_cells))
            for subset in combinations(fov_cells, subset_size)
                push!(actions, SensingAction(agent.id, collect(subset), false))
            end
        end
    end
    
    return actions
end

"""
Get available observations for an agent
"""
function get_available_observations(agent, env)
    # For now, return a simple set of observations
    # In a full implementation, this would generate all possible observations
    observations = GridObservation[]
    
    # Add empty observation (no events detected)
    push!(observations, GridObservation(agent.id, Tuple{Int, Int}[], EventState[], []))
    
    # Add observations with events
    for event_state in [NO_EVENT, EVENT_PRESENT]
        push!(observations, GridObservation(agent.id, [(1,1)], [event_state], []))
    end
    
    return observations
end

"""
Get agent's current position based on the agent's trajectory cycle phase
"""
function get_agent_position(agent, env)
    # Calculate position directly using trajectory and phase offset
    t = get_current_time(env, agent)
    
    # Apply phase offset
    adjusted_time = t + agent.phase_offset
    
    # Calculate position based on trajectory type
    if typeof(agent.trajectory) <: CircularTrajectory
        angle = 2π * (adjusted_time % agent.trajectory.period) / agent.trajectory.period
        x = agent.trajectory.center_x + round(Int, agent.trajectory.radius * cos(angle))
        y = agent.trajectory.center_y + round(Int, agent.trajectory.radius * sin(angle))
        return (x, y)
    elseif typeof(agent.trajectory) <: LinearTrajectory
        t_normalized = (adjusted_time % agent.trajectory.period) / agent.trajectory.period
        x = round(Int, agent.trajectory.start_x + t_normalized * (agent.trajectory.end_x - agent.trajectory.start_x))
        y = round(Int, agent.trajectory.start_y + t_normalized * (agent.trajectory.end_y - agent.trajectory.start_y))
        return (x, y)
    else
        return (1, 1)  # fallback
    end
end

# Helper to get the current time step for the agent (assume env has a time_step or pass as argument)
function get_current_time(env, agent)
    if hasproperty(env, :time_step)
        return env.time_step
    elseif hasfield(typeof(env), :time_step)
        return getfield(env, :time_step)
    else
        return 0
    end
end

"""
Get field of view for an agent at a position
"""
function get_field_of_view(agent, position, env)
    # For now, return a simple FOV
    # In a full implementation, this would use the agent's sensor model
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    for dx in -1:1
        for dy in -1:1
            nx, ny = x + dx, y + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                push!(fov_cells, (nx, ny))
            end
        end
    end
    
    return fov_cells
end

"""
Simulate environment transition using the environment's transition model
"""
function simulate_transition(env, current_state, joint_action, rng::AbstractRNG)
    # For now, use a simple transition model
    # In a full implementation, this would use the environment's proper transition model
    
    # Create a new state with updated event map
    new_state = copy(current_state)
    
    # Update events using the environment's event dynamics
    if hasfield(typeof(env), :event_dynamics)
        # Convert to DBN model for environment update
        dbn_model = DBNTransitionModel2(env.event_dynamics)
        
        # Convert EventState to EventState2 for DBN update
        event_map_2 = Matrix{EventState2}(undef, env.height, env.width)
        for y in 1:env.height
            for x in 1:env.width
                event_map_2[y, x] = new_state.event_map[y, x] == EVENT_PRESENT ? EVENT_PRESENT_2 : NO_EVENT_2
            end
        end
        
        # Update environment
        EventDynamicsModule.update_events!(dbn_model, event_map_2, rng)
        
        # Convert back to EventState
        for y in 1:env.height
            for x in 1:env.width
                new_state.event_map[y, x] = event_map_2[y, x] == EVENT_PRESENT_2 ? EVENT_PRESENT : NO_EVENT
            end
        end
    end
    
    # Update agent positions based on their trajectories
    new_state.time_step += 1
    for (i, agent) in enumerate(env.agents)
        new_state.agent_positions[i] = get_position_at_time(agent.trajectory, new_state.time_step)
    end
    
    return new_state
end

"""
Calculate reward for an action (information gain - cost)
"""
function calculate_reward(env, state, action::SensingAction, belief)
    if isempty(action.target_cells)
        return 0.0  # Wait action
    end
    
    # Calculate information gain
    information_gain = 0.0
    for cell in action.target_cells
        # Get current uncertainty at this cell
        uncertainty = get_cell_uncertainty(belief, cell)
        information_gain += uncertainty
    end
    
    # Subtract observation cost
    observation_cost = 0.1  # Cost per sensing action
    
    return information_gain - observation_cost
end

"""
Update belief with observation
"""
function update_belief_with_observation!(belief, action::SensingAction, observation::GridObservation, env)
    # For now, do nothing
    # In a full implementation, this would update the belief using Bayes rule
end

"""
Get cell uncertainty from belief
"""
function get_cell_uncertainty(belief, cell)
    # Get the belief distribution for this cell
    cell_belief = belief[cell[1], cell[2]]
    
    # Calculate Shannon entropy: -sum(p * log(p)) for non-zero probabilities
    entropy = 0.0
    for (event_type, prob) in enumerate(cell_belief)
        if prob > 0.0
            entropy -= prob * log(prob)
        end
    end
    
    return entropy
end

"""
Generate combinations of elements
"""
function combinations(elements, k)
    if k == 0
        return [Tuple{}[]]
    elseif k == 1
        return [[element] for element in elements]
    else
        result = []
        for i in 1:length(elements)
            for combo in combinations(elements[i+1:end], k-1)
                push!(result, [elements[i]; combo])
            end
        end
        return result
    end
end

end # module 
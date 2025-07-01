module MacroPlanner

using POMDPs
using POMDPTools
using Random
using ..Types
import ..Agents.BeliefManagement: sample_from_belief
# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..EventState2, ..NO_EVENT_2, ..EVENT_PRESENT_2
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..RangeLimitedSensor
import ..get_position_at_time

export best_script

"""
best_script(env, belief::GlobalBeliefState, agent::Agent, C::Int, other_scripts)::Vector{SensingAction}
  – Enumerate every |A|^C open-loop action sequence for `agent`.
  – For each sequence:
        • Roll out C steps:  (simulate using env.transition & env.observation)
        • Propagate local belief only with *predicted* observations
          (use expectation, i.e. marginalise over obs distribution).
        • Plug reward = ∑ γ^k R( … )   [use existing reward() helper]
        • For other agents use `other_scripts[k]` (deterministic vector passed in)
  – Return argmax sequence (ties → first).
"""
function best_script(env, belief, agent, C::Int, other_scripts; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Get available actions for the agent
    available_actions = get_available_actions(agent, env)
    
    if isempty(available_actions)
        return SensingAction[]
    end
    
    # Enumerate all possible action sequences of length C
    action_sequences = generate_action_sequences(available_actions, C)
    
    best_sequence = SensingAction[]  # Default to empty sequence
    best_value = -Inf
    
    println("🔍 Evaluating $(length(action_sequences)) action sequences for agent $(agent.id)")
    
    for (i, sequence) in enumerate(action_sequences)
        # Evaluate this sequence
        value = evaluate_action_sequence(env, belief, agent, sequence, other_scripts, C, rng)
        
        if value > best_value
            best_value = value
            best_sequence = sequence
        end
        
        if i % 100 == 0
            println("  Evaluated $(i)/$(length(action_sequences)) sequences, best value: $(round(best_value, digits=3))")
        end
    end
    
    println("✅ Best sequence found with value: $(round(best_value, digits=3))")
    return best_sequence
end

"""
Generate all possible action sequences of length C
"""
function generate_action_sequences(actions::Vector{SensingAction}, C::Int)
    if C == 0
        return Vector{SensingAction}[]
    elseif C == 1
        return [[action] for action in actions]
    else
        sequences = Vector{SensingAction}[]
        shorter_sequences = generate_action_sequences(actions, C - 1)
        
        for action in actions
            for shorter_seq in shorter_sequences
                new_seq = [action; shorter_seq]
                push!(sequences, new_seq)
            end
        end
        
        return sequences
    end
end

"""
Evaluate a single action sequence using Monte Carlo rollouts
"""
function evaluate_action_sequence(env, belief, agent, sequence::Vector{SensingAction}, other_scripts, C::Int, rng::AbstractRNG)
    num_rollouts = 50  # Number of Monte Carlo samples
    total_value = 0.0
    
    for rollout in 1:num_rollouts
        # Copy belief for this rollout
        current_belief = copy(belief)
        current_state = sample_from_belief(current_belief, rng)
        
        rollout_value = 0.0
        discount_factor = 1.0
        
        for k in 1:min(length(sequence), C)
            # Get current action
            action = sequence[k]
            
            # Get other agents' actions for this step
            other_actions = get_other_actions(other_scripts, k)
            
            # Simulate environment transition
            joint_action = [action; other_actions]
            next_state = simulate_transition(env, current_state, joint_action, rng)
            
            # Calculate reward (information gain - cost)
            reward = calculate_reward(env, current_state, action, current_belief)
            
            # Update belief with predicted observations (expectation over observation distribution)
            observation = predict_observation(env, current_state, action, rng)
            update_belief_with_observation!(current_belief, action, observation, env)
            
            # Accumulate discounted reward
            rollout_value += discount_factor * reward
            
            # Update state and discount factor
            current_state = next_state
            discount_factor *= 0.95  # gamma discount
            
            # Early termination if no more actions
            if k >= length(sequence)
                break
            end
        end
        
        total_value += rollout_value
    end
    
    return total_value / num_rollouts
end

"""
Get available actions for an agent based on its current position and sensor capabilities
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
Get other agents' actions for a given time step
"""
function get_other_actions(other_scripts, step::Int)
    actions = SensingAction[]
    
    for script in other_scripts
        if script !== nothing && step <= length(script)
            push!(actions, script[step])
        else
            # If script is shorter, repeat last action or use wait
            if script !== nothing && !isempty(script)
                push!(actions, script[end])
            else
                # Default wait action
                push!(actions, SensingAction(0, Tuple{Int, Int}[], false))
            end
        end
    end
    
    return actions
end

"""
Simulate environment transition
"""
function simulate_transition(env, current_state, joint_action, rng::AbstractRNG)
    # For now, return a copy of current state
    # In a full implementation, this would use the environment's transition model
    return copy(current_state)
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
Predict observation for an action (expectation over observation distribution)
"""
function predict_observation(env, state, action::SensingAction, rng::AbstractRNG)
    # For now, return a simple observation
    # In a full implementation, this would sample from P(O|a, s)
    return GridObservation(action.agent_id, action.target_cells, EventState[], [])
end

"""
Update belief with observation
"""
function update_belief_with_observation!(belief, action::SensingAction, observation::GridObservation, env)
    # For now, do nothing
    # In a full implementation, this would update the belief using Bayes rule
end

"""
Get agent's current position
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
    # Try to get time_step from env, fallback to 0 if not present
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
Get cell uncertainty from belief
"""
function get_cell_uncertainty(belief, cell)
    # For now, return a default uncertainty
    # In a full implementation, this would get the actual uncertainty from the belief
    return 0.5
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
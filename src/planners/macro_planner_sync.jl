module MacroPlannerSync

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Combinatorics
using ..Types
import ..Agents.BeliefManagement: sample_from_belief
# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..EventState2, ..NO_EVENT_2, ..EVENT_PRESENT_2
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..RangeLimitedSensor
# Import trajectory functions
import ..Agents.TrajectoryPlanner.get_position_at_time
# Import DBN functions for transition modeling
import ..Environment.EventDynamicsModule.DBNTransitionModel2, ..Environment.EventDynamicsModule.predict_next_belief_dbn
# Import belief management functions
import ..Agents.BeliefManagement.predict_belief_evolution_dbn, ..Agents.BeliefManagement.Belief,
       ..Agents.BeliefManagement.calculate_uncertainty_from_distribution

export plan_synchronous_cycle, update_global_belief_sync, evaluate_joint_action_sequence

"""
JointAction - Represents a joint action for all agents
"""
struct JointAction
    actions::Vector{SensingAction}  # One action per agent
end

"""
JointObservation - Represents joint observations from all agents
"""
struct JointObservation
    observations::Vector{GridObservation}  # One observation per agent
end

"""
plan_synchronous_cycle(env, global_belief::Belief, agents::Vector{Agent}, C::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
Plans a synchronous cycle where all agents coordinate their actions for the next C timesteps.
Uses POMDPs.jl solvers for joint action optimization.
"""
function plan_synchronous_cycle(env, global_belief::Belief, agents::Vector{Agent}, C::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    println("🔄 Planning synchronous cycle for $(length(agents)) agents over $(C) timesteps")
    
    # Generate all possible joint action sequences
    joint_action_sequences = generate_joint_action_sequences(env, agents, C)
    
    if isempty(joint_action_sequences)
        println("⚠️ No valid joint action sequences found")
        return Vector{Vector{SensingAction}}[]  # Return empty sequences for each agent
    end
    
    println("🔍 Evaluating $(length(joint_action_sequences)) joint action sequences")
    
    best_sequence = joint_action_sequences[1]  # Default to first sequence
    best_value = -Inf
    
    for (i, joint_sequence) in enumerate(joint_action_sequences)
        # Evaluate this joint action sequence
        value = evaluate_joint_action_sequence(env, global_belief, agents, joint_sequence, C, rng)
        
        if value > best_value
            best_value = value
            best_sequence = joint_sequence
        end
        
        if i % 50 == 0
            println("  Evaluated $(i)/$(length(joint_action_sequences)) sequences, best value: $(round(best_value, digits=3))")
        end
    end
    
    println("✅ Best joint sequence found with value: $(round(best_value, digits=3))")
    
    # Convert joint sequence to individual agent sequences
    agent_sequences = Vector{Vector{SensingAction}}(undef, length(agents))
    for agent_idx in 1:length(agents)
        agent_sequences[agent_idx] = [joint_step[agent_idx] for joint_step in best_sequence]
    end
    
    return agent_sequences
end

"""
generate_joint_action_sequences(env, agents::Vector{Agent}, C::Int)
Generates all possible joint action sequences of length C for all agents.
"""
function generate_joint_action_sequences(env, agents::Vector{Agent}, C::Int)
    if C == 0
        return Vector{Vector{Vector{SensingAction}}}()
    end
    
    # 1. Get available actions for each agent at each timestep
    actions_per_agent_per_timestep = Vector{Vector{Vector{SensingAction}}}()
    
    for agent in agents
        agent_actions_per_timestep = Vector{Vector{SensingAction}}()
        
        for t in 1:C
            # Get agent's position at this timestep
            agent_pos = get_agent_position_at_time(agent, env, t-1)
            
            # Get available actions for this agent at this timestep
            available_actions = get_available_actions_at_time(agent, agent_pos, env)
            push!(agent_actions_per_timestep, available_actions)
        end
        
        push!(actions_per_agent_per_timestep, agent_actions_per_timestep)
    end
    
    # 2. Generate all joint action sequences using a simpler approach
    joint_sequences = generate_joint_sequences_simple(actions_per_agent_per_timestep, C)
    
    return joint_sequences
end

"""
generate_joint_sequences_simple(actions_per_agent_per_timestep, C::Int)
Generates joint action sequences using a simpler, non-recursive approach.
"""
function generate_joint_sequences_simple(actions_per_agent_per_timestep, C::Int)
    num_agents = length(actions_per_agent_per_timestep)
    
    # For each timestep, generate all possible joint actions
    timestep_joint_actions = Vector{Vector{Vector{SensingAction}}}()
    
    for t in 1:C
        # Get actions for each agent at this timestep
        agent_actions = [actions_per_agent_per_timestep[agent_idx][t] for agent_idx in 1:num_agents]
        
        # Generate all combinations of individual actions
        action_combinations = collect(Iterators.product(agent_actions...))
        
        # Convert each combination to a joint action
        timestep_actions = Vector{Vector{SensingAction}}()
        for combo in action_combinations
            joint_action = collect(combo)  # Vector{SensingAction}
            push!(timestep_actions, joint_action)
        end
        
        push!(timestep_joint_actions, timestep_actions)
    end
    
    # Now generate all sequences by taking one joint action from each timestep
    sequences = Vector{Vector{Vector{SensingAction}}}()
    
    # Use a simple cartesian product approach
    if C == 1
        # Just return all joint actions for the single timestep
        for joint_action in timestep_joint_actions[1]
            push!(sequences, [joint_action])
        end
    else
        # Generate all combinations across timesteps
        for t1_action in timestep_joint_actions[1]
            if C == 2
                for t2_action in timestep_joint_actions[2]
                    push!(sequences, [t1_action, t2_action])
                end
            elseif C == 3
                for t2_action in timestep_joint_actions[2]
                    for t3_action in timestep_joint_actions[3]
                        push!(sequences, [t1_action, t2_action, t3_action])
                    end
                end
            else
                # For longer sequences, use a more general approach
                remaining_sequences = generate_joint_sequences_simple(actions_per_agent_per_timestep[2:end], C-1)
                for remaining_seq in remaining_sequences
                    push!(sequences, [t1_action; remaining_seq])
                end
            end
        end
    end
    
    return sequences
end

"""
get_available_actions_at_time(agent::Agent, position::Tuple{Int, Int}, env)
Gets available actions for an agent at a specific position and time.
"""
function get_available_actions_at_time(agent::Agent, position::Tuple{Int, Int}, env)
    # Get Field of Regard (FOR) at this position
    for_cells = get_field_of_regard_at_position(agent, position, env)
    
    actions = SensingAction[]
    
    # Add wait action
    push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))
    
    # Add single-cell sensing actions
    for cell in for_cells
        push!(actions, SensingAction(agent.id, [cell], false))
    end
    
    # Add multi-cell sensing actions (up to max_sensing_targets)
    if length(for_cells) > 1 && env.max_sensing_targets > 1
        for subset_size in 2:min(env.max_sensing_targets, length(for_cells))
            for subset in combinations(for_cells, subset_size)
                push!(actions, SensingAction(agent.id, collect(subset), false))
            end
        end
    end
    
    return actions
end

"""
evaluate_joint_action_sequence(env, global_belief::Belief, agents::Vector{Agent}, joint_sequence, C::Int, rng::AbstractRNG)
Evaluates a joint action sequence using the global belief and environment model.
"""
function evaluate_joint_action_sequence(env, global_belief::Belief, agents::Vector{Agent}, joint_sequence, C::Int, rng::AbstractRNG)
    γ = env.discount
    value = 0.0
    belief = deepcopy(global_belief)  # Copy belief for deterministic evolution
    
    for k in 1:min(length(joint_sequence), C)
        # Get joint action for this timestep
        joint_action = joint_sequence[k]
        
        # Calculate expected joint information gain
        joint_gain = calculate_joint_information_gain(belief, joint_action, env, agents, k-1)
        
        # Accumulate discounted reward
        value += (γ)^(k-1) * joint_gain
        
        # Simulate the step: belief evolution + joint observations
        belief = simulate_joint_step(belief, joint_action, env, agents, k-1)
    end
    
    return value
end

"""
calculate_joint_information_gain(belief::Belief, joint_action::Vector{SensingAction}, env, agents::Vector{Agent}, timestep_offset::Int)
Calculates the joint information gain from all agents' actions.
"""
function calculate_joint_information_gain(belief::Belief, joint_action::Vector{SensingAction}, env, agents::Vector{Agent}, timestep_offset::Int)
    total_gain = 0.0
    
    # Track which cells are being observed by multiple agents
    cell_observation_count = Dict{Tuple{Int, Int}, Int}()
    
    for (agent_idx, action) in enumerate(joint_action)
        if isempty(action.target_cells)
            continue  # Wait action
        end
        
        # Get agent's position at this timestep
        agent = agents[agent_idx]
        agent_pos = get_agent_position_at_time(agent, env, timestep_offset)
        
        # Get the Field of Regard at this position
        for_cells = get_field_of_regard_at_position(agent, agent_pos, env)
        
        # Calculate information gain for each sensed cell
        for cell in action.target_cells
            x, y = cell
            if 1 <= x <= env.width && 1 <= y <= env.height
                # Check if cell is actually observable from this position
                if cell in for_cells
                    # Count how many agents are observing this cell
                    cell_observation_count[cell] = get(cell_observation_count, cell, 0) + 1
                    
                    # Calculate information gain for this cell
                    cell_gain = calculate_cell_information_gain(belief.event_distributions[:, y, x])
                    total_gain += cell_gain
                end
            end
        end
    end
    
    # Apply coordination penalty for overlapping observations
    coordination_penalty = 0.0
    for (cell, count) in cell_observation_count
        if count > 1
            # Penalize redundant observations (diminishing returns)
            coordination_penalty += 0.1 * (count - 1)
        end
    end
    
    # Apply observation costs
    total_actions = sum(length(action.target_cells) for action in joint_action)
    observation_cost = 0.1 * total_actions
    
    return total_gain - coordination_penalty - observation_cost
end

"""
calculate_cell_information_gain(prob_vector::Vector{Float64})
Calculates information gain for a single cell: G(b_k) = H(b_k) * P(event)
"""
function calculate_cell_information_gain(prob_vector::Vector{Float64})
    # Calculate entropy: H(b_k) = -∑ p_i * log(p_i)
    entropy = calculate_entropy_from_distribution(prob_vector)
    
    # Weight by event probability: G(b_k) = H(b_k) * P(event)
    # P(event) is the sum of all event state probabilities (states 2 and beyond)
    if length(prob_vector) >= 2
        event_probability = sum(prob_vector[2:end])
    else
        event_probability = 0.0
    end
    
    return entropy * event_probability
end

"""
simulate_joint_step(belief::Belief, joint_action::Vector{SensingAction}, env, agents::Vector{Agent}, timestep_offset::Int)
Simulates one step with joint actions and observations.
"""
function simulate_joint_step(belief::Belief, joint_action::Vector{SensingAction}, env, agents::Vector{Agent}, timestep_offset::Int)
    # First, apply DBN belief evolution
    evolved_belief = predict_belief_evolution_dbn(belief, env.event_dynamics, 1)
    
    # Then, simulate joint observations and update belief
    updated_belief = copy(evolved_belief)
    
    # Track which cells have been observed to avoid double-counting
    observed_cells = Set{Tuple{Int, Int}}()
    
    for (agent_idx, action) in enumerate(joint_action)
        if isempty(action.target_cells)
            continue  # Wait action
        end
        
        # Get agent's position at this timestep
        agent = agents[agent_idx]
        agent_pos = get_agent_position_at_time(agent, env, timestep_offset)
        
        # Get the Field of Regard at this position
        for_cells = get_field_of_regard_at_position(agent, agent_pos, env)
        
        # Simulate observations for this agent's action
        for cell in action.target_cells
            x, y = cell
            if 1 <= x <= env.width && 1 <= y <= env.height && cell in for_cells
                # Only update if this cell hasn't been observed yet (priority to first observer)
                if !(cell in observed_cells)
                    # Get current belief distribution for this cell
                    current_dist = updated_belief.event_distributions[:, y, x]
                    
                    # Simulate perfect observation (most likely state becomes certain)
                    most_likely_state_idx = argmax(current_dist)
                    
                    # Create a more certain distribution around the most likely state
                    new_dist = fill(0.1, length(current_dist))
                    new_dist[most_likely_state_idx] = 0.7
                    
                    # Normalize
                    new_dist ./= sum(new_dist)
                    updated_belief.event_distributions[:, y, x] = new_dist
                    
                    # Mark this cell as observed
                    push!(observed_cells, cell)
                end
            end
        end
    end
    
    return updated_belief
end

"""
update_global_belief_sync(global_belief::Belief, joint_action::Vector{SensingAction}, joint_observation::Vector{GridObservation}, env)
Updates the global belief with joint observations from all agents.
"""
function update_global_belief_sync(global_belief::Belief, joint_action::Vector{SensingAction}, joint_observation::Vector{GridObservation}, env)
    # First, evolve belief using DBN
    evolved_belief = predict_belief_evolution_dbn(global_belief, env.event_dynamics, 1)
    
    # Then update with all observations (most recent observations have priority)
    updated_belief = copy(evolved_belief)
    
    # Process observations in reverse order to give priority to most recent
    for obs_idx in length(joint_observation):-1:1
        observation = joint_observation[obs_idx]
        action = joint_action[obs_idx]
        
        # Update belief with this observation
        for (i, cell) in enumerate(observation.sensed_cells)
            x, y = cell
            if 1 <= x <= env.width && 1 <= y <= env.height
                # Get observed state
                observed_state = observation.event_states[i]
                
                # Update belief for this cell with perfect observation
                if observed_state == EVENT_PRESENT
                    # Set to certain event present
                    updated_belief.event_distributions[2, y, x] = 1.0  # EVENT_PRESENT
                    updated_belief.event_distributions[1, y, x] = 0.0  # NO_EVENT
                else
                    # Set to certain no event
                    updated_belief.event_distributions[1, y, x] = 1.0  # NO_EVENT
                    updated_belief.event_distributions[2, y, x] = 0.0  # EVENT_PRESENT
                end
            end
        end
    end
    
    return updated_belief
end

"""
calculate_entropy_from_distribution(prob_vector::Vector{Float64})
Calculates entropy for a multi-state belief distribution
H(b_k) = -∑ p_i * log(p_i)
"""
function calculate_entropy_from_distribution(prob_vector::Vector{Float64})
    entropy = 0.0
    for prob in prob_vector
        if prob > 0.0
            entropy -= prob * log(prob)
        end
    end
    return entropy
end

"""
get_agent_position_at_time(agent::Agent, env, timestep_offset::Int)
Gets agent's position at a specific future timestep.
"""
function get_agent_position_at_time(agent::Agent, env, timestep_offset::Int)
    t = get_current_time(env, agent) + timestep_offset
    return get_position_at_time(agent.trajectory, t, agent.phase_offset)
end

"""
get_field_of_regard_at_position(agent::Agent, position::Tuple{Int, Int}, env)
Gets field of regard for an agent at a specific position.
"""
function get_field_of_regard_at_position(agent::Agent, position::Tuple{Int, Int}, env)
    # For now, return a simple FOR based on sensor range
    # In a full implementation, this would use the agent's sensor model
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Use sensor range to determine FOR
    sensor_range = round(Int, agent.sensor.range)
    
    for dx in -sensor_range:sensor_range
        for dy in -sensor_range:sensor_range
            nx, ny = x + dx, y + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                # Check if within sensor range
                distance = sqrt(dx^2 + dy^2)
                if distance <= agent.sensor.range
                    push!(fov_cells, (nx, ny))
                end
            end
        end
    end
    
    return fov_cells
end

"""
get_current_time(env, agent)
Helper to get the current time step for the agent.
"""
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
combinations(elements, k)
Generates combinations of elements.
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

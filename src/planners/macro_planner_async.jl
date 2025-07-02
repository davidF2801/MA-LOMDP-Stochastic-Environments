module MacroPlannerAsync

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Infiltrator
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

export best_script

"""
best_script(env, belief::Belief, agent::Agent, C::Int, other_scripts, gs_state)::Vector{SensingAction}
  – Enumerate every |A|^C open-loop action sequence for `agent`.
  – For each sequence:
        • Roll out C steps:  (simulate using env.transition & env.observation)
        • Propagate local belief only with *predicted* observations
          (use expectation, i.e. marginalise over obs distribution).
        • Plug reward = ∑ γ^k R( … )   [use existing reward() helper]
        • For other agents use `other_scripts[k]` (deterministic vector passed in)
  – Return argmax sequence (ties → first).
"""
function best_script(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Enumerate all possible action sequences of length C considering trajectory
    action_sequences = generate_action_sequences(agent, env, C)
    
    if isempty(action_sequences)
        return SensingAction[]
    end
    
    best_sequence = SensingAction[]  # Default to empty sequence
    best_value = -Inf
    
    println("🔍 Evaluating $(length(action_sequences)) action sequences for agent $(agent.id)")
    
    for (i, sequence) in enumerate(action_sequences)
        # Evaluate this sequence
        value = evaluate_action_sequence(env, belief, agent, sequence, other_scripts, C, gs_state, rng)
        
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
Generate all possible action sequences of length C considering agent trajectory
"""
function generate_action_sequences(agent, env, C::Int)
    if C == 0
        return Vector{SensingAction}[]
    end
    
    # 1. Propagate agent trajectory for C timesteps
    trajectory_positions = Vector{Tuple{Int, Int}}()
    for t in 0:(C-1)
        pos = get_agent_position_at_time(agent, env, t)
        push!(trajectory_positions, pos)
    end
    
    # 2. Get available actions for each timestep
    actions_per_timestep = Vector{Vector{SensingAction}}()
    for t in 1:C
        pos = trajectory_positions[t]
        for_cells = get_field_of_regard_at_position(agent, pos, env)
        
        # Generate actions for this timestep
        timestep_actions = SensingAction[]
        
        # Add wait action
        push!(timestep_actions, SensingAction(agent.id, Tuple{Int, Int}[], false))
        
        # Add single-cell sensing actions
        for cell in for_cells
            push!(timestep_actions, SensingAction(agent.id, [cell], false))
        end
        
        # Add multi-cell sensing actions (up to max_sensing_targets)
        if length(for_cells) > 1 && env.max_sensing_targets > 1
            for subset_size in 2:min(env.max_sensing_targets, length(for_cells))
                for subset in combinations(for_cells, subset_size)
                    push!(timestep_actions, SensingAction(agent.id, collect(subset), false))
                end
            end
        end
        
        push!(actions_per_timestep, timestep_actions)
    end
    
    # 3. Generate all sequences by selecting one action per timestep
    sequences = generate_sequences_from_actions_per_timestep(actions_per_timestep)
    
    return sequences
end

"""
Generate all sequences by selecting one action per timestep
"""
function generate_sequences_from_actions_per_timestep(actions_per_timestep::Vector{Vector{SensingAction}})
    if isempty(actions_per_timestep)
        return Vector{SensingAction}[]
    elseif length(actions_per_timestep) == 1
        return [[action] for action in actions_per_timestep[1]]
    else
        sequences = Vector{SensingAction}[]
        
        # Get actions for current timestep
        current_actions = actions_per_timestep[1]
        
        # Recursively generate sequences for remaining timesteps
        remaining_sequences = generate_sequences_from_actions_per_timestep(actions_per_timestep[2:end])
        
        # Combine current actions with remaining sequences
        for action in current_actions
            for remaining_seq in remaining_sequences
                new_seq = [action; remaining_seq]
                push!(sequences, new_seq)
            end
        end
        
        return sequences
    end
end

"""
Evaluate a single action sequence using deterministic open-loop macro-script evaluation
"""
function evaluate_action_sequence(env, belief, agent, sequence::Vector{SensingAction}, other_scripts, C::Int, gs_state, rng::AbstractRNG)
    γ = env.discount  # Use environment discount factor
    value = 0.0
    B = deepcopy(belief)  # Copy belief for deterministic evolution
    
    for k in 1:min(length(sequence), C)
        # Get current action
        action = sequence[k]
        
        # Get other agents' actions for this step (deterministic)
        other_actions = get_other_actions(other_scripts, k)
        
        # Calculate expected information gain considering agent's position at timestep k-1
        gain = calculate_expected_information_gain_at_time(B, action, other_actions, env, agent, k-1, gs_state, sequence)
        
        # Accumulate discounted reward
        value += (γ)^(k-1) * gain
        
        # Simulate the step: belief evolution + other agents' observations
        B = simulate_step_with_other_agents(B, other_actions, env)
    end
    
    return value
end

"""
Get available actions for an agent based on its current position and sensor capabilities
"""
function get_available_actions(agent, env)
    # Get agent's current position
    current_pos = get_agent_position(agent, env)
    
    # Get cells within sensor range (Field of View)
    fov_cells = get_field_of_regard(agent, current_pos, env)
    
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





# """
# Calculate expected information gain considering other agents' actions and observations
# """
# function calculate_expected_information_gain_with_other_agents(belief, action::SensingAction, other_actions::Vector{SensingAction}, env)
#     if isempty(action.target_cells)
#         return 0.0  # Wait action
#     end
    
#     # Calculate expected information gain for each sensed cell
#     information_gain = 0.0
    
#     for cell in action.target_cells
#         x, y = cell
#         if 1 <= x <= env.width && 1 <= y <= env.height
#             # Calculate expected information gain for this cell considering other agents
#             cell_information_gain = calculate_cell_information_gain_with_other_agents(
#                 belief, cell, other_actions, env, 1, 0, 0, gs_state  # agent_id=1, current_timestep=0, evaluation_timestep=0, gs_state for current step
#             )
#             information_gain += cell_information_gain
#         end
#     end
    
#     # Subtract observation cost: c_obs * number of non-wait actions
#     observation_cost = 0.1 * length(action.target_cells)  # Cost per cell sensed
    
#     return information_gain - observation_cost
# end

"""
Calculate expected information gain considering agent's position at a specific timestep
"""
function calculate_expected_information_gain_at_time(belief, action::SensingAction, other_actions::Vector{SensingAction}, env, agent, phase::Int, gs_state, sequence)
    if isempty(action.target_cells)
        return 0.0  # Wait action
    end
    
    # Get agent's position at this timestep
    agent_pos = get_agent_position_at_time(agent, env, phase)
    
    # Get the Field of Regard at this position
    for_cells = get_field_of_regard_at_position(agent, agent_pos, env)
    
    # Calculate expected information gain for each sensed cell
    information_gain = 0.0
    
    for cell in action.target_cells
        x, y = cell
        if 1 <= x <= env.width && 1 <= y <= env.height
            # Check if cell is actually observable from this position
            if cell in for_cells
                # Calculate expected information gain for this cell considering other agents
                cell_information_gain = calculate_cell_information_gain_with_other_agents(
                    belief, cell, other_actions, env, agent.id, phase, gs_state, sequence  # agent_id, current_timestep, evaluation_timestep, gs_state
                )
                information_gain += cell_information_gain
            else
                # Cell is not observable from this position, no information gain
                information_gain += 0.0
            end
        end
    end
    
    # Subtract observation cost: c_obs * number of non-wait actions
    observation_cost = 0.1 * length(action.target_cells)  # Cost per cell sensed
    
    return information_gain - observation_cost
end

"""
Calculate expected information gain for a cell considering other agents' future observations
This implements the sophisticated approach that:
1. Identifies which other agents will observe the cell before our evaluation time
2. Records the specific times of those observations  
3. Conditions our estimate on possible observation outcomes
4. Averages information gains weighted by observation probabilities
"""
function calculate_cell_information_gain_with_other_agents(belief, cell::Tuple{Int, Int}, other_actions::Vector{SensingAction}, env, current_agent_id::Int, phase::Int, gs_state, sequence)
    x, y = cell
    
    # Get current belief distribution for this cell
    current_belief_dist = belief.event_distributions[:, y, x]
    
    # Step 1: Identify which other agents have observed or will observe this cell before evaluation_time
    # We need to collect ALL scheduled observations of this cell, accounting for phase offsets
    past_observations = collect_all_scheduled_observations(cell, env, current_agent_id, phase, gs_state, sequence)
    
    if isempty(past_observations)
        # No other agents will observe this cell, use simple information gain
        return calculate_cell_information_gain(current_belief_dist)
    end
    
    # Step 2: Generate all possible observation outcome sequences
    observation_outcomes = generate_observation_outcomes(past_observations, current_belief_dist)
    
    # Step 3: Calculate expected information gain by averaging over outcomes
    total_expected_gain = 0.0
    
    for (outcome_sequence, outcome_probability) in observation_outcomes
        # Simulate belief evolution with this outcome sequence
        evolved_belief_dist = simulate_belief_evolution_with_observations(
            current_belief_dist, outcome_sequence, past_observations, env, current_agent_id, phase + gs_state.time_step
        )
        
        # Calculate information gain for our observation at evaluation_time
        cell_information_gain = calculate_cell_information_gain(evolved_belief_dist)
        
        # Weight by probability of this outcome sequence
        total_expected_gain += outcome_probability * cell_information_gain
    end
    
    return total_expected_gain
end

"""
Collect ALL scheduled observations of a cell by other agents that happen before the current agent's evaluation time
Returns a vector of (agent_id, global_timestep) tuples where global_timestep is the global simulation time
These are observations that have already happened or will happen before the current agent evaluates this cell
"""
function collect_all_scheduled_observations(cell::Tuple{Int, Int}, env, current_agent_id::Int, current_agent_timestep::Int, gs_state, sequence)
    past_observations = Vector{Tuple{Int, Int}}()  # (agent_id, global_timestep)
    # Check if the current agent has already observed this cell in its own sequence
    # (up to but not including the current timestep)
    for (seq_idx, planned_action) in enumerate(sequence)
        if seq_idx >= current_agent_timestep
            break  # Stop checking at current timestep (exclusive)
        end
        if cell in planned_action.target_cells
            push!(past_observations, (current_agent_id, gs_state.time_step+seq_idx))
        end
    end
    # Get the current agent to understand its phase offset
    current_agent = nothing
    if hasproperty(env, :agents)
        for agent in env.agents
            if agent.id == current_agent_id
                current_agent = agent
                break
            end
        end
    end
    
    if current_agent === nothing
        return past_observations
    end
    
    # Convert current agent's evaluation timestep to global time
    global_evaluation_time = gs_state.time_step
    
    # Check all other agents' planned sequences for observations of this cell
    # Use the ground station's mapping of agent plans
    for (other_agent_id, other_agent_plan) in gs_state.agent_plans
        # Skip the current agent
        if other_agent_id == current_agent_id
            continue
        end
        
        # Skip if no plan available
        if other_agent_plan === nothing
            continue
        end
        
        # Get the other agent to understand its phase offset
        other_agent = nothing
        for agent in env.agents
            if agent.id == other_agent_id
                other_agent = agent
                break
            end
        end
        
        if other_agent === nothing
            continue
        end
        # Check each timestep in the other agent's planned sequence
        if gs_state.agent_plan_types[other_agent_id] == :script
            # For macro-scripts, check each action in the sequence
            for (seq_idx, planned_action) in enumerate(other_agent_plan)
                agent_timestep = seq_idx  # This is the agent's own timestep
                
                # Convert agent timestep to global timestep
                # Global timestep = agent_timestep + agent_phase_offset
                global_timestep_observation = global_evaluation_time - (current_agent.trajectory.period-current_agent.phase_offset+other_agent.phase_offset) + agent_timestep

                if cell in planned_action.target_cells
                    push!(past_observations, (other_agent_id, global_timestep_observation))
                end
                # Check if this observation happens before the current agent's evaluation time in global time
            end
        elseif gs_state.agent_plan_types[other_agent_id] == :policy
            # For policy trees, we can't easily determine future observations
            # For now, skip policy trees (they're closed-loop anyway)
            continue
        end
    end
    
    # Sort by global time to ensure chronological order
    sort!(past_observations, by = obs -> obs[2])
    
    return past_observations
end

"""
Generate all possible observation outcome sequences for past observations
Returns a vector of (outcome_sequence, probability) tuples
"""
function generate_observation_outcomes(past_observations::Vector{Tuple{Int, Int}}, current_belief_dist::Vector{Float64})
    if isempty(past_observations)
        return [([], 1.0)]
    end
    
    num_states = length(current_belief_dist)
    
    # Sort observations by time to process them chronologically
    sorted_observations = sort(past_observations, by = obs -> obs[2])  # Sort by time
    
    # Generate all possible outcome sequences recursively
    outcomes = generate_outcome_sequences_recursive(sorted_observations, current_belief_dist, 1)
    
    return outcomes
end

"""
Recursively generate all possible outcome sequences for multiple observations
"""
function generate_outcome_sequences_recursive(observations::Vector{Tuple{Int, Int}}, current_belief_dist::Vector{Float64}, obs_idx::Int)
    if obs_idx > length(observations)
        return [([], 1.0)]
    end
    
    num_states = length(current_belief_dist)
    outcomes = Vector{Tuple{Vector{Int}, Float64}}()
    
    # For each possible observation outcome (what the agent might observe)
    # In a 2-state system: 1 = observed NO_EVENT, 2 = observed EVENT_PRESENT
    for observed_state in 1:num_states
        # Calculate probability of observing this state given current belief
        # This is a simplified observation model - in reality it would depend on sensor characteristics
        if observed_state == 1  # Observed NO_EVENT
            # Probability of observing NO_EVENT = P(true_state=1) + some noise
            observation_prob = current_belief_dist[1] * 0.9 + current_belief_dist[2] * 0.1
        else  # Observed EVENT_PRESENT  
            # Probability of observing EVENT_PRESENT = P(true_state=2) + some noise
            observation_prob = current_belief_dist[2] * 0.9 + current_belief_dist[1] * 0.1
        end
        
        if observation_prob > 0.0
            # Generate outcomes for remaining observations
            remaining_outcomes = generate_outcome_sequences_recursive(observations, current_belief_dist, obs_idx + 1)
            
            for (remaining_sequence, remaining_prob) in remaining_outcomes
                # Combine this outcome with remaining outcomes
                full_sequence = [observed_state; remaining_sequence]
                full_probability = observation_prob * remaining_prob
                
                push!(outcomes, (full_sequence, full_probability))
            end
        end
    end
    
    return outcomes
end

"""
Simulate belief evolution with a sequence of observations
"""
function simulate_belief_evolution_with_observations(
    initial_belief_dist::Vector{Float64}, 
    outcome_sequence::Vector{Int}, 
    past_observations::Vector{Tuple{Int, Int}}, 
    env,
    current_agent_id::Int,
    evaluation_timestep::Int
)
    # Create a temporary belief object to use existing DBN functions
    temp_belief = Belief(
        reshape(initial_belief_dist, :, 1, 1),  # Reshape to 3D array [states, height, width]
        zeros(1, 1),  # Dummy uncertainty map
        0,  # Dummy last_update
        []  # Dummy history
    )
    
    # Get current agent to understand its phase offset
    current_agent = nothing
    for agent in env.agents
        if agent.id == current_agent_id
            current_agent = agent
            break
        end
    end
    
    if current_agent === nothing
        return initial_belief_dist
    end
    
    # Convert evaluation timestep to global timestep
    # evaluation_timestep is the agent's own timestep, convert to global
    global_evaluation_time = evaluation_timestep
        
    last_cell_observation_time = past_observations[end][2]
    
    # Apply observations in chronological order
    for (i, observed_state) in enumerate(outcome_sequence)
        if i <= length(past_observations)
            
            # Calculate how many global timesteps have passed since the last observation of this cell
            timesteps_since_last_cell_observation = global_evaluation_time - last_cell_observation_time
            
            # Apply DBN evolution for the timesteps between observations of this cell
            if timesteps_since_last_cell_observation > 0
                temp_belief = predict_belief_evolution_dbn(temp_belief, env.event_dynamics, timesteps_since_last_cell_observation)
            end
            
            # Apply observation update based on what was observed
            # This is a simplified Bayesian update
            if observed_state == 1  # Observed NO_EVENT
                # Update belief: increase probability of NO_EVENT, decrease EVENT_PRESENT
                temp_belief.event_distributions[1, 1, 1] *= 1  # Increase confidence in NO_EVENT
                temp_belief.event_distributions[2, 1, 1] *= 0  # Decrease confidence in EVENT_PRESENT
            else  # Observed EVENT_PRESENT
                # Update belief: increase probability of EVENT_PRESENT, decrease NO_EVENT
                temp_belief.event_distributions[1, 1, 1] *= 0  # Decrease confidence in NO_EVENT
                temp_belief.event_distributions[2, 1, 1] *= 1  # Increase confidence in EVENT_PRESENT
            end
            
            # Normalize the distribution
            total_prob = sum(temp_belief.event_distributions[:, 1, 1])
            if total_prob > 0.0
                temp_belief.event_distributions[:, 1, 1] ./= total_prob
            end
            
            # # Update the last observation time for this cell
            # last_cell_observation_time = global_timestep
        end
    end
    
    # Finally, apply DBN evolution from the last observation to our evaluation time
    timesteps_to_evaluation = global_evaluation_time - last_cell_observation_time
    if timesteps_to_evaluation > 0
        temp_belief = predict_belief_evolution_dbn(temp_belief, env.event_dynamics, timesteps_to_evaluation)
    end
    
    # Return the final distribution
    return vec(temp_belief.event_distributions[:, 1, 1])
end



"""
Calculate information gain for a single cell: G(b_k) = H(b_k) * P(event)
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
Simulate one step considering other agents' actions and observations
"""
function simulate_step_with_other_agents(belief, other_actions::Vector{SensingAction}, env)
    # First, apply DBN belief evolution
    evolved_belief = predict_belief_evolution_dbn(belief, env.event_dynamics, 1)
    
    # Then, simulate other agents' observations and update belief
    updated_belief = copy(evolved_belief)
    
    for other_action in other_actions
        if other_action !== nothing && !isempty(other_action.target_cells)
            # Simulate observations for this agent's action
            for cell in other_action.target_cells
                x, y = cell
                if 1 <= x <= env.width && 1 <= y <= env.height
                    # Get current belief distribution for this cell
                    current_dist = updated_belief.event_distributions[:, y, x]
                    
                    # Simulate observation (perfect observation model)
                    # In a more sophisticated model, this would be probabilistic
                    # For now, we'll assume the observation reduces uncertainty
                    # This is a simplified approach - in reality, we'd need to consider
                    # the actual observation model and update accordingly
                    
                    # Simple update: find most likely state and update accordingly
                    most_likely_state_idx = argmax(current_dist)
                    
                    # Create a more certain distribution around the most likely state
                    new_dist = fill(0.1, length(current_dist))
                    new_dist[most_likely_state_idx] = 0.7
                    
                    # Normalize
                    new_dist ./= sum(new_dist)
                    updated_belief.event_distributions[:, y, x] = new_dist
                end
            end
        end
    end
    
    return updated_belief
end

"""
Calculate entropy for a multi-state belief distribution
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
Calculate entropy for a binary belief state (event vs no event) - Legacy function
H(b_k) = -b_k * log(b_k) - (1-b_k) * log(1-b_k)
"""
function calculate_entropy(probability::Float64)
    if probability <= 0.0 || probability >= 1.0
        return 0.0  # No uncertainty if probability is 0 or 1
    end
    return -(probability * log(probability) + (1 - probability) * log(1 - probability))
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
function get_field_of_regard(agent, position, env)
    # Use the same logic as get_field_of_regard_at_position for consistency
    return get_field_of_regard_at_position(agent, position, env)
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

"""
Get agent's position at a specific future timestep
"""
function get_agent_position_at_time(agent, env, timestep_offset::Int)
    # Calculate position at future timestep using trajectory and phase offset
    t = get_current_time(env, agent) + timestep_offset
    
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

"""
Get field of regard for an agent at a specific position
"""
function get_field_of_regard_at_position(agent, position, env)
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

end # module 
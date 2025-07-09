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
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..RangeLimitedSensor, ..EventMap
# Import trajectory functions
import ..Agents.TrajectoryPlanner.get_position_at_time
# Import DBN functions for transition modeling
import ..Environment.EventDynamicsModule.DBNTransitionModel2, ..Environment.EventDynamicsModule.predict_next_belief_dbn
# Import belief management functions
import ..Agents.BeliefManagement
import ..Agents.BeliefManagement.predict_belief_evolution_dbn, ..Agents.BeliefManagement.Belief,
       ..Agents.BeliefManagement.calculate_uncertainty_from_distribution, ..Agents.BeliefManagement.predict_belief_rsp,
       ..Agents.BeliefManagement.evolve_no_obs, ..Agents.BeliefManagement.get_neighbor_beliefs,
       ..Agents.BeliefManagement.enumerate_joint_states, ..Agents.BeliefManagement.product,
       ..Agents.BeliefManagement.normalize_belief_distributions, ..Agents.BeliefManagement.collapse_belief_to,
       ..Agents.BeliefManagement.enumerate_all_possible_outcomes, ..Agents.BeliefManagement.merge_equivalent_beliefs,
       ..Agents.BeliefManagement.calculate_cell_entropy, ..Agents.BeliefManagement.get_event_probability
# Remove circular imports - these functions will be available through the environment

export best_script, evaluate_action_sequence_exact, calculate_macro_script_reward

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
        #rintln("Evaluating sequence $(i) of $(length(action_sequences))")
        # Evaluate this sequence using exact enumeration if worlds are pre-computed
        value = evaluate_action_sequence_exact(env, belief, agent, sequence, other_scripts, C, gs_state, rng)
        
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
Collect all scheduled observations of a specific cell by other agents
"""
function collect_all_scheduled_observations(cell::Tuple{Int, Int}, env, current_agent_id::Int, phase::Int, gs_state, sequence)
    observations = Vector{Tuple{Int, EventState}}()  # (timestep, observed_state)
    
    # Check other agents' plans for observations of this cell
    for (agent_id, plan) in gs_state.agent_plans
        if agent_id != current_agent_id && plan !== nothing && gs_state.agent_plan_types[agent_id] == :script
            # Check each timestep in the plan
            for (timestep, action) in enumerate(plan)
                if cell in action.target_cells
                    # This agent will observe this cell at this timestep
                    global_timestep = gs_state.agent_last_sync[agent_id] + timestep - 1
                    
                    # Only include observations that happen before our current phase
                    if global_timestep < gs_state.time_step + phase
                        # Get the actual observed state from ground station history
                        observed_state = get_observed_state_from_world(cell, global_timestep, env, gs_state)
                        push!(observations, (global_timestep, observed_state))
        end
    end
            end
        end
    end
    
    return observations
end

"""
Get observed state from world state at a specific timestep
"""
function get_observed_state_from_world(cell::Tuple{Int, Int}, timestep::Int, env, gs_state)
    # Look through all agents' observation histories to find if this cell was observed at this timestep
    for (agent_id, obs_history) in gs_state.agent_observation_history
        for (obs_timestep, obs_cell, obs_state) in obs_history
            if obs_timestep == timestep && obs_cell == cell
                return obs_state  # Return the actual observed state
            end
        end
    end
    
    # If no observation found, we need to simulate what would be observed from the world state
    # This is a fallback for when we don't have the actual observation
    # In a full implementation, this would access the world state at the given timestep
    return NO_EVENT
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
    adjusted_time = t % agent.trajectory.period
    # Calculate position based on trajectory type
    if typeof(agent.trajectory) <: CircularTrajectory
        angle = 2π * (adjusted_time % agent.trajectory.period) / agent.trajectory.period
        x = agent.trajectory.center_x + round(Int, agent.trajectory.radius * cos(angle))
        y = agent.trajectory.center_y + round(Int, agent.trajectory.radius * sin(angle))
        return (x, y)
    elseif typeof(agent.trajectory) <: LinearTrajectory
        #t_normalized = (adjusted_time % agent.trajectory.period) / agent.trajectory.period
        step_x = abs(agent.trajectory.end_x - agent.trajectory.start_x)/(agent.trajectory.period-1)
        step_y = abs(agent.trajectory.end_y - agent.trajectory.start_y)/(agent.trajectory.period-1)
        x = round(Int, agent.trajectory.start_x + adjusted_time * step_x)
        y = round(Int, agent.trajectory.start_y + adjusted_time * step_y)
        return (x, y)
    else
        return (1, 1)  # fallback
    end
end

"""
Get field of regard for an agent at a specific position
"""
function get_field_of_regard_at_position(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Check if we want row-only visibility (sensor range = 0 means row-only)
    if agent.sensor.range == 0.0
        # Row-only visibility: agent can only see cells in its current row
        for nx in 1:env.width
            push!(fov_cells, (nx, y))
        end
    else
        # Standard sensor range visibility
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
    end
    return fov_cells
end

"""
Evaluate a single action sequence using exact belief evolution
"""
function evaluate_action_sequence_exact(env, belief₀, agent, seq, other_scripts, C, gs_state, rng::AbstractRNG)
    # For exact evaluation, we don't use pre-enumerated worlds
    # Instead, we directly simulate belief evolution using the Díaz-Avalos formula
    return calculate_macro_script_reward(seq, other_scripts, C, env, agent, belief₀, gs_state)
end

"""
Calculate reward for a macro-script using exact asynchronous belief evolution
"""
function calculate_macro_script_reward(seq::Vector{SensingAction}, other_scripts, C::Int, env, agent, belief₀, gs_state)
    γ = env.discount
    c_obs = 0.0 # Cost of performing an observation
    
    # Step 1: Determine the last time where all observation outcomes are known
    tau_i = gs_state.time_step  # Current sync time of agent i
    tau = gs_state.agent_last_sync  # Last sync times of all agents
    t_clean = minimum([tau[j] for j in keys(tau) if j != agent.id])
    
    # Step 2: Roll forward deterministically from uniform belief to t_clean using known observations
    # Start with uniform belief distribution (we knew nothing at t=0)
    B = initialize_uniform_belief(env)
    for t in 0:(t_clean-1)
        B = evolve_no_obs(B, env)  # Contagion-aware update
        # Apply known observations (perfect observations)
        for (agent_j, action_j) in get_known_observations_at_time(t, gs_state)
            for cell in action_j.target_cells
                if has_known_observation(t, cell, gs_state)
                    observed_value = get_known_observation(t, cell, gs_state)
                    B = collapse_belief_to(B, cell, observed_value)
                end
            end
        end
    end
    # Step 3: Initialize branching structure at t_clean
    B_branches = Dict{Int, Vector{Tuple{Belief, Float64}}}()
    B_branches[t_clean] = [(B, 1.0)]
    # Step 4: Create branching windows for each agent j ≠ i
    branch_windows = Vector{Tuple{Int, Int, Int}}()
    for (j, last_sync) in tau
        if j != agent.id
            t_start = last_sync     # We have known observations up to here
            t_end = tau_i        # Branch until current agent i sync
            if t_start < t_end
                push!(branch_windows, (t_start, t_end, j))
            end
        end
    end
    
    # Sort and merge overlapping windows to avoid double branching
    sort!(branch_windows)

    # Step 5: Forward branch over unknown observations from other agents before tau_i
    for (start_t, end_t, agent_j) in branch_windows
        for t in start_t:(end_t-1)
            new_branches = Vector{Tuple{Belief, Float64}}()
            for (B_cur, p_branch) in B_branches[t]

                B_evolved = evolve_no_obs(B_cur, env)
                
                # Check which observations need branching based on the current window
                obs_set = Vector{Tuple{Int, SensingAction}}()  # Unknown observations for branching
                
                # First, apply known observations from history (deterministic)
                # if gs_state.time_step == 6
                #     @infiltrate
                # end
                for (agent_k, action_k) in get_known_observations_at_time(t, gs_state)
                    if agent_k != agent.id
                        for cell in action_k.target_cells
                            observed_value = get_known_observation(t, cell, gs_state)
                            B_evolved = collapse_belief_to(B_evolved, cell, observed_value)
                        end
                    end
                end
                # if gs_state.time_step == 6
                #     @infiltrate
                # end
                # Then, get unknown observations from plans for branching
                for (agent_k, action_k) in get_scheduled_observations_at_time(t, gs_state)
                    if agent_k != agent.id && agent_k == agent_j  # Only branch over the agent in current window
                        push!(obs_set, (agent_k, action_k))
                    end
                end
                
                if !isempty(obs_set)
                    # if gs_state.time_step == 6
                    #     @infiltrate
                    # end
                    for obs_combo in enumerate_all_possible_outcomes(B_evolved, obs_set)
                        # obs_combo = (cell, observed_state, probability)
                        B_new = deepcopy(B_evolved)
                        cell, observed_state, probability = obs_combo
                        B_new = collapse_belief_to(B_new, cell, observed_state)
                        push!(new_branches, (B_new, p_branch * probability))
                    end
                else
                    push!(new_branches, (B_evolved, p_branch))
                end
            end
            B_branches[t + 1] = merge_equivalent_beliefs(new_branches)
        end
    end
    # if gs_state.time_step == 6
    #     @infiltrate
    # end
    # Step 6: Simulate macro-script from tau_i forward (agent i)
    R_seq = zeros(length(seq))
    B_post = Dict{Int, Vector{Tuple{Belief, Float64}}}()
    B_post[tau_i] = B_branches[tau_i]
    
    for k in 1:length(seq)
        a_i = seq[k]
        t_global = tau_i + k - 1
        new_branches = Vector{Tuple{Belief, Float64}}()
        for (B, p_branch) in B_post[t_global]
            obs_set = Vector{Tuple{Int, SensingAction}}(get_scheduled_observations_at_time(t_global, gs_state))
            push!(obs_set, (agent.id, a_i))  # Include our own planned action
            # Check if all actions in obs_set are wait actions
            all_wait_actions = all(action.target_cells == Tuple{Int,Int}[] for (_, action) in obs_set)
            
            if !all_wait_actions
                # At least one action is a sensing action - branch over all possible observation outcomes
                for obs_combo in enumerate_all_possible_outcomes(B, obs_set)
                    B_new = deepcopy(B)
                    cell, observed_state, probability = obs_combo
                    B_new = collapse_belief_to(B_new, cell, observed_state)
                    B_next = evolve_no_obs(B_new, env)
                    push!(new_branches, (B_next, p_branch * probability))
                end
            else
                # All actions are wait actions - just evolve belief without observations
                B_next = evolve_no_obs(B, env)
                push!(new_branches, (B_next, p_branch))
            end
        end
        
        B_post[t_global + 1] = merge_equivalent_beliefs(new_branches)
        # if gs_state.time_step == 6
        #     @infiltrate
        # end
        # Infiltrate if the sequence has an action different than wait (empty action)
        has_non_wait_action = false
        for a in seq
            if !isempty(a.target_cells)
                has_non_wait_action = true
                break
            end
        end
        # Step 6.2: Compute expected reward at time t_global
        expected_reward = 0.0
        for (B_cur, p_branch) in B_post[t_global]
            for cell in a_i.target_cells

                H_before = calculate_cell_entropy(B_cur, cell)
                H_after = 0.0
                info_gain = H_before - H_after
                weighted_gain = info_gain * get_event_probability(B_cur, cell)
                expected_reward += p_branch * weighted_gain
            end
            
            if !isempty(a_i.target_cells)
                expected_reward -= p_branch * c_obs
            end
        end
        R_seq[k] = expected_reward
        # if gs_state.time_step == 6
        #     @infiltrate
        # end
    end
    
    # Step 7: Return total discounted reward
    return sum((γ^(k-1)) * R_seq[k] for k in 1:length(seq))
end




"""
Get known observations at a specific time (from observation history)
"""
function get_known_observations_at_time(t::Int, gs_state)
    observations = Vector{Tuple{Int, SensingAction}}()
    
    # Check all agents' observation histories for observations at this timestep
    for (agent_id, obs_history) in gs_state.agent_observation_history
        # Look for observations at the specific timestep t
        for (obs_timestep, obs_cell, obs_state) in obs_history
            if obs_timestep == t
                # Create a SensingAction from the observation
                action = SensingAction(agent_id, [obs_cell], false)
                push!(observations, (agent_id, action))
            end
        end
    end
    return observations
end

"""
Get scheduled observations at a specific time (from agent plans)
"""
function get_scheduled_observations_at_time(t::Int, gs_state)
    observations = Vector{Tuple{Int, SensingAction}}()
    
    # Check all agents' plans for observations at this time
    for (agent_id, plan) in gs_state.agent_plans
        if plan !== nothing && gs_state.agent_plan_types[agent_id] == :script
            # For macro-scripts, check if this timestep has an action
            plan_timestep = t - gs_state.agent_last_sync[agent_id]
            if 1 <= plan_timestep <= length(plan)
                action = plan[plan_timestep]
                if !isempty(action.target_cells)
                    push!(observations, (agent_id, action))
                end
            end
        end
    end
    
    return observations
end

"""
Get actual observations for a specific agent at a specific time
"""
function get_agent_observations_at_time(agent_id::Int, t::Int, gs_state)
    observations = Vector{Tuple{Tuple{Int, Int}, EventState}}()  # (cell, observed_state)
    
    # Look through the agent's observation history
    if haskey(gs_state.agent_observation_history, agent_id)
        for (obs_timestep, obs_cell, obs_state) in gs_state.agent_observation_history[agent_id]
            if obs_timestep == t
                push!(observations, (obs_cell, obs_state))
            end
        end
    end
    
    return observations
end

"""
Get all observations for a specific cell across all agents
"""
function get_cell_observations(cell::Tuple{Int, Int}, gs_state)
    observations = Vector{Tuple{Int, Int, EventState}}()  # (agent_id, timestep, observed_state)
    
    # Look through all agents' observation histories
    for (agent_id, obs_history) in gs_state.agent_observation_history
        for (obs_timestep, obs_cell, obs_state) in obs_history
            if obs_cell == cell
                push!(observations, (agent_id, obs_timestep, obs_state))
            end
        end
    end
    
    return observations
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
Initialize uniform belief distribution (we knew nothing at t=0)
"""
function initialize_uniform_belief(env)
    # For 2-state model: [NO_EVENT, EVENT_PRESENT]
    num_states = 2
    uniform_distribution = fill(1.0/num_states, num_states)
    
    return BeliefManagement.initialize_belief(env.width, env.height, uniform_distribution)
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
Check if we have a known observation for a cell at a time
"""
function has_known_observation(t::Int, cell::Tuple{Int, Int}, gs_state)
    # Check if any agent has observed this cell at this exact timestep
    for (agent_id, obs_history) in gs_state.agent_observation_history
        for (obs_timestep, obs_cell, obs_state) in obs_history
            if obs_timestep == t && obs_cell == cell
                return true
            end
        end
    end
    return false
end

"""
Get known observation for a cell at a time
"""
function get_known_observation(t::Int, cell::Tuple{Int, Int}, gs_state)
    # Look through all agents' observation histories to find the actual observation
    for (agent_id, obs_history) in gs_state.agent_observation_history
        for (obs_timestep, obs_cell, obs_state) in obs_history
            if obs_timestep == t && obs_cell == cell
                return obs_state  # Return the actual observed state
            end
        end
    end
end

end # module 
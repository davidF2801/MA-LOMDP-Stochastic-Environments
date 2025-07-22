module MacroPlannerAsyncApprox

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Infiltrator
using Logging
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
       ..Agents.BeliefManagement.calculate_cell_entropy, ..Agents.BeliefManagement.get_event_probability,
       ..Agents.BeliefManagement.clear_belief_evolution_cache!, ..Agents.BeliefManagement.get_cache_stats
# Remove circular imports - these functions will be available through the environment

export best_script, calculate_macro_script_reward

"""
best_script(env, belief::Belief, agent::Agent, C::Int, other_scripts, gs_state)::Vector{SensingAction}
  – Pre-compute belief branches up to t_clean (independent of action sequences)
  – Enumerate every |A|^C open-loop action sequence for `agent`.
  – For each sequence:
        • Evaluate using pre-computed belief branches
        • Simulate macro-script from tau_i forward (agent i)
        • Plug reward = ∑ γ^k R( … )   [use existing reward() helper]
        • For other agents use `other_scripts[k]` (deterministic vector passed in)
  – Return argmax sequence (ties → first).
"""
function best_script(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG, mode::Symbol=:script)
    # Start timing
    start_time = time()
    
    # Clear belief evolution cache at the start of each planning session
    clear_belief_evolution_cache!()
    
    # Enumerate all possible action sequences of length C considering trajectory
    action_sequences = generate_action_sequences(agent, env, C)

    if isempty(action_sequences)
        end_time = time()
        return SensingAction[], end_time - start_time
    end
    
    # Step 1 & 2: Pre-compute belief branches up to t_clean (independent of action sequences)
    println("🔄 Pre-computing belief branches for agent $(agent.id)...")
    B_branches = precompute_belief_branches(env, agent, gs_state)
    @infiltrate
    best_sequence = SensingAction[]  # Default to empty sequence
    best_value = -Inf
    println("🔍 Evaluating $(length(action_sequences)) action sequences for agent $(agent.id)")
    for (i, sequence) in enumerate(action_sequences)
        # Evaluate this sequence using pre-computed belief branches
        value = calculate_macro_script_reward(sequence, other_scripts, C, env, agent, B_branches, gs_state, mode==:future_actions)
        
        if value > best_value
            best_value = value
            best_sequence = sequence
        end
        if i % 100 == 0
            println("  Evaluated $(i)/$(length(action_sequences)) sequences, best value: $(round(best_value, digits=3))")
        end
    end
    
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ Best sequence found with value: $(round(best_value, digits=3)) in $(round(planning_time, digits=3)) seconds")
    
    # Report cache statistics
    cache_stats = get_cache_stats()
    println("📊 Cache statistics: $(cache_stats[:hits]) hits, $(cache_stats[:misses]) misses, $(round(cache_stats[:hit_rate] * 100, digits=1))% hit rate")
    
    return best_sequence, planning_time
end

"""
Generate all possible action sequences of length C considering agent trajectory
"""
function generate_action_sequences(agent, env, C::Int, phase_offset::Int=0)
    if C == 0
        return Vector{SensingAction}[]
    end
    
    # 1. Propagate agent trajectory for C timesteps
    trajectory_positions = Vector{Tuple{Int, Int}}()
    for t in 0:(C-1)
        pos = get_position_at_time(agent.trajectory, t, phase_offset)
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
    # Use the same log2ic as get_field_of_regard_at_position for consistency
    return get_field_of_regard_at_position(agent, position, env)
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
Pre-compute belief branches up to t_clean (Steps 1 & 2 of the algorithm)
This is independent of the specific action sequence being evaluated
"""
function precompute_belief_branches(env, agent, gs_state)
    # Step 1: Determine the last time where all observation outcomes are known
    tau_i = gs_state.time_step  # Current sync time of agent i
    tau = gs_state.agent_last_sync  # Last sync times of all agents
    
    # Handle initial case: if any agent hasn't synced yet (last_sync = -1), start from t=0
    other_agent_sync_times = [tau[j] for j in keys(tau) if j != agent.id]
    if any(sync_time == -1 for sync_time in other_agent_sync_times)
        t_clean = 0  # Start from t=0 if any agent hasn't synced yet
        println("  🔄 Initial case: some agents haven't synced yet, starting from t_clean = 0")
    else
        t_clean = minimum(other_agent_sync_times)
        println("  🔄 Using t_clean = $(t_clean) (minimum of other agent sync times)")
    end
    # Special case: if t_clean = 0 but there are scheduled observations at t=0 that we don't know,
    # we need to branch over them starting from t=0
    if t_clean == 0
        # Check if there are any scheduled observations at t=0 that we don't have in our history
        scheduled_at_t0 = get_scheduled_observations_at_time(0, gs_state)
        known_at_t0 = get_known_observations_at_time(0, gs_state)
        
        # If there are scheduled observations at t=0 that we don't know, we need to branch
        if !isempty(scheduled_at_t0) && isempty(known_at_t0)
            println("  🔄 Found scheduled observations at t=0 that we don't know, will branch over them")
            # We'll handle this in the branching windows
        end
    end
    
    # Step 2: Roll forward deterministically from uniform belief to t_clean-1 using known observations
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
    # At t_clean, we start branching over unknown observations
    B_branches = Dict{Int, Vector{Tuple{Belief, Float64}}}()
    B_branches[t_clean] = [(B, 1.0)]
    
    # Step 4: Create branching windows for each agent j ≠ i
    branch_windows = Vector{Tuple{Int, Int, Int}}()
    for (j, last_sync) in tau
        if j != agent.id
            # Special case: if this agent has a plan and we're at t_clean=0,
            # we need to branch over their scheduled observations starting from t=0
            if t_clean == 0 && haskey(gs_state.agent_plans, j) && gs_state.agent_plans[j] !== nothing
                # Check if this agent has scheduled observations at t=0
                plan = gs_state.agent_plans[j]
                if 1 <= length(plan) && !isempty(plan[1].target_cells)
                    # This agent has a scheduled observation at t=0, branch from t=0
                    push!(branch_windows, (0, tau_i, j))
                end
            else
                # Use t_clean as the starting point to avoid negative timesteps
                t_start = max(t_clean, last_sync)  # We have known observations up to here
                t_end = tau_i        # Branch until current agent i sync
                if t_start < t_end
                    push!(branch_windows, (t_start, t_end, j))
                end
            end
        end
    end
    
    # Sort and merge overlapping windows to avoid double branching
    sort!(branch_windows)
    
    # If we're branching from t=0, make sure we have the belief at t=0
    if t_clean == 0 && any(start_t == 0 for (start_t, _, _) in branch_windows)
        B_branches[0] = [(B, 1.0)]
    end
    
    println("  🔄 Branching windows: $(branch_windows)")
    # Step 5: Forward branch over unknown observations from other agents before tau_i
    for (start_t, end_t, agent_j) in branch_windows
        println("  🔄 Processing branching window: $(start_t) to $(end_t) for agent $(agent_j)")
        for t in start_t:(end_t-1)
            println("    🔄 Processing timestep $(t)")
            new_branches = Vector{Tuple{Belief, Float64}}()
            for (B_cur, p_branch) in B_branches[t]

                B_evolved = evolve_no_obs(B_cur, env)
                
                # Check which observations need branching based on the current window
                obs_set = Vector{Tuple{Int, SensingAction}}()  # Unknown observations for branching
                
                # First, apply known observations from history (deterministic)
                known_obs = get_known_observations_at_time(t, gs_state)
                for (agent_k, action_k) in known_obs
                    if agent_k != agent.id
                        for cell in action_k.target_cells
                            observed_value = get_known_observation(t, cell, gs_state)
                            B_evolved = collapse_belief_to(B_evolved, cell, observed_value)
                        end
                    end
                end
                scheduled_obs = get_scheduled_observations_at_time(t, gs_state)
                println("    🔄 Scheduled observations at t=$(t): $(scheduled_obs)")
                for (agent_k, action_k) in scheduled_obs
                    # Only branch over observations that are not already known
                    # Check if this observation is already known by comparing agent_id and target_cells
                    is_known = false
                    for (known_agent, known_action) in known_obs
                        if agent_k == known_agent && action_k.target_cells == known_action.target_cells
                            is_known = true
                            break
                        end
                    end
                    
                    if !is_known && agent_k != agent.id && agent_k == agent_j
                        push!(obs_set, (agent_k, action_k))
                        println("    🔄 Adding observation from agent $(agent_k): $(action_k)")
                    end
                end
                if !isempty(obs_set)
                    for (observation_combo, probability) in enumerate_all_possible_outcomes(B_evolved, obs_set)
                        # observation_combo = Vector{(cell, observed_state)}
                        B_new = deepcopy(B_evolved)
                        # Apply all observations in the combination together
                        for (cell, observed_state) in observation_combo
                            B_new = collapse_belief_to(B_new, cell, observed_state)
                        end
                        push!(new_branches, (B_new, p_branch * probability))

                    end
                else
                    push!(new_branches, (B_evolved, p_branch))
                end
            end
            new_branches = branch_pruning(new_branches, env, agent, gs_state)
            B_branches[t + 1] = merge_equivalent_beliefs(new_branches)
        end
    end
    
    return B_branches
end
function branch_pruning(branches::Vector{Tuple{Belief, Float64}}, env, agent, gs_state)
    # Keep branches up to max_prob_mass threshold, then renormalize
    # Returns the branches that are kept (not pruned) for a single timestep
    # Guarantee: we never discard more than (1 - env.max_prob_mass) of total probability
    
    if isempty(branches)
        return branches
    end
    
    # Calculate the maximum probability we can discard
    max_discard_prob = 1.0 - env.max_prob_mass
    
    # Sort branches by probability mass (descending order)
    sorted_branches = sort(branches, by=x->x[2], rev=true)
    
    # Calculate total probability mass
    total_prob = sum(prob for (_, prob) in branches)
    
    # If total probability is very small, keep all branches
    if total_prob < 1e-10
        return branches
    end
    
    # Keep branches until the NEXT branch to discard would exceed max_discard_prob
    selected_branches = Vector{Tuple{Belief, Float64}}()
    cumulative_prob = 0.0
    
    for (belief, prob) in sorted_branches
        # Check if discarding this branch would exceed max_discard_prob
        # If we discard this branch, we keep cumulative_prob and discard (total_prob - cumulative_prob)
        discarded_if_we_stop_here = total_prob - cumulative_prob
        
        if discarded_if_we_stop_here <= max_discard_prob
            # We can safely stop here (discarding the rest would be <= max_discard_prob)
            break
        else
            # We must keep this branch to avoid discarding too much
            push!(selected_branches, (belief, prob))
            cumulative_prob += prob
        end
    end
    
    # If no branches were kept, keep at least the highest probability branch
    if isempty(selected_branches) && !isempty(sorted_branches)
        push!(selected_branches, (sorted_branches[1][1], sorted_branches[1][2]))
        cumulative_prob = sorted_branches[1][2]
    end
    
    # Renormalize the kept branches
    if cumulative_prob > 0.0
        normalized_branches = Vector{Tuple{Belief, Float64}}()
        for (belief, prob) in selected_branches
            normalized_prob = prob / cumulative_prob
            push!(normalized_branches, (belief, normalized_prob))
        end
        
        # Verify we didn't discard too much
        actual_discarded = total_prob - cumulative_prob
        if actual_discarded > max_discard_prob + 1e-10  # Allow small numerical errors
            @warn "Discarded more than allowed: $(actual_discarded) > $(max_discard_prob)"
        end
        if length(normalized_branches) < length(branches) && length(normalized_branches) > 3
            @infiltrate
        end
        return normalized_branches
    else
        # Fallback: keep original branches if no probability mass
        return branches
    end
end

"""
Calculate reward for a macro-script using pre-computed belief branches
"""
function calculate_macro_script_reward(seq::Vector{SensingAction}, other_scripts, C::Int, env, agent, B_branches, gs_state, predict_future_actions::Bool=false)
    γ = env.discount
    c_obs = 0.0 # Cost of performing an observation
    
    # Get current sync time for agent i
    tau_i = gs_state.time_step
    
    # Step 6: Simulate macro-script from tau_i forward (agent i)
    R_seq = zeros(length(seq))
    B_post = Dict{Int, Vector{Tuple{Belief, Float64}}}()
    B_post[tau_i] = B_branches[tau_i]
    
    # Track which agents we've already generated sub-plans for
    agents_with_sub_plans = Set{Int}()
    
    # Store sub-plans mapping: agent_id -> Dict{global_timestep -> action}
    sub_plans_mapping = Dict{Int, Dict{Int, SensingAction}}()
    
    for k in 1:length(seq)
        a_i = seq[k]
        t_global = tau_i + k - 1
        new_branches = Vector{Tuple{Belief, Float64}}()
        for (B, p_branch) in B_post[t_global]
            obs_set = Vector{Tuple{Int, SensingAction}}(get_scheduled_observations_at_time(t_global, gs_state))
            
            if predict_future_actions
                # Check if any other agent is missing from obs_set (their plan ran out)
                all_agent_ids = Set(keys(gs_state.agent_plans))
                agents_with_obs = Set(agent_id for (agent_id, _) in obs_set)
                missing_agents = setdiff(all_agent_ids, agents_with_obs, Set([agent.id]))  # Exclude current agent
                
                # If some other agent is missing and we haven't generated sub-plans for them yet, generate sub-plans
                for missing_agent_id in missing_agents
                    if missing_agent_id ∉ agents_with_sub_plans
                        t_start = t_global
                        t_end = gs_state.time_step + length(seq)
                        agent_sub_plan = generate_sub_plans(seq, C, env, B_post[t_start], gs_state, t_start, t_end, missing_agent_id)
                        sub_plans_mapping[missing_agent_id] = agent_sub_plan
                        push!(agents_with_sub_plans, missing_agent_id)  # Mark as processed
                    end
                end
            
                # Add sub-plan actions for the current timestep to obs_set
                for (agent_id, timestep_actions) in sub_plans_mapping
                    if haskey(timestep_actions, t_global)
                        push!(obs_set, (agent_id, timestep_actions[t_global]))
                    end
                end
            end

            push!(obs_set, (agent.id, a_i))  # Include our own planned action
            all_wait_actions = all(action.target_cells == Tuple{Int,Int}[] for (_, action) in obs_set)
            
            if !all_wait_actions
                # At least one action is a sensing action - branch over all possible observation outcomes
                for (observation_combo, probability) in enumerate_all_possible_outcomes(B, obs_set)
                    B_new = deepcopy(B)
                    # Apply all observations in the combination together
                    for (cell, observed_state) in observation_combo
                        B_new = collapse_belief_to(B_new, cell, observed_state)
                    end
                    B_next = evolve_no_obs(B_new, env)
                    push!(new_branches, (B_next, p_branch * probability))
                    if sum(p_branch for (_, p_branch) in new_branches)>1.1
                        @infiltrate
                    end
                end
            else
                # All actions are wait actions - just evolve belief without observations
                B_next = evolve_no_obs(B, env)
                push!(new_branches, (B_next, p_branch))
                            end
            end
            new_branches = branch_pruning(new_branches, env, agent, gs_state)
            B_post[t_global + 1] = merge_equivalent_beliefs(new_branches)
        
        # Check that probability branches sum to 1.0
        total_prob = sum(p_branch for (_, p_branch) in B_post[t_global + 1])        
        if abs(total_prob-1.0) > 1e-6
            @warn "Probability branches don't sum to 1.0: $(total_prob)"
        end
        # Step 6.2: Compute expected reward at time t_global
        R_seq[k] = compute_expected_reward(B_post[t_global], a_i, c_obs)
    end
    # Step 7: Return total discounted reward
    return sum((γ^(k-1)) * R_seq[k] for k in 1:length(seq))
end

function generate_sub_plans(seq, C, env, belief_branches_vector, gs_state, t_start, t_end, missing_agent_id)
    # Generate random actions for the missing agent from t_start to t_end
    # Returns: Dict{global_timestep -> SensingAction}
    
    agent = env.agents[missing_agent_id]
    timestep_actions = Dict{Int, SensingAction}()
    phase_offset = t_start % agent.trajectory.period
    seq_length = t_end - t_start
    action_sequences = generate_action_sequences(agent, env, seq_length, phase_offset)
    # Evaluate all possible action sequences
    best_sequence = SensingAction[]
    best_value = -Inf
    for sequence in action_sequences
        # Evaluate this sequence using belief branches
        value = evaluate_sub_sequence(sequence, env, agent, t_start, gs_state, belief_branches_vector)
        if value > best_value
            best_value = value
            best_sequence = sequence
        end
    end
    
    # Map the best sequence to timesteps
    for (i, action) in enumerate(best_sequence)
        timestep_actions[t_start + i - 1] = action
    end
    
    return timestep_actions
end

"""
Compute expected reward for an action given belief branches
"""
function compute_expected_reward(belief_branches, action, c_obs)
    expected_reward = 0.0
    
    for (B_cur, p_branch) in belief_branches
        for cell in action.target_cells
            H_before = calculate_cell_entropy(B_cur, cell)
            H_after = 0.0
            info_gain = H_before - H_after
            weighted_gain = info_gain*get_event_probability(B_cur, cell)

            expected_reward += p_branch * weighted_gain
        end
        
        if !isempty(action.target_cells)
            expected_reward -= p_branch * c_obs
        end
    end
    
    return expected_reward
end

"""
Evaluate a sub-sequence for a missing agent using belief branches
"""
function evaluate_sub_sequence(seq::Vector{SensingAction}, env, agent, t_start::Int, gs_state, belief_branches_vector)
    γ = env.discount
    c_obs = 0.0  # Cost of performing an observation
    
    # Use belief branches starting from t_start
    B_post = Dict{Int, Vector{Tuple{Belief, Float64}}}()
    B_post[t_start] = belief_branches_vector
    
    total_reward = 0.0
    
    for (k, action) in enumerate(seq)
        t_global = t_start + k - 1
        # Calculate reward using belief branches
        action_reward = compute_expected_reward(B_post[t_global], action, c_obs)
        # Apply discount
        total_reward += (γ^(k-1)) * action_reward
        
        # Update belief branches for next timestep (simplified - just evolve without observations)
        new_branches = Vector{Tuple{Belief, Float64}}()
        for (B, p_branch) in B_post[t_global]
            B_next = evolve_no_obs(B, env)
            push!(new_branches, (B_next, p_branch))
        end
        B_post[t_global + 1] = merge_equivalent_beliefs(new_branches)
    end
    
    return total_reward
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
        if plan !== nothing
            # For macro-scripts, check if this timestep has an action
            plan_timestep = (t - gs_state.agent_last_sync[agent_id]) + 1
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

# Import utility functions from Types module
import ..Types.calculate_entropy_from_distribution, ..Types.calculate_cell_information_gain, ..Types.combinations

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
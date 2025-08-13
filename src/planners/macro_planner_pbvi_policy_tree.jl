module AsyncPBVIPolicyTree

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Infiltrator
using Statistics
using Base.Threads
using ..Types
import ..Agents.BeliefManagement: sample_from_belief
import ..Types: check_battery_feasible, simulate_battery_evolution
# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..EventState2, ..NO_EVENT_2, ..EVENT_PRESENT_2
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..ComplexTrajectory, ..RangeLimitedSensor, ..EventMap
# Import trajectory functions
import ..Agents.TrajectoryPlanner.get_position_at_time
# Import DBN functions for transition modeling
import ..Environment.EventDynamicsModule.DBNTransitionModel2, ..Environment.EventDynamicsModule.predict_next_belief_dbn
# Import belief management functions
import ..Agents.BeliefManagement
import ..Agents.BeliefManagement.predict_belief_evolution_dbn, ..Agents.BeliefManagement.Belief,
       ..Agents.BeliefManagement.calculate_uncertainty_from_distribution, ..Agents.BeliefManagement.predict_belief_rsp,
       ..Agents.BeliefManagement.evolve_no_obs,..Agents.BeliefManagement.evolve_no_obs_fast, ..Agents.BeliefManagement.get_neighbor_beliefs,
       ..Agents.BeliefManagement.enumerate_joint_states, ..Agents.BeliefManagement.product,
       ..Agents.BeliefManagement.normalize_belief_distributions, ..Agents.BeliefManagement.collapse_belief_to,
       ..Agents.BeliefManagement.enumerate_all_possible_outcomes, ..Agents.BeliefManagement.merge_equivalent_beliefs,
       ..Agents.BeliefManagement.calculate_cell_entropy, ..Agents.BeliefManagement.get_event_probability,
       ..Agents.BeliefManagement.clear_belief_evolution_cache!, ..Agents.BeliefManagement.get_cache_stats,
       ..Agents.BeliefManagement.beliefs_are_equivalent

export best_policy_tree, calculate_policy_tree_reward, create_debuggable_policy_tree, print_policy_tree_structure, export_policy_tree_to_dict, inspect_policy_node, find_policy_nodes

# PBVI-specific types for policy trees
struct ClockVector
    phases::Vector{Int}  # Phases of all agents in the trajectory
end

# Add copy method for ClockVector
Base.copy(cv::ClockVector) = ClockVector(copy(cv.phases))
Base.deepcopy(cv::ClockVector) = ClockVector(deepcopy(cv.phases))

# Policy tree node structure - now with observation histories for ALL agents
struct PolicyTreeNode
    clock::ClockVector
    agent_observation_histories::Dict{Int, Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}}  # agent_id -> (timestep, observations)
    digest::UInt64          # immutable, pre-computed hash
    belief::Belief          # still carried for look-ups
end

# Add copy method for PolicyTreeNode
Base.copy(ptn::PolicyTreeNode) = PolicyTreeNode(copy(ptn.clock), deepcopy(ptn.agent_observation_histories), ptn.digest, deepcopy(ptn.belief))
Base.deepcopy(ptn::PolicyTreeNode) = PolicyTreeNode(deepcopy(ptn.clock), deepcopy(ptn.agent_observation_histories), ptn.digest, deepcopy(ptn.belief))

# Add hash and equality methods for dictionary keys
Base.hash(cv::ClockVector, h::UInt) = hash(cv.phases, h)
Base.hash(ptn::PolicyTreeNode, h::UInt) = hash(ptn.clock, hash(ptn.digest, h))

Base.isequal(cv1::ClockVector, cv2::ClockVector) = cv1.phases == cv2.phases
Base.isequal(ptn1::PolicyTreeNode, ptn2::PolicyTreeNode) = isequal(ptn1.clock, ptn2.clock) && 
                                                           ptn1.agent_observation_histories == ptn2.agent_observation_histories &&
                                                           beliefs_are_equivalent(ptn1.belief, ptn2.belief)
Base.:(==)(cv1::ClockVector, cv2::ClockVector) = isequal(cv1, cv2)
Base.:(==)(ptn1::PolicyTreeNode, ptn2::PolicyTreeNode) = isequal(ptn1, ptn2)

"""
Create a PolicyTreeNode with pre-computed digest
"""
function PolicyTreeNode(clock::ClockVector, agent_observation_histories::Dict{Int, Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}}, belief::Belief)
    digest = hash(belief.event_distributions, UInt(0))
    return PolicyTreeNode(clock, agent_observation_histories, digest, belief)
end

"""
best_policy_tree(env, belief::Belief, agent::Agent, C::Int, other_scripts, gs_state)::Function
  – Use PBVI to find the best reactive policy
  – Build belief set, run value iteration, extract policy tree
  – Return a reactive policy function AND the policy tree structure for debugging
"""
function best_policy_tree(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG, 
                    N_seed::Int=30, N_particles::Int=32, N_sweeps::Int=50, ε::Float64=0.1)
    # Start timing
    start_time = time()
    
    # Clear belief evolution cache at the start of each planning session
    clear_belief_evolution_cache!()
    
    # Get parameters from the pseudocode
    B_clean = deepcopy(belief)
    agent_i = agent
    τ_i = gs_state.time_step
    agents_j = [env.agents[j] for j in keys(env.agents) if j != agent.id]
    τ_js_vector = gs_state.agent_last_sync
    H = C  # Horizon length
    
    println("🔄 Building belief set for PBVI policy tree...")
    # Build belief set
    𝔅 = build_belief_set(B_clean, agent_i, τ_i, agents_j, τ_js_vector, H, env, gs_state, N_seed)
    println("🔄 Running PBVI with $(length(𝔅)) belief points...")
    # Run PBVI
    VALUE, POLICY = pbvi_policy_tree(𝔅, N_particles, N_sweeps, ε, agent_i, env, gs_state)
    
    # Create a clear policy tree structure for debugging
    policy_tree = create_debuggable_policy_tree(𝔅, POLICY, VALUE, agent_i, env)
    
    # Extract reactive policy from policy tree
    reactive_policy = extract_reactive_policy(POLICY, VALUE, 𝔅, agent_i, env, gs_state, H)
    @infiltrate
    # Debug: Print policy tree structure clearly
    println("🔍 Policy Tree Structure:")
    print_policy_tree_structure(policy_tree)
    
    # Store the reactive policy in the agent
    agent.reactive_policy = reactive_policy
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ PBVI policy tree found in $(round(planning_time, digits=3)) seconds")
    
    # Report cache statistics
    cache_stats = get_cache_stats()
    println("📊 Cache statistics: $(cache_stats[:hits]) hits, $(cache_stats[:misses]) misses, $(round(cache_stats[:hit_rate] * 100, digits=1))% hit rate")
    
    return reactive_policy, planning_time, policy_tree
end

"""
Sample system state at τ_i (fully-informed system belief)
"""
function sample_system_state_at_τi(B_clean::Belief, agent_i::Agent, τ_i::Int, agents_j::Vector{Agent}, 
                                  τ_js_vector::Dict{Int, Int}, env, gs_state)
    b = deepcopy(B_clean)
    
    # Initialize observation histories for all agents
    agent_obs_histories = Dict{Int, Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}}()
    agent_obs_histories[agent_i.id] = Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}()
    for j in agents_j
        agent_obs_histories[j.id] = Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}()
    end
    
    # Determine t_clean (minimum of other agent sync times)
    other_agent_sync_times = [τ_js_vector[j.id] for j in agents_j]
    t_clean = any(sync_time == -1 for sync_time in other_agent_sync_times) ? 0 : minimum(other_agent_sync_times)
    
    # Roll forward from t_clean to τ_i-1
    for t in t_clean:(τ_i-1)
        for j in agents_j
            if t > τ_js_vector[j.id]
                # Get agent j's current observation history
                agent_j_obs_history = get(agent_obs_histories, j.id, Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}())
                if j_senses_at_time(j, t, gs_state, agent_j_obs_history)
                    cell = scheduled_cell(j, t, gs_state, agent_j_obs_history)
                    if cell !== nothing
                        state = sample_event_state_from(b, cell)
                        b = collapse_belief_to(b, cell, state)
                        
                        # Update agent j's observation history with timestep
                        obs_symbol_j = [(cell, state)]
                        if !haskey(agent_obs_histories, j.id)
                            agent_obs_histories[j.id] = Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}()
                        end
                        push!(agent_obs_histories[j.id], (t, obs_symbol_j))
                    end
                end
            end
        end
        b = evolve_no_obs_fast(b, env, calculate_uncertainty=false)
    end
    
    # Create clock vector at τ_i
    # Calculate phases relative to agent_i (which is at phase 0)
    agent_phases = Int[]
    
    # Get all agents in the same order as the clock vector
    all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
    
    # Find agent_i's index in the clock vector
    agent_i_index = find_agent_index(agent_i, env)
    
    # Calculate phases for all agents relative to agent_i
    for (i, agent) in enumerate(all_agents)
        if i == agent_i_index
            # Agent_i is at phase 0
            push!(agent_phases, 0)
        else
            # Other agents: relative phase offset
            relative_offset = mod((agent.phase_offset - agent_i.phase_offset), agent.trajectory.period)
            push!(agent_phases, relative_offset)
        end
    end
    τ_clock = ClockVector(agent_phases)
    
    return (τ_clock, b, agent_obs_histories)
end

"""
Simulate one step forward for policy tree - returns local observation symbol
"""
function simulate_one_step_policy_tree(τ_clock::ClockVector, b_sys::Belief, action_i::SensingAction, 
                          agent_i::Agent, agents_j::Vector{Agent}, env, gs_state,
                          agent_obs_histories::Dict{Int, Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}})

    # Calculate information gain from current system belief
    r_step = 0.0
    if !isempty(action_i.target_cells)
        for cell in action_i.target_cells
            H_before = calculate_cell_entropy(b_sys, cell)
            # Simplified: assume perfect observation
            H_after = 0.0
            info_gain = H_before - H_after
            r_step += info_gain
        end
    end

    # Calculate global time from agent_i's phase in the clock vector
    agent_i_index = find_agent_index(agent_i, env)
    if agent_i_index === nothing
        t_global = gs_state.time_step  # Fallback
    else
        # Calculate global time based on agent_i's phase
        agent_i_phase = τ_clock.phases[agent_i_index]
        t_global = gs_state.time_step + agent_i_phase
    end
    
    # Sample Agent i's own observation outcome (this is what the policy branches on)
    obs_symbol_i = Vector{Tuple{Tuple{Int, Int}, EventState}}()
    if !isempty(action_i.target_cells)
        for cell in action_i.target_cells
            state_i = sample_event_state_from(b_sys, cell)
            push!(obs_symbol_i, (cell, state_i))
            # Collapse belief based on agent i's observation
            b_sys = collapse_belief_to(b_sys, cell, state_i)
        end
    end
    
    # Update agent i's observation history with timestep
    if !isempty(obs_symbol_i)
        if !haskey(agent_obs_histories, agent_i.id)
            agent_obs_histories[agent_i.id] = Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}()
        end
        push!(agent_obs_histories[agent_i.id], (t_global, obs_symbol_i))
    end
    
    # Sample other agents' observations and apply them immediately (hidden from agent i)
    for j in agents_j
        # Get agent j's current observation history
        agent_j_obs_history = get(agent_obs_histories, j.id, Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}())
        if j_senses_at_time(j, t_global, gs_state, agent_j_obs_history)
            cell_j = scheduled_cell(j, t_global, gs_state, agent_j_obs_history)
            if cell_j !== nothing
                state_j = sample_event_state_from(b_sys, cell_j)
                # Apply sampled observation immediately (Monte Carlo sampling)
                b_sys = collapse_belief_to(b_sys, cell_j, state_j)
                
                # Update agent j's observation history with timestep
                obs_symbol_j = [(cell_j, state_j)]
                if !haskey(agent_obs_histories, j.id)
                    agent_obs_histories[j.id] = Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}()
                end
                push!(agent_obs_histories[j.id], (t_global, obs_symbol_j))
            end
        end
    end
    
    # Predict one step ahead (use fast vectorized version without cache)
    evolve_start = time()
    b_sys = evolve_no_obs_fast(b_sys, env, calculate_uncertainty=false)
    evolve_time = time() - evolve_start
    τ_clock = advance_clock_vector(τ_clock, [agent_i; agents_j])
    
    return (r_step, τ_clock, b_sys, obs_symbol_i, agent_obs_histories, evolve_time)
end

"""
Build belief set for PBVI policy tree
"""
function build_belief_set(B_clean::Belief, agent_i::Agent, τ_i::Int, agents_j::Vector{Agent}, 
                         τ_js_vector::Dict{Int, Int}, H::Int, env, gs_state, N_seed::Int)
    𝔅 = Set{PolicyTreeNode}()
    total_generated = 0
    
    for seed in 1:N_seed
        (τ, b_sys, initial_agent_obs_histories) = sample_system_state_at_τi(B_clean, agent_i, τ_i, agents_j, τ_js_vector, env, gs_state)
        # Start with empty observation histories for all agents
        empty_agent_histories = Dict{Int, Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}}()
        for agent in [agent_i; agents_j]
            empty_agent_histories[agent.id] = Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}()
        end
        push!(𝔅, PolicyTreeNode(τ, empty_agent_histories, deepcopy(b_sys)))
        
        # Start with the initial observation histories from sampling
        all_agent_histories = deepcopy(initial_agent_obs_histories)
        
        for h in 1:H
            total_generated += 1
            # Take a random action and simulate
            a_rand = random_pointing(agent_i, τ, env)
            
            # Initialize all agents with their current observation histories
            agent_obs_histories = deepcopy(all_agent_histories)
            (_, τ, b_sys, obs_symbol, all_agent_obs_histories, _) = simulate_one_step_policy_tree(τ, b_sys, a_rand, agent_i, agents_j, env, gs_state, agent_obs_histories)
            if agent_i.id == 2
                # Debug point for agent 2
            end
            # Update all agent histories with the results from simulation
            all_agent_histories = all_agent_obs_histories
            
            # Create a single policy tree node with all agents' observation histories
            push!(𝔅, PolicyTreeNode(τ, deepcopy(all_agent_histories), deepcopy(b_sys)))
        end
    end
    
    # Ensure we have coverage for all possible phase combinations
    println("🔍 Ensuring phase coverage...")
    all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
    agent_i_index = find_agent_index(agent_i, env)
    
    # Generate additional belief points for missing phase combinations
    for phase_combination in generate_all_phase_combinations(all_agents, agent_i)
        clock = ClockVector(phase_combination)
        clock_key = Tuple(clock.phases)
        
        # Check if we already have nodes for this phase combination
        existing_nodes = [ptn for ptn in 𝔅 if Tuple(ptn.clock.phases) == clock_key]
        
        if length(existing_nodes) < 2  # Ensure at least 2 nodes per phase combination
            # Create additional belief points for this phase combination
            for extra_seed in 1:2
                # Sample a belief state
                b_extra = deepcopy(B_clean)
                # Add some random observations to create variety
                for _ in 1:rand(0:3)
                    cell = (rand(1:env.width), rand(1:env.height))
                    state = rand([NO_EVENT, EVENT_PRESENT])
                    b_extra = collapse_belief_to(b_extra, cell, state)
                end
                
                # Create empty observation histories for this phase
                empty_histories = Dict{Int, Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}}()
                for agent in all_agents
                    empty_histories[agent.id] = Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}()
                end
                
                push!(𝔅, PolicyTreeNode(clock, empty_histories, b_extra))
                total_generated += 1
            end
        end
    end
    
    final_count = length(𝔅)
    println("📊 Belief set: generated $(total_generated) points, kept $(final_count) unique points (removed $(total_generated - final_count) duplicates)")
    
    return collect(𝔅)
end

"""
PBVI algorithm for policy trees
"""
function pbvi_policy_tree(𝔅::Vector{PolicyTreeNode}, N_particles::Int, N_sweeps::Int, ε::Float64, 
              agent_i::Agent, env, gs_state)
    VALUE = Dict{PolicyTreeNode, Float64}()
    POLICY = Dict{PolicyTreeNode, SensingAction}()
    # make a Dict from phase-tuple → vector of points in that slice
    SLICE_BUCKET  = Dict{Tuple{Vararg{Int}}, Vector{PolicyTreeNode}}()
    DIGEST_LOOKUP = Dict{Tuple{Vararg{Int}}, Dict{UInt64,PolicyTreeNode}}()

    for ptn in 𝔅
        key = Tuple(ptn.clock.phases)          # hashable
        push!(get!(SLICE_BUCKET, key, PolicyTreeNode[]), ptn)
    end
    for (key, vec) in SLICE_BUCKET
        lkp = Dict{UInt64,PolicyTreeNode}()
        for ptn in vec
            lkp[ptn.digest] = ptn
        end
        DIGEST_LOOKUP[key] = lkp              # digest → ptn in that slice
    end

    
    # Initialize values
    for ptn in 𝔅
        VALUE[ptn] = 0.0
    end
    
    γ = env.discount
    
    for sweep in 1:N_sweeps
        Δ = 0.0
        shuffled_𝔅 = shuffle(𝔅)
        sim_times = Float64[]  # Track simulation times for this sweep

        
        # Sequential belief point processing (thread-safe)
        for ptn in shuffled_𝔅
            best_Q = -Inf
            best_act = nothing
            
            # Get all feasible actions
            action_set = all_pointings(agent_i, ptn.clock, env)
            
            for a in action_set
                sum_Q = 0.0
                
                # Sequential particle simulation
                for particle in 1:N_particles
                    step_start = time()
                    # Use the observation histories from the current policy tree node
                    agent_obs_histories = deepcopy(ptn.agent_observation_histories)
                    (r, τ′, b′, obs_symbol, all_agent_obs_histories, evolve_time) = simulate_one_step_policy_tree(copy(ptn.clock), copy(ptn.belief), a, 
                                                    agent_i, get_other_agents(agent_i, env), env, gs_state, agent_obs_histories)
                    step_time = time() - step_start
                    
                    # Create next policy tree node with updated observation histories
                    next_ptn = PolicyTreeNode(τ′, all_agent_obs_histories, b′)
                    
                    # Find nearest belief point based on observation history and clock
                    nearest_ptn = find_nearest_belief_policy_tree(next_ptn, SLICE_BUCKET, DIGEST_LOOKUP)
                    v_next = VALUE[nearest_ptn]
                    
                    sum_Q += r + γ * v_next
                end
                
                Q_hat = sum_Q / N_particles
                
                if Q_hat > best_Q
                    best_Q = Q_hat
                    best_act = a
                end
            end
            
            Δ = max(Δ, abs(best_Q - VALUE[ptn]))
            VALUE[ptn] = best_Q
            POLICY[ptn] = best_act
        end
        
        println("  Sweep $(sweep): max change = $(round(Δ, digits=4))")

        
        if Δ < ε
            break
        end
    end
    return VALUE, POLICY
end

"""
Extract reactive policy from policy tree
"""
function extract_reactive_policy(POLICY::Dict{PolicyTreeNode, SensingAction}, VALUE::Dict{PolicyTreeNode, Float64}, 
                             𝔅::Vector{PolicyTreeNode}, agent_i::Agent, env, gs_state, H::Int)
    
    # Create a reactive policy function that takes observation history and current time, returns action
    function reactive_policy(observation_history::Vector{GridObservation}, current_time::Int)
        # Convert observation history to standardized format
        standardized_history = Vector{Tuple{Tuple{Int, Int}, EventState}}[]
        
        for obs in observation_history
            obs_standardized = Tuple{Tuple{Int, Int}, EventState}[]
            for (i, cell) in enumerate(obs.sensed_cells)
                if i <= length(obs.event_states)
                    push!(obs_standardized, (cell, obs.event_states[i]))
                end
            end
            if !isempty(obs_standardized)
                push!(standardized_history, obs_standardized)
            end
        end
        
        # Find the policy tree node that best matches current state
        # Calculate current phases based on current time and agent trajectories
        current_phases = Int[]
        all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
        agent_i_index = find_agent_index(agent_i, env)
        
        for (i, agent) in enumerate(all_agents)
            # Calculate actual phase based on current time and agent's phase offset
            actual_phase = mod((current_time + agent.phase_offset), agent.trajectory.period)
            push!(current_phases, actual_phase)
        end
        current_clock = ClockVector(current_phases)
        # Find matching policy tree nodes based on clock phases and agent i's observation history
        candidate_ptns = PolicyTreeNode[]
        if current_time == 7 && agent_i.id == 2
            @infiltrate
        end
        for ptn in 𝔅
            if ptn.clock.phases == current_clock.phases
                # Extract agent i's observation history from the policy tree node
                agent_i_history = get(ptn.agent_observation_histories, agent_i.id, Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}())
                # Convert to standardized format for comparison
                agent_i_standardized = Vector{Vector{Tuple{Tuple{Int, Int}, EventState}}}()
                for (_, obs_symbols) in agent_i_history
                    push!(agent_i_standardized, obs_symbols)
                end
                if agent_i_standardized == standardized_history
                    push!(candidate_ptns, ptn)
                end
            end
        end
        if isempty(candidate_ptns)
            # If no exact phase match, generate a feasible action for the current phase
            # This ensures we never return an infeasible action
            
            # Generate feasible actions for the current phase
            feasible_actions = all_pointings(agent_i, current_clock, env)
            
            # Ensure we always have at least the WAIT action
            if isempty(feasible_actions)
                push!(feasible_actions, SensingAction(agent_i.id, Tuple{Int, Int}[], false))
            end
            
            # Choose the best feasible action based on current belief
            best_action = SensingAction(agent_i.id, Tuple{Int, Int}[], false)  # Default to wait
            best_value = -Inf
            
            for action in feasible_actions
                # Simple heuristic: prefer sensing actions over wait, prefer cells with higher uncertainty
                if !isempty(action.target_cells)
                    # Calculate expected information gain
                    info_gain = 0.0
                    for cell in action.target_cells
                        # Use agent's local belief if available, otherwise use a simple heuristic
                        if agent_i.belief !== nothing && haskey(agent_i.belief, :uncertainty_map)
                            x, y = cell
                            if 1 <= x <= size(agent_i.belief.uncertainty_map, 2) && 1 <= y <= size(agent_i.belief.uncertainty_map, 1)
                                info_gain += agent_i.belief.uncertainty_map[y, x]
                            end
                        else
                            # Simple heuristic: prefer cells closer to agent's current position
                            agent_index = find_agent_index(agent_i, env)
                            if agent_index !== nothing
                                phase = current_clock.phases[agent_index]
                                agent_pos = get_position_at_time(agent_i.trajectory, phase)
                                distance = sqrt((cell[1] - agent_pos[1])^2 + (cell[2] - agent_pos[2])^2)
                                info_gain += 1.0 / (1.0 + distance)  # Closer cells get higher score
                            else
                                info_gain += 1.0  # Default score if we can't calculate distance
                            end
                        end
                    end
                    
                    if info_gain > best_value
                        best_value = info_gain
                        best_action = action
                    end
                end
            end
            
            return best_action
        else
            # Take the policy tree node with highest value among candidates
            best_value = -Inf
            best_ptn = candidate_ptns[1]
            for ptn in candidate_ptns
                if haskey(VALUE, ptn) && VALUE[ptn] > best_value
                    best_value = VALUE[ptn]
                    best_ptn = ptn
                end
            end
            
            # Get the action from the best policy tree node
            if haskey(POLICY, best_ptn)
                action = POLICY[best_ptn]
                
                # CRITICAL: Verify that this action is feasible for the current phase
                if is_action_feasible_for_phase(action, agent_i, current_clock, env)
                    return action
                else
                    # Generate a feasible action instead
                    feasible_actions = all_pointings(agent_i, current_clock, env)
                    if !isempty(feasible_actions)
                        return feasible_actions[1]  # Return first feasible action
                    else
                        return SensingAction(agent_i.id, Tuple{Int, Int}[], false)
                    end
                end
            else
                return SensingAction(agent_i.id, Tuple{Int, Int}[], false)
            end
        end
    end
    
    return reactive_policy
end

"""
Check if an action is feasible for the current phase
"""
function is_action_feasible_for_phase(action::SensingAction, agent::Agent, clock::ClockVector, env)
    # Get agent position at this phase
    agent_index = find_agent_index(agent, env)
    if agent_index === nothing
        return false
    end
    
    phase = clock.phases[agent_index]
    pos = get_position_at_time(agent.trajectory, phase)
    
    # Get available cells in field of view for this phase
    available_cells = get_field_of_regard_at_position(agent, pos, env)
    
    # Check if action targets are in available cells
    for target_cell in action.target_cells
        if !(target_cell in available_cells)
            return false
        end
    end
    
    # Check battery feasibility
    if !check_battery_feasible(agent, action, agent.battery_level)
        return false
    end
    
    return true
end

# Helper functions

"""
Create a debuggable policy tree structure that's easy to inspect
"""
function create_debuggable_policy_tree(𝔅::Vector{PolicyTreeNode}, POLICY::Dict{PolicyTreeNode, SensingAction}, 
                                     VALUE::Dict{PolicyTreeNode, Float64}, agent_i::Agent, env)
    # Group policy tree nodes by clock phases for easier inspection
    policy_tree = Dict{Tuple{Vararg{Int}}, Vector{Dict{String, Any}}}()
    
    for ptn in 𝔅
        clock_key = Tuple(ptn.clock.phases)
        
        # Create a readable representation of this node
        node_info = Dict{String, Any}()
        
        # Clock information
        node_info["clock_phases"] = ptn.clock.phases
        node_info["clock_description"] = describe_clock_phases(ptn.clock, agent_i, env)
        
        # Belief information
        node_info["belief_summary"] = belief_summary(ptn.belief)
        node_info["total_uncertainty"] = sum(ptn.belief.uncertainty_map)
        
        # Agent observation histories
        node_info["agent_obs_histories"] = Dict{Int, Vector{String}}()
        for (agent_id, obs_history) in ptn.agent_observation_histories
            obs_strings = String[]
            for (timestep, obs_symbols) in obs_history
                obs_str = "t$(timestep): ["
                for (cell, state) in obs_symbols
                    state_str = state == EVENT_PRESENT ? "EVENT" : "NO_EVENT"
                    obs_str *= "($(cell[1]),$(cell[2]))=$state_str, "
                end
                obs_str = chop(obs_str, tail=2) * "]"
                push!(obs_strings, obs_str)
            end
            node_info["agent_obs_histories"][agent_id] = obs_strings
        end
        
        # Policy information
        if haskey(POLICY, ptn)
            action = POLICY[ptn]
            node_info["action"] = action_to_string(action)
            node_info["action_type"] = isempty(action.target_cells) ? "WAIT" : "SENSE"
        else
            node_info["action"] = "NO_ACTION"
            node_info["action_type"] = "NONE"
        end
        
        # Value information
        if haskey(VALUE, ptn)
            node_info["value"] = VALUE[ptn]
        else
            node_info["value"] = 0.0
        end
        
        # Add to policy tree
        if !haskey(policy_tree, clock_key)
            policy_tree[clock_key] = Vector{Dict{String, Any}}()
        end
        push!(policy_tree[clock_key], node_info)
    end
    
    return policy_tree
end

"""
Print the policy tree structure in a clear, readable format
"""
function print_policy_tree_structure(policy_tree::Dict{Tuple{Vararg{Int}}, Vector{Dict{String, Any}}})
    println("📋 Policy Tree Summary:")
    println("  Total clock configurations: $(length(policy_tree))")
    
    total_nodes = sum(length(nodes) for nodes in values(policy_tree))
    println("  Total policy nodes: $total_nodes")
    
    # Group by clock configuration
    for (clock_key, nodes) in policy_tree
        println("\n🕐 Clock Configuration: $clock_key")
        println("  Description: $(describe_clock_phases_general(clock_key))")
        println("  Nodes in this configuration: $(length(nodes))")
        
        # Show action distribution for this clock configuration
        action_counts = Dict{String, Int}()
        for node in nodes
            action_type = get(node, "action_type", "NONE")
            action_counts[action_type] = get(action_counts, action_type, 0) + 1
        end
        
        println("  Action distribution:")
        for (action_type, count) in action_counts
            println("    $action_type: $count")
        end
        
        # Show a few example nodes
        println("  Example nodes:")
        for (i, node) in enumerate(nodes[1:min(3, length(nodes))])
            println("    Node $i:")
            println("      Action: $(get(node, "action", "NO_ACTION"))")
            println("      Value: $(round(get(node, "value", 0.0), digits=3))")
            println("      Uncertainty: $(round(get(node, "total_uncertainty", 0.0), digits=3))")
            
            # Show agent i's observation history
            agent_i_id = 1  # Assuming agent i is agent 1
            if haskey(node["agent_obs_histories"], agent_i_id)
                obs_history = node["agent_obs_histories"][agent_i_id]
                if !isempty(obs_history)
                    println("      Agent $agent_i_id observations: $(obs_history[1:min(2, length(obs_history))])")
                end
            end
        end
        
        if length(nodes) > 3
            println("    ... and $(length(nodes) - 3) more nodes")
        end
    end
end

"""
Convert action to readable string
"""
function action_to_string(action::SensingAction)
    if isempty(action.target_cells)
        return "WAIT"
    else
        cells_str = join(["($(c[1]),$(c[2]))" for c in action.target_cells], ", ")
        return "SENSE[$cells_str]"
    end
end

"""
Describe clock phases in human-readable format
"""
function describe_clock_phases(clock::ClockVector, agent_i::Agent, env)
    all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
    agent_i_index = find_agent_index(agent_i, env)
    
    if agent_i_index === nothing
        return "Unknown agent configuration"
    end
    
    descriptions = String[]
    for (i, phase) in enumerate(clock.phases)
        if i == agent_i_index
            push!(descriptions, "Agent$(all_agents[i].id)(phase $phase - CURRENT)")
        else
            push!(descriptions, "Agent$(all_agents[i].id)(phase $phase)")
        end
    end
    
    return join(descriptions, ", ")
end

"""
Describe clock phases from clock key
"""
function describe_clock_phases_general(clock_key::Tuple{Vararg{Int}})
    return "Phases: $clock_key"
end

"""
Calculate similarity between observation histories
"""
function calculate_history_similarity(history1, history2)
    if length(history1) != length(history2)
        return 0.0
    end
    
    similarity = 0.0
    for (obs1, obs2) in zip(history1, history2)
        if obs1 == obs2
            similarity += 1.0
        end
    end
    
    return similarity / max(length(history1), 1)
end

"""
Calculate similarity between observation histories with partial matching
"""
function calculate_history_similarity_partial(history1::Dict{Int, Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}}}, history2::Vector{Vector{Tuple{Tuple{Int, Int}, EventState}}})
    # Extract agent i's observations from history1 (ignore timesteps for comparison)
    # For now, assume agent i is the first agent in the dictionary
    agent_i_obs = Vector{Tuple{Tuple{Int, Int}, EventState}}()
    for (agent_id, obs_history) in history1
        for (_, obs_symbols) in obs_history
            append!(agent_i_obs, obs_symbols)
        end
        break  # Only use the first agent's observations for now
    end
    
    # Allow partial matching for histories of different lengths
    min_len = min(length(agent_i_obs), length(history2))
    max_len = max(length(agent_i_obs), length(history2))
    
    if min_len == 0
        return 0.0
    end
    
    similarity = 0.0
    for i in 1:min_len
        if agent_i_obs[i] == history2[i]
            similarity += 1.0
        end
    end
    
    return similarity / max_len
end

"""
Sample event state from belief for a cell
"""
function sample_event_state_from(belief::Belief, cell::Tuple{Int, Int})
    # Get probability of event in this cell
    p_event = get_event_probability(belief, cell)
    
    # Sample based on probability
    if rand() < p_event
        return EVENT_PRESENT  # Use EventState, not EventState2
    else
        return NO_EVENT  # Use EventState, not EventState2
    end
end

"""
Check if agent j senses at time t using reactive policy
"""
function j_senses_at_time(j::Agent, t::Int, gs_state, agent_j_obs_history::Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}})
    # Check if agent j has a reactive policy
    if j.reactive_policy === nothing
        return false
    end
    
    # Convert observation history to GridObservation format for the reactive policy
    grid_observations = Vector{GridObservation}()
    for (_, obs_symbols) in agent_j_obs_history
        cells = Tuple{Int, Int}[]
        event_states = EventState[]
        for (cell, state) in obs_symbols
            push!(cells, cell)
            push!(event_states, state)
        end
        if !isempty(cells)
            push!(grid_observations, GridObservation(j.id, cells, event_states, []))
        end
    end
    
    # Get action from agent j's reactive policy
    action = j.reactive_policy(grid_observations, t)
    return !isempty(action.target_cells)
end

"""
Get scheduled cell for agent j at time t using reactive policy
"""
function scheduled_cell(j::Agent, t::Int, gs_state, agent_j_obs_history::Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}})
    # Check if agent j has a reactive policy
    if j.reactive_policy === nothing
        return nothing
    end
    
    # Convert observation history to GridObservation format for the reactive policy
    grid_observations = Vector{GridObservation}()
    for (_, obs_symbols) in agent_j_obs_history
        cells = Tuple{Int, Int}[]
        event_states = EventState[]
        for (cell, state) in obs_symbols
            push!(cells, cell)
            push!(event_states, state)
        end
        if !isempty(cells)
            push!(grid_observations, GridObservation(j.id, cells, event_states, []))
        end
    end
    # Get action from agent j's reactive policy
    action = j.reactive_policy(grid_observations, t)
    if !isempty(action.target_cells)
        return action.target_cells[1]  # Return first cell
    end
    
    return nothing
end

"""
Get next sync time for agent
"""
function next_sync_time(agent::Agent, gs_state)
    # Simplified: assume sync every C timesteps
    C = agent.trajectory.period  # This should come from environment or agent parameters
    return gs_state.time_step + C
end

"""
Get next sync time for agent at a specific global time
"""
function next_sync_time_at_global_time(agent::Agent, t_global::Int)
    # Simplified: assume sync every C timesteps
    C = agent.trajectory.period  # This should come from environment or agent parameters
    return t_global + C
end

"""
Advance clock vector
"""
function advance_clock_vector(τ_clock::ClockVector, agents)
    # Advance phases of all agents
    new_phases = Int[]
    for (i, agent) in enumerate(agents)
        new_phase = mod(τ_clock.phases[i] + 1, agent.trajectory.period)
        push!(new_phases, new_phase)
    end
    return ClockVector(new_phases)
end

"""
Generate random pointing action
"""
function random_pointing(agent::Agent, τ_clock::ClockVector, env)
    # Get agent position at this time using agent's phase
    agent_index = find_agent_index(agent, env)
    if agent_index === nothing
        return SensingAction(agent.id, Tuple{Int, Int}[], false)
    end
    phase = τ_clock.phases[agent_index]
    pos = get_position_at_time(agent.trajectory, phase)
    
    # Get available cells in field of view
    available_cells = get_field_of_regard_at_position(agent, pos, env)
    
    # Create action set: wait action + pointing actions
    actions = SensingAction[]
    push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))  # Wait action
    
    # Add pointing actions for available cells
    for cell in available_cells
        action = SensingAction(agent.id, [cell], false)
        if check_battery_feasible(agent, action, agent.battery_level)
            push!(actions, action)
        end
    end
    
    # Pick random action (including wait)
    return rand(actions)
end

"""
Get all pointing actions for agent
"""
function all_pointings(agent::Agent, τ_clock::ClockVector, env)
    actions = SensingAction[]
    
    # Get agent position at this time using agent's phase
    agent_index = find_agent_index(agent, env)
    if agent_index === nothing
        return [SensingAction(agent.id, Tuple{Int, Int}[], false)]
    end
    phase = τ_clock.phases[agent_index]
    pos = get_position_at_time(agent.trajectory, phase)
    # Get available cells in field of view
    available_cells = get_field_of_regard_at_position(agent, pos, env)
    
    # Add wait action
    push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))
    
    # Add single cell actions
    for cell in available_cells
        action = SensingAction(agent.id, [cell], false)
        if check_battery_feasible(agent, action, agent.battery_level)
            push!(actions, action)
        end
    end
    
    return actions
end

"""
Generate all possible phase combinations for the agents
"""
function generate_all_phase_combinations(all_agents::Vector{Agent}, agent_i::Agent)
    phase_combinations = Vector{Int}[]
    
    # Get the period from agent_i's trajectory (assume all agents have same period)
    period = agent_i.trajectory.period
    
    # Generate all combinations of phases
    for phase1 in 0:(period-1)
        for phase2 in 0:(period-1)
            if length(all_agents) == 2
                push!(phase_combinations, [phase1, phase2])
            else
                for phase3 in 0:(period-1)
                    push!(phase_combinations, [phase1, phase2, phase3])
                end
            end
        end
    end
    
    return phase_combinations
end

@inline function find_nearest_belief_policy_tree(target::PolicyTreeNode, SLICE_BUCKET, DIGEST_LOOKUP)
    key = Tuple(target.clock.phases)

    # 1. bucket for the correct slice
    bucket = get(SLICE_BUCKET, key, nothing)
    if bucket === nothing
        return target            # should not happen if 𝔅 covered all slices
    end

    # 2. exact digest match (O(1))
    fp = get(DIGEST_LOOKUP[key], target.digest, nothing)
    if fp !== nothing
        return fp
    end

    # 3. fall-back: find closest based on observation history similarity
    nearest   = bucket[1]
    best_dist = calculate_history_similarity(target.agent_observation_histories, nearest.agent_observation_histories)
    @inbounds for ptn in bucket
        d = calculate_history_similarity(target.agent_observation_histories, ptn.agent_observation_histories)
        if d > best_dist  # Higher similarity is better
            best_dist = d
            nearest   = ptn
        end
    end
    return nearest
end

"""
Calculate distance between two beliefs using low-dimensional summary
"""
function belief_distance(b1::Belief, b2::Belief)
    # Use digest for fast comparison first
    digest1 = hash(b1.event_distributions, UInt(0))
    digest2 = hash(b2.event_distributions, UInt(0))
    
    if digest1 == digest2
        return 0.0  # Exact match
    end
    
    # Fallback to L1 distance on low-dimensional summary
    # Create summary: vector of per-cell event probabilities
    summary1 = belief_summary(b1)
    summary2 = belief_summary(b2)
    
    return sum(abs.(summary1 .- summary2))
end

"""
Create low-dimensional summary of belief for distance calculation
"""
function belief_summary(belief::Belief)
    # Get dimensions from event_distributions array [state, y, x]
    num_states, height, width = size(belief.event_distributions)
    
    # Create summary: vector of event probabilities for each cell
    summary = Float64[]
    
    for x in 1:width, y in 1:height
        # Get event probability for this cell (sum over event states)
        event_prob = 0.0
        for state in 1:num_states
            if state == 2  # Assuming state 2 is EVENT_PRESENT
                event_prob += belief.event_distributions[state, y, x]
            end
        end
        push!(summary, event_prob)
    end
    
    return summary
end

"""
Find agent index in the clock vector
"""
function find_agent_index(agent::Agent, env)
    # Get all agents in the same order as they appear in the clock vector
    all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
    
    # Find the index of this agent
    for (i, env_agent) in enumerate(all_agents)
        if env_agent.id == agent.id
            return i
        end
    end
    
    return nothing
end

"""
Get other agents (excluding agent_i)
"""
function get_other_agents(agent_i::Agent, env)
    return [env.agents[j] for j in keys(env.agents) if j != agent_i.id]
end

"""
Get field of regard for an agent at a specific position
"""
function get_field_of_regard_at_position(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Check sensor pattern
    if agent.sensor.pattern == :cross
        # Cross-shaped sensor: agent's position and adjacent cells
        ax, ay = position
        for dx in -1:1, dy in -1:1
            nx, ny = ax + dx, ay + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                # Only include cross pattern (not diagonal)
                if (dx == 0 && dy == 0) || (dx == 0 && dy != 0) || (dx != 0 && dy == 0)
                    push!(fov_cells, (nx, ny))
                end
            end
        end
    elseif agent.sensor.pattern == :row_only || agent.sensor.range == 0.0
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
Export policy tree to a simple dictionary format for easy inspection
"""
function export_policy_tree_to_dict(policy_tree::Dict{Tuple{Vararg{Int}}, Vector{Dict{String, Any}}})
    export_dict = Dict{String, Any}()
    
    for (clock_key, nodes) in policy_tree
        clock_str = string(clock_key)
        export_dict[clock_str] = []
        
        for node in nodes
            # Create a simplified node representation
            simple_node = Dict{String, Any}()
            simple_node["action"] = get(node, "action", "NO_ACTION")
            simple_node["value"] = get(node, "value", 0.0)
            simple_node["uncertainty"] = get(node, "total_uncertainty", 0.0)
            
            # Simplify observation histories
            simple_obs = Dict{String, Vector{String}}()
            for (agent_id, obs_history) in get(node, "agent_obs_histories", Dict{Int, Vector{String}}())
                simple_obs[string(agent_id)] = obs_history
            end
            simple_node["observations"] = simple_obs
            
            push!(export_dict[clock_str], simple_node)
        end
    end
    
    return export_dict
end

"""
Inspect a specific policy tree node
"""
function inspect_policy_node(policy_tree::Dict{Tuple{Vararg{Int}}, Vector{Dict{String, Any}}}, clock_key::Tuple{Vararg{Int}}, node_index::Int)
    nodes = policy_tree[clock_key]
    if node_index < 1 || node_index > length(nodes)
        println("Node index $node_index out of bounds for clock key $clock_key.")
        return
    end

    node = nodes[node_index]
    println("Inspecting Policy Node $node_index for Clock Key: $clock_key")
    println("  Action: $(get(node, "action", "NO_ACTION"))")
    println("  Value: $(round(get(node, "value", 0.0), digits=3))")
    println("  Uncertainty: $(round(get(node, "total_uncertainty", 0.0), digits=3))")
    
    # Show agent i's observation history
    agent_i_id = 1  # Assuming agent i is agent 1
    if haskey(node["agent_obs_histories"], agent_i_id)
        obs_history = node["agent_obs_histories"][agent_i_id]
        println("  Agent $agent_i_id Observations:")
        for (i, obs_str) in enumerate(obs_history)
            println("    t$(i): $obs_str")
        end
    end
end

"""
Find policy tree nodes based on specific criteria
Returns a vector of node dictionaries that match the criteria
"""
function find_policy_nodes(policy_tree::Dict{Tuple{Vararg{Int}}, Vector{Dict{String, Any}}};
                           clock_key::Tuple{Vararg{Int}}=nothing,
                           action_type::String="ANY",
                           value_range::Tuple{Float64, Float64}=(-Inf, Inf),
                           uncertainty_range::Tuple{Float64, Float64}=(-Inf, Inf),
                           observation_history::Vector{Tuple{Tuple{Int, Int}, EventState}}=nothing)
    found_nodes = Dict{String, Any}[]
    
    for (ck, nodes) in policy_tree
        if clock_key !== nothing && ck != clock_key
            continue
        end
        
        for (i, node) in enumerate(nodes)
            # Check action type
            if action_type != "ANY" && get(node, "action_type", "NONE") != action_type
                continue
            end
            
            # Check value range
            value = get(node, "value", 0.0)
            if value < value_range[1] || value > value_range[2]
                continue
            end
            
            # Check uncertainty range
            uncertainty = get(node, "total_uncertainty", 0.0)
            if uncertainty < uncertainty_range[1] || uncertainty > uncertainty_range[2]
                continue
            end
            
            # Check observation history (if provided)
            if observation_history !== nothing
                # Extract agent i's observation history from the node
                agent_i_history_str = get(node, "agent_obs_histories", Dict{Int, Vector{String}}())
                agent_i_history_vec = Vector{Vector{Tuple{Tuple{Int, Int}, EventState}}}()
                for (_, obs_symbols_str) in agent_i_history_str
                    obs_symbols_vec = Vector{Tuple{Tuple{Int, Int}, EventState}}()
                    for (cell_str, state_str) in obs_symbols_str
                        cell = Tuple{Int, Int}(parse.(Int, split(cell_str, ",")))
                        state = state_str == "EVENT" ? EVENT_PRESENT : NO_EVENT
                        push!(obs_symbols_vec, (cell, state))
                    end
                    push!(agent_i_history_vec, obs_symbols_vec)
                end
                
                # Compare with the provided observation history
                if agent_i_history_vec != observation_history
                    continue
                end
            end
            
            # If all criteria match, add to found nodes
            push!(found_nodes, node)
        end
    end
    
    return found_nodes
end

end # module 
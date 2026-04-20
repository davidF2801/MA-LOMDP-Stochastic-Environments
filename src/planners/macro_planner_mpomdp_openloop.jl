module MacroPlannerMPOMDPOpenLoop

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Infiltrator
using Statistics
using Base.Iterators
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
       ..Agents.BeliefManagement.evolve_no_obs, ..Agents.BeliefManagement.evolve_no_obs_fast, ..Agents.BeliefManagement.get_neighbor_beliefs,
       ..Agents.BeliefManagement.enumerate_joint_states, ..Agents.BeliefManagement.prob_product,
       ..Agents.BeliefManagement.normalize_belief_distributions, ..Agents.BeliefManagement.collapse_belief_to,
       ..Agents.BeliefManagement.enumerate_all_possible_outcomes, ..Agents.BeliefManagement.merge_equivalent_beliefs,
       ..Agents.BeliefManagement.get_event_probability,
       ..Agents.BeliefManagement.clear_belief_evolution_cache!, ..Agents.BeliefManagement.get_cache_stats,
       ..Agents.BeliefManagement.beliefs_are_equivalent
import ..MacroPlannerPBVI: calculate_sophisticated_reward

export best_joint_plan_mpomdp, calculate_joint_sequence_reward

"""
JointAction - Represents a joint action for all agents at a timestep
"""
struct JointAction
    actions::Vector{SensingAction}  # One action per agent (ordered by agent.id)
end

"""
best_joint_plan_mpomdp(env, belief::Belief, agents::Vector{Agent}, C::Int, gs_state)
MPOMDP Open-Loop Planner - Centralized planning with joint actions.

This planner ignores communication constraints (centralized info at every step),
so it is not directly comparable to ABBA/SB-ABBA which respect sync intervals.
- Has access to ALL agents' observations (centralized information)
- Plans joint actions (one action per agent per timestep)
- Plans C steps ahead (open-loop)
- Evaluates each sequence with the true expected discounted reward (exact marginalization over observation outcomes)

Returns: Vector of JointAction (one per timestep), planning_time
"""
function best_joint_plan_mpomdp(env, belief::Belief, agents::Vector{Agent}, C::Int, gs_state; 
                                rng::AbstractRNG=Random.GLOBAL_RNG,
                                N_sequences::Int=0)
    # Ensure agents are sorted by id for consistent ordering
    sorted_agents = sort(agents, by=a -> a.id)
    start_time = time()
    
    println("🤖 MPOMDP Open-Loop Planner: Planning for $(length(agents)) agents, horizon C=$(C)")
    println("  📊 Using centralized observations from all agents")
    println("  🎯 Planning joint actions (one per agent per timestep)")
    
    # Generate joint action sequences
    #
    # For small problems we keep the original exact enumeration.
    # For larger instances (or when N_sequences > 0), we can instead
    # sample up to N_sequences joint sequences from the full space.
    joint_sequences = generate_joint_action_sequences(sorted_agents, env, C, gs_state, rng; N_sequences=N_sequences)
    
    println("  🔍 Using $(length(joint_sequences)) joint action sequences for evaluation")
    
    if isempty(joint_sequences)
        println("  ⚠️  No feasible joint sequences found, returning wait actions")
        wait_sequence = [JointAction([SensingAction(agent.id, Tuple{Int, Int}[], false) for agent in agents]) for _ in 1:C]
        return wait_sequence, time() - start_time
    end
    
    # Evaluate each sequence using the global belief
    best_sequence = nothing
    best_value = -Inf
    
    total_sequences = length(joint_sequences)
    println("  🔍 Evaluating $(total_sequences) joint sequences...")
    # Print progress roughly every 5% of the work (at least every 1 sequence)
    progress_step = max(1, Int(cld(total_sequences, 20)))
    for (i, joint_seq) in enumerate(joint_sequences)
        value = calculate_joint_sequence_reward(joint_seq, belief, env, sorted_agents, C, gs_state)
        
        if value > best_value
            best_value = value
            best_sequence = joint_seq
        end
        
        if (i % progress_step == 0) || (i == total_sequences)
            perc = round(100 * i / total_sequences; digits=1)
            println("    Progress: $(i)/$(total_sequences) ($(perc)%), best value: $(round(best_value, digits=3))")
        end
    end
    
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ MPOMDP Open-Loop: Best joint plan found in $(round(planning_time, digits=3)) seconds")
    println("  Best value: $(round(best_value, digits=3))")
    
    return best_sequence, planning_time
end

"""
generate_joint_action_sequences(agents, env, C, gs_state, rng; N_sequences=0)
Generate joint action sequences of length C.

If N_sequences <= 0, performs the original exact enumeration of *all*
joint sequences. If N_sequences > 0, samples up to N_sequences distinct
sequences uniformly at random from the (implicit) full space.
"""
function generate_joint_action_sequences(agents::Vector{Agent}, env, C::Int, gs_state,
                                         rng::AbstractRNG=Random.GLOBAL_RNG;
                                         N_sequences::Int=0)
    if C == 0
        return Vector{Vector{JointAction}}[]
    end
    
    # Get available actions for each agent at each timestep
    actions_per_agent_per_timestep = Vector{Vector{Vector{SensingAction}}}()
    
    for t in 1:C
        # Calculate absolute timestep
        global_timestep = gs_state.time_step + t - 1
        
        # Get actions for each agent at this timestep
        agent_actions_at_t = Vector{Vector{SensingAction}}()
        
        for agent in agents
            # Get agent position at this timestep
            pos = get_position_at_time(agent.trajectory, global_timestep, agent.phase_offset)
            
            # Get field of regard
            for_cells = get_field_of_regard_at_position(agent, pos, env)
            
            # Generate available actions for this agent at this timestep
            available_actions = SensingAction[]
            
            # Add wait action
            push!(available_actions, SensingAction(agent.id, Tuple{Int, Int}[], false))
            
            # Add single-cell sensing actions
            for cell in for_cells
                push!(available_actions, SensingAction(agent.id, [cell], false))
            end
            
            # Add multi-cell sensing actions (up to max_sensing_targets)
            if length(for_cells) > 1 && env.max_sensing_targets > 1
                for subset_size in 2:min(env.max_sensing_targets, length(for_cells))
                    for subset in combinations(for_cells, subset_size)
                        push!(available_actions, SensingAction(agent.id, collect(subset), false))
                    end
                end
            end
            
            push!(agent_actions_at_t, available_actions)
        end
        
        push!(actions_per_agent_per_timestep, agent_actions_at_t)
    end
    
    # For each timestep, generate all joint actions (Cartesian product of agent actions)
    joint_actions_per_timestep = Vector{Vector{JointAction}}()
    
    for t in 1:C
        agent_actions_at_t = actions_per_agent_per_timestep[t]
        
        timestep_joint_actions = Vector{JointAction}()
        for action_combo in Iterators.product(agent_actions_at_t...)
            ordered_actions = [action for action in action_combo]
            push!(timestep_joint_actions, JointAction(ordered_actions))
        end
        push!(joint_actions_per_timestep, timestep_joint_actions)
    end
    
    # If N_sequences <= 0, do full enumeration (original behaviour)
    if N_sequences <= 0
        joint_sequences = Vector{Vector{JointAction}}()
        for joint_action_combo in Iterators.product(joint_actions_per_timestep...)
            sequence = [ja for ja in joint_action_combo]
            push!(joint_sequences, sequence)
        end
        if length(joint_sequences) > 50000
            println("  ⚠️  Large sequence space: $(length(joint_sequences)) sequences (exact search may be slow)")
        end
        return joint_sequences
    end

    # Otherwise, sample up to N_sequences sequences uniformly at random
    joint_sequences = Vector{Vector{JointAction}}()
    seen = Set{Vector{Int}}()  # store indices to avoid trivial duplicates
    
    # Precompute counts per timestep for index-based sampling
    counts_per_timestep = [length(joint_actions_per_timestep[t]) for t in 1:C]
    
    max_trials = 10 * N_sequences
    trials = 0
    while length(joint_sequences) < N_sequences && trials < max_trials
        trials += 1
        idxs = [rand(rng, 1:counts_per_timestep[t]) for t in 1:C]
        key = idxs
        if key in seen
            continue
        end
        push!(seen, key)
        sequence = [joint_actions_per_timestep[t][idxs[t]] for t in 1:C]
        push!(joint_sequences, sequence)
    end

    println("  📉 Sampled $(length(joint_sequences)) joint sequences (target N_sequences=$(N_sequences))")
    return joint_sequences
end

"""
calculate_joint_sequence_reward(joint_sequence, belief, env, agents, C, gs_state)
Evaluates a joint action sequence using the global belief.

Exact expected reward: at each step marginalizes over all possible observation
outcomes (EVENT_PRESENT / NO_EVENT per observed cell) with their probabilities,
so the returned value is the true expected discounted reward.
"""
function calculate_joint_sequence_reward(joint_sequence::Vector{JointAction}, belief::Belief, env, agents::Vector{Agent}, C::Int, gs_state)
    γ = env.discount
    return _exact_expected_reward_from(deepcopy(belief), joint_sequence, 1, env, γ)
end

"""
Deduplicate observed cells: when multiple agents observe the same cell, count it once
and use a single probability. Returns (cell, P(event)) for each unique cell (first occurrence).
"""
function _unique_cells_and_probs(belief::Belief, joint_action::JointAction)
    seen = Set{Tuple{Int, Int}}()
    cells_and_probs = Tuple{Tuple{Int, Int}, Float64}[]
    for action in joint_action.actions
        for cell in action.target_cells
            if cell ∉ seen
                push!(seen, cell)
                p = get_event_probability(belief, cell)
                push!(cells_and_probs, (cell, p))
            end
        end
    end
    return cells_and_probs
end

"""
_exact_expected_reward_from(belief, joint_sequence, t, env, γ)
Recursively computes exact expected discounted reward from step t onward,
marginalizing over all possible observation outcomes (per unique observed cell).
Uses unique cells only so the expectation is the true expected discounted reward.
"""
function _exact_expected_reward_from(belief::Belief, joint_sequence::Vector{JointAction}, t::Int, env, γ::Float64)
    if t > length(joint_sequence)
        return 0.0
    end
    joint_action = joint_sequence[t]

    # Unique (cell, P(event)) only — multiple agents sensing same cell counted once
    cells_and_probs = _unique_cells_and_probs(belief, joint_action)

    # Expected immediate reward (each unique cell once) - use ABBA/PBVI reward
    immediate_reward = 0.0
    for (cell, p) in cells_and_probs
        immediate_reward += calculate_sophisticated_reward(belief, cell)
    end
    discounted_imm = (γ^(t - 1)) * immediate_reward

    # If no observations this step, just evolve belief and recurse
    if isempty(cells_and_probs)
        b_next = evolve_no_obs_fast(deepcopy(belief), env, calculate_uncertainty=false)
        return discounted_imm + _exact_expected_reward_from(b_next, joint_sequence, t + 1, env, γ)
    end

    # Marginalize over 2^N outcomes for N unique cells (correct expectation)
    n_cells = length(cells_and_probs)
    exp_continuation = 0.0
    for outcome_bits in 0:(2^n_cells - 1)
        prob = 1.0
        b_new = deepcopy(belief)
        for i in 1:n_cells
            cell, p = cells_and_probs[i]
            observed = ((outcome_bits >> (i - 1)) & 1 == 1) ? EVENT_PRESENT : NO_EVENT
            prob *= (observed == EVENT_PRESENT ? p : (1.0 - p))
            b_new = collapse_belief_to(b_new, cell, observed)
        end
        b_next = evolve_no_obs_fast(b_new, env, calculate_uncertainty=false)
        exp_continuation += prob * _exact_expected_reward_from(b_next, joint_sequence, t + 1, env, γ)
    end

    return discounted_imm + exp_continuation
end

"""
get_field_of_regard_at_position(agent, position, env)
Gets field of regard for an agent at a specific position.
"""
function get_field_of_regard_at_position(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Check sensor pattern
    if agent.sensor.pattern == :cross
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
combinations(elements, k)
Generate all combinations of k elements from the collection.
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


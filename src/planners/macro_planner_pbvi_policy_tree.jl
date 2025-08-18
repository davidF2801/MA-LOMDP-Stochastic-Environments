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

export best_policy_tree, create_debuggable_policy_tree, print_policy_tree_structure, export_policy_tree_to_dict, inspect_policy_node, find_policy_nodes, set_reward_config_from_main

############################
# Types and small utilities
############################

# Local observation symbol for Agent i.
# Here: :none for wait, :event / :no_event for binary sensing.
@enum ObsSym::UInt8 begin
    OBS_NONE = 0
    OBS_NO_EVENT = 1
    OBS_EVENT = 2
end

struct ClockVector
    phases::Vector{Int}  # Phases of all agents in the trajectory
end

# Add copy method for ClockVector
Base.copy(cv::ClockVector) = ClockVector(copy(cv.phases))
Base.deepcopy(cv::ClockVector) = ClockVector(deepcopy(cv.phases))

struct Particle
    belief::Belief
    w::Float64
end

# Policy-tree node = (clock, depth, local history of Agent i, particle set)
struct PolicyTreeNode
    clock::ClockVector
    depth::Int
    hist::Vector{ObsSym}     # only Agent i local history
    particles::Vector{Particle}
    digest::UInt64           # e.g., hash of weighted-mean belief for fast lookup
end

# Add hash and equality methods for dictionary keys
Base.hash(cv::ClockVector, h::UInt) = hash(cv.phases, h)
Base.hash(ptn::PolicyTreeNode, h::UInt) = hash((ptn.clock.phases, ptn.depth, ptn.hist), h)

Base.isequal(cv1::ClockVector, cv2::ClockVector) = cv1.phases == cv2.phases
Base.isequal(ptn1::PolicyTreeNode, ptn2::PolicyTreeNode) = 
    isequal(ptn1.clock, ptn2.clock) && ptn1.depth == ptn2.depth && ptn1.hist == ptn2.hist
Base.:(==)(cv1::ClockVector, cv2::ClockVector) = isequal(cv1, cv2)
Base.:(==)(ptn1::PolicyTreeNode, ptn2::PolicyTreeNode) = isequal(ptn1, ptn2)

# Weighted mean belief for a node (used for NN / digest)
function node_mean_belief(ptn::PolicyTreeNode)::Belief
    # Make a shallow copy of shape from the first particle, then weighted average
    @assert !isempty(ptn.particles)
    acc = deepcopy(ptn.particles[1].belief)
    acc.event_distributions .= 0.0
    Z = 0.0
    for prt in ptn.particles
        acc.event_distributions .+= prt.w .* prt.belief.event_distributions
        Z += prt.w
    end
    if Z > 0.0
        acc.event_distributions ./= Z
    end
    return acc
end

# Recompute digest from mean belief (quantize to be stable)
function recompute_digest!(ptn::PolicyTreeNode)
    if !isempty(ptn.particles)
        mb = node_mean_belief(ptn)
        ptn.digest = hash(round.(mb.event_distributions; digits=10), UInt(0))
    else
        ptn.digest = UInt(0)
    end
    return ptn
end

# Merge a particle into a child node (accumulate weight, optional cap/resample later)
function push_particle!(ptn::PolicyTreeNode, b::Belief, w::Float64)
    push!(ptn.particles, Particle(deepcopy(b), w))
end

# Create or fetch child node keyed by (clock, depth+1, hist ⊕ σ)
function get_child_node!(
    nodes_index::Dict{Tuple{Vector{Int},Int,Vector{ObsSym}}, PolicyTreeNode},
    parent::PolicyTreeNode, next_clock::ClockVector, σ::ObsSym
)::PolicyTreeNode
    key = (copy(next_clock.phases), parent.depth+1, vcat(parent.hist, σ))
    if haskey(nodes_index, key)
        return nodes_index[key]
    else
        child = PolicyTreeNode(next_clock, parent.depth+1, vcat(parent.hist, σ),
                               Particle[], UInt(0))
        recompute_digest!(child)
        nodes_index[key] = child
        return child
    end
end

# Sample one particle index proportional to weights
function sample_particle_idx(ws::Vector{Float64})
    # cumulative roulette
    s = sum(ws); 
    if s <= 0.0
        return rand(1:length(ws))  # uniform fallback
    end
    r = rand() * s
    c = 0.0
    @inbounds for i in 1:length(ws)
        c += ws[i]
        if r <= c
            return i
        end
    end
    return length(ws)
end

# All possible local symbols for (agent_i, action a)
function possible_local_symbols(agent_i::Agent, a::SensingAction)::Vector{ObsSym}
    isempty(a.target_cells) && return [OBS_NONE]
    # noiseless binary
    return [OBS_NO_EVENT, OBS_EVENT]
end

# Belief evolution cache for performance
const BELIEF_EVOLUTION_CACHE = Dict{UInt64, Belief}()
const CACHE_STATS = Dict{Symbol, Int}(:hits => 0, :misses => 0, :size => 0)

# Reward function configuration - these will be set from main.jl
const DEFAULT_ENTROPY_WEIGHT = Ref(0.5)    # w_H: Weight for entropy reduction (coordination)
const DEFAULT_VALUE_WEIGHT = Ref(0.5)      # w_F: Weight for state value (detection priority)
const DEFAULT_INFORMATION_STATES = Ref([1, 2])  # I_1: No event, I_2: Event
const DEFAULT_STATE_VALUES = Ref([0.1, 0.9])    # F_1: No event value, F_2: Event value

"""
Calculate sophisticated reward for sensing actions
"""
function calculate_sophisticated_reward(belief::Belief, cell::Tuple{Int, Int})
    # 1. Entropy-based reward: w_H * (H_prior - H_post)
    H_before = calculate_cell_entropy(belief, cell)
    H_after = 0.0  # Simplified: assume perfect observation
    entropy_reward = DEFAULT_ENTROPY_WEIGHT[] * (H_before - H_after)
    
    # 2. State value reward: w_F * E[F_I_j]
    # Calculate expected value under current belief
    event_prob = get_event_probability(belief, cell)
    no_event_prob = 1.0 - event_prob
    
    # E[F_I_j] = Σ_k p(I_k) * F_k
    expected_value = no_event_prob * DEFAULT_STATE_VALUES[][1] + event_prob * DEFAULT_STATE_VALUES[][2]
    value_reward = DEFAULT_VALUE_WEIGHT[] * expected_value
    
    # Total reward for this cell
    return entropy_reward + value_reward
end

"""
Set reward configuration from main.jl constants
"""
function set_reward_config_from_main(entropy_weight::Float64, value_weight::Float64, 
                                   information_states::Vector{Int}, state_values::Vector{Float64})
    DEFAULT_ENTROPY_WEIGHT[] = entropy_weight
    DEFAULT_VALUE_WEIGHT[] = value_weight
    DEFAULT_INFORMATION_STATES[] = information_states
    DEFAULT_STATE_VALUES[] = state_values
    
    println("🎯 Policy Tree Reward configuration synced from main.jl:")
    println("   • Entropy weight (w_H): $entropy_weight")
    println("   • Value weight (w_F): $value_weight")
    println("   • Information states: $information_states")
    println("   • State values: $state_values")
end

########################################
# One-step evolution + branching kernel
########################################

"""
Given a *system* belief `b` and action `a` by Agent i at `clock`,
(1) apply other agents' scheduled observations by *sampling* their outcomes,
(2) evolve one step,
(3) return:
    - expected immediate reward r_i under `b` and `a` (no sampling for Agent i's symbol),
    - next clock `τ′`,
    - a Dict σ→(pσ, bσ) for Agent i's local symbols (no extra sampling for σ).
"""
function step_and_branch(
    clock::ClockVector, b::Belief, a::SensingAction,
    agent_i::Agent, agents_j::Vector{Agent}, env, gs_state
)
    # 1) Reward under current b (your sophisticated reward)
    r_step = 0.0
    if !isempty(a.target_cells)
        for cell in a.target_cells
            r_step += calculate_sophisticated_reward(b, cell)
        end
    end

    # 2) Compute global time from Agent i's phase
    agent_i_index = find_agent_index(agent_i, env)
    t_global = (agent_i_index === nothing) ? gs_state.time_step :
               gs_state.time_step + clock.phases[agent_i_index]

    # 3) Sample *other* agents' observations and collapse into b
    b_after = deepcopy(b)
    for j in agents_j
        if j_senses_at_time(j, t_global, gs_state)
            cell = scheduled_cell(j, t_global, gs_state)
            if cell !== nothing
                state = sample_event_state_from(b_after, cell)
                b_after = collapse_belief_to(b_after, cell, state)
            end
        end
    end

    # 4) Advance dynamics once (system)
    b_after = evolve_no_obs_fast(b_after, env, calculate_uncertainty=false)
    next_clock = advance_clock_vector(clock, [agent_i; agents_j])

    # 5) Split on Agent i's *local* obs symbol σ (no extra sampling):
    #    For noiseless binary sensing on one cell:
    children = Dict{ObsSym, Tuple{Float64,Belief}}()
    if isempty(a.target_cells)
        # wait → deterministic σ=∅
        children[OBS_NONE] = (1.0, deepcopy(b_after))
    else
        cell = a.target_cells[1]  # if 1-at-a-time, adjust if multi-cell
        p_event = get_event_probability(b_after, cell)
        # σ = NO_EVENT
        b_no = collapse_belief_to(deepcopy(b_after), cell, NO_EVENT)
        children[OBS_NO_EVENT] = (1.0 - p_event, b_no)
        # σ = EVENT
        b_yes = collapse_belief_to(deepcopy(b_after), cell, EVENT_PRESENT)
        children[OBS_EVENT] = (p_event, b_yes)
    end

    return r_step, next_clock, children
end

########################################
# PBVI for policy trees (modified backup)
########################################

function pbvi_policy_tree(
    nodes::Vector{PolicyTreeNode}, N_particles::Int, N_sweeps::Int, ε::Float64,
    agent_i::Agent, env, gs_state;
    max_particles_per_node::Int=256, prune_eps::Float64=0.0
)
    # Index nodes by (phases, depth, hist) so children can be created/found quickly
    nodes_index = Dict{Tuple{Vector{Int},Int,Vector{ObsSym}}, PolicyTreeNode}()
    for ptn in nodes
        key = (copy(ptn.clock.phases), ptn.depth, copy(ptn.hist))
        nodes_index[key] = ptn
    end

    VALUE  = Dict{PolicyTreeNode, Float64}(ptn => 0.0 for ptn in nodes)
    POLICY = Dict{PolicyTreeNode, SensingAction}()
    # Optional: record explicit children for debugging/rollout
    CHILDREN = Dict{PolicyTreeNode, Dict{ObsSym, PolicyTreeNode}}()

    γ = env.discount

    for sweep in 1:N_sweeps
        sweep_start = time()
        Δ = 0.0
        for ptn in shuffle(nodes)
            # Actions feasible at this clock
            action_set = all_pointings(agent_i, ptn.clock, env)

            best_Q = -Inf
            best_act = nothing
            best_child_map = Dict{ObsSym, PolicyTreeNode}()

            # Pre-extract particle arrays for sampling
            ws = [p.w for p in ptn.particles]
            isempty(ws) && (ws = [1.0])  # guard; shouldn't happen

            for a in action_set
                # Expected immediate reward accumulator and per-symbol weight mass
                reward_acc = 0.0
                sym_mass   = Dict{ObsSym, Float64}()                # Σ_n w_n pσ^(n)
                sym_child  = Dict{ObsSym, PolicyTreeNode}()         # σ → child node
                # Clear temporary particle bags for this (ptn,a) evaluation
                temp_particles = Dict{ObsSym, Vector{Particle}}()

                # Monte Carlo over node's particle set (resampling with replacement)
                # This averages over *other agents* uncertainties.
                for n in 1:N_particles
                    # sample particle index proportional to weight
                    idx = sample_particle_idx(ws)
                    par = ptn.particles[min(idx, length(ptn.particles))]

                    r_step, next_clock, children = step_and_branch(
                        ptn.clock, par.belief, a, agent_i, get_other_agents(agent_i, env), env, gs_state
                    )

                    reward_acc += par.w * r_step

                    # Distribute this particle's weight into each σ branch
                    for (σ, (pσ, bσ)) in children
                        mass = par.w * pσ
                        if mass <= prune_eps
                            continue
                        end
                        # Create/fetch child node and accumulate particle
                        child = get_child_node!(nodes_index, ptn, next_clock, σ)
                        if !in(child, nodes)
                            push!(nodes, child)           # in case it's new
                        end
                        sym_child[σ] = child
                        sym_mass[σ] = get(sym_mass, σ, 0.0) + mass
                        push!(get!(temp_particles, σ, Particle[]), Particle(bσ, mass))
                    end
                end

                # Push the temporary particles into the actual child nodes
                for (σ, bag) in temp_particles
                    child = sym_child[σ]
                    # Merge and optionally cap/resample
                    append!(child.particles, bag)
                    # (Optional) light resampling to keep node size bounded
                    if length(child.particles) > max_particles_per_node
                        # simple stratified downsample proportional to weights
                        ws_child = [p.w for p in child.particles]
                        newbag = Particle[]
                        for _ in 1:max_particles_per_node
                            j = sample_particle_idx(ws_child)
                            push!(newbag, child.particles[j])
                        end
                        child.particles = newbag
                    end
                    recompute_digest!(child)
                    # Ensure VALUE dict has an entry so next sweeps can see it
                    if !haskey(VALUE, child)
                        VALUE[child] = 0.0
                    end
                end

                # Compute backup: average reward + γ * mixture of child values
                # Normalize by total node mass to keep scale stable
                total_w = sum(ws)
                R_bar   = (total_w > 0 ? reward_acc/total_w : 0.0)

                V_mix = 0.0
                if total_w > 0
                    for (σ, mσ) in sym_mass
                        V_mix += (mσ/total_w) * VALUE[sym_child[σ]]
                    end
                end

                Q_hat = R_bar + γ * V_mix

                if Q_hat > best_Q
                    best_Q = Q_hat
                    best_act = a
                    best_child_map = sym_child
                end
            end

            # Bellman update
            Δ = max(Δ, abs(best_Q - VALUE[ptn]))
            VALUE[ptn] = best_Q
            if best_act !== nothing
                POLICY[ptn] = best_act
                CHILDREN[ptn] = best_child_map
            end
        end

        sweep_time = time() - sweep_start
        println("  Sweep $sweep: max Δ = $(round(Δ, digits=4)) in $(round(sweep_time, digits=2))s")
        if Δ < ε
            println("  Converged.")
            break
        end
    end

    return VALUE, POLICY, CHILDREN
end

########################################
# Policy tree construction and execution
########################################

"""
Build initial belief set for PBVI policy tree by depositing particles into nodes
"""
function build_belief_set(B_clean::Belief, agent_i::Agent, τ_i::Int, agents_j::Vector{Agent}, 
                         τ_js_vector::Dict{Int, Int}, H::Int, env, gs_state, N_seed::Int)
    nodes = PolicyTreeNode[]
    
    # Calculate initial phases relative to agent_i
    all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
    agent_i_index = find_agent_index(agent_i, env)
    
    initial_phases = Int[]
    for (i, agent) in enumerate(all_agents)
        if i == agent_i_index
            push!(initial_phases, 0)  # Agent_i is at phase 0
        else
            relative_offset = mod((agent.phase_offset - agent_i.phase_offset), agent.trajectory.period)
            push!(initial_phases, relative_offset)
        end
    end
    initial_clock = ClockVector(initial_phases)
    
    # Create root node at depth 0 with empty history
    root_node = PolicyTreeNode(initial_clock, 0, ObsSym[], Particle[], UInt(0))
    
    # Deposit initial particles into root node
    for seed in 1:N_seed
        # Sample system state at τ_i
        b = deepcopy(B_clean)
        
        # Determine t_clean and roll forward with other agents' observations
        other_agent_sync_times = [τ_js_vector[j.id] for j in agents_j]
        t_clean = any(sync_time == -1 for sync_time in other_agent_sync_times) ? 0 : minimum(other_agent_sync_times)
        
        for t in t_clean:(τ_i-1)
            for j in agents_j
                if t > τ_js_vector[j.id] && j_senses_at_time(j, t, gs_state)
                    cell = scheduled_cell(j, t, gs_state)
                    if cell !== nothing
                        state = sample_event_state_from(b, cell)
                        b = collapse_belief_to(b, cell, state)
                    end
                end
            end
            b = evolve_no_obs_fast(b, env, calculate_uncertainty=false)
        end
        
        # Add particle to root node
        push_particle!(root_node, b, 1.0)
    end
    
    recompute_digest!(root_node)
    push!(nodes, root_node)
    
    # Generate rollout particles for different depths
    for depth in 1:H
        # Create nodes at this depth by rolling out from existing nodes
        current_depth_nodes = PolicyTreeNode[]
        
        for parent_node in filter(n -> n.depth == depth-1, nodes)
            # Sample some rollout trajectories from this parent
            n_rollouts = max(1, N_seed ÷ (2^depth))  # Fewer rollouts at deeper levels
            
            for _ in 1:n_rollouts
                if isempty(parent_node.particles)
                    continue
                end
                
                # Sample a particle from parent
                ws = [p.w for p in parent_node.particles]
                idx = sample_particle_idx(ws)
                par = parent_node.particles[idx]
                
                # Take a random action
                action_set = all_pointings(agent_i, parent_node.clock, env)
                if isempty(action_set)
                    continue
                end
                a_rand = rand(action_set)
                
                # Step and branch
                r_step, next_clock, children = step_and_branch(
                    parent_node.clock, par.belief, a_rand, agent_i, agents_j, env, gs_state
                )
                
                # Create child nodes for each observation symbol
                for (σ, (pσ, bσ)) in children
                    if pσ > 1e-6  # Skip very low probability branches
                        child_hist = vcat(parent_node.hist, σ)
                        # Find or create child node
                        child_node = nothing
                        for existing in current_depth_nodes
                            if existing.clock.phases == next_clock.phases && 
                               existing.depth == depth && existing.hist == child_hist
                                child_node = existing
                                break
                            end
                        end
                        
                        if child_node === nothing
                            child_node = PolicyTreeNode(next_clock, depth, child_hist, Particle[], UInt(0))
                            push!(current_depth_nodes, child_node)
                        end
                        
                        # Add particle to child
                        push_particle!(child_node, bσ, par.w * pσ)
                    end
                end
            end
        end
        
        # Recompute digests and add to main nodes list
        for node in current_depth_nodes
            recompute_digest!(node)
            push!(nodes, node)
        end
    end
    
    println("📊 Policy tree belief set: generated $(length(nodes)) nodes across $(H+1) depths")
    
    return nodes
end

"""
Extract reactive policy from policy tree
"""
function extract_reactive_policy(POLICY::Dict{PolicyTreeNode, SensingAction}, 
                                VALUE::Dict{PolicyTreeNode, Float64}, 
                                CHILDREN::Dict{PolicyTreeNode, Dict{ObsSym, PolicyTreeNode}},
                                nodes::Vector{PolicyTreeNode}, agent_i::Agent, env, gs_state, H::Int)
    
    # Create a reactive policy function that takes observation history and current time, returns action
    function reactive_policy(observation_history::Vector{GridObservation}, current_time::Int)
        # Convert observation history to local observation symbols
        local_hist = ObsSym[]
        
        for obs in observation_history
            if isempty(obs.sensed_cells)
                push!(local_hist, OBS_NONE)
            else
                # Assume binary sensing on first cell
                if !isempty(obs.event_states) && obs.event_states[1] == EVENT_PRESENT
                    push!(local_hist, OBS_EVENT)
                else
                    push!(local_hist, OBS_NO_EVENT)
                end
            end
        end
        
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
        
        # Find matching policy tree node
        best_node = nothing
        best_match_score = -1
        
        for node in nodes
            if node.clock.phases == current_clock.phases
                # Calculate history match score
                hist_match = 0
                min_len = min(length(node.hist), length(local_hist))
                for i in 1:min_len
                    if node.hist[i] == local_hist[i]
                        hist_match += 1
                    end
                end
                
                match_score = hist_match - abs(length(node.hist) - length(local_hist))
                if match_score > best_match_score
                    best_match_score = match_score
                    best_node = node
                end
            end
        end
        
        if best_node !== nothing && haskey(POLICY, best_node)
            action = POLICY[best_node]
            
            # Verify action is feasible for current phase
            if is_action_feasible_for_phase(action, agent_i, current_clock, env)
                return action
            end
        end
        
        # Fallback: generate a feasible action
        feasible_actions = all_pointings(agent_i, current_clock, env)
        if !isempty(feasible_actions)
            return feasible_actions[1]
        else
            return SensingAction(agent_i.id, Tuple{Int, Int}[], false)
        end
    end
    
    return reactive_policy
end

"""
Main entry point for policy tree planning
"""
function best_policy_tree(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG, 
                    N_seed::Int=8, N_particles::Int=2, N_sweeps::Int=5, ε::Float64=0.5)
    # Start timing
    start_time = time()
    
    # Get parameters from the pseudocode
    B_clean = deepcopy(belief)
    agent_i = agent
    τ_i = gs_state.time_step
    agents_j = [env.agents[j] for j in keys(env.agents) if j != agent.id]
    τ_js_vector = gs_state.agent_last_sync
    H = min(3, C)  # Much smaller horizon for testing
    
    println("🔄 Building belief set for PBVI policy tree...")
    # Build belief set
    nodes = build_belief_set(B_clean, agent_i, τ_i, agents_j, τ_js_vector, H, env, gs_state, N_seed)
    
    println("🔄 Running PBVI with $(length(nodes)) policy tree nodes...")
    # Run PBVI
    VALUE, POLICY, CHILDREN = pbvi_policy_tree(nodes, N_particles, N_sweeps, ε, agent_i, env, gs_state)
    
    # Create a clear policy tree structure for debugging
    policy_tree = create_debuggable_policy_tree(nodes, POLICY, VALUE, agent_i, env)
    
    # Extract reactive policy from policy tree
    reactive_policy = extract_reactive_policy(POLICY, VALUE, CHILDREN, nodes, agent_i, env, gs_state, H)
    
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
    println("📊 Belief management cache: $(cache_stats[:hits]) hits, $(cache_stats[:misses]) misses, $(round(cache_stats[:hit_rate] * 100, digits=1))% hit rate")
    
    return reactive_policy, planning_time, policy_tree
end

# Helper functions that remain the same
function sample_event_state_from(belief::Belief, cell::Tuple{Int, Int})
    p_event = get_event_probability(belief, cell)
    if rand() < p_event
        return EVENT_PRESENT
    else
        return NO_EVENT
    end
end

function j_senses_at_time(j::Agent, t::Int, gs_state)
    if !haskey(gs_state.agent_plans, j.id) || gs_state.agent_plans[j.id] === nothing
        return false
    end
    
    plan = gs_state.agent_plans[j.id]
    plan_timestep = (t - gs_state.agent_last_sync[j.id]) + 1
    
    if 1 <= plan_timestep <= length(plan)
        action = plan[plan_timestep]
        return !isempty(action.target_cells)
    end
    
    return false
end

function scheduled_cell(j::Agent, t::Int, gs_state)
    if !haskey(gs_state.agent_plans, j.id) || gs_state.agent_plans[j.id] === nothing
        return nothing
    end
    
    plan = gs_state.agent_plans[j.id]
    plan_timestep = (t - gs_state.agent_last_sync[j.id]) + 1
    
    if 1 <= plan_timestep <= length(plan)
        action = plan[plan_timestep]
        if !isempty(action.target_cells)
            return action.target_cells[1]
        end
    end
    
    return nothing
end

function advance_clock_vector(τ_clock::ClockVector, agents)
    new_phases = Int[]
    for (i, agent) in enumerate(agents)
        new_phase = mod(τ_clock.phases[i] + 1, agent.trajectory.period)
        push!(new_phases, new_phase)
    end
    return ClockVector(new_phases)
end

function all_pointings(agent::Agent, τ_clock::ClockVector, env)
    actions = SensingAction[]
    
    agent_index = find_agent_index(agent, env)
    if agent_index === nothing
        return [SensingAction(agent.id, Tuple{Int, Int}[], false)]
    end
    
    phase = τ_clock.phases[agent_index]
    pos = get_position_at_time(agent.trajectory, phase)
    available_cells = get_field_of_regard_at_position(agent, pos, env)
    
    push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))
    
    for cell in available_cells
        action = SensingAction(agent.id, [cell], false)
        if check_battery_feasible(agent, action, agent.battery_level)
            push!(actions, action)
        end
    end
    
    return actions
end

function is_action_feasible_for_phase(action::SensingAction, agent::Agent, clock::ClockVector, env)
    agent_index = find_agent_index(agent, env)
    if agent_index === nothing
        return false
    end
    
    phase = clock.phases[agent_index]
    pos = get_position_at_time(agent.trajectory, phase)
    available_cells = get_field_of_regard_at_position(agent, pos, env)
    
    for target_cell in action.target_cells
        if !(target_cell in available_cells)
            return false
        end
    end
    
    if !check_battery_feasible(agent, action, agent.battery_level)
        return false
    end
    
    return true
end

function find_agent_index(agent::Agent, env)
    all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
    for (i, env_agent) in enumerate(all_agents)
        if env_agent.id == agent.id
            return i
        end
    end
    return nothing
end

function get_other_agents(agent_i::Agent, env)
    return [env.agents[j] for j in keys(env.agents) if j != agent_i.id]
end

function get_field_of_regard_at_position(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    if agent.sensor.pattern == :cross
        ax, ay = position
        for dx in -1:1, dy in -1:1
            nx, ny = ax + dx, ay + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                if (dx == 0 && dy == 0) || (dx == 0 && dy != 0) || (dx != 0 && dy == 0)
                    push!(fov_cells, (nx, ny))
                end
            end
        end
    elseif agent.sensor.pattern == :row_only || agent.sensor.range == 0.0
        for nx in 1:env.width
            push!(fov_cells, (nx, y))
        end
    else
        sensor_range = round(Int, agent.sensor.range)
        for dx in -sensor_range:sensor_range
            for dy in -sensor_range:sensor_range
                nx, ny = x + dx, y + dy
                if 1 <= nx <= env.width && 1 <= ny <= env.height
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

# Debugging and visualization functions (simplified versions)
function create_debuggable_policy_tree(nodes::Vector{PolicyTreeNode}, POLICY::Dict{PolicyTreeNode, SensingAction}, 
                                     VALUE::Dict{PolicyTreeNode, Float64}, agent_i::Agent, env)
    policy_tree = Dict{Tuple{Vararg{Int}}, Vector{Dict{String, Any}}}()
    
    for node in nodes
        clock_key = Tuple(node.clock.phases)
        
        node_info = Dict{String, Any}()
        node_info["clock_phases"] = node.clock.phases
        node_info["depth"] = node.depth
        node_info["history"] = string.(node.hist)
        node_info["num_particles"] = length(node.particles)
        
        if haskey(POLICY, node)
            action = POLICY[node]
            node_info["action"] = isempty(action.target_cells) ? "WAIT" : "SENSE[$(action.target_cells)]"
        else
            node_info["action"] = "NO_ACTION"
        end
        
        if haskey(VALUE, node)
            node_info["value"] = VALUE[node]
        else
            node_info["value"] = 0.0
        end
        
        if !haskey(policy_tree, clock_key)
            policy_tree[clock_key] = Vector{Dict{String, Any}}()
        end
        push!(policy_tree[clock_key], node_info)
    end
    
    return policy_tree
end

function print_policy_tree_structure(policy_tree::Dict{Tuple{Vararg{Int}}, Vector{Dict{String, Any}}})
    println("📋 Policy Tree Summary:")
    println("  Total clock configurations: $(length(policy_tree))")
    
    total_nodes = sum(length(nodes) for nodes in values(policy_tree))
    println("  Total policy nodes: $total_nodes")
    
    for (clock_key, nodes) in policy_tree
        println("\n🕐 Clock Configuration: $clock_key")
        println("  Nodes in this configuration: $(length(nodes))")
        
        # Group by depth
        depth_groups = Dict{Int, Vector{Dict{String, Any}}}()
        for node in nodes
            depth = get(node, "depth", 0)
            if !haskey(depth_groups, depth)
                depth_groups[depth] = Vector{Dict{String, Any}}()
            end
            push!(depth_groups[depth], node)
        end
        
        for depth in sort(collect(keys(depth_groups)))
            depth_nodes = depth_groups[depth]
            println("    Depth $depth: $(length(depth_nodes)) nodes")
            
            # Show action distribution
            action_counts = Dict{String, Int}()
            for node in depth_nodes
                action = get(node, "action", "NONE")
                action_counts[action] = get(action_counts, action, 0) + 1
            end
            
            for (action, count) in action_counts
                println("      $action: $count")
            end
        end
    end
end

# Placeholder functions for compatibility
export_policy_tree_to_dict(policy_tree) = policy_tree
inspect_policy_node(policy_tree, clock_key, node_index) = println("Inspecting node $node_index at clock $clock_key")
find_policy_nodes(policy_tree; kwargs...) = []

end # module 
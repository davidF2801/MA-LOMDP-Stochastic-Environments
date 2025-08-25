module MacroPlannerPBVI

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Infiltrator
using Statistics
using Base.Threads
# Removed Divergences.jl - using custom KL divergence implementation instead
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

export best_script, calculate_macro_script_reward, calculate_sophisticated_reward, configure_reward_weights, set_reward_config_from_main, get_belief_cache_stats, get_timing_stats, get_detailed_timing_analysis, analyze_cache_efficiency, test_kl_performance, test_blas_performance

# PBVI-specific types
struct ClockVector
    phases::Vector{Int}  # Phases of all agents in the trajectory
end

# Add copy method for ClockVector
Base.copy(cv::ClockVector) = ClockVector(copy(cv.phases))
Base.deepcopy(cv::ClockVector) = ClockVector(deepcopy(cv.phases))

struct BeliefPoint
    clock::ClockVector
    digest::UInt64          # immutable, pre-computed hash
    belief::Belief          # still carried for look-ups
end

# Add copy method for BeliefPoint
Base.copy(bp::BeliefPoint) = BeliefPoint(copy(bp.clock), bp.digest, deepcopy(bp.belief))
Base.deepcopy(bp::BeliefPoint) = BeliefPoint(deepcopy(bp.clock), bp.digest, deepcopy(bp.belief))

# Add hash and equality methods for dictionary keys
Base.hash(cv::ClockVector, h::UInt) = hash(cv.phases, h)
Base.hash(bp::BeliefPoint, h::UInt) = hash(bp.clock, hash(bp.digest, h))

Base.isequal(cv1::ClockVector, cv2::ClockVector) = cv1.phases == cv2.phases
Base.isequal(bp1::BeliefPoint, bp2::BeliefPoint) = isequal(bp1.clock, bp2.clock) && beliefs_are_equivalent(bp1.belief, bp2.belief)
Base.:(==)(cv1::ClockVector, cv2::ClockVector) = isequal(cv1, cv2)
Base.:(==)(bp1::BeliefPoint, bp2::BeliefPoint) = isequal(bp1, bp2)

# Belief evolution cache for performance
const BELIEF_EVOLUTION_CACHE = Dict{UInt64, Belief}()
const CACHE_STATS = Dict{Symbol, Int}(:hits => 0, :misses => 0, :size => 0)
const TIMING_STATS = Dict{Symbol, Float64}(:total_cache_time => 0.0, :total_direct_time => 0.0, :cache_calls => 0, :direct_calls => 0)

# BLAS-based KL divergence optimization
const T = Float64
@inline clamp01(x::T) where {T<:AbstractFloat} = clamp(x, eps(T), one(T)-eps(T))

"""
Prepare candidates for fast KL divergence search using BLAS
"""
function prepare_kl_candidates(𝔅::Vector{BeliefPoint})
    if isempty(𝔅)
        return nothing
    end
    
    # Get dimensions from first belief
    num_states, height, width = size(𝔅[1].belief.event_distributions)
    K, F = num_states, height * width  # K categories per factor, F factors
    N = length(𝔅)
    
    # Stack all candidates into 3D array Q[K, F, N]
    Q = Array{T}(undef, K, F, N)
    
    for (j, bp) in enumerate(𝔅)
        belief = bp.belief
        # Extract and normalize the belief distributions
        for x in 1:width, y in 1:height
            f = (y - 1) * width + x  # Factor index
            for k in 1:num_states
                Q[k, f, j] = clamp01(belief.event_distributions[k, y, x])
            end
        end
        
        # Normalize each factor to sum to 1
        for f in 1:F
            col_sum = sum(Q[:, f, j])
            if col_sum > 0
                Q[:, f, j] ./= col_sum
            end
        end
    end
    
    # Precompute log(Q) and reshape for BLAS
    LQ = log.(Q)
    LQmat = reshape(LQ, K*F, N)  # (KF) × N
    
    # Pre-organize by clock vector for O(1) lookup
    clock_groups = Dict{Vector{Int}, Vector{Int}}()  # clock -> indices
    for (i, bp) in enumerate(𝔅)
        key = collect(bp.clock.phases)  # Convert to Vector{Int} for Dict key
        if !haskey(clock_groups, key)
            clock_groups[key] = Int[]
        end
        push!(clock_groups[key], i)
    end
    
    return (LQmat=LQmat, K=K, F=F, N=N, beliefs=𝔅, clock_groups=clock_groups)
end

"""
Fast nearest belief search using BLAS-based KL divergence optimization
"""
function find_nearest_belief_blas(target::BeliefPoint, prep, SLICE_BUCKET, DIGEST_LOOKUP)
    if prep === nothing
        return find_nearest_belief(target, SLICE_BUCKET, DIGEST_LOOKUP)
    end
    
    # O(1) lookup for same-clock candidates using pre-organized groups
    key = collect(target.clock.phases)
    candidate_indices = get(prep.clock_groups, key, Int[])
    
    if isempty(candidate_indices)
        return target
    end
    
    # Use BLAS optimization: find Q that maximizes sum(P * log(Q))
    # This is equivalent to minimizing KL(P||Q) for fixed P
    target_belief = target.belief
    num_states, height, width = size(target_belief.event_distributions)
    
    # Prepare target belief P as a vector
    P_vec = Vector{T}(undef, prep.K * prep.F)
    idx = 1
    
    for x in 1:width, y in 1:height
        for k in 1:num_states
            P_vec[idx] = clamp01(target_belief.event_distributions[k, y, x])
            idx += 1
        end
    end
    
    # Normalize P_vec to sum to 1
    P_sum = sum(P_vec)
    if P_sum > 0
        P_vec ./= P_sum
    end
    
    # Use pre-organized indices for fast subset access
    LQmat_subset = prep.LQmat[:, candidate_indices]
    
    # Compute scores = LQ_subset' * P_vec using BLAS
    scores = LQmat_subset' * P_vec  # length(candidate_indices) × 1 vector
    
    # Find the candidate with maximum score (minimum KL divergence)
    jmax = argmax(scores)
    
    return prep.beliefs[candidate_indices[jmax]]
end

"""
Clear the belief evolution cache
"""
function clear_belief_evolution_cache!()
    empty!(BELIEF_EVOLUTION_CACHE)
    CACHE_STATS[:hits] = 0
    CACHE_STATS[:misses] = 0
    CACHE_STATS[:size] = 0
    TIMING_STATS[:total_cache_time] = 0.0
    TIMING_STATS[:total_direct_time] = 0.0
    TIMING_STATS[:cache_calls] = 0
    TIMING_STATS[:direct_calls] = 0
    

    

end

"""
Get cache statistics
"""
function get_belief_cache_stats()
    return copy(CACHE_STATS)
end





"""
Get timing statistics
"""
function get_timing_stats()
    return copy(TIMING_STATS)
end

"""
Get detailed timing analysis
"""
function get_detailed_timing_analysis()
    timing_stats = get_timing_stats()
    belief_cache_stats = get_belief_cache_stats()
    
    if timing_stats[:direct_calls] == 0
        return "No timing data available yet"
    end
    
    # Calculate performance metrics
    total_calls = timing_stats[:cache_calls] + timing_stats[:direct_calls]
    cache_hit_rate = timing_stats[:cache_calls] / total_calls * 100
    
    avg_direct_time = timing_stats[:total_direct_time] / timing_stats[:direct_calls]
    total_time_saved = timing_stats[:cache_calls] * avg_direct_time
    
    # Use actual measured cache time
    total_time_with_cache = timing_stats[:total_direct_time] + timing_stats[:total_cache_time]
    total_time_without_cache = total_calls * avg_direct_time
    
    speedup = total_time_without_cache / total_time_with_cache
    
    analysis = """
📊 Detailed Timing Analysis:
• Total calls: $total_calls
• Cache hits: $(timing_stats[:cache_calls]) ($(round(cache_hit_rate, digits=1))%)
• Direct calls: $(timing_stats[:direct_calls])
• Average direct evolution time: $(round(avg_direct_time * 1000, digits=2)) ms
• Total direct evolution time: $(round(timing_stats[:total_direct_time] * 1000, digits=2)) ms
• Total cache access time: $(round(timing_stats[:total_cache_time] * 1000, digits=2)) ms
• Average cache access time: $(round((timing_stats[:total_cache_time] / max(timing_stats[:cache_calls], 1)) * 1000, digits=4)) ms
• Estimated time saved: $(round(total_time_saved * 1000, digits=2)) ms
• Overall speedup: $(round(speedup, digits=2))x
• Cache size: $(belief_cache_stats[:size]) entries
"""
    
    return analysis
end

"""
Analyze cache efficiency and provide debugging information
"""
function analyze_cache_efficiency()
    belief_cache_stats = get_belief_cache_stats()
    timing_stats = get_timing_stats()
    
    total_calls = timing_stats[:cache_calls] + timing_stats[:direct_calls]
    if total_calls == 0
        return "No cache data available yet"
    end
    
    cache_hit_rate = timing_stats[:cache_calls] / total_calls * 100
    cache_utilization = belief_cache_stats[:size] / 1000 * 100  # Assuming max size of 1000
    
    analysis = """
🔍 Cache Efficiency Analysis:
• Cache hit rate: $(round(cache_hit_rate, digits=1))%
• Cache utilization: $(round(cache_utilization, digits=1))%
• Total unique beliefs processed: $(total_calls)
• Cache entries stored: $(belief_cache_stats[:size])
• Cache efficiency: $(round(cache_hit_rate / cache_utilization, digits=2)) hits per entry

💡 Interpretation:
• High hit rate ($(round(cache_hit_rate, digits=1))%) suggests beliefs are being reused effectively
• Cache utilization ($(round(cache_utilization, digits=1))%) shows how much of cache capacity is used
• If hit rate is much higher than utilization, beliefs may be very similar
• Consider increasing cache size if utilization is high and hit rate could improve
"""
    
    return analysis
end

"""
Cached belief evolution using evolve_no_obs_fast
"""
function cached_belief_evolution(belief::Belief, env)
    # Create a more robust cache key that considers numerical tolerance
    # Use a quantized version of the belief for the cache key
    cache_key = create_belief_cache_key(belief, env)
    
    # Check cache first with timing
    cache_start = time()
    if haskey(BELIEF_EVOLUTION_CACHE, cache_key)
        CACHE_STATS[:hits] += 1
        TIMING_STATS[:cache_calls] += 1
        evolved_belief = deepcopy(BELIEF_EVOLUTION_CACHE[cache_key])
        cache_time = time() - cache_start
        TIMING_STATS[:total_cache_time] += cache_time
        return evolved_belief
    end
    
    # Cache miss - compute evolution with timing
    CACHE_STATS[:misses] += 1
    TIMING_STATS[:direct_calls] += 1
    
    start_time = time()
    evolved_belief = evolve_no_obs_fast(belief, env, calculate_uncertainty=false)
    direct_time = time() - start_time
    
    TIMING_STATS[:total_direct_time] += direct_time
    
    # Store in cache (with size limit to prevent memory issues)
    if CACHE_STATS[:size] < 1000  # Limit cache size
        BELIEF_EVOLUTION_CACHE[cache_key] = deepcopy(evolved_belief)
        CACHE_STATS[:size] += 1
    end
    
    return evolved_belief
end

"""
Create a robust cache key for beliefs that accounts for numerical tolerance
"""
function create_belief_cache_key(belief::Belief, env)
    # Quantize the belief distributions to handle numerical precision issues
    # This should match the tolerance used in beliefs_are_equivalent (1e-10)
    tolerance = 1e-10
    
    # Create a quantized version of the belief
    quantized_distributions = round.(belief.event_distributions ./ tolerance) .* tolerance
    
    # Create cache key from quantized belief and environment
    belief_hash = hash(quantized_distributions, UInt(0))
    env_hash = hash(env.width, hash(env.height, hash(env.dynamics)))
    cache_key = hash(belief_hash, env_hash)
    
    return cache_key
end

"""
Create a BeliefPoint with pre-computed digest
"""
function BeliefPoint(clock::ClockVector, belief::Belief)
    digest = hash(belief.event_distributions, UInt(0))
    return BeliefPoint(clock, digest, belief)
end

"""
best_script(env, belief::Belief, agent::Agent, C::Int, other_scripts, gs_state)::Vector{SensingAction}
  – Use PBVI to find the best action sequence
  – Build belief set, run value iteration, extract policy
  – Return the best sequence
"""
function best_script(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG, 
                    N_seed::Int=50, N_particles::Int=64, N_sweeps::Int=50, ε::Float64=0.1)
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
    
    println("🔄 Building belief set for PBVI...")
    # Build belief set
    𝔅 = build_belief_set(B_clean, agent_i, τ_i, agents_j, τ_js_vector, H, env, gs_state, N_seed)
    println("🔄 Running PBVI with $(length(𝔅)) belief points...")
    # Run PBVI
    VALUE, POLICY = pbvi(𝔅, N_particles, N_sweeps, ε, agent_i, env, gs_state)
    # Extract best sequence from policy
    
    best_sequence = extract_best_sequence(POLICY, VALUE, 𝔅, agent_i, env, gs_state, H)
    @infiltrate
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ PBVI sequence found in $(round(planning_time, digits=3)) seconds")
    
    # Report cache statistics
    cache_stats = get_cache_stats()
    belief_cache_stats = get_belief_cache_stats()
    timing_stats = get_timing_stats()
    
    println("📊 Belief management cache: $(cache_stats[:hits]) hits, $(cache_stats[:misses]) misses, $(round(cache_stats[:hit_rate] * 100, digits=1))% hit rate")
    println("📊 Belief evolution cache: $(belief_cache_stats[:hits]) hits, $(belief_cache_stats[:misses]) misses, size: $(belief_cache_stats[:size])")
    
    # Report timing statistics
    if timing_stats[:direct_calls] > 0
        println("⏱️  Belief evolution timing: $(timing_stats[:cache_calls]) cache hits, $(timing_stats[:direct_calls]) direct calls")
        println(get_detailed_timing_analysis())
        println(analyze_cache_efficiency())
    end
    
    return best_sequence, planning_time
end

"""
Sample system state at τ_i (fully-informed system belief)
"""
function sample_system_state_at_τi(B_clean::Belief, agent_i::Agent, τ_i::Int, agents_j::Vector{Agent}, 
                                  τ_js_vector::Dict{Int, Int}, env, gs_state)
    b = deepcopy(B_clean)
    
    # Determine t_clean (minimum of other agent sync times)
    other_agent_sync_times = [τ_js_vector[j.id] for j in agents_j]
    t_clean = any(sync_time == -1 for sync_time in other_agent_sync_times) ? 0 : minimum(other_agent_sync_times)
    
    # Roll forward from t_clean to τ_i-1
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
    
    return (τ_clock, b)
end

"""
Simulate one step forward
"""
function simulate_one_step(τ_clock::ClockVector, b_sys::Belief, action_i::SensingAction, 
                          agent_i::Agent, agents_j::Vector{Agent}, env, gs_state)

    # Track timing for different operations
    timing_breakdown = Dict{Symbol, Float64}()
    
    # Calculate sophisticated reward: R = Σ[j∈U_t] [w_H·(H_prior-H_post) + w_F·E[F_I_j]]
    info_start = time()
    r_step = 0.0
    
    if !isempty(action_i.target_cells)
        for cell in action_i.target_cells
            # Use the sophisticated reward function with global configuration
            cell_reward = calculate_sophisticated_reward(b_sys, cell)
            r_step += cell_reward
        end
    end
    timing_breakdown[:info_gain] = time() - info_start

    # Calculate global time from agent_i's phase in the clock vector
    time_calc_start = time()
    agent_i_index = find_agent_index(agent_i, env)
    # if agent_i_index == 2
    #     
    # end
    if agent_i_index === nothing
        t_global = gs_state.time_step  # Fallback
    else
        # Calculate global time based on agent_i's phase
        # The phase represents how many steps ahead we are from the current gs_state.time_step
        agent_i_phase = τ_clock.phases[agent_i_index]
        t_global = gs_state.time_step + agent_i_phase
    end
    timing_breakdown[:time_calculation] = time() - time_calc_start
    
    # Sample outcomes actually happening this step
    sampling_start = time()
    state_i = nothing
    if !isempty(action_i.target_cells)
        cell_i = action_i.target_cells[1]  # Assume single cell for now
        state_i = sample_event_state_from(b_sys, cell_i)
    end
    # Sample other agents' observations and apply them immediately
    for j in agents_j
        if j_senses_at_time(j, t_global, gs_state)
            cell_j = scheduled_cell(j, t_global, gs_state)
            if cell_j !== nothing
                state_j = sample_event_state_from(b_sys, cell_j)
                # Apply sampled observation immediately (Monte Carlo sampling)
                b_sys = collapse_belief_to(b_sys, cell_j, state_j)
            end
        end
    end
    # Collapse belief based on current agent's action
    if state_i !== nothing && !isempty(action_i.target_cells)
        b_sys = collapse_belief_to(b_sys, action_i.target_cells[1], state_i)
    end
    timing_breakdown[:sampling_and_collapse] = time() - sampling_start
    
    # Predict one step ahead (use cached belief evolution)
    evolve_start = time()
    b_sys = evolve_no_obs_fast(b_sys, env, calculate_uncertainty=false)
    evolve_time = time() - evolve_start
    timing_breakdown[:belief_evolution] = evolve_time
    
    # Clock vector update
    clock_start = time()
    τ_clock = advance_clock_vector(τ_clock, [agent_i; agents_j])
    timing_breakdown[:clock_update] = time() - clock_start
    
    # if agent_i_index == 2
    #     
    # end
    return (r_step, τ_clock, b_sys, evolve_time, timing_breakdown)
end

"""
Build belief set for PBVI
"""
function build_belief_set(B_clean::Belief, agent_i::Agent, τ_i::Int, agents_j::Vector{Agent}, 
                         τ_js_vector::Dict{Int, Int}, H::Int, env, gs_state, N_seed::Int)
    𝔅 = Set{BeliefPoint}()
    total_generated = 0
    
    for seed in 1:N_seed
        (τ, b_sys) = sample_system_state_at_τi(B_clean, agent_i, τ_i, agents_j, τ_js_vector, env, gs_state)
        
        for h in 0:(H-1)
            total_generated += 1
            push!(𝔅, BeliefPoint(τ, deepcopy(b_sys)))
            
            # Take a random action and simulate
            a_rand = random_pointing(agent_i, τ, env)
            (_, τ, b_sys, _, _) = simulate_one_step(τ, b_sys, a_rand, agent_i, agents_j, env, gs_state)
        end
    end
    
    final_count = length(𝔅)
    println("📊 Belief set: generated $(total_generated) points, kept $(final_count) unique points (removed $(total_generated - final_count) duplicates)")
    
    return collect(𝔅)
end

"""
PBVI algorithm
"""
function pbvi(𝔅::Vector{BeliefPoint}, N_particles::Int, N_sweeps::Int, ε::Float64, 
              agent_i::Agent, env, gs_state)
    VALUE = Dict{BeliefPoint, Float64}()
    POLICY = Dict{BeliefPoint, SensingAction}()
    # make a Dict from phase-tuple → vector of points in that slice
    SLICE_BUCKET  = Dict{Tuple{Vararg{Int}}, Vector{BeliefPoint}}()
    DIGEST_LOOKUP = Dict{Tuple{Vararg{Int}}, Dict{UInt64,BeliefPoint}}()

    # Timing statistics
    timing_stats = Dict{Symbol, Float64}(
        :total_simulate_one_step => 0.0,
        :total_nearest_neighbor => 0.0,
        :total_action_set_gen => 0.0,
        :total_belief_copy => 0.0
    )
    operation_counts = Dict{Symbol, Int}(
        :simulate_one_step_calls => 0,
        :nearest_neighbor_calls => 0,
        :action_set_gen_calls => 0,
        :belief_copy_calls => 0
    )

    for bp in 𝔅
        key = Tuple(bp.clock.phases)          # hashable
        push!(get!(SLICE_BUCKET, key, BeliefPoint[]), bp)
    end
    for (key, vec) in SLICE_BUCKET
        lkp = Dict{UInt64,BeliefPoint}()
        for bp in vec
            lkp[bp.digest] = bp
        end
        DIGEST_LOOKUP[key] = lkp              # digest → bp in that slice
    end

    
    # Initialize values
    for bp in 𝔅
        VALUE[bp] = 0.0
    end
    
    # Prepare candidates for fast KL divergence search
    prep = prepare_kl_candidates(𝔅)
    
    γ = env.discount
    
    for sweep in 1:N_sweeps
        sweep_start = time()
        Δ = 0.0
        shuffled_𝔅 = shuffle(𝔅)
        sim_times = Float64[]  # Track simulation times for this sweep

        
        # Sequential belief point processing (thread-safe)
        for bp in shuffled_𝔅
            best_Q = -Inf
            best_act = nothing
            
            # Get all feasible actions with timing
            action_start = time()
            action_set = all_pointings(agent_i, bp.clock, env)
            action_time = time() - action_start
            timing_stats[:total_action_set_gen] += action_time
            operation_counts[:action_set_gen_calls] += 1
            
            for a in action_set
                sum_Q = 0.0
                
                # Sequential particle simulation
                for particle in 1:N_particles
                    # Time belief copying
                    copy_start = time()
                    clock_copy = copy(bp.clock)
                    belief_copy = deepcopy(bp.belief)
                    copy_time = time() - copy_start
                    timing_stats[:total_belief_copy] += copy_time
                    operation_counts[:belief_copy_calls] += 1
                    
                    # Time simulate_one_step
                    sim_start = time()
                    (r, τ′, b′, evolve_time, timing_breakdown) = simulate_one_step(clock_copy, belief_copy, a, 
                                                    agent_i, get_other_agents(agent_i, env), env, gs_state)                    
                    sim_time = time() - sim_start
                    timing_stats[:total_simulate_one_step] += sim_time
                    operation_counts[:simulate_one_step_calls] += 1
                    
                    # Time the nearest neighbor search
                    nn_start = time()
                    nearest_bp = find_nearest_belief_blas(BeliefPoint(τ′, b′), prep, SLICE_BUCKET, DIGEST_LOOKUP)
                    nn_time = time() - nn_start
                    timing_stats[:total_nearest_neighbor] += nn_time
                    operation_counts[:nearest_neighbor_calls] += 1
                    
                    v_next = VALUE[nearest_bp]
                    
                    sum_Q += r + γ * v_next
                end
                
                Q_hat = sum_Q / N_particles
                
                if Q_hat > best_Q
                    best_Q = Q_hat
                    best_act = a
                end
            end
            
            Δ = max(Δ, abs(best_Q - VALUE[bp]))
            VALUE[bp] = best_Q
            POLICY[bp] = best_act
        end
        
        sweep_time = time() - sweep_start
        println("  Sweep $(sweep): max change = $(round(Δ, digits=4)) in $(round(sweep_time, digits=2))s")

        
        if Δ < ε
            break
        end
    end
    
    # Print timing summary
    println("\n📊 PBVI Timing Breakdown:")
    println("• Simulate one step: $(round(timing_stats[:total_simulate_one_step] * 1000, digits=1)) ms ($(operation_counts[:simulate_one_step_calls]) calls)")
    println("• Nearest neighbor search: $(round(timing_stats[:total_nearest_neighbor] * 1000, digits=1)) ms ($(operation_counts[:nearest_neighbor_calls]) calls)")
    println("• Action set generation: $(round(timing_stats[:total_action_set_gen] * 1000, digits=1)) ms ($(operation_counts[:action_set_gen_calls]) calls)")
    println("• Belief copying: $(round(timing_stats[:total_belief_copy] * 1000, digits=1)) ms ($(operation_counts[:belief_copy_calls]) calls)")
    
    return VALUE, POLICY
end
"""
Extract best sequence from policy using belief points from PBVI
"""
function extract_best_sequence(POLICY::Dict{BeliefPoint, SensingAction}, VALUE::Dict{BeliefPoint, Float64}, 
                             𝔅::Vector{BeliefPoint}, agent_i::Agent, env, gs_state, H::Int)
    sequence = SensingAction[]
    
    # Find the belief point that best represents the current state
    # Look for belief points with clock phases matching current agent phases
    # Calculate phases relative to agent_i (which is at phase 0)
    current_phases = Int[]
    
    # Get all agents in the same order as the clock vector
    all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
    
    # Find agent_i's index in the clock vector
    agent_i_index = find_agent_index(agent_i, env)
    
    # Calculate phases for all agents relative to agent_i
    for (i, agent) in enumerate(all_agents)
        if i == agent_i_index
            # Agent_i is at phase 0
            push!(current_phases, 0)
        else
            # Other agents: relative phase offset
            relative_offset = mod((agent.phase_offset - agent_i.phase_offset), agent.trajectory.period)
            push!(current_phases, relative_offset)
        end
    end
    @infiltrate
    current_clock = ClockVector(current_phases)
    if agent_i.id == 2
        
    end
    candidate_bps = BeliefPoint[]
    
    for bp in 𝔅
        if bp.clock.phases == current_clock.phases
            push!(candidate_bps, bp)
        end
    end
    if isempty(candidate_bps)
        # If no exact match, find the belief point with closest phases
        if !isempty(𝔅)
            # Sort by phase difference and take the closest
            sort!(𝔅, by=bp -> sum(abs.(bp.clock.phases .- current_clock.phases)))
            current_bp = 𝔅[1]
        else
            # Fallback: return empty sequence
            return SensingAction[]
        end
    else
        # Take the belief point with highest value among candidates
        best_value = -Inf
        current_bp = candidate_bps[1]
        for bp in candidate_bps
            if haskey(VALUE, bp) && VALUE[bp] > best_value
                best_value = VALUE[bp]
                current_bp = bp
            end
        end
    end
    
    # Extract sequence by following the policy
    for h in 1:H
        if haskey(POLICY, current_bp)
            action = POLICY[current_bp]
            push!(sequence, action)
            
            # Find the next belief point by simulating forward
            # We need to find a belief point that represents the next state
            next_phases = Int[]
            for (i, agent) in enumerate([agent_i; get_other_agents(agent_i, env)])
                next_phase = mod((current_bp.clock.phases[i] + 1), agent.trajectory.period)
                push!(next_phases, next_phase)
            end
            next_clock = ClockVector(next_phases)
            next_candidates = BeliefPoint[]
            
            for bp in 𝔅
                if bp.clock.phases == next_clock.phases
                    push!(next_candidates, bp)
                end
            end
            
            if !isempty(next_candidates)
                # Find the closest belief point to the simulated next state
                # For simplicity, just take the first one
                current_bp = next_candidates[1]
            else
                # If no next belief point found, break
                break
            end
        else
            # Fallback to wait action
            push!(sequence, SensingAction(agent_i.id, Tuple{Int, Int}[], false))
            break
        end
    end
    
    return sequence
end

# function extract_best_sequence(POLICY::Dict{BeliefPoint, SensingAction},
#     VALUE::Dict{BeliefPoint, Float64},
#     𝔅::Vector{BeliefPoint},
#     agent_i::Agent, env, gs_state, H::Int)

#     sequence = SensingAction[]

#     # Compute current clock phases relative to agent_i
#     all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
#     agent_i_index = find_agent_index(agent_i, env)
#     current_phases = [i == agent_i_index ? 0 :
#                       mod((agent.phase_offset - agent_i.phase_offset), agent.trajectory.period)
#                       for (i, agent) in enumerate(all_agents)]
#     current_clock = ClockVector(current_phases)

#     # Find matching beliefs for current clock
#     candidate_bps = [bp for bp in 𝔅 if bp.clock.phases == current_clock.phases]
#     if isempty(candidate_bps)
#         sort!(𝔅, by=bp -> sum(abs.(bp.clock.phases .- current_clock.phases)))
#         candidate_bps = [𝔅[1]]
#     end

#     current_candidates = candidate_bps

#     # Iterate horizon steps
#     for h in 1:H
#         action_scores = Dict{SensingAction, Vector{Float64}}()

#         # For each candidate belief at this clock, get policy action and value
#         for bp in current_candidates
#             if haskey(POLICY, bp)
#                 act = POLICY[bp]
#                 push!(get!(action_scores, act, Float64[]), VALUE[bp])
#             end
#         end

#         if isempty(action_scores)
#             push!(sequence, SensingAction(agent_i.id, Tuple{Int,Int}[], false))
#             break
#         end

#         # Compute average value per action
#         avg_scores = Dict(act => mean(vals) for (act, vals) in action_scores)
#         best_action = argmax(avg_scores)  # action with highest avg value
#         push!(sequence, best_action)

#         # Advance to next clock & get next candidate beliefs
#         next_phases = [mod((current_clock.phases[i] + 1), all_agents[i].trajectory.period)
#                        for i in 1:length(all_agents)]
#         next_clock = ClockVector(next_phases)
#         current_candidates = [bp for bp in 𝔅 if bp.clock.phases == next_clock.phases]

#         if isempty(current_candidates)
#             break
#         end
#         current_clock = next_clock
#     end

#     return sequence
# end




# Helper functions

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
Check if agent j senses at time t
"""
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

"""
Get scheduled cell for agent j at time t
"""
function scheduled_cell(j::Agent, t::Int, gs_state)
    if !haskey(gs_state.agent_plans, j.id) || gs_state.agent_plans[j.id] === nothing
        return nothing
    end
    
    plan = gs_state.agent_plans[j.id]
    plan_timestep = (t - gs_state.agent_last_sync[j.id]) + 1
    
    if 1 <= plan_timestep <= length(plan)
        action = plan[plan_timestep]
        if !isempty(action.target_cells)
            return action.target_cells[1]  # Return first cell
        end
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
        new_phase = (τ_clock.phases[i] + 1) % agent.trajectory.period
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
    
    # Get position - should work for all trajectory types with just phase
    pos = get_position_at_time(agent.trajectory, phase)
    
    # Get available cells in field of view
    available_cells = get_field_of_regard_at_position(agent, pos, env)
    # if agent.id == 2
    #     
    # end
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
    
    # Get position - should work for all trajectory types with just phase
    pos = get_position_at_time(agent.trajectory, phase)
    
    # Get available cells in field of view
    available_cells = get_field_of_regard_at_position(agent, pos, env)
    # if agent.id == 2
    #     
    # end
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

@inline function find_nearest_belief(target::BeliefPoint, SLICE_BUCKET, DIGEST_LOOKUP)
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
    # 3. fall-back: cheapest distance inside the bucket
    nearest   = bucket[1]
    best_dist = belief_distance(target.belief, nearest.belief)
    for bp in bucket
        d = belief_distance(target.belief, bp.belief)
        if d < best_dist
            best_dist = d
            nearest   = bp
        end
    end
    return nearest
end


"""
Calculate distance between two beliefs using KL divergence with caching
"""
function belief_distance(b1::Belief, b2::Belief)
    # Use digest for fast comparison first
    digest1 = hash(b1.event_distributions, UInt(0))
    digest2 = hash(b2.event_distributions, UInt(0))
    
    if digest1 == digest2
        return 0.0  # Exact match
    end
    
    # Check distance cache (order-independent)
    cache_key = digest1 < digest2 ? (digest1, digest2) : (digest2, digest1)
    if haskey(DISTANCE_CACHE, cache_key)
        DISTANCE_CACHE_STATS[:hits] += 1
        return DISTANCE_CACHE[cache_key]
    end
    
    # Cache miss - calculate KL divergence
    DISTANCE_CACHE_STATS[:misses] += 1
    distance = belief_distance_kl(b1, b2)
    
    # Store in cache (with size limit)
    if DISTANCE_CACHE_STATS[:size] < 10000  # Larger limit for distances
        DISTANCE_CACHE[cache_key] = distance
        DISTANCE_CACHE_STATS[:size] += 1
    end
    
    return distance
end

"""
Calculate KL divergence between two beliefs using optimized summary method
"""
function belief_distance_kl(b1::Belief, b2::Belief)
    # Get belief summaries efficiently
    summary1 = belief_summary_optimized(b1)
    summary2 = belief_summary_optimized(b2)
    
    # Add small epsilon to avoid log(0) and ensure positivity
    ε = 1e-10
    summary1 = summary1 .+ ε
    summary2 = summary2 .+ ε
    
    # Normalize to proper probabilities
    summary1 = summary1 ./ sum(summary1)
    summary2 = summary2 ./ sum(summary2)
    
    # Use optimized symmetric KL divergence from Types module
    return symmetric_kl_divergence_fast(summary1, summary2)
end

"""
Create optimized belief summary without allocations
"""
function belief_summary_optimized(belief::Belief)
    num_states, height, width = size(belief.event_distributions)
    
    # Pre-allocate the summary array
    summary = Vector{Float64}(undef, height * width)
    idx = 1
    
    @inbounds for x in 1:width, y in 1:height
        # Sum over event states (assuming state 2 is EVENT_PRESENT)
        event_prob = 0.0
        for state in 1:num_states
            if state == 2  # EVENT_PRESENT
                event_prob += belief.event_distributions[state, y, x]
            end
        end
        summary[idx] = event_prob
        idx += 1
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
    elseif agent.sensor.pattern == :circular
        # Circular sensor: agent's position and all 8 adjacent cells (9-cell pattern)
        # This is for the 9x9 grid circular trajectory agents
        for dx in -1:1, dy in -1:1
            nx, ny = x + dx, y + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                push!(fov_cells, (nx, ny))
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

# Reward function configuration - these will be set from main.jl
# Default values if not set externally
const DEFAULT_ENTROPY_WEIGHT = get(ENV, "ENTROPY_WEIGHT", 1.0)    # w_H: Weight for entropy reduction (coordination)
const DEFAULT_VALUE_WEIGHT = get(ENV, "VALUE_WEIGHT", 0.5)        # w_F: Weight for state value (detection priority)
const DEFAULT_INFORMATION_STATES = get(ENV, "INFORMATION_STATES", [1, 2])  # I_1: No event, I_2: Event
const DEFAULT_STATE_VALUES = get(ENV, "STATE_VALUES", [0.1, 1.0])        # F_1: No event value, F_2: Event value

"""
Calculate sophisticated reward for sensing actions

R = Σ[j∈U_t] [w_H·(H_prior(b_j) - H_post(b_j)) + w_F·E[F_I_j]]

Where:
- w_H: Weight for entropy reduction (inter-agent coordination)
- w_F: Weight for state value (detection priority)
- E[F_I_j]: Expected value of information state I_k under current belief

Uses global configuration constants from main.jl
"""
function calculate_sophisticated_reward(belief::Belief, cell::Tuple{Int, Int})
    
    # 1. Entropy-based reward: w_H * (H_prior - H_post)
    H_before = calculate_cell_entropy(belief, cell)
    H_after = 0.0  # Simplified: assume perfect observation
    entropy_reward = DEFAULT_ENTROPY_WEIGHT * (H_before - H_after)
    
    # 2. State value reward: w_F * E[F_I_j]
    # Calculate expected value under current belief
    event_prob = get_event_probability(belief, cell)
    no_event_prob = 1.0 - event_prob
    
    # E[F_I_j] = Σ_k p(I_k) * F_k
    expected_value = no_event_prob * DEFAULT_STATE_VALUES[1] + event_prob * DEFAULT_STATE_VALUES[2]
    value_reward = DEFAULT_VALUE_WEIGHT * expected_value
    
    # Total reward for this cell
    return entropy_reward + value_reward
end

"""
Configure reward function weights

Parameters:
- w_H: Weight for entropy reduction (coordination). Default: 1.0
- w_F: Weight for state value (detection priority). Default: 0.5

Examples:
- w_H=1.0, w_F=0.0: Pure entropy-based reward (original behavior)
- w_H=0.5, w_F=1.0: Balanced coordination and detection
- w_H=0.0, w_F=1.0: Pure value-based reward (detection-focused)
"""
function configure_reward_weights(; w_H::Float64=1.0, w_F::Float64=0.5)
    global DEFAULT_ENTROPY_WEIGHT = w_H
    global DEFAULT_VALUE_WEIGHT = w_F
    
    println("🎯 Reward weights configured:")
    println("   • Entropy weight (w_H): $w_H (coordination)")
    println("   • Value weight (w_F): $w_F (detection priority)")
    
    if w_H == 1.0 && w_F == 0.0
        println("   → Pure entropy-based reward (original behavior)")
    elseif w_H == 0.0 && w_F == 1.0
        println("   → Pure value-based reward (detection-focused)")
    else
        println("   → Balanced reward function")
    end
end

"""
Set reward configuration from main.jl constants
This function should be called from the main script to sync configuration
"""
function set_reward_config_from_main(entropy_weight::Float64, value_weight::Float64, 
                                   information_states::Vector{Int}, state_values::Vector{Float64})
    global DEFAULT_ENTROPY_WEIGHT = entropy_weight
    global DEFAULT_VALUE_WEIGHT = value_weight
    global DEFAULT_INFORMATION_STATES = information_states
    global DEFAULT_STATE_VALUES = state_values
    
    println("🎯 Reward configuration synced from main.jl:")
    println("   • Entropy weight (w_H): $entropy_weight")
    println("   • Value weight (w_F): $value_weight")
    println("   • Information states: $information_states")
    println("   • State values: $state_values")
end

end # module 

# module MacroPlannerPBVI

# using POMDPs
# using POMDPTools
# using Random
# using LinearAlgebra
# using Infiltrator
# using Statistics
# using Base.Threads
# using ..Types
# import ..Agents.BeliefManagement: sample_from_belief
# import ..Types: check_battery_feasible, simulate_battery_evolution
# # Import types from the parent module (Planners)
# import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
# import ..EventState2, ..NO_EVENT_2, ..EVENT_PRESENT_2
# import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..ComplexTrajectory, ..RangeLimitedSensor, ..EventMap
# # Import trajectory functions
# import ..Agents.TrajectoryPlanner.get_position_at_time
# # Import DBN functions for transition modeling
# import ..Environment.EventDynamicsModule.DBNTransitionModel2, ..Environment.EventDynamicsModule.predict_next_belief_dbn
# # Import belief management functions
# import ..Agents.BeliefManagement
# import ..Agents.BeliefManagement.predict_belief_evolution_dbn, ..Agents.BeliefManagement.Belief,
#        ..Agents.BeliefManagement.calculate_uncertainty_from_distribution, ..Agents.BeliefManagement.predict_belief_rsp,
#        ..Agents.BeliefManagement.evolve_no_obs,..Agents.BeliefManagement.evolve_no_obs_fast, ..Agents.BeliefManagement.get_neighbor_beliefs,
#        ..Agents.BeliefManagement.enumerate_joint_states, ..Agents.BeliefManagement.product,
#        ..Agents.BeliefManagement.normalize_belief_distributions, ..Agents.BeliefManagement.collapse_belief_to,
#        ..Agents.BeliefManagement.enumerate_all_possible_outcomes, ..Agents.BeliefManagement.merge_equivalent_beliefs,
#        ..Agents.BeliefManagement.calculate_cell_entropy, ..Agents.BeliefManagement.get_event_probability,
#        ..Agents.BeliefManagement.clear_belief_evolution_cache!, ..Agents.BeliefManagement.get_cache_stats,
#        ..Agents.BeliefManagement.beliefs_are_equivalent

# export best_script, calculate_macro_script_reward

# # PBVI-specific types
# struct ClockVector
#     phases::Vector{Int}  # Phases of all agents in the trajectory
# end

# # Add copy method for ClockVector
# Base.copy(cv::ClockVector) = ClockVector(copy(cv.phases))
# Base.deepcopy(cv::ClockVector) = ClockVector(deepcopy(cv.phases))

# struct BeliefPoint
#     clock::ClockVector
#     digest::UInt64          # immutable, pre-computed hash
#     belief::Belief          # still carried for look-ups
# end

# # Add copy method for BeliefPoint
# Base.copy(bp::BeliefPoint) = BeliefPoint(copy(bp.clock), bp.digest, deepcopy(bp.belief))
# Base.deepcopy(bp::BeliefPoint) = BeliefPoint(deepcopy(bp.clock), bp.digest, deepcopy(bp.belief))

# # Add hash and equality methods for dictionary keys
# Base.hash(cv::ClockVector, h::UInt) = hash(cv.phases, h)
# Base.hash(bp::BeliefPoint, h::UInt) = hash(bp.clock, hash(bp.digest, h))

# Base.isequal(cv1::ClockVector, cv2::ClockVector) = cv1.phases == cv2.phases
# Base.isequal(bp1::BeliefPoint, bp2::BeliefPoint) = isequal(bp1.clock, bp2.clock) && beliefs_are_equivalent(bp1.belief, bp2.belief)
# Base.:(==)(cv1::ClockVector, cv2::ClockVector) = isequal(cv1, cv2)
# Base.:(==)(bp1::BeliefPoint, bp2::BeliefPoint) = isequal(bp1, bp2)

# """
# Create a BeliefPoint with pre-computed digest
# """
# function BeliefPoint(clock::ClockVector, belief::Belief)
#     digest = hash(belief.event_distributions, UInt(0))
#     return BeliefPoint(clock, digest, belief)
# end

# """
# best_script(env, belief::Belief, agent::Agent, C::Int, other_scripts, gs_state)::Vector{SensingAction}
#   – Use PBVI to find the best action sequence
#   – Build belief set, run value iteration, extract policy
#   – Return the best sequence
# """
# function best_script(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG, 
#                     N_seed::Int=10, N_particles::Int=64, N_sweeps::Int=50, ε::Float64=0.1)
#     # Start timing
#     start_time = time()
    
#     # Clear belief evolution cache at the start of each planning session
#     clear_belief_evolution_cache!()
    
#     # Get parameters from the pseudocode
#     B_clean = deepcopy(belief)
#     agent_i = agent
#     τ_i = gs_state.time_step
#     agents_j = [env.agents[j] for j in keys(env.agents) if j != agent.id]
#     τ_js_vector = gs_state.agent_last_sync
#     H = C  # Horizon length
    
#     println("🔄 Building belief set for PBVI...")
#     # Build belief set
#     𝔅 = build_belief_set(B_clean, agent_i, τ_i, agents_j, τ_js_vector, H, env, gs_state, N_seed)
#     
#     println("🔄 Running PBVI with $(length(𝔅)) belief points...")
#     # Run PBVI
#     VALUE, POLICY = pbvi(𝔅, N_particles, N_sweeps, ε, agent_i, env, gs_state)
#     
#     # Extract best sequence from policy
#     best_sequence = extract_best_sequence(POLICY, VALUE, 𝔅, agent_i, env, gs_state, H)
#     
#     # End timing
#     end_time = time()
#     planning_time = end_time - start_time
    
#     println("✅ PBVI sequence found in $(round(planning_time, digits=3)) seconds")
    
#     # Report cache statistics
#     cache_stats = get_cache_stats()
#     println("📊 Cache statistics: $(cache_stats[:hits]) hits, $(cache_stats[:misses]) misses, $(round(cache_stats[:hit_rate] * 100, digits=1))% hit rate")
    
#     return best_sequence, planning_time
# end

# """
# Sample system state at τ_i (fully-informed system belief)
# """
# function sample_system_state_at_τi(B_clean::Belief, agent_i::Agent, τ_i::Int, agents_j::Vector{Agent}, 
#                                   τ_js_vector::Dict{Int, Int}, env, gs_state)
#     b = deepcopy(B_clean)
    
#     # Determine t_clean (minimum of other agent sync times)
#     other_agent_sync_times = [τ_js_vector[j.id] for j in agents_j]
#     t_clean = any(sync_time == -1 for sync_time in other_agent_sync_times) ? 0 : minimum(other_agent_sync_times)
    
#     # Roll forward from t_clean to τ_i-1
#     for t in t_clean:(τ_i-1)
#         for j in agents_j
#             if t > τ_js_vector[j.id] && j_senses_at_time(j, t, gs_state)
#                 cell = scheduled_cell(j, t, gs_state)
#                 if cell !== nothing
#                     state = sample_event_state_from(b, cell)
#                     b = collapse_belief_to(b, cell, state)
#                 end
#             end
#         end
#         b = evolve_no_obs_fast(b, env, calculate_uncertainty=false)
#     end
    
#     # Create clock vector at τ_i
#     # Calculate phases relative to agent_i (which is at phase 0)
#     agent_phases = Int[]
    
#     # Get all agents in the same order as the clock vector
#     all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
    
#     # Find agent_i's index in the clock vector
#     agent_i_index = find_agent_index(agent_i, env)
    
#     # Calculate phases for all agents relative to agent_i
#     for (i, agent) in enumerate(all_agents)
#         if i == agent_i_index
#             # Agent_i is at phase 0
#             push!(agent_phases, 0)
#         else
#             # Other agents: relative phase offset
#             relative_offset = mod((agent.phase_offset - agent_i.phase_offset), agent.trajectory.period)
#             push!(agent_phases, relative_offset)
#         end
#     end
#     τ_clock = ClockVector(agent_phases)
    
#     return (τ_clock, b)
# end

# """
# Simulate one step forward
# """
# function simulate_one_step(τ_clock::ClockVector, b_sys::Belief, action_i::SensingAction, 
#                           agent_i::Agent, agents_j::Vector{Agent}, env, gs_state)

#     # Calculate information gain
#     r_step = 0.0
#     if !isempty(action_i.target_cells)
#         for cell in action_i.target_cells
#             H_before = calculate_cell_entropy(b_sys, cell)
#             # Simplified: assume perfect observation
#             H_after = 0.0
#             info_gain = H_before - H_after
#             #event_prob = get_event_probability(b_sys, cell)
#             r_step += info_gain
#         end
#     end

#     # Calculate global time from agent_i's phase in the clock vector
#     agent_i_index = find_agent_index(agent_i, env)
#     # if agent_i_index == 2
#     #     
#     # end
#     if agent_i_index === nothing
#         t_global = gs_state.time_step  # Fallback
#     else
#         # Calculate global time based on agent_i's phase
#         # The phase represents how many steps ahead we are from the current gs_state.time_step
#         agent_i_phase = τ_clock.phases[agent_i_index]
#         t_global = gs_state.time_step + agent_i_phase
#     end
#     # Sample outcomes actually happening this step
#     state_i = nothing
#     if !isempty(action_i.target_cells)
#         cell_i = action_i.target_cells[1]  # Assume single cell for now
#         state_i = sample_event_state_from(b_sys, cell_i)
#     end
#     # Sample other agents' observations and apply them immediately
#     for j in agents_j
#         if j_senses_at_time(j, t_global, gs_state)
#             cell_j = scheduled_cell(j, t_global, gs_state)
#             if cell_j !== nothing
#                 state_j = sample_event_state_from(b_sys, cell_j)
#                 # Apply sampled observation immediately (Monte Carlo sampling)
#                 b_sys = collapse_belief_to(b_sys, cell_j, state_j)
#             end
#         end
#     end
#     # Collapse belief based on current agent's action
#     if state_i !== nothing && !isempty(action_i.target_cells)
#         b_sys = collapse_belief_to(b_sys, action_i.target_cells[1], state_i)
#     end
#     # Predict one step ahead (use fast vectorized version without cache)
#     evolve_start = time()
#     b_sys = evolve_no_obs_fast(b_sys, env, calculate_uncertainty=false)
#     evolve_time = time() - evolve_start
#     τ_clock = advance_clock_vector(τ_clock, [agent_i; agents_j])
#     # if agent_i_index == 2
#     #     
#     # end
#     return (r_step, τ_clock, b_sys, evolve_time)
# end

# """
# Build belief set for PBVI
# """
# function build_belief_set(B_clean::Belief, agent_i::Agent, τ_i::Int, agents_j::Vector{Agent}, 
#                          τ_js_vector::Dict{Int, Int}, H::Int, env, gs_state, N_seed::Int)
#     𝔅 = Set{BeliefPoint}()
#     total_generated = 0
    
#     for seed in 1:N_seed
#         (τ, b_sys) = sample_system_state_at_τi(B_clean, agent_i, τ_i, agents_j, τ_js_vector, env, gs_state)
        
#         for h in 0:(H-1)
#             total_generated += 1
#             push!(𝔅, BeliefPoint(τ, deepcopy(b_sys)))
            
#             # Take a random action and simulate
#             a_rand = random_pointing(agent_i, τ, env)
#             (_, τ, b_sys) = simulate_one_step(τ, b_sys, a_rand, agent_i, agents_j, env, gs_state)
#         end
#     end
    
#     final_count = length(𝔅)
#     println("📊 Belief set: generated $(total_generated) points, kept $(final_count) unique points (removed $(total_generated - final_count) duplicates)")
    
#     return collect(𝔅)
# end

# """
# PBVI algorithm
# """
# function pbvi(𝔅::Vector{BeliefPoint}, N_particles::Int, N_sweeps::Int, ε::Float64, 
#               agent_i::Agent, env, gs_state)
#     VALUE = Dict{BeliefPoint, Float64}()
#     POLICY = Dict{BeliefPoint, SensingAction}()
#     # make a Dict from phase-tuple → vector of points in that slice
#     SLICE_BUCKET  = Dict{Tuple{Vararg{Int}}, Vector{BeliefPoint}}()
#     DIGEST_LOOKUP = Dict{Tuple{Vararg{Int}}, Dict{UInt64,BeliefPoint}}()

#     for bp in 𝔅
#         key = Tuple(bp.clock.phases)          # hashable
#         push!(get!(SLICE_BUCKET, key, BeliefPoint[]), bp)
#     end
#     for (key, vec) in SLICE_BUCKET
#         lkp = Dict{UInt64,BeliefPoint}()
#         for bp in vec
#             lkp[bp.digest] = bp
#         end
#         DIGEST_LOOKUP[key] = lkp              # digest → bp in that slice
#     end

    
#     # Initialize values
#     for bp in 𝔅
#         VALUE[bp] = 0.0
#     end
    
#     γ = env.discount
    
#     for sweep in 1:N_sweeps
#         Δ = 0.0
#         shuffled_𝔅 = shuffle(𝔅)
#         sim_times = Float64[]  # Track simulation times for this sweep

        
#         # Sequential belief point processing (thread-safe)
#         for bp in shuffled_𝔅
#             best_Q = -Inf
#             best_act = nothing
            
#             # Get all feasible actions
#             action_set = all_pointings(agent_i, bp.clock, env)
            
#             for a in action_set
#                 sum_Q = 0.0
                
#                 # Sequential particle simulation
#                 for particle in 1:N_particles
#                     (r, τ′, b′, evolve_time) = simulate_one_step(copy(bp.clock), copy(bp.belief), a, 
#                                                     agent_i, get_other_agents(agent_i, env), env, gs_state)                    
#                     # Find nearest belief point
#                     nearest_bp = find_nearest_belief(BeliefPoint(τ′, b′), SLICE_BUCKET, DIGEST_LOOKUP)
#                     v_next = VALUE[nearest_bp]
                    
#                     sum_Q += r + γ * v_next
#                 end
                
#                 Q_hat = sum_Q / N_particles
                
#                 if Q_hat > best_Q
#                     best_Q = Q_hat
#                     best_act = a
#                 end
#             end
            
#             Δ = max(Δ, abs(best_Q - VALUE[bp]))
#             VALUE[bp] = best_Q
#             POLICY[bp] = best_act
#         end
        
#         println("  Sweep $(sweep): max change = $(round(Δ, digits=4))")

        
#         if Δ < ε
#             break
#         end
#     end
    
#     return VALUE, POLICY
# end

# """
# Extract best sequence from policy using belief points from PBVI
# """
# function extract_best_sequence(POLICY::Dict{BeliefPoint, SensingAction}, VALUE::Dict{BeliefPoint, Float64}, 
#                              𝔅::Vector{BeliefPoint}, agent_i::Agent, env, gs_state, H::Int)
#     sequence = SensingAction[]
    
#     # Find the belief point that best represents the current state
#     # Look for belief points with clock phases matching current agent phases
#     # Calculate phases relative to agent_i (which is at phase 0)
#     current_phases = Int[]
    
#     # Get all agents in the same order as the clock vector
#     all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
    
#     # Find agent_i's index in the clock vector
#     agent_i_index = find_agent_index(agent_i, env)
    
#     # Calculate phases for all agents relative to agent_i
#     for (i, agent) in enumerate(all_agents)
#         if i == agent_i_index
#             # Agent_i is at phase 0
#             push!(current_phases, 0)
#         else
#             # Other agents: relative phase offset
#             relative_offset = mod((agent.phase_offset - agent_i.phase_offset), agent.trajectory.period)
#             push!(current_phases, relative_offset)
#         end
#     end
#     current_clock = ClockVector(current_phases)
#     if agent_i.id == 2
#         
#     end
#     candidate_bps = BeliefPoint[]
    
#     for bp in 𝔅
#         if bp.clock.phases == current_clock.phases
#             push!(candidate_bps, bp)
#         end
#     end
#     if isempty(candidate_bps)
#         # If no exact match, find the belief point with closest phases
#         if !isempty(𝔅)
#             # Sort by phase difference and take the closest
#             sort!(𝔅, by=bp -> sum(abs.(bp.clock.phases .- current_clock.phases)))
#             current_bp = 𝔅[1]
#         else
#             # Fallback: return empty sequence
#             return SensingAction[]
#         end
#     else
#         # Take the belief point with highest value among candidates
#         best_value = -Inf
#         current_bp = candidate_bps[1]
#         for bp in candidate_bps
#             if haskey(VALUE, bp) && VALUE[bp] > best_value
#                 best_value = VALUE[bp]
#                 current_bp = bp
#             end
#         end
#     end
    
#     # Extract sequence by following the policy
#     for h in 1:H
#         if haskey(POLICY, current_bp)
#             action = POLICY[current_bp]
#             push!(sequence, action)
            
#             # Find the next belief point by simulating forward
#             # We need to find a belief point that represents the next state
#             next_phases = Int[]
#             for (i, agent) in enumerate([agent_i; get_other_agents(agent_i, env)])
#                 next_phase = (current_bp.clock.phases[i] + 1) % agent.trajectory.period
#                 push!(next_phases, next_phase)
#             end
#             next_clock = ClockVector(next_phases)
#             next_candidates = BeliefPoint[]
            
#             for bp in 𝔅
#                 if bp.clock.phases == next_clock.phases
#                     push!(next_candidates, bp)
#                 end
#             end
            
#             if !isempty(next_candidates)
#                 # Find the closest belief point to the simulated next state
#                 # For simplicity, just take the first one
#                 current_bp = next_candidates[1]
#             else
#                 # If no next belief point found, break
#                 break
#             end
#         else
#             # Fallback to wait action
#             push!(sequence, SensingAction(agent_i.id, Tuple{Int, Int}[], false))
#             break
#         end
#     end
    
#     return sequence
# end

# # Helper functions

# """
# Sample event state from belief for a cell
# """
# function sample_event_state_from(belief::Belief, cell::Tuple{Int, Int})
#     # Get probability of event in this cell
#     p_event = get_event_probability(belief, cell)
    
#     # Sample based on probability
#     if rand() < p_event
#         return EVENT_PRESENT  # Use EventState, not EventState2
#     else
#         return NO_EVENT  # Use EventState, not EventState2
#     end
# end

# """
# Check if agent j senses at time t
# """
# function j_senses_at_time(j::Agent, t::Int, gs_state)
#     if !haskey(gs_state.agent_plans, j.id) || gs_state.agent_plans[j.id] === nothing
#         return false
#     end
    
#     plan = gs_state.agent_plans[j.id]
#     plan_timestep = (t - gs_state.agent_last_sync[j.id]) + 1
    
#     if 1 <= plan_timestep <= length(plan)
#         action = plan[plan_timestep]
#         return !isempty(action.target_cells)
#     end
    
#     return false
# end

# """
# Get scheduled cell for agent j at time t
# """
# function scheduled_cell(j::Agent, t::Int, gs_state)
#     if !haskey(gs_state.agent_plans, j.id) || gs_state.agent_plans[j.id] === nothing
#         return nothing
#     end
    
#     plan = gs_state.agent_plans[j.id]
#     plan_timestep = (t - gs_state.agent_last_sync[j.id]) + 1
    
#     if 1 <= plan_timestep <= length(plan)
#         action = plan[plan_timestep]
#         if !isempty(action.target_cells)
#             return action.target_cells[1]  # Return first cell
#         end
#     end
    
#     return nothing
# end

# """
# Get next sync time for agent
# """
# function next_sync_time(agent::Agent, gs_state)
#     # Simplified: assume sync every C timesteps
#     C = agent.trajectory.period  # This should come from environment or agent parameters
#     return gs_state.time_step + C
# end

# """
# Get next sync time for agent at a specific global time
# """
# function next_sync_time_at_global_time(agent::Agent, t_global::Int)
#     # Simplified: assume sync every C timesteps
#     C = agent.trajectory.period  # This should come from environment or agent parameters
#     return t_global + C
# end

# """
# Advance clock vector
# """
# function advance_clock_vector(τ_clock::ClockVector, agents)
#     # Advance phases of all agents
#     new_phases = Int[]
#     for (i, agent) in enumerate(agents)
#         new_phase = (τ_clock.phases[i] + 1) % agent.trajectory.period
#         push!(new_phases, new_phase)
#     end
#     return ClockVector(new_phases)
# end

# """
# Generate random pointing action
# """
# function random_pointing(agent::Agent, τ_clock::ClockVector, env)
#     # Get agent position at this time using agent's phase
#     agent_index = find_agent_index(agent, env)
#     if agent_index === nothing
#         return SensingAction(agent.id, Tuple{Int, Int}[], false)
#     end
#     phase = τ_clock.phases[agent_index]
#     pos = get_position_at_time(agent.trajectory, phase)
    
#     # Get available cells in field of view
#     available_cells = get_field_of_regard_at_position(agent, pos, env)
#     # if agent.id == 2
#     #     
#     # end
#     # Create action set: wait action + pointing actions
#     actions = SensingAction[]
#     push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))  # Wait action
    
#     # Add pointing actions for available cells
#     for cell in available_cells
#         action = SensingAction(agent.id, [cell], false)
#         if check_battery_feasible(agent, action, agent.battery_level)
#             push!(actions, action)
#         end
#     end
    
#     # Pick random action (including wait)
#     return rand(actions)
# end

# """
# Get all pointing actions for agent
# """
# function all_pointings(agent::Agent, τ_clock::ClockVector, env)
#     actions = SensingAction[]
    
#     # Get agent position at this time using agent's phase
#     agent_index = find_agent_index(agent, env)
#     if agent_index === nothing
#         return [SensingAction(agent.id, Tuple{Int, Int}[], false)]
#     end
#     phase = τ_clock.phases[agent_index]
#     pos = get_position_at_time(agent.trajectory, phase)
#     # Get available cells in field of view
#     available_cells = get_field_of_regard_at_position(agent, pos, env)
#     # if agent.id == 2
#     #     
#     # end
#     # Add wait action
#     push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))
    
#     # Add single cell actions
#     for cell in available_cells
#         action = SensingAction(agent.id, [cell], false)
#         if check_battery_feasible(agent, action, agent.battery_level)
#             push!(actions, action)
#         end
#     end
    
#     return actions
# end

# @inline function find_nearest_belief(target::BeliefPoint, SLICE_BUCKET, DIGEST_LOOKUP)
#     key = Tuple(target.clock.phases)

#     # 1. bucket for the correct slice
#     bucket = get(SLICE_BUCKET, key, nothing)
#     if bucket === nothing
#         return target            # should not happen if 𝔅 covered all slices
#     end

#     # 2. exact digest match (O(1))
#     fp = get(DIGEST_LOOKUP[key], target.digest, nothing)
#     if fp !== nothing
#         return fp
#     end

#     # 3. fall-back: cheapest distance inside the bucket
#     nearest   = bucket[1]
#     best_dist = belief_distance(target.belief, nearest.belief)
#     @inbounds for bp in bucket
#         d = belief_distance(target.belief, bp.belief)
#         if d < best_dist
#             best_dist = d
#             nearest   = bp
#         end
#     end
#     return nearest
# end


# """
# Calculate distance between two beliefs using low-dimensional summary
# """
# function belief_distance(b1::Belief, b2::Belief)
#     # Use digest for fast comparison first
#     digest1 = hash(b1.event_distributions, UInt(0))
#     digest2 = hash(b2.event_distributions, UInt(0))
    
#     if digest1 == digest2
#         return 0.0  # Exact match
#     end
    
#     # Fallback to L1 distance on low-dimensional summary
#     # Create summary: vector of per-cell event probabilities
#     summary1 = belief_summary(b1)
#     summary2 = belief_summary(b2)
    
#     return sum(abs.(summary1 .- summary2))
# end

# """
# Create low-dimensional summary of belief for distance calculation
# """
# function belief_summary(belief::Belief)
#     # Get dimensions from event_distributions array [state, y, x]
#     num_states, height, width = size(belief.event_distributions)
    
#     # Create summary: vector of event probabilities for each cell
#     summary = Float64[]
    
#     for x in 1:width, y in 1:height
#         # Get event probability for this cell (sum over event states)
#         event_prob = 0.0
#         for state in 1:num_states
#             if state == 2  # Assuming state 2 is EVENT_PRESENT
#                 event_prob += belief.event_distributions[state, y, x]
#             end
#         end
#         push!(summary, event_prob)
#     end
    
#     return summary
# end

# """
# Find agent index in the clock vector
# """
# function find_agent_index(agent::Agent, env)
#     # Get all agents in the same order as they appear in the clock vector
#     all_agents = [env.agents[j] for j in sort(collect(keys(env.agents)))]
    
#     # Find the index of this agent
#     for (i, env_agent) in enumerate(all_agents)
#         if env_agent.id == agent.id
#             return i
#         end
#     end
    
#     return nothing
# end

# """
# Get other agents (excluding agent_i)
# """
# function get_other_agents(agent_i::Agent, env)
#     return [env.agents[j] for j in keys(env.agents) if j != agent_i.id]
# end

# """
# Get field of regard for an agent at a specific position
# """
# function get_field_of_regard_at_position(agent, position, env)
#     x, y = position
#     fov_cells = Tuple{Int, Int}[]
    
#     # Check sensor pattern
#     if agent.sensor.pattern == :cross
#         # Cross-shaped sensor: agent's position and adjacent cells
#         ax, ay = position
#         for dx in -1:1, dy in -1:1
#             nx, ny = ax + dx, ay + dy
#             if 1 <= nx <= env.width && 1 <= ny <= env.height
#                 # Only include cross pattern (not diagonal)
#                 if (dx == 0 && dy == 0) || (dx == 0 && dy != 0) || (dx != 0 && dy == 0)
#                     push!(fov_cells, (nx, ny))
#                 end
#             end
#         end
#     elseif agent.sensor.pattern == :row_only || agent.sensor.range == 0.0
#         # Row-only visibility: agent can only see cells in its current row
#         for nx in 1:env.width
#             push!(fov_cells, (nx, y))
#         end
#     else
#         # Standard sensor range visibility
#         sensor_range = round(Int, agent.sensor.range)
#         for dx in -sensor_range:sensor_range
#             for dy in -sensor_range:sensor_range
#                 nx, ny = x + dx, y + dy
#                 if 1 <= nx <= env.width && 1 <= ny <= env.height
#                     # Check if within sensor range
#                     distance = sqrt(dx^2 + dy^2)
#                     if distance <= agent.sensor.range
#                         push!(fov_cells, (nx, ny))
#                     end
#                 end
#             end
#         end
#     end
#     return fov_cells
# end
# """
# Set reward configuration from main.jl constants
# This function should be called from the main script to sync configuration
# """
# function set_reward_config_from_main(entropy_weight::Float64, value_weight::Float64, 
#                                    information_states::Vector{Int}, state_values::Vector{Float64})
#     global DEFAULT_ENTROPY_WEIGHT = entropy_weight
#     global DEFAULT_VALUE_WEIGHT = value_weight
#     global DEFAULT_INFORMATION_STATES = information_states
#     global DEFAULT_STATE_VALUES = state_values
    
#     println("🎯 Reward configuration synced from main.jl:")
#     println("   • Entropy weight (w_H): $entropy_weight")
#     println("   • Value weight (w_F): $value_weight")
#     println("   • Information states: $information_states")
#     println("   • State values: $state_values")
# end

# end # module 
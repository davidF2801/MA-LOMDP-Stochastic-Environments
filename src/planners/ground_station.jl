module GroundStation

using POMDPs
using POMDPTools
using Random
using Infiltrator
using ..Types
using ..Environment
using ..Agents

# Import planner modules
include("macro_planner_async.jl")
include("macro_planner_approx.jl")
include("policy_tree_planner.jl")
include("macro_planner_random.jl")
include("macro_planner_sweep.jl")
include("macro_planner_greedy.jl")
include("macro_planner_prior_based.jl")
include("macro_planner_pbvi.jl")
include("macro_planner_pbvi_policy_tree.jl")

using .MacroPlannerAsync
using .MacroPlannerAsyncApprox
using .PolicyTreePlanner
using .MacroPlannerRandom
using .MacroPlannerSweep
using .MacroPlannerGreedy
using .MacroPlannerPriorBased
using .MacroPlannerPBVI
using .AsyncPBVIPolicyTree

# Import Agent type from TrajectoryPlanner
include("../agents/trajectory_planner.jl")
using .TrajectoryPlanner: get_position_at_time

# Import types from the parent module (Planners)
import ..Types.EventState, ..Types.NO_EVENT, ..Types.EVENT_PRESENT
import ..Types.SensingAction, ..Types.GridObservation, ..Types.Agent
import ..Types.CircularTrajectory, ..Types.LinearTrajectory, ..Types.RangeLimitedSensor
import ..Types.EventState2, ..Types.NO_EVENT_2, ..Types.EVENT_PRESENT_2
# Import EventDynamicsModule functions through Environment
import ..Environment.EventDynamicsModule.DBNTransitionModel2, ..Environment.EventDynamicsModule.predict_next_belief_dbn
import ..Agents.BeliefManagement: update_cell_distribution, get_neighbor_event_probabilities
import ..Agents.BeliefManagement: predict_belief_rsp, evolve_no_obs, collapse_belief_to
import ..Environment.EventDynamicsModule: rsp_transition_probs
# Import functions from MacroPlannerAsync
import .MacroPlannerAsync: initialize_uniform_belief, get_known_observations_at_time, has_known_observation, get_known_observation

export maybe_sync!, GroundStationState, update_global_belief_rsp!, precompute_worlds!

"""
GroundStationState - Maintains global belief and agent synchronization info
"""
mutable struct GroundStationState
    global_belief::Any  # Global belief over the environment
    agent_last_sync::Dict{Int, Int}  # Last sync time for each agent
    agent_plans::Dict{Int, Any}  # Current plans for each agent
    agent_plan_types::Dict{Int, Symbol}  # :script or :policy for each agent
    agent_observation_history::Dict{Int, Vector{Tuple{Int, Tuple{Int, Int}, EventState}}}  # (timestep, cell, observed_state) for each agent
    time_step::Int
    planning_times::Dict{Int, Vector{Float64}}  # Planning times for each agent
    total_planning_time::Float64  # Total planning time across all agents
    num_plans_computed::Int  # Total number of plans computed
end

"""
maybe_sync!(env, gs_state, agents, t)
  – For each agent i:
        • if in-range(agents[i], t)           # FoV(contact_region)
            – upload obs since last contact  → update global_belief
            – compute *either* macro-script or policy-tree:
                 if mode == :script   → MacroPlanner.best_script(...)
                 if mode == :policy   → PolicyTreePlanner.best_policy_tree(...)
            – send plan back to agent (store in agent.plan)
            – reset local log2/clock for that agent
"""
function maybe_sync!(env, gs_state::GroundStationState, agents, t::Int; 
                    planning_mode::Symbol=:script, rng::AbstractRNG=Random.GLOBAL_RNG)
    
    println("🛰️  Ground Station: Checking for sync opportunities at time $(t)")
    gs_state.time_step = t
    for (i, agent) in enumerate(agents)
        agent_id = agent.id
        # Check if agent is in range for synchronization
        if in_range(agent, t, env, env.ground_station_pos)
            println("📡 Agent $(agent_id) in range at time $(t)")
            
            # Upload observations since last contact
            observations = get_agent_observations_since_sync(agent, gs_state.agent_last_sync[agent_id], t)
            println("📊 Agent $(agent_id) uploading $(length(observations)) observations since last sync (t=$(gs_state.agent_last_sync[agent_id]))")
            
            # Store observations in ground station history
            # Each observation corresponds to one timestep since last sync
            for (obs_idx, observation) in enumerate(observations)
                for (i, cell) in enumerate(observation.sensed_cells)
                    if i <= length(observation.event_states)
                        observed_state = observation.event_states[i]
                        # Calculate the actual timestep when this observation occurred
                        # obs_idx is 1-based, so we add it to the last sync time
                        obs_timestep = gs_state.agent_last_sync[agent_id] + obs_idx-1
                        push!(gs_state.agent_observation_history[agent_id], (obs_timestep, cell, observed_state))
                    end
                end
            end
            update_global_belief!(gs_state.global_belief, observations, env, gs_state, t)
            # Calculate contact horizon (steps until next sync)
            C_i = calculate_contact_horizon(agent, t, env)
            println("⏰ Contact horizon for agent $(agent_id): $(C_i) steps")
            
            # Get other agents' current plans
            other_plans = get_other_agent_plans(gs_state, agent_id)
            # Compute new plan based on planning mode
            if planning_mode == :script
                println("📋 Computing macro-script for agent $(agent_id)")
                new_plan, planning_time = MacroPlannerAsync.best_script(env, gs_state.global_belief, agent, C_i, other_plans, gs_state, rng=rng)
                gs_state.agent_plan_types[agent_id] = :script
                
                # Track planning time
                push!(gs_state.planning_times[agent_id], planning_time)
                gs_state.total_planning_time += planning_time
                gs_state.num_plans_computed += 1
                println("⏱️  Agent $(agent_id) planning time: $(round(planning_time, digits=3)) seconds")
            elseif planning_mode == :policy
                println("🌳 Computing policy tree for agent $(agent_id)")
                reactive_policy, planning_time, policy_tree = PolicyTreePlanner.best_policy_tree(env, gs_state.global_belief, agent, C_i, gs_state, rng=rng)
                gs_state.agent_plan_types[agent_id] = :policy
                
                # For policy planning, store the reactive policy in the agent
                # The agent already has the reactive policy stored from best_policy_tree
                # We don't store it in gs_state.agent_plans since that's for action sequences
                
                # Track planning time
                push!(gs_state.planning_times[agent_id], planning_time)
                gs_state.total_planning_time += planning_time
                gs_state.num_plans_computed += 1
                println("⏱️  Agent $(agent_id) policy tree planning time: $(round(planning_time, digits=3)) seconds")
            elseif planning_mode == :random
                println("🎲 Computing random plan for agent $(agent_id)")
                new_plan, planning_time = MacroPlannerRandom.best_script_random(env, gs_state.global_belief, agent, C_i, other_plans, gs_state, rng=rng)
                gs_state.agent_plan_types[agent_id] = :random
                # Track planning time
                push!(gs_state.planning_times[agent_id], planning_time)
                gs_state.total_planning_time += planning_time
                gs_state.num_plans_computed += 1
                println("⏱️  Agent $(agent_id) random planning time: $(round(planning_time, digits=3)) seconds")
            elseif planning_mode == :future_actions
                println("🔮 Computing future actions for agent $(agent_id)")
                new_plan, planning_time = MacroPlannerAsync.best_script(env, gs_state.global_belief, agent, C_i, other_plans, gs_state, rng=rng, mode=:future_actions)
                gs_state.agent_plan_types[agent_id] = :future_actions
                # Track planning time
                push!(gs_state.planning_times[agent_id], planning_time)
                gs_state.total_planning_time += planning_time
                gs_state.num_plans_computed += 1
                println("⏱️  Agent $(agent_id) future actions planning time: $(round(planning_time, digits=3)) seconds")
            elseif planning_mode == :sweep
                println("🧹 Computing sweep plan for agent $(agent_id)")
                new_plan, planning_time = MacroPlannerSweep.best_script(env, gs_state.global_belief, agent, C_i, other_plans, gs_state, rng=rng)
                gs_state.agent_plan_types[agent_id] = :sweep
                
                # Track planning time
                push!(gs_state.planning_times[agent_id], planning_time)
                gs_state.total_planning_time += planning_time
                gs_state.num_plans_computed += 1
                println("⏱️  Agent $(agent_id) sweep planning time: $(round(planning_time, digits=3)) seconds")
            elseif planning_mode == :greedy
                println("🤖 Computing greedy plan for agent $(agent_id)")
                new_plan, planning_time = MacroPlannerGreedy.best_script(env, gs_state.global_belief, agent, C_i, other_plans, gs_state, rng=rng)
                gs_state.agent_plan_types[agent_id] = :greedy
                
                # Track planning time
                push!(gs_state.planning_times[agent_id], planning_time)
                gs_state.total_planning_time += planning_time
                gs_state.num_plans_computed += 1
                println("⏱️  Agent $(agent_id) greedy planning time: $(round(planning_time, digits=3)) seconds")
            elseif planning_mode == :macro_approx || planning_mode == :macro_approx_099 || planning_mode == :macro_approx_095 || planning_mode == :macro_approx_090
                println("🔧 Computing macro-approximate plan for agent $(agent_id)")
                new_plan, planning_time = MacroPlannerAsyncApprox.best_script(env, gs_state.global_belief, agent, C_i, other_plans, gs_state, rng=rng)
                gs_state.agent_plan_types[agent_id] = planning_mode
                
                # Track planning time
                push!(gs_state.planning_times[agent_id], planning_time)
                gs_state.total_planning_time += planning_time
                gs_state.num_plans_computed += 1
                println("⏱️  Agent $(agent_id) macro-approximate planning time: $(round(planning_time, digits=3)) seconds")
            elseif planning_mode == :prior_based
                println("📊 Computing prior-based plan for agent $(agent_id)")
                new_plan, planning_time = MacroPlannerPriorBased.best_script(env, gs_state.global_belief, agent, C_i, other_plans, gs_state, rng=rng)
                gs_state.agent_plan_types[agent_id] = :prior_based
                
                # Track planning time
                push!(gs_state.planning_times[agent_id], planning_time)
                gs_state.total_planning_time += planning_time
                gs_state.num_plans_computed += 1
                println("⏱️  Agent $(agent_id) prior-based planning time: $(round(planning_time, digits=3)) seconds")
            elseif planning_mode == :pbvi
                println("🧠 Computing PBVI plan for agent $(agent_id)")
                @infiltrate
                new_plan, planning_time = MacroPlannerPBVI.best_script(env, gs_state.global_belief, agent, C_i, other_plans, gs_state, rng=rng)
                gs_state.agent_plan_types[agent_id] = :pbvi
                
                # Track planning time
                push!(gs_state.planning_times[agent_id], planning_time)
                gs_state.total_planning_time += planning_time
                gs_state.num_plans_computed += 1
                println("⏱️  Agent $(agent_id) PBVI planning time: $(round(planning_time, digits=3)) seconds")
            elseif planning_mode == :pbvi_policy_tree
                println("🌳 Computing PBVI policy tree for agent $(agent_id)")
                reactive_policy, planning_time, policy_tree = AsyncPBVIPolicyTree.best_policy_tree(env, gs_state.global_belief, agent, C_i, other_plans, gs_state, rng=rng)
                
                gs_state.agent_plan_types[agent_id] = :pbvi_policy_tree
                
                # Track planning time
                push!(gs_state.planning_times[agent_id], planning_time)
                gs_state.total_planning_time += planning_time
                gs_state.num_plans_computed += 1
                println("⏱️  Agent $(agent_id) PBVI policy tree planning time: $(round(planning_time, digits=3)) seconds")
            else
                error("Unknown planning mode: $(planning_mode)")
            end
            
            # Store plan in ground station state (for non-policy modes)
            if planning_mode != :policy && planning_mode != :pbvi_policy_tree
                gs_state.agent_plans[agent_id] = new_plan
                # Reset agent's plan index for the new plan
                agent.plan_index = 1
            else
                # For policy modes, the reactive policy is already stored in the agent
                # No need to store anything in gs_state.agent_plans
                # Reset agent's plan index (though not used for policies)
                agent.plan_index = 1
            end
            
            # Update last sync time
            gs_state.agent_last_sync[agent_id] = t
            
            println("✅ Agent $(agent_id) synchronized with new plan")
        end
    end
    
    # Update ground station time
end

"""
Check if agent is in range for synchronization
"""
function in_range(agent, t::Int, env, ground_station_pos)
    # Get agent's current position with phase offset
    current_pos = get_position_at_time(agent.trajectory, t, agent.phase_offset)
    @infiltrate
    # Use the provided ground station position
    return current_pos == ground_station_pos
end

"""
Get agent's observations since last synchronization
"""
function get_agent_observations_since_sync(agent, last_sync::Int, current_time::Int)
    # Get observations that occurred after the last sync time
    # Assumes observations are added to history in chronolog2ical order (one per time step)
    
    if last_sync == -1
        # First sync - return all observations
        return copy(agent.observation_history)
    end
    
    # Calculate how many observations to skip (one per time step since last sync)
    observations_since_sync = current_time - last_sync
    
    if observations_since_sync <= 0
        # No new observations since last sync
        return GridObservation[]
    end
    # Get the most recent observations (last observations_since_sync observations)
    total_observations = length(agent.observation_history)
    start_index = max(1, total_observations - observations_since_sync + 1)
    
    if start_index > total_observations
        # No observations available
        return GridObservation[]
    end
    
    # Return the observations since last sync
    return agent.observation_history[start_index:end]
end

"""
Update global belief with new observations using the same log2ic as macro planner async
Updates belief till t_clean (last time where all observation outcomes are known)
"""
function update_global_belief!(global_belief, observations::Vector{GridObservation}, env, gs_state::GroundStationState, current_time::Int)
    if isempty(observations)
        println("🔄 No new observations to update global belief")
        return
    end
    
    println("🔄 Updating global belief with $(length(observations)) observations")
    
    # Step 1: Determine the last time where all observation outcomes are known
    tau = gs_state.agent_last_sync  # Last sync times of all agents
    t_clean = minimum([tau[j] for j in keys(tau)])
    
    println("  📊 t_clean = $(t_clean) (last time where all observation outcomes are known)")
    
    # Step 2: Roll forward deterministically from prior belief to t_clean using known observations
    # Start with prior-based belief distribution
    B = initialize_global_belief(env)
    
    for t in 0:(t_clean-1)
        # Apply known observations (perfect observations)
        for (agent_j, action_j) in get_known_observations_at_time(t, gs_state)
            for cell in action_j.target_cells
                if has_known_observation(t, cell, gs_state)
                    observed_value = get_known_observation(t, cell, gs_state)
                    B = collapse_belief_to(B, cell, observed_value)
                end
            end
        end
        B = evolve_no_obs(B, env)  # Contagion-aware update
    end
    
    # Update the global belief with the belief at t_clean
    global_belief.event_distributions = B.event_distributions
    global_belief.uncertainty_map = B.uncertainty_map
    global_belief.last_update = t_clean
    
    println("✅ Global belief updated till t_clean = $(t_clean)")
end

"""
Get neighbor beliefs for a cell (helper function for transition model)
"""
function get_neighbor_beliefs(probabilities::Matrix{Float64}, x::Int, y::Int)
    neighbor_beliefs = Float64[]
    height, width = size(probabilities)
    
    for dx in -1:1
        for dy in -1:1
            if dx == 0 && dy == 0
                continue
            end
            
            nx, ny = x + dx, y + dy
            if 1 <= nx <= width && 1 <= ny <= height
                push!(neighbor_beliefs, probabilities[ny, nx])
            end
        end
    end
    
    return neighbor_beliefs
end

"""
Calculate contact horizon (steps until next sync opportunity)
"""
function calculate_contact_horizon(agent, current_time::Int, env)
    # For a grid with period equal to height, agents can sync every period steps
    # Return a shorter horizon for more frequent planning
    return agent.trajectory.period  # Steps until next sync (matches grid period)
end

"""
Get other agents' current plans
"""
function get_other_agent_plans(gs_state::GroundStationState, current_agent_id::Int)
    other_plans = []
    
    for (agent_id, plan) in gs_state.agent_plans
        if agent_id != current_agent_id
            push!(other_plans, plan)
        end
    end
    
    return other_plans
end

"""
Initialize ground station state
"""
function initialize_ground_station(env, agents; num_states::Int=2)
    # Initialize global belief
    global_belief = initialize_global_belief(env, num_states=num_states)
    
    # Initialize agent tracking
    agent_last_sync = Dict{Int, Int}()
    agent_plans = Dict{Int, Any}()
    agent_plan_types = Dict{Int, Symbol}()
    agent_observation_history = Dict{Int, Vector{Tuple{Int, Tuple{Int, Int}, EventState}}}()
    planning_times = Dict{Int, Vector{Float64}}()
    total_planning_time = 0.0
    num_plans_computed = 0
    
    for agent in agents
        agent_last_sync[agent.id] = -1  # No sync yet
        agent_plans[agent.id] = nothing
        agent_plan_types[agent.id] = :script
        agent_observation_history[agent.id] = Tuple{Int, Tuple{Int, Int}, EventState}[]
        planning_times[agent.id] = Float64[]
    end
    
    return GroundStationState(global_belief, agent_last_sync, agent_plans, agent_plan_types, agent_observation_history, 0, planning_times, total_planning_time, num_plans_computed)
end

"""
Initialize global belief
"""
function initialize_global_belief(env; num_states::Int=2)
    # Initialize with prior distribution over event states using Types + prior-based planner logic
    # For 2-state model: [NO_EVENT, EVENT_PRESENT]
    # Build per-cell P(EVENT_PRESENT) prior using the same method as prior-based macro planner
    prior_map = MacroPlannerPriorBased.calculate_prior_probability_map(env)
    
    return BeliefManagement.initialize_belief_from_event_prior(prior_map)
end

"""
Get agent's current plan from ground station
"""
function get_agent_plan(agent, gs_state::GroundStationState)
    agent_id = agent.id
    
    # Get plan from ground station state
    plan = get(gs_state.agent_plans, agent_id, nothing)
    plan_type = get(gs_state.agent_plan_types, agent_id, :script)
    
    return plan, plan_type
end

"""
Print ground station status
"""
function print_status(gs_state::GroundStationState)
    println("🛰️  Ground Station Status:")
    println("  Time step: $(gs_state.time_step)")
    println("  Agent sync times: $(gs_state.agent_last_sync)")
    println("  Agent plan types: $(gs_state.agent_plan_types)")
    
    for (agent_id, plan) in gs_state.agent_plans
        if plan !== nothing
            plan_type = gs_state.agent_plan_types[agent_id]
            if plan_type == :script
                println("  Agent $(agent_id): $(length(plan))-step script")
            else
                println("  Agent $(agent_id): policy tree")
            end
        else
            println("  Agent $(agent_id): no plan")
        end
    end
    
    # Print planning time statistics
    print_planning_time_stats(gs_state)
end

"""
Print planning time statistics
"""
function print_planning_time_stats(gs_state::GroundStationState)
    if gs_state.num_plans_computed > 0
        avg_time = gs_state.total_planning_time / gs_state.num_plans_computed
        println("⏱️  Planning Time Statistics:")
        println("  Total plans computed: $(gs_state.num_plans_computed)")
        println("  Total planning time: $(round(gs_state.total_planning_time, digits=3)) seconds")
        println("  Average planning time: $(round(avg_time, digits=3)) seconds")
        
        # Per-agent statistics
        for (agent_id, times) in gs_state.planning_times
            if !isempty(times)
                agent_avg = sum(times) / length(times)
                agent_min = minimum(times)
                agent_max = maximum(times)
                println("  Agent $(agent_id): avg=$(round(agent_avg, digits=3))s, min=$(round(agent_min, digits=3))s, max=$(round(agent_max, digits=3))s ($(length(times)) plans)")
            end
        end
    else
        println("⏱️  No planning time data available yet")
    end
end

"""
Get average planning time
"""
function get_average_planning_time(gs_state::GroundStationState)
    if gs_state.num_plans_computed > 0
        return gs_state.total_planning_time / gs_state.num_plans_computed
    else
        return 0.0
    end
end

"""
update_global_belief_rsp!(gs_state, obs_batch, λmap, now)
Update global belief using RSP dynamics after receiving observations
"""
function update_global_belief_rsp!(gs_state::GroundStationState, obs_batch::Vector{Tuple{Agent, GridObservation}}, λmap::Matrix{Float64}, now::Int)
    # 1. Collapse belief with new observations
    for (agent, observation) in obs_batch
        # Update belief with this observation using existing function
        # This assumes we have a dummy action for the observation
        dummy_action = SensingAction(agent.id, observation.sensed_cells, false)
        
        # For now, we'll do a simple update - in a full implementation,
        # we'd call update_belief_state with proper event dynamics
        if gs_state.global_belief !== nothing
            # Simple collapse to certainty for observed cells
            for (i, cell) in enumerate(observation.sensed_cells)
                x, y = cell
                if 1 <= x <= size(λmap, 2) && 1 <= y <= size(λmap, 1)
                    observed_state = observation.event_states[i]
                    
                    # Update belief distribution for this cell
                    if observed_state == EVENT_PRESENT
                        gs_state.global_belief.event_distributions[1, y, x] = 0.0  # NO_EVENT
                        gs_state.global_belief.event_distributions[2, y, x] = 1.0  # EVENT_PRESENT
                    else
                        gs_state.global_belief.event_distributions[1, y, x] = 1.0  # NO_EVENT
                        gs_state.global_belief.event_distributions[2, y, x] = 0.0  # EVENT_PRESENT
                    end
                end
            end
        end
    end
    
    # 2. Calculate Δt since last update
    Δt = now - gs_state.time_step
    
    # 3. Propagate unobserved part using RSP
    if gs_state.global_belief !== nothing && Δt > 0
        gs_state.global_belief = predict_belief_rsp(gs_state.global_belief, λmap, Δt)
    end
    
    gs_state.time_step = now
end

"""
precompute_worlds!(env, B0::Belief, C::Int)
Pre-compute all possible world trajectories for the next C timesteps
"""
function precompute_worlds!(env, B0::Belief, C::Int)
    if !hasfield(typeof(env), :ignition_prob)
        # Create a simple ignition probability map if not present
        height, width = size(B0.event_distributions)[2:3]
        env.ignition_prob = fill(0.1, height, width)  # Uniform ignition probability
    end
    
    env.pre_enumerated_worlds = enumerate_worlds(B0, C, env.ignition_prob)
    
end

"""
enumerate_worlds(B0::Belief, C::Int, λmap::Matrix{Float64}) -> Vector{Tuple{Vector{EventMap}, Float64}}
Enumerate all possible world trajectories using exact enumeration
"""
function enumerate_worlds(B0::Belief, C::Int, λmap::Matrix{Float64})
    num_states, height, width = size(B0.event_distributions)
    
    # Convert belief to event map probabilities
    event_probs = Matrix{Float64}(undef, height, width)
    for y in 1:height, x in 1:width
        if num_states >= 2
            event_probs[y, x] = B0.event_distributions[2, y, x]  # P(EVENT_PRESENT)
        else
            event_probs[y, x] = 0.0
        end
    end
    
    # For small grids, enumerate all possible initial states
    total_cells = height * width
    max_initial_states = 2^total_cells
    
    # For computational tractability, limit to reasonable size
    if max_initial_states > 1000
        println("⚠️  Grid too large for exact enumeration. Limiting to 1000 initial states.")
        max_initial_states = 1000
    end
    
    trajectories = Vector{Tuple{Vector{EventMap}, Float64}}()
    
    # Enumerate all possible initial states
    for state_idx in 0:(max_initial_states-1)
        # Create initial state from binary representation
        initial_map = Matrix{EventState}(undef, height, width)
        initial_weight = 1.0
        
        for cell_idx in 0:(total_cells-1)
            y = div(cell_idx, width) + 1
            x = mod(cell_idx, width) + 1
            
            # Check if this cell should have an event based on state_idx
            bit = (state_idx >> cell_idx) & 1
            
            if bit == 1
                initial_map[y, x] = EVENT_PRESENT
                initial_weight *= event_probs[y, x]
            else
                initial_map[y, x] = NO_EVENT
                initial_weight *= (1.0 - event_probs[y, x])
            end
        end
        
        if initial_weight > 0.0
            # Recursively enumerate all possible trajectories from this initial state
            enumerate_trajectories_dfs!(trajectories, initial_map, initial_weight, λmap, C, 0, EventMap[])
        end
    end
    return trajectories
end

"""
enumerate_trajectories_dfs!(trajectories, current_map, current_weight, λmap, C, depth, current_trajectory)
Recursively enumerate all possible trajectories using depth-first search
"""
function enumerate_trajectories_dfs!(trajectories::Vector{Tuple{Vector{EventMap}, Float64}}, 
                                   current_map::EventMap, 
                                   current_weight::Float64, 
                                   λmap::Matrix{Float64}, 
                                   C::Int, 
                                   depth::Int,
                                   current_trajectory::Vector{EventMap})
    
    # Add current state to trajectory
    trajectory_with_current = [current_trajectory; [copy(current_map)]]
    
    if depth == C - 1
        # We've reached the end of the trajectory (C-1 evolutions from initial state)
        push!(trajectories, (trajectory_with_current, current_weight))
        return
    end
    
    # Get all possible next states and their probabilities
    transition_probs = rsp_transition_probs(current_map, λmap)
    # Branch on all possible next states
    for (next_map, transition_prob) in transition_probs
        if transition_prob > 0.0
            new_weight = current_weight * transition_prob
            
            # Recursively enumerate from this next state
            enumerate_trajectories_dfs!(trajectories, next_map, new_weight, λmap, C, depth + 1, trajectory_with_current)
        end
    end
end

"""
Check if agent j senses at time t using their reactive policy (policy tree)
"""
function j_senses_at_time_policy_tree(j::Agent, t::Int, agent_j_obs_history::Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}})
    # Check if agent j has a reactive policy
    if j.reactive_policy === nothing
        return false
    end
    
    # Convert observation history to GridObservation format for the reactive policy
    grid_observations = Vector{GridObservation}()
    for (timestep, obs_symbols) in agent_j_obs_history
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
Get scheduled cell for agent j at time t using their reactive policy (policy tree)
"""
function scheduled_cell_policy_tree(j::Agent, t::Int, agent_j_obs_history::Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}})
    # Check if agent j has a reactive policy
    if j.reactive_policy === nothing
        return nothing
    end
    
    # Convert observation history to GridObservation format for the reactive policy
    grid_observations = Vector{GridObservation}()
    for (timestep, obs_symbols) in agent_j_obs_history
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
Check if agent j senses at time t using ground station plans (fallback for non-policy agents)
"""
function j_senses_at_time_gs_plan(j::Agent, t::Int, gs_state)
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
Get scheduled cell for agent j at time t using ground station plans (fallback for non-policy agents)
"""
function scheduled_cell_gs_plan(j::Agent, t::Int, gs_state)
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
Check if agent j senses at time t - unified function that handles both policy trees and ground station plans
"""
function j_senses_at_time(j::Agent, t::Int, gs_state, agent_j_obs_history::Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}} = Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}[])
    # Check agent's plan type to determine which method to use
    plan_type = get(gs_state.agent_plan_types, j.id, :script)
    
    if plan_type == :policy || plan_type == :pbvi_policy_tree
        # Use policy tree method
        return j_senses_at_time_policy_tree(j, t, agent_j_obs_history)
    else
        # Use ground station plan method
        return j_senses_at_time_gs_plan(j, t, gs_state)
    end
end

"""
Get scheduled cell for agent j at time t - unified function that handles both policy trees and ground station plans
"""
function scheduled_cell(j::Agent, t::Int, gs_state, agent_j_obs_history::Vector{Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}} = Tuple{Int, Vector{Tuple{Tuple{Int, Int}, EventState}}}[])
    # Check agent's plan type to determine which method to use
    plan_type = get(gs_state.agent_plan_types, j.id, :script)
    
    if plan_type == :policy || plan_type == :pbvi_policy_tree
        # Use policy tree method
        return scheduled_cell_policy_tree(j, t, agent_j_obs_history)
    else
        # Use ground station plan method
        return scheduled_cell_gs_plan(j, t, gs_state)
    end
end

end # module 
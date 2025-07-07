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
include("policy_tree_planner.jl")

using .MacroPlannerAsync
using .PolicyTreePlanner

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
import ..Agents.BeliefManagement: predict_belief_rsp
import ..Environment.EventDynamicsModule: rsp_transition_probs

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
            – reset local log/clock for that agent
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
                        obs_timestep = gs_state.agent_last_sync[agent_id] + obs_idx
                        push!(gs_state.agent_observation_history[agent_id], (obs_timestep, cell, observed_state))
                    end
                end
            end
            
            update_global_belief!(gs_state.global_belief, observations, env)
            
            # Calculate contact horizon (steps until next sync)
            C_i = calculate_contact_horizon(agent, t, env)
            println("⏰ Contact horizon for agent $(agent_id): $(C_i) steps")
            
            # Get other agents' current plans
            other_plans = get_other_agent_plans(gs_state, agent_id)
            
            # Compute new plan based on planning mode
            if planning_mode == :script
                println("📋 Computing macro-script for agent $(agent_id)")
                new_plan = MacroPlannerAsync.best_script(env, gs_state.global_belief, agent, C_i, other_plans, gs_state, rng=rng)
                gs_state.agent_plan_types[agent_id] = :script
            elseif planning_mode == :policy
                println("🌳 Computing policy tree for agent $(agent_id)")
                new_plan = PolicyTreePlanner.best_policy_tree(env, gs_state.global_belief, agent, C_i, other_plans, rng=rng)
                gs_state.agent_plan_types[agent_id] = :policy
            else
                error("Unknown planning mode: $(planning_mode)")
            end
            
            # Store plan in ground station state
            gs_state.agent_plans[agent_id] = new_plan
            
            # Reset agent's plan index for the new plan
            agent.plan_index = 1
            
            # Store plan in ground station state (agents don't store plans directly)
            # The plan will be executed by the ground station when needed
            
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
    # Use the provided ground station position
    return current_pos == ground_station_pos
end

"""
Get agent's observations since last synchronization
"""
function get_agent_observations_since_sync(agent, last_sync::Int, current_time::Int)
    # Get observations that occurred after the last sync time
    # Assumes observations are added to history in chronological order (one per time step)
    
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
Update global belief with new observations using proper Bayesian updating
Processes observations chronologically and applies transition models to unobserved cells
"""
function update_global_belief!(global_belief, observations::Vector{GridObservation}, env)
    if isempty(observations)
        println("🔄 No new observations to update global belief")
        return
    end
    
    println("🔄 Updating global belief with $(length(observations)) observations")
    
    # Create DBN transition model for belief propagation
    dbn_model = DBNTransitionModel2(env.event_dynamics)
    
    # Get current belief distributions
    current_distributions = copy(global_belief.event_distributions)
    num_states, height, width = size(current_distributions)
    
    # Process observations chronologically (they should be in order from get_agent_observations_since_sync)
    for (obs_idx, observation) in enumerate(observations)
        agent_id = observation.agent_id
        sensed_cells = observation.sensed_cells
        event_states = observation.event_states
        
        println("  📡 Processing observation $(obs_idx): Agent $(agent_id) observed $(length(sensed_cells)) cells")
        
        # First, apply transition model to ALL cells (simulate environment evolution)
        # This represents what happens between observations
        new_distributions = similar(current_distributions)
        
        for y in 1:height
            for x in 1:width
                # Get neighbor beliefs for transition model
                neighbor_event_probs = get_neighbor_event_probabilities(current_distributions, x, y)
                
                # Apply DBN transition model to predict next belief for each state
                current_cell_dist = current_distributions[:, y, x]
                new_cell_dist = similar(current_cell_dist)
                
                # Simplified transition model for multi-state belief
                E_neighbors = sum(neighbor_event_probs)
                
                # State transitions based on number of states
                if num_states == 2
                    # Simple 2-state model: NO_EVENT ↔ EVENT_PRESENT
                    new_cell_dist[1] = current_cell_dist[1] * (1.0 - env.event_dynamics.birth_rate - env.event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * env.event_dynamics.death_rate
                    
                    new_cell_dist[2] = current_cell_dist[1] * (env.event_dynamics.birth_rate + env.event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * (1.0 - env.event_dynamics.death_rate)
                elseif num_states == 4
                    # 4-state model: NO_EVENT → EVENT_PRESENT → EVENT_SPREADING → EVENT_DECAYING → NO_EVENT
                    new_cell_dist[1] = current_cell_dist[1] * (1.0 - env.event_dynamics.birth_rate - env.event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * env.event_dynamics.death_rate +
                                       current_cell_dist[4] * env.event_dynamics.decay_rate
                    
                    new_cell_dist[2] = current_cell_dist[1] * (env.event_dynamics.birth_rate + env.event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * (1.0 - env.event_dynamics.death_rate - env.event_dynamics.spread_rate) +
                                       current_cell_dist[3] * env.event_dynamics.decay_rate
                    
                    new_cell_dist[3] = current_cell_dist[2] * env.event_dynamics.spread_rate +
                                       current_cell_dist[3] * (1.0 - env.event_dynamics.decay_rate)
                    
                    new_cell_dist[4] = current_cell_dist[3] * env.event_dynamics.decay_rate +
                                       current_cell_dist[4] * (1.0 - env.event_dynamics.decay_rate)
                else
                    # Default: simple 2-state model
                    new_cell_dist[1] = current_cell_dist[1] * (1.0 - env.event_dynamics.birth_rate - env.event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * env.event_dynamics.death_rate
                    
                    new_cell_dist[2] = current_cell_dist[1] * (env.event_dynamics.birth_rate + env.event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * (1.0 - env.event_dynamics.death_rate)
                end
                
                # Normalize
                total = sum(new_cell_dist)
                if total > 0
                    new_cell_dist ./= total
                end
                
                new_distributions[:, y, x] = new_cell_dist
            end
        end
        
        # Update current distributions with transition results
        current_distributions = new_distributions
        
        # Now apply perfect observations for observed cells
        for (i, cell) in enumerate(sensed_cells)
            if i <= length(event_states)
                event_state = event_states[i]
                x, y = cell
                
                if 1 <= x <= width && 1 <= y <= height
                    # Get prior distribution (after transition)
                    prior_distribution = current_distributions[:, y, x]
                    
                    # Apply perfect observation: belief collapses to certainty
                    updated_distribution = update_cell_distribution(prior_distribution, event_state)
                    current_distributions[:, y, x] = updated_distribution
                    
                    # Print update information
                    if length(prior_distribution) >= 2
                        prior_event_prob = sum(prior_distribution[2:end])
                        posterior_event_prob = sum(updated_distribution[2:end])
                    else
                        prior_event_prob = 0.0
                        posterior_event_prob = 0.0
                    end
                    println("    🎯 Cell ($(x),$(y)): $(event_state) observed, belief collapsed from $(round(prior_event_prob, digits=3)) to $(round(posterior_event_prob, digits=3))")
                end
            end
        end
    end
    
    # Update the global belief with final distributions
    global_belief.event_distributions = current_distributions
    
    println("✅ Global belief updated with proper Bayesian inference")
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
    return env.height  # Steps until next sync (matches grid period)
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
    
    for agent in agents
        agent_last_sync[agent.id] = -1  # No sync yet
        agent_plans[agent.id] = nothing
        agent_plan_types[agent.id] = :script
        agent_observation_history[agent.id] = Tuple{Int, Tuple{Int, Int}, EventState}[]
    end
    
    return GroundStationState(global_belief, agent_last_sync, agent_plans, agent_plan_types, agent_observation_history, 0)
end

"""
Initialize global belief
"""
function initialize_global_belief(env; num_states::Int=2)
    # Initialize with uniform distribution over event states using proper Belief type
    # For 2-state model: [NO_EVENT, EVENT_PRESENT]
    # For 4-state model: [NO_EVENT, EVENT_PRESENT, EVENT_SPREADING, EVENT_DECAYING]
    
    uniform_distribution = fill(1.0/num_states, num_states)
    
    return BeliefManagement.initialize_belief(env.width, env.height, uniform_distribution)
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

end # module 
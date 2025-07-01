module GroundStation

using POMDPs
using POMDPTools
using Random
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
import ..Agents.BeliefManagement
export maybe_sync!, GroundStationState

"""
GroundStationState - Maintains global belief and agent synchronization info
"""
mutable struct GroundStationState
    global_belief::Any  # Global belief over the environment
    agent_last_sync::Dict{Int, Int}  # Last sync time for each agent
    agent_plans::Dict{Int, Any}  # Current plans for each agent
    agent_plan_types::Dict{Int, Symbol}  # :script or :policy for each agent
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
    
    for (i, agent) in enumerate(agents)
        agent_id = agent.id
        
        # Check if agent is in range for synchronization
        if in_range(agent, t, env, env.ground_station_pos)
            println("📡 Agent $(agent_id) in range at time $(t)")
            
            # Upload observations since last contact
            observations = get_agent_observations_since_sync(agent, gs_state.agent_last_sync[agent_id], t)
            println("📊 Agent $(agent_id) uploading $(length(observations)) observations since last sync (t=$(gs_state.agent_last_sync[agent_id]))")
            update_global_belief!(gs_state.global_belief, observations, env)
            
            # Calculate contact horizon (steps until next sync)
            C_i = calculate_contact_horizon(agent, t, env)
            println("⏰ Contact horizon for agent $(agent_id): $(C_i) steps")
            
            # Get other agents' current plans
            other_plans = get_other_agent_plans(gs_state, agent_id)
            
            # Compute new plan based on planning mode
            if planning_mode == :script
                println("📋 Computing macro-script for agent $(agent_id)")
                new_plan = MacroPlannerAsync.best_script(env, gs_state.global_belief, agent, C_i, other_plans, rng=rng)
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
            
            # Store plan in ground station state (agents don't store plans directly)
            # The plan will be executed by the ground station when needed
            
            # Update last sync time
            gs_state.agent_last_sync[agent_id] = t
            
            println("✅ Agent $(agent_id) synchronized with new plan")
        end
    end
    
    # Update ground station time
    gs_state.time_step = t
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
    
    # Get current belief probabilities
    current_probabilities = copy(global_belief.event_probabilities)
    height, width = size(current_probabilities)
    
    # Process observations chronologically (they should be in order from get_agent_observations_since_sync)
    for (obs_idx, observation) in enumerate(observations)
        agent_id = observation.agent_id
        sensed_cells = observation.sensed_cells
        event_states = observation.event_states
        
        println("  📡 Processing observation $(obs_idx): Agent $(agent_id) observed $(length(sensed_cells)) cells")
        
        # First, apply transition model to ALL cells (simulate environment evolution)
        # This represents what happens between observations
        new_probabilities = similar(current_probabilities)
        
        for y in 1:height
            for x in 1:width
                # Get neighbor beliefs for transition model
                neighbor_beliefs = get_neighbor_beliefs(current_probabilities, x, y)
                
                # Apply DBN transition model to predict next belief
                new_probabilities[y, x] = predict_next_belief_dbn(
                    current_probabilities[y, x], 
                    neighbor_beliefs, 
                    dbn_model
                )
            end
        end
        
        # Update current probabilities with transition results
        current_probabilities = new_probabilities
        
        # Now apply Bayesian updating for observed cells
        for (i, cell) in enumerate(sensed_cells)
            if i <= length(event_states)
                event_state = event_states[i]
                x, y = cell
                
                if 1 <= x <= width && 1 <= y <= height
                    # Get prior probability (after transition)
                    prior_prob = current_probabilities[y, x]
                    
                    # Apply Bayesian updating based on observation
                    if event_state == EVENT_PRESENT
                        # P(event|observation) = P(observation|event) * P(event) / P(observation)
                        # Assuming P(observation|event) = 0.9 and P(observation|no_event) = 0.1
                        likelihood_event = 1
                        likelihood_no_event = 0
                        posterior = (likelihood_event * prior_prob) / (likelihood_event * prior_prob + likelihood_no_event * (1 - prior_prob))
                        current_probabilities[y, x] = posterior
                        println("    🎯 Cell ($(x),$(y)): Event observed, updated from $(round(prior_prob, digits=3)) to $(round(posterior, digits=3))")
                    elseif event_state == NO_EVENT
                        # P(no_event|observation) = P(observation|no_event) * P(no_event) / P(observation)
                        likelihood_event = 0
                        likelihood_no_event = 1
                        posterior = (likelihood_no_event * (1 - prior_prob)) / (likelihood_event * prior_prob + likelihood_no_event * (1 - prior_prob))
                        current_probabilities[y, x] = 1.0 - posterior
                        println("    🎯 Cell ($(x),$(y)): No event observed, updated from $(round(prior_prob, digits=3)) to $(round(1.0 - posterior, digits=3))")
                    end
                end
            end
        end
    end
    
    # Update the global belief with final probabilities
    global_belief.event_probabilities = current_probabilities
    
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
function initialize_ground_station(env, agents)
    # Initialize global belief
    global_belief = initialize_global_belief(env)
    
    # Initialize agent tracking
    agent_last_sync = Dict{Int, Int}()
    agent_plans = Dict{Int, Any}()
    agent_plan_types = Dict{Int, Symbol}()
    
    for agent in agents
        agent_last_sync[agent.id] = -1  # No sync yet
        agent_plans[agent.id] = nothing
        agent_plan_types[agent.id] = :script
    end
    
    return GroundStationState(global_belief, agent_last_sync, agent_plans, agent_plan_types, 0)
end

"""
Initialize global belief
"""
function initialize_global_belief(env)
    # Initialize with uniform prior probability using proper Belief type
    # In a real system, this could be based on historical data or domain knowledge
    prior_probability = 0.1  # Low prior probability for events (most cells are empty)
    return BeliefManagement.initialize_belief(env.width, env.height, prior_probability)
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

end # module 
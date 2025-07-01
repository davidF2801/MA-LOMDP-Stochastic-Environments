module GroundStation

using POMDPs
using POMDPTools
using Random
using ..Types
using ..Agents

# Import planner modules
include("macro_planner.jl")
include("policy_tree_planner.jl")

using .MacroPlanner
using .PolicyTreePlanner

# Import Agent type from TrajectoryPlanner
include("../agents/trajectory_planner.jl")
using .TrajectoryPlanner: Agent, get_position_at_time

# Import types from the parent module (Planners)
import ..Types.EventState, ..Types.NO_EVENT, ..Types.EVENT_PRESENT
import ..Types.SensingAction, ..Types.GridObservation, ..Types.Agent
import ..Types.CircularTrajectory, ..Types.LinearTrajectory, ..Types.RangeLimitedSensor
import ..Types.EventState2, ..Types.NO_EVENT_2, ..Types.EVENT_PRESENT_2
import ..EventDynamicsModule
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
            update_global_belief!(gs_state.global_belief, observations, env)
            
            # Calculate contact horizon (steps until next sync)
            C_i = calculate_contact_horizon(agent, t, env)
            println("⏰ Contact horizon for agent $(agent_id): $(C_i) steps")
            
            # Get other agents' current plans
            other_plans = get_other_agent_plans(gs_state, agent_id)
            
            # Compute new plan based on planning mode
            if planning_mode == :script
                println("📋 Computing macro-script for agent $(agent_id)")
                new_plan = MacroPlanner.best_script(env, gs_state.global_belief, agent, C_i, other_plans, rng=rng)
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
    # For now, return empty observations
    # In a full implementation, this would retrieve stored observations
    return GridObservation[]
end

"""
Update global belief with new observations
"""
function update_global_belief!(global_belief, observations::Vector{GridObservation}, env)
    # For now, do nothing
    # In a full implementation, this would update the global belief using Bayes rule
    println("🔄 Updating global belief with $(length(observations)) observations")
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
    # For now, return a simple belief structure
    # In a full implementation, this would initialize a proper belief state
    return Dict("probabilities" => fill(0.1, env.height, env.width))
end

"""
Get agent's current position
"""
function get_agent_position(agent, env)
    # For now, return a default position
    # In a full implementation, this would get the agent's actual position
    return (1, 1)
end

"""
Execute agent's current plan
"""
function execute_plan(agent, gs_state::GroundStationState, local_obs_history::Vector{GridObservation})
    agent_id = agent.id
    
    # Get plan from ground station state
    plan = get(gs_state.agent_plans, agent_id, nothing)
    plan_type = get(gs_state.agent_plan_types, agent_id, :script)
    
    if plan === nothing
        # No plan available, use default wait action
        return SensingAction(agent.id, Tuple{Int, Int}[], false)
    end
    
    if plan_type == :script
        # Execute macro-script (open-loop)
        # For now, just return the first action in the script
        if !isempty(plan)
            return plan[1]
        else
            # Script empty, use wait action
            return SensingAction(agent.id, Tuple{Int, Int}[], false)
        end
        
    elseif plan_type == :policy
        # Execute policy tree (closed-loop)
        action = get_action_from_tree(plan, local_obs_history)
        if action === nothing
            # No policy found, use wait action
            return SensingAction(agent.id, Tuple{Int, Int}[], false)
        else
            return action
        end
        
    else
        error("Unknown plan type: $(plan_type)")
    end
end

"""
Get action from policy tree based on observation history
"""
function get_action_from_tree(tree, obs_history::Vector{GridObservation})
    # Try to find exact match
    if haskey(tree, obs_history)
        return tree[obs_history]
    end
    
    # Try to find partial match (use longest matching prefix)
    for i in length(obs_history)-1:-1:0
        prefix = obs_history[1:i]
        if haskey(tree, prefix)
            return tree[prefix]
        end
    end
    
    return nothing
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
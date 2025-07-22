"""
TrajectoryPlanner - Manages deterministic periodic trajectories for agents
"""
module TrajectoryPlanner

using POMDPs
using POMDPTools
using Distributions
using Random
using ..Types

# Import types from the parent module
import ..Types.Trajectory, ..Types.CircularTrajectory, ..Types.LinearTrajectory, ..Types.RangeLimitedSensor, ..Types.GridObservation

export get_position_at_time, calculate_trajectory_period, execute_plan, get_action_from_tree

"""
get_position_at_time(trajectory::CircularTrajectory, time::Int)
Gets agent position at a specific time for circular trajectory
"""
function get_position_at_time(trajectory::CircularTrajectory, time::Int)
    # TODO: Implement circular trajectory position calculation
    # - Calculate angle based on time and period
    # - Convert to x,y coordinates
    
    angle = 2π * (time % trajectory.period) / trajectory.period
    x = trajectory.center_x + round(Int, trajectory.radius * cos(angle))
    y = trajectory.center_y + round(Int, trajectory.radius * sin(angle))
    
    return (x, y)
end

"""
get_position_at_time(trajectory::LinearTrajectory, time::Int)
Gets agent position at a specific time for linear trajectory
"""
function get_position_at_time(trajectory::LinearTrajectory, time::Int)
    step = (time % trajectory.period)
    n_steps = trajectory.period - 1
    x = round(Int, trajectory.start_x + step * (trajectory.end_x - trajectory.start_x) / n_steps)
    y = round(Int, trajectory.start_y + step * (trajectory.end_y - trajectory.start_y) / n_steps)
    return (x, y)
end

"""
get_position_at_time(trajectory::LinearTrajectory, time::Int, phase_offset::Int)
Gets agent position at a specific time for linear trajectory with phase offset
"""
function get_position_at_time(trajectory::LinearTrajectory, time::Int, phase_offset::Int)
    # Apply phase offset to time
    adjusted_time = time + phase_offset
    return get_position_at_time(trajectory, adjusted_time)
end

"""
calculate_trajectory_period(trajectory::Trajectory)
Calculates the period of a trajectory
"""
function calculate_trajectory_period(trajectory::CircularTrajectory)
    return trajectory.period
end

function calculate_trajectory_period(trajectory::LinearTrajectory)
    return trajectory.period
end

# """
# update_agent_position!(agent::Agent, time::Int)
# Updates agent position based on current time
# """
# function update_agent_position!(agent::Agent, time::Int)
#     # TODO: Implement position update
#     # Note: Agent now uses phase_offset instead of current_time
#     # Position is calculated dynamically from trajectory and time
# end

"""
get_trajectory_waypoints(trajectory::Trajectory, num_points::Int)
Gets waypoints along the trajectory for visualization
"""
function get_trajectory_waypoints(trajectory::CircularTrajectory, num_points::Int)
    # TODO: Implement waypoint calculation for circular trajectory
    waypoints = Tuple{Int, Int}[]
    
    for i in 0:num_points-1
        time = round(Int, i * trajectory.period / num_points)
        push!(waypoints, get_position_at_time(trajectory, time))
    end
    
    return waypoints
end

function get_trajectory_waypoints(trajectory::LinearTrajectory, num_points::Int)
    # TODO: Implement waypoint calculation for linear trajectory
    waypoints = Tuple{Int, Int}[]
    
    for i in 0:num_points-1
        time = round(Int, i * trajectory.period / num_points)
        push!(waypoints, get_position_at_time(trajectory, time))
    end
    
    return waypoints
end

"""
create_circular_trajectory(center_x::Int, center_y::Int, radius::Float64, period::Int)
Creates a circular trajectory
"""
function create_circular_trajectory(center_x::Int, center_y::Int, radius::Float64, period::Int)
    return CircularTrajectory(center_x, center_y, radius, period)
end

"""
create_linear_trajectory(start_x::Int, start_y::Int, end_x::Int, end_y::Int, period::Int)
Creates a linear trajectory
"""
function create_linear_trajectory(start_x::Int, start_y::Int, end_x::Int, end_y::Int, period::Int)
    return LinearTrajectory(start_x, start_y, end_x, end_y, period)
end

"""
get_action_from_tree(policy_tree, local_obs_history::Vector{GridObservation})
Gets the appropriate action from a policy tree based on observation history
"""
function get_action_from_tree(policy_tree, local_obs_history::Vector{GridObservation})
    # Traverse the policy tree based on observation history
    current_node = policy_tree
    
    # Follow the tree based on recent observations
    for obs in local_obs_history
        # Find the child node that matches this observation
        matching_child = nothing
        
        for (child_obs, child_node) in current_node.children
            # Check if this child's observation matches our observation
            if observations_match(child_obs, obs)
                matching_child = child_node
                break
            end
        end
        
        if matching_child !== nothing
            current_node = matching_child
        else
            # No matching child found, stay at current node
            break
        end
    end
    
    # Return the action at the current node
    return current_node.action
end

"""
observations_match(tree_obs::Vector{Tuple{Tuple{Int, Int}, EventState}}, actual_obs::GridObservation)
Check if the tree observation matches the actual observation
"""
function observations_match(tree_obs::Vector{Tuple{Tuple{Int, Int}, EventState}}, actual_obs::GridObservation)
    # Convert actual observation to the same format as tree observations
    actual_obs_formatted = Vector{Tuple{Tuple{Int, Int}, EventState}}()
    
    for (i, cell) in enumerate(actual_obs.sensed_cells)
        if i <= length(actual_obs.event_states)
            push!(actual_obs_formatted, (cell, actual_obs.event_states[i]))
        end
    end
    
    # Check if the observations match
    if length(tree_obs) != length(actual_obs_formatted)
        return false
    end
    
    # Sort both observations to ensure order doesn't matter
    sorted_tree_obs = sort(tree_obs, by = x -> x[1])
    sorted_actual_obs = sort(actual_obs_formatted, by = x -> x[1])
    
    for (tree_obs_item, actual_obs_item) in zip(sorted_tree_obs, sorted_actual_obs)
        if tree_obs_item != actual_obs_item
            return false
        end
    end
    
    return true
end

"""
execute_plan(agent::Agent, plan, plan_type::Symbol, local_obs_history::Vector{GridObservation})
Execute agent's current plan and return the next action to take
"""
function execute_plan(agent::Agent, plan, plan_type::Symbol, local_obs_history::Vector{GridObservation})
    agent_id = agent.id
    
    if plan === nothing
        # No plan available, use default wait action
        return SensingAction(agent_id, Tuple{Int, Int}[], false)
    end
    
    if plan_type == :script || plan_type == :random || plan_type == :future_actions || plan_type == :sweep || plan_type == :greedy || plan_type == :macro_approx
        # Execute macro-script (open-loop), random sequence, future actions sequence, sweep sequence, greedy sequence, or macro-approximate sequence
        if !isempty(plan)
            # Get the action at the current plan index
            if agent.plan_index <= length(plan)
                action = plan[agent.plan_index]
                # Increment plan index for next execution
                agent.plan_index += 1
                return action
            else
                # Plan exhausted, use wait action
                return SensingAction(agent_id, Tuple{Int, Int}[], false)
            end
        else
            # Script empty, use wait action
            return SensingAction(agent_id, Tuple{Int, Int}[], false)
        end
        
    elseif plan_type == :policy
        # Execute policy tree (closed-loop)
        action = get_action_from_tree(plan, local_obs_history)
        if action === nothing
            # No policy found, use wait action
            return SensingAction(agent_id, Tuple{Int, Int}[], false)
        else
            return action
        end
        
    else
        error("Unknown plan type: $(plan_type)")
    end
end

end # module 
"""
TrajectoryPlanner - Manages deterministic periodic trajectories for agents
"""
module TrajectoryPlanner

using POMDPs
using POMDPTools
using Distributions
using Random
using ..Types
using Infiltrator
# Import types from the parent module
import ..Types.Trajectory, ..Types.CircularTrajectory, ..Types.LinearTrajectory, ..Types.ComplexTrajectory, ..Types.RangeLimitedSensor, ..Types.GridObservation
# Import battery management functions
import ..Types.update_battery!, ..Types.check_battery_feasible

export get_position_at_time, calculate_trajectory_period, execute_plan, get_action_from_tree

"""
get_position_at_time(trajectory::CircularTrajectory, time::Int)
Gets agent position at a specific time for circular trajectory
"""
function get_position_at_time(trajectory::CircularTrajectory, time::Int)
    # Calculate angle based on time and period
    # Phase 0 is aligned with the Ground Station at (5,5)
    
    # Calculate the angle from circle center to Ground Station (5,5)
    gs_x, gs_y = 5, 5  # Ground Station position
    gs_angle = atan(gs_y - trajectory.center_y, gs_x - trajectory.center_x)
    # Calculate the trajectory angle, starting from GS direction at phase 0
    angle = gs_angle + 2π * mod(time, trajectory.period) / trajectory.period
    x = trajectory.center_x + round(Int, trajectory.radius * cos(angle))
    y = trajectory.center_y + round(Int, trajectory.radius * sin(angle))
    
    return (x, y)
end

"""
get_position_at_time(trajectory::CircularTrajectory, time::Int, phase_offset::Int)
Gets agent position at a specific time for circular trajectory with phase offset
"""
function get_position_at_time(trajectory::CircularTrajectory, time::Int, phase_offset::Int)
    # Apply phase offset to the time
    # Phase 0 is aligned with the Ground Station at (5,5)
    
    # Calculate the angle from circle center to Ground Station (5,5)
    gs_x, gs_y = 5, 5  # Ground Station position
    gs_angle = atan(gs_y - trajectory.center_y, gs_x - trajectory.center_x)
    
    # Calculate the trajectory angle, starting from GS direction at phase 0
    adjusted_time = mod((time + phase_offset), trajectory.period)
    angle = gs_angle + 2π * adjusted_time / trajectory.period
    x = trajectory.center_x + round(Int, trajectory.radius * cos(angle))
    y = trajectory.center_y + round(Int, trajectory.radius * sin(angle))
    
    return (x, y)
end

"""
get_position_at_time(trajectory::LinearTrajectory, time::Int)
Gets agent position at a specific time for linear trajectory
"""
function get_position_at_time(trajectory::LinearTrajectory, time::Int)
    step = mod(time,trajectory.period)
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
get_position_at_time(trajectory::ComplexTrajectory, time::Int)
Gets agent position at a specific time for complex trajectory
"""
function get_position_at_time(trajectory::ComplexTrajectory, time::Int)
    # Calculate which waypoint we're at based on time
    waypoint_index = (time % trajectory.period) + 1
    if waypoint_index > length(trajectory.waypoints)
        waypoint_index = 1  # Wrap around
    end
    return trajectory.waypoints[waypoint_index]
end

"""
get_position_at_time(trajectory::ComplexTrajectory, time::Int, phase_offset::Int)
Gets agent position at a specific time for complex trajectory with phase offset
"""
function get_position_at_time(trajectory::ComplexTrajectory, time::Int, phase_offset::Int)
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

function calculate_trajectory_period(trajectory::ComplexTrajectory)
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

function get_trajectory_waypoints(trajectory::ComplexTrajectory, num_points::Int)
    # For complex trajectory, return the actual waypoints
    return copy(trajectory.waypoints)
end

"""
create_circular_trajectory(center_x, center_y, radius::Float64, period::Int)
Creates a circular trajectory
"""
function create_circular_trajectory(center_x, center_y, radius::Float64, period::Int)
    # Convert center coordinates to integers
    center_x_int = round(Int, center_x)
    center_y_int = round(Int, center_y)
    return CircularTrajectory(center_x_int, center_y_int, radius, period, 1.0)  # Default step_size = 1.0
end

"""
create_linear_trajectory(start_x::Int, start_y::Int, end_x::Int, end_y::Int, period::Int)
Creates a linear trajectory
"""
function create_linear_trajectory(start_x::Int, start_y::Int, end_x::Int, end_y::Int, period::Int)
    return LinearTrajectory(start_x, start_y, end_x, end_y, period, 1.0)  # Default step_size = 1.0
end

"""
create_complex_trajectory(waypoints::Vector{Tuple{Int, Int}}, period::Int)
Creates a complex trajectory with multiple waypoints
"""
function create_complex_trajectory(waypoints::Vector{Tuple{Int, Int}}, period::Int)
    return ComplexTrajectory(waypoints, period, 1.0)
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
execute_plan(agent::Agent, plan, plan_type::Symbol, local_obs_history::Vector{GridObservation}, current_time::Int)
Execute agent's current plan and return the next action to take.
Note: Charging happens in the main simulation loop, not here
"""
function execute_plan(agent::Agent, plan, plan_type::Symbol, local_obs_history::Vector{GridObservation}, current_time::Int)
    agent_id = agent.id
    # DEBUG: Always print what plan_type we have
    if current_time % 5 == 0 || plan_type == :oracle
        println("📋 execute_plan called: t=$(current_time), agent=$(agent_id), plan_type=$(plan_type)")
    end
    
    # ORACLE: Make decision at EVERY timestep based on current ground truth!
    if plan_type == :oracle
        println("🔮 ORACLE at t=$(current_time) agent $(agent_id)")
        # Get environment reference
        if !hasproperty(agent, :env_ref) || agent.env_ref === nothing
            println("  ❌ ERROR: No env_ref")
            return SensingAction(agent_id, Tuple{Int, Int}[], false)
        end
        env = agent.env_ref
        
        # Get CURRENT ground truth RIGHT NOW
        if !hasproperty(env, :current_state) || env.current_state === nothing
            println("  ❌ ERROR: No current_state in env")
            return SensingAction(agent_id, Tuple{Int, Int}[], false)
        end
        
        ground_truth = env.current_state
        total_events = Base.count(==(Types.EVENT_PRESENT), ground_truth)
        
        # Get agent position NOW
        pos = get_position_at_time(agent.trajectory, current_time, agent.phase_offset)
        
        # Get field of regard NOW
        for_cells = get_oracle_field_of_regard(agent, pos, env)
        # Initialize observation tracking if not exists
        if agent.oracle_obs_history === nothing
            agent.oracle_obs_history = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()  # cell -> (count, last_time)
        end
        
        # Two-cell contiguous mode: only when enabled (does not affect main.jl / main_5x5)
        use_two_cell = hasproperty(env, :max_sensing_targets) && env.max_sensing_targets >= 2 && Types.CONTIGUOUS_PAIRS_ONLY[]
        if use_two_cell
            pairs = Types.contiguous_pairs(for_cells)
            chosen_pair = nothing
            if !isempty(pairs)
                # Score each pair: (event_count_in_pair, -total_obs_count, time_since_obs) -> prefer more events, least observed, longest since
                pair_scores = []
                for pair in pairs
                    event_count = sum((1 for c in pair if ground_truth[c[2], c[1]] == Types.EVENT_PRESENT); init=0)
                    total_obs = 0
                    time_since = typemax(Int)
                    for c in pair
                        if haskey(agent.oracle_obs_history, c)
                            obs_count, last_t = agent.oracle_obs_history[c]
                            total_obs += obs_count
                            time_since = min(time_since, current_time - last_t)
                        else
                            time_since = min(time_since, typemax(Int))
                        end
                    end
                    push!(pair_scores, (pair, event_count, total_obs, time_since == typemax(Int) ? -1 : time_since))
                end
                # Prefer pairs with at least one event; then least total_obs; then longest time_since
                filter!(ps -> ps[2] > 0, pair_scores)
                if !isempty(pair_scores)
                    sort!(pair_scores, by = x -> (-x[2], x[3], -x[4]))  # event_count desc, total_obs asc, time_since desc
                    best_obs = pair_scores[1][3]
                    best_time = pair_scores[1][4]
                    tied = [ps[1] for ps in pair_scores if ps[2] == pair_scores[1][2] && ps[3] == best_obs && ps[4] == best_time]
                    chosen_pair = rand(tied)
                end
            end
            if chosen_pair !== nothing
                for c in chosen_pair
                    if haskey(agent.oracle_obs_history, c)
                        obs_count, _ = agent.oracle_obs_history[c]
                        agent.oracle_obs_history[c] = (obs_count + 1, current_time)
                    else
                        agent.oracle_obs_history[c] = (1, current_time)
                    end
                end
                println("  🎯 Agent $(agent_id) observing contiguous pair $(chosen_pair)")
                agent.battery_level = max(0.0, agent.battery_level - agent.observation_cost * 2)
                return SensingAction(agent_id, collect(chosen_pair), false)
            else
                println("  ⏸️  Agent $(agent_id) waiting - no events in FOR (two-cell mode)")
                return SensingAction(agent_id, Tuple{Int, Int}[], false)
            end
        end
        
        # Find ALL events in FOR (single-cell mode)
        event_cells = Tuple{Int, Int}[]
        for cell in for_cells
            x, y = cell
            if ground_truth[y, x] == Types.EVENT_PRESENT
                push!(event_cells, cell)
            end
        end
        # Choose best cell using priority rule
        chosen_cell = nothing
        if !isempty(event_cells)
            # Apply priority rule directly here - no function call
            if length(event_cells) == 1
                chosen_cell = event_cells[1]
            else
                # Calculate scores: (cell, obs_count, time_since_last_obs)
                cell_scores = []
                for cell in event_cells
                    if haskey(agent.oracle_obs_history, cell)
                        obs_count, last_obs_time = agent.oracle_obs_history[cell]
                        time_since_obs = current_time - last_obs_time
                    else
                        obs_count = 0
                        time_since_obs = typemax(Int)
                    end
                    push!(cell_scores, (cell, obs_count, time_since_obs))
                end
                
                # Sort: least observed → longest since obs
                sort!(cell_scores, by = x -> (x[2], -x[3]))
                
                # Find all tied cells
                best_obs_count = cell_scores[1][2]
                best_time_since = cell_scores[1][3]
                tied_cells = [x[1] for x in cell_scores if x[2] == best_obs_count && x[3] == best_time_since]
                
                # Pick randomly from tied cells
                chosen_cell = rand(tied_cells)
            end
            
            # Update observation history
            if haskey(agent.oracle_obs_history, chosen_cell)
                obs_count, _ = agent.oracle_obs_history[chosen_cell]
                agent.oracle_obs_history[chosen_cell] = (obs_count + 1, current_time)
            else
                agent.oracle_obs_history[chosen_cell] = (1, current_time)
            end
            
            println("  🎯 Agent $(agent_id) observing event at $(chosen_cell) ($(length(event_cells)) events available)")
        else
            println("  ⏸️  Agent $(agent_id) waiting - no events in FOR ($(length(for_cells)) cells, $(total_events) total events)")
        end
        
        # Return action
        if chosen_cell !== nothing
            action = SensingAction(agent_id, [chosen_cell], false)
            # Note: Battery check removed per user request - battery doesn't matter
            # Update battery for tracking (charging happens in main loop)
            agent.battery_level = max(0.0, agent.battery_level - agent.observation_cost)
            return action
        else
            return SensingAction(agent_id, Tuple{Int, Int}[], false)
        end
    end
    
    if plan === nothing && plan_type != :policy && plan_type != :pbvi_policy_tree && plan_type != :pomcp_online && plan_type != :dec_sb_abba_online && plan_type != :do_sb_abba_online
        # No plan available, use default wait action (pomcp_online uses reactive_policy, not plan)
        return SensingAction(agent_id, Tuple{Int, Int}[], false)
    end
    if plan_type == :script || plan_type == :joint_abba || plan_type == :random || plan_type == :future_actions || plan_type == :sweep || plan_type == :greedy || plan_type == :macro_approx || plan_type == :macro_approx_099 || plan_type == :macro_approx_095 || plan_type == :macro_approx_090 || plan_type == :prior_based || plan_type == :pbvi || plan_type == :pbvi_rollout || plan_type == :pbvi_mis || plan_type == :mpomdp_openloop || plan_type == :pomcp || plan_type == :klolop || plan_type == :posts
        # Execute macro-script (open-loop), random, sweep, greedy, prior-based, PBVI, PBVI+MIS, MPOMDP, POMCP, or KL-OLOP sequence
        if !isempty(plan)
            # Get the action at the current plan index
            if agent.plan_index <= length(plan)
                planned_action = plan[agent.plan_index]
                
                # Note: Battery check removed per user request - battery doesn't matter
                # Execute the planned action (planners now generate feasible actions)
                num_observations = length(planned_action.target_cells)
                # Discharge for observations (charging happens in main loop)
                total_cost = agent.observation_cost * num_observations
                agent.battery_level = max(0.0, agent.battery_level - total_cost)
                agent.plan_index += 1
                return planned_action
            else
                # Plan exhausted, use wait action
                return SensingAction(agent_id, Tuple{Int, Int}[], false)
            end
        else
            # Script empty, use wait action
            return SensingAction(agent_id, Tuple{Int, Int}[], false)
        end
        
    elseif plan_type == :policy || plan_type == :pbvi_policy_tree || plan_type == :pomcp_online || plan_type == :dec_sb_abba_online || plan_type == :do_sb_abba_online
        # Execute reactive policy (closed-loop) for other policy-based planners
        if agent.reactive_policy !== nothing
            # Use the reactive policy function directly
            # Pass the current time to the reactive policy
            planned_action = agent.reactive_policy(local_obs_history, current_time)
        else
            # Fallback to old policy tree method (only if plan is not nothing)
            if plan !== nothing
                planned_action = get_action_from_tree(plan, local_obs_history)
            else
                planned_action = nothing
            end
        end
        
        if planned_action === nothing
            # No policy found, use wait action
            return SensingAction(agent_id, Tuple{Int, Int}[], false)
        else
            # Note: Battery check removed per user request - battery doesn't matter
            # Execute the planned action (planners now generate feasible actions)
            num_observations = length(planned_action.target_cells)
            # Discharge for observations (charging happens in main loop)
            total_cost = agent.observation_cost * num_observations
            agent.battery_level = max(0.0, agent.battery_level - total_cost)
            return planned_action
        end
        
    else
        error("Unknown plan type: $(plan_type)")
    end
end

"""
Select best event cell to observe based on priority rule:
1. Least observed (lowest observation count)
2. If tie: longest time since last observation
3. If still tie: random uniform
"""
function select_best_event_cell(event_cells::Vector{Tuple{Int, Int}}, 
                                obs_history::Dict{Tuple{Int, Int}, Tuple{Int, Int}},
                                current_time::Int)
    if length(event_cells) == 1
        return event_cells[1]
    end
    
    # Calculate scores for each cell: (cell, obs_count, time_since_last_obs)
    cell_scores = []
    for cell in event_cells
        if haskey(obs_history, cell)
            obs_count, last_obs_time = obs_history[cell]
            time_since_obs = current_time - last_obs_time
        else
            # Never observed - highest priority
            obs_count = 0
            time_since_obs = typemax(Int)  # Infinite time
        end
        push!(cell_scores, (cell, obs_count, time_since_obs))
    end
    
    # Sort by: 1) least observed (ascending), 2) longest since last obs (descending)
    sort!(cell_scores, by = x -> (x[2], -x[3]))
    
    # Find all cells with same best score
    best_obs_count = cell_scores[1][2]
    best_time_since = cell_scores[1][3]
    
    tied_cells = [x[1] for x in cell_scores if x[2] == best_obs_count && x[3] == best_time_since]
    
    # Break tie randomly
    return rand(tied_cells)
end

"""
Get ground truth for oracle (helper function in trajectory planner)
"""
function get_oracle_ground_truth(env)
    height, width = env.height, env.width
    ground_truth = Matrix{Int}(undef, height, width)
    
    # Get current state from environment
    if hasproperty(env, :current_state) && env.current_state !== nothing
        for y in 1:height, x in 1:width
            cell_state = env.current_state[y, x]
            if cell_state == Types.NO_EVENT
                ground_truth[y, x] = 1
            elseif cell_state == Types.EVENT_PRESENT
                ground_truth[y, x] = 2
            else
                ground_truth[y, x] = 1
            end
        end
    else
        error("Oracle Error: env.current_state is not available!")
    end
    
    return ground_truth
end

"""
Get field of regard for oracle (helper function in trajectory planner)
"""
function get_oracle_field_of_regard(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Check sensor pattern
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
    elseif agent.sensor.pattern == :circular
        for dx in -1:1, dy in -1:1
            nx, ny = x + dx, y + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                push!(fov_cells, (nx, ny))
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

end # module 
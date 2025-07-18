module MacroPlannerSweep

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Infiltrator
using ..Types
import ..Agents.BeliefManagement: sample_from_belief
# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..EventState2, ..NO_EVENT_2, ..EVENT_PRESENT_2
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..RangeLimitedSensor, ..EventMap
# Import trajectory functions
import ..Agents.TrajectoryPlanner.get_position_at_time
# Import belief management functions
import ..Agents.BeliefManagement
import ..Agents.BeliefManagement.predict_belief_evolution_dbn, ..Agents.BeliefManagement.Belief,
       ..Agents.BeliefManagement.calculate_uncertainty_from_distribution, ..Agents.BeliefManagement.predict_belief_rsp,
       ..Agents.BeliefManagement.evolve_no_obs, ..Agents.BeliefManagement.get_neighbor_beliefs,
       ..Agents.BeliefManagement.enumerate_joint_states, ..Agents.BeliefManagement.product,
       ..Agents.BeliefManagement.normalize_belief_distributions, ..Agents.BeliefManagement.collapse_belief_to,
       ..Agents.BeliefManagement.enumerate_all_possible_outcomes, ..Agents.BeliefManagement.merge_equivalent_beliefs,
       ..Agents.BeliefManagement.beliefs_are_equivalent, ..Agents.BeliefManagement.calculate_cell_entropy, 
       ..Agents.BeliefManagement.get_event_probability, ..Agents.BeliefManagement.clear_belief_evolution_cache!, 
       ..Agents.BeliefManagement.get_cache_stats

export best_script, evaluate_action_sequence_exact, calculate_macro_script_reward

"""
best_script(env, belief::Belief, agent::Agent, C::Int, other_scripts, gs_state)::Vector{SensingAction}
  – Generate a systematic sweep sequence for the agent
  – Sweep over columns in the agent's row in a systematic pattern
  – Return the sweep sequence
"""
function best_script(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Start timing
    start_time = time()
    
    # Generate systematic sweep sequence
    sweep_sequence = generate_sweep_sequence(agent, env, C, gs_state)
    
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ Sweep sequence generated in $(round(planning_time, digits=3)) seconds")
    
    return sweep_sequence, planning_time
end

"""
Generate a systematic sweep sequence for the agent
"""
function generate_sweep_sequence(agent, env, C::Int, gs_state)
    if C == 0
        return SensingAction[]
    end
    
    # Get agent's row (y-coordinate) - use the actual position at time 0
    agent_pos = get_position_at_time(agent.trajectory, 0, agent.phase_offset)
    agent_row = agent_pos[2]
    
    # Get the actual number of columns and agent's trajectory period
    num_columns = env.width
    trajectory_period = agent.trajectory.period
    
    # Get the current global timestep when this planning is happening
    current_global_time = gs_state.time_step
    
    # Create different periodic sweep patterns for different agents to avoid conflicts
    # Each agent will sweep one column per period, cycling through columns
    if agent.id == 1
        # Agent 1: start from rightmost column, then leftmost, then work inward
        sweep_pattern = collect(num_columns:-1:1)  # Right to left
    elseif agent.id == 2
        # Agent 2: start from leftmost column, then rightmost, then work inward
        sweep_pattern = collect(1:num_columns)  # Left to right
    else
        # Other agents: alternate patterns to avoid conflicts
        if agent.id % 2 == 1
            # Odd agents: center outward pattern
            center = div(num_columns, 2) + 1
            sweep_pattern = [center]
            for offset in 1:max(center-1, num_columns-center)
                if center + offset <= num_columns
                    push!(sweep_pattern, center + offset)
                end
                if center - offset >= 1
                    push!(sweep_pattern, center - offset)
                end
            end
        else
            # Even agents: alternating from ends
            sweep_pattern = Int[]
            for i in 1:num_columns
                if i % 2 == 1
                    push!(sweep_pattern, div(i, 2) + 1)  # Start from left
                else
                    push!(sweep_pattern, num_columns - div(i, 2) + 1)  # Then from right
                end
            end
        end
    end
    
    # Create sweep sequence
    sweep_sequence = SensingAction[]
    
    println("🔍 DEBUG: Sweep sequence generation for agent $(agent.id)")
    println("  Current global time: $(current_global_time)")
    println("  Trajectory period: $(trajectory_period)")
    println("  Sweep pattern: $(sweep_pattern)")
    println("  Planning horizon C: $(C)")
    
    for t in 1:C
        # Calculate which period we're in based on the global timestep when this planning happens
        # Each period consists of trajectory_period timesteps
        global_timestep = current_global_time + t - 1  # Global timestep for this action
        period = div(global_timestep-agent.phase_offset, trajectory_period)  # Which period we're in (0-based)
        
        # Determine which column to sense for this period
        # Cycle through the sweep pattern based on the period
        pattern_index = mod(period, length(sweep_pattern)) + 1
        column_to_sense = sweep_pattern[pattern_index]
        
        # Get agent's actual position at this timestep
        agent_pos = get_position_at_time(agent.trajectory, global_timestep, agent.phase_offset)
        
        # Get field of regard at this position
        for_cells = get_field_of_regard_at_position(agent, agent_pos, env)
        
        # Check if the target column is in the field of regard
        target_cell = (column_to_sense, agent_pos[2])  # Use agent's actual row
        
        println("  Step $(t): global_timestep=$(global_timestep), period=$(period), pattern_index=$(pattern_index), column_to_sense=$(column_to_sense)")
        println("    Agent pos: $(agent_pos), target_cell: $(target_cell), in_for: $(target_cell in for_cells)")
        println("    Period calculation: div($(global_timestep), $(trajectory_period)) = $(period)")
        if target_cell in for_cells
            # Target cell is reachable, create sensing action
            action = SensingAction(agent.id, [target_cell], false)
        else
            # Target cell is not reachable, use wait action
            action = SensingAction(agent.id, Tuple{Int, Int}[], false)
        end
        
        push!(sweep_sequence, action)
    end
    
    println("🔄 Generated periodic sweep sequence for agent $(agent.id) in row $(agent_row):")
    println("  Current global time: $(current_global_time)")
    println("  Trajectory period: $(trajectory_period)")
    println("  Number of columns: $(num_columns)")
    println("  Sweep pattern: $(sweep_pattern) (one column per period)")
    println("  Sequence: $(length(sweep_sequence)) actions")
    println("  Pattern description: $(get_pattern_description(sweep_pattern))")
    println("  Sweep behavior:")
    for (i, col) in enumerate(sweep_pattern)
        period_start = (i-1) * trajectory_period
        period_end = i * trajectory_period - 1
        println("    Period $(i-1) (global steps $(period_start)-$(period_end)): observe column $(col)")
    end
    println("    Then cycles back to Period 0...")
    return sweep_sequence
end

"""
Get a human-readable description of the sweep pattern
"""
function get_pattern_description(pattern)
    if length(pattern) == 3
        if pattern == [3, 2, 1]
            return "Right → Center → Left"
        elseif pattern == [1, 2, 3]
            return "Left → Center → Right"
        elseif pattern == [2, 3, 1]
            return "Center → Right → Left"
        elseif pattern == [2, 1, 3]
            return "Center → Left → Right"
        end
    elseif length(pattern) == 4
        if pattern == [4, 3, 2, 1]
            return "Right → Center-right → Center-left → Left"
        elseif pattern == [1, 2, 3, 4]
            return "Left → Center-left → Center-right → Right"
        end
    end
    
    # For dynamic patterns, create a generic description
    if pattern[1] == maximum(pattern)
        return "Right to left sweep"
    elseif pattern[1] == minimum(pattern)
        return "Left to right sweep"
    elseif pattern[1] == div(length(pattern), 2) + 1
        return "Center outward sweep"
    else
        return "Alternating sweep: $(pattern)"
    end
end

"""
Evaluate a sweep sequence (simplified - just return a fixed value since sweep is deterministic)
"""
function evaluate_action_sequence_exact(env, belief₀, agent, seq, other_scripts, C, gs_state, rng::AbstractRNG)
    # For sweep strategy, we don't need complex evaluation
    # Just return a fixed value since the sequence is deterministic
    return 1.0
end

"""
Calculate reward for sweep sequence (simplified)
"""
function calculate_macro_script_reward(seq::Vector{SensingAction}, other_scripts, C::Int, env, agent, B_branches, gs_state)
    # For sweep strategy, return a simple reward based on sequence length
    # This is a baseline, so we use a simple heuristic
    return length(seq) * 0.1  # Simple reward proportional to sequence length
end

"""
Get field of regard for an agent at a specific position
"""
function get_field_of_regard_at_position(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Check if we want row-only visibility (sensor range = 0 means row-only)
    if agent.sensor.range == 0.0
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

end # module 
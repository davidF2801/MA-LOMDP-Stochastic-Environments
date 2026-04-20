module MacroPlannerGreedy

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Statistics
using ..Types
import ..Types: check_battery_feasible, simulate_battery_evolution
# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..ComplexTrajectory, ..RangeLimitedSensor
# Import trajectory functions
import ..Agents.TrajectoryPlanner.get_position_at_time
# Import belief management functions
import ..Agents.BeliefManagement
import ..Agents.BeliefManagement.Belief, ..Agents.BeliefManagement.evolve_no_obs_fast, 
       ..Agents.BeliefManagement.collapse_belief_to, ..Agents.BeliefManagement.calculate_cell_entropy, 
       ..Agents.BeliefManagement.get_event_probability, ..Agents.BeliefManagement.enumerate_all_possible_outcomes

export best_script

"""
best_script(env, belief::Belief, agent::Agent, C::Int, other_scripts, gs_state)::Vector{SensingAction}
  – Generate a simple greedy sequence for the agent
  – At each step, choose the action that maximizes entropy * event_probability
  – Much simpler than before - no complex branching or battery simulation
  – Return the greedy sequence
"""
function best_script(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Start timing
    start_time = time()
    
    # Generate simple greedy sequence using absolute timesteps from gs_state
    greedy_sequence = generate_simple_greedy_sequence(agent, env, C, belief, gs_state)
    
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ Simple greedy sequence generated in $(round(planning_time, digits=3)) seconds")
    
    return greedy_sequence, planning_time
end

"""
Generate a simple greedy sequence for the agent - much faster than before
Uses absolute timesteps from gs_state to ensure actions are feasible at execution time
"""
function generate_simple_greedy_sequence(agent, env, C::Int, belief::Belief, gs_state)
    if C == 0
        return SensingAction[]
    end
    
    greedy_sequence = SensingAction[]
    current_belief = deepcopy(belief)
    
    for t in 1:C
        # Calculate absolute timestep when this action will be executed
        global_timestep = gs_state.time_step + t - 1
        # Get agent's position at the actual execution time
        agent_pos = get_position_at_time(agent.trajectory, global_timestep, agent.phase_offset)
        
        # Get available cells based on sensor pattern
        available_cells = get_field_of_regard_at_position(agent, agent_pos, env)
        
        # Two-cell-only mode: only consider contiguous pairs (no single-cell) when CONTIGUOUS_PAIRS_ONLY and max_sensing_targets >= 2
        two_cell_only = env.max_sensing_targets >= 2 && Types.CONTIGUOUS_PAIRS_ONLY[]
        best_cell = nothing
        best_pair = nothing
        best_value = -Inf
        
        if two_cell_only && length(available_cells) > 1
            for pair in Types.contiguous_pairs(available_cells)
                pair_value = 0.0
                for c in pair
                    pair_value += calculate_cell_entropy(current_belief, c) * get_event_probability(current_belief, c)
                end
                if pair_value > best_value
                    best_value = pair_value
                    best_pair = pair
                end
            end
        else
            for cell in available_cells
                entropy = calculate_cell_entropy(current_belief, cell)
                event_prob = get_event_probability(current_belief, cell)
                greedy_value = entropy * event_prob
                if greedy_value > best_value
                    best_value = greedy_value
                    best_cell = cell
                end
            end
            if env.max_sensing_targets >= 2 && Types.CONTIGUOUS_PAIRS_ONLY[]
                for pair in Types.contiguous_pairs(available_cells)
                    pair_value = 0.0
                    for c in pair
                        pair_value += calculate_cell_entropy(current_belief, c) * get_event_probability(current_belief, c)
                    end
                    if pair_value > best_value
                        best_value = pair_value
                        best_cell = nothing
                        best_pair = pair
                    end
                end
            end
        end
        
        # Create action for the best cell/pair (or wait if none found)
        if best_pair !== nothing && best_value > 0.0
            action = SensingAction(agent.id, collect(best_pair), false)
            push!(greedy_sequence, action)
            for c in best_pair
                current_belief = collapse_belief_to(current_belief, c, EVENT_PRESENT)
            end
        elseif best_cell !== nothing && best_value > 0.0
            action = SensingAction(agent.id, [best_cell], false)
            push!(greedy_sequence, action)
            
            # Update belief with expected observation (assume EVENT_PRESENT for simplicity)
            current_belief = collapse_belief_to(current_belief, best_cell, EVENT_PRESENT)
        else
            # Wait action
            action = SensingAction(agent.id, Tuple{Int, Int}[], false)
            push!(greedy_sequence, action)
        end
        
        # Evolve belief for next timestep
        current_belief = evolve_no_obs_fast(current_belief, env, calculate_uncertainty=false)
    end
    
    println("🔄 Generated simple greedy sequence for agent $(agent.id):")
    println("  Agent position: starts at $(get_position_at_time(agent.trajectory, 0))")
    println("  Sequence: $(length(greedy_sequence)) actions")
    sensing_actions = count(a -> !isempty(a.target_cells), greedy_sequence)
    println("  Sensing actions: $(sensing_actions)/$(length(greedy_sequence))")
    
    return greedy_sequence
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

end # module 
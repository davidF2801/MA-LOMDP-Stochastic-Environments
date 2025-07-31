module MacroPlannerRandom

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
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..ComplexTrajectory, ..RangeLimitedSensor, ..EventMap
# Import trajectory functions
import ..Agents.TrajectoryPlanner.get_position_at_time

export best_script_random, generate_random_action_sequences

"""
best_script_random(env, belief::Belief, agent, C::Int, other_scripts, gs_state)::Vector{SensingAction}
  – Generate a random action sequence of length C for the agent
  – Each action is randomly selected from available actions at each timestep
  – Returns the random sequence
"""
function best_script_random(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Start timing
    start_time = time()
    
    println("🎲 Generating random action sequence for agent $(agent.id)...")
    
    # Generate random action sequence
    random_sequence = generate_random_action_sequences(agent, env, C, rng)
    
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ Random sequence generated in $(round(planning_time, digits=3)) seconds")
    println("  Sequence length: $(length(random_sequence))")
    println("  Actions: $([isempty(a.target_cells) ? "wait" : "sense$(a.target_cells)" for a in random_sequence])")
    
    return random_sequence, planning_time
end

"""
Generate random action sequences of length C considering agent trajectory
"""
function generate_random_action_sequences(agent, env, C::Int, rng::AbstractRNG)
    if C == 0
        return SensingAction[]
    end
    
    # 1. Propagate agent trajectory for C timesteps
    trajectory_positions = Vector{Tuple{Int, Int}}()
    for t in 0:(C-1)
        pos = get_position_at_time(agent.trajectory, t)
        push!(trajectory_positions, pos)
    end
    
    # 2. Generate random actions for each timestep
    random_sequence = SensingAction[]
    for t in 1:C
        pos = trajectory_positions[t]
        for_cells = get_field_of_regard_at_position(agent, pos, env)
        
        # Generate available actions for this timestep
        available_actions = SensingAction[]
        
        # Add wait action
        push!(available_actions, SensingAction(agent.id, Tuple{Int, Int}[], false))
        
        # Add single-cell sensing actions
        for cell in for_cells
            push!(available_actions, SensingAction(agent.id, [cell], false))
        end
        
        # Add multi-cell sensing actions (up to max_sensing_targets)
        if length(for_cells) > 1 && env.max_sensing_targets > 1
            for subset_size in 2:min(env.max_sensing_targets, length(for_cells))
                for subset in combinations(for_cells, subset_size)
                    push!(available_actions, SensingAction(agent.id, collect(subset), false))
                end
            end
        end
        
        # Randomly select one action from available actions
        if !isempty(available_actions)
            random_action = rand(rng, available_actions)
            push!(random_sequence, random_action)
        else
            # Fallback to wait action if no actions available
            push!(random_sequence, SensingAction(agent.id, Tuple{Int, Int}[], false))
        end
    end
    
    return random_sequence
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

"""
Generate combinations of elements
"""
function combinations(elements, k)
    if k == 0
        return [Tuple{}[]]
    elseif k == 1
        return [[element] for element in elements]
    else
        result = []
        for i in 1:length(elements)
            for combo in combinations(elements[i+1:end], k-1)
                push!(result, [elements[i]; combo])
            end
        end
        return result
    end
end

end # module 
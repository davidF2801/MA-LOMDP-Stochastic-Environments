module MacroPlannerPriorBased

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Infiltrator
using Distributions
using ..Types
import ..Agents.BeliefManagement: sample_from_belief
# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..EventState2, ..NO_EVENT_2, ..EVENT_PRESENT_2
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..ComplexTrajectory, ..RangeLimitedSensor, ..EventMap
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
  – Generate actions based on prior probability map without belief updates
  – Calculate initial event probabilities using ignition + contagion
  – Make decisions based on these static probabilities
"""
function best_script(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Start timing
    start_time = time()
    
    # Calculate prior probability map based on environment parameters (model-based approach)
    prior_prob_map = calculate_prior_probability_map(env)
    
    # Generate actions based on prior probabilities
    action_sequence = generate_prior_based_sequence(agent, env, C, gs_state, prior_prob_map)
    
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ Prior-based sequence generated in $(round(planning_time, digits=3)) seconds")
    
    return action_sequence, planning_time
end

"""
Calculate prior probability map using ignition probability + contagion from neighbors
"""
function calculate_prior_probability_map(env)
    height, width = env.height, env.width
    prior_map = Matrix{Float64}(undef, height, width)
    
    # Initialize with ignition probabilities
    for y in 1:height, x in 1:width
        # Get cell-specific RSP parameters
        cell_params = Types.get_cell_rsp_params(env.rsp_params, y, x)
        
        # Initial probability is based on spontaneous ignition (beta0) + external ignition (lambda)
        initial_prob = cell_params.beta0 + cell_params.lambda
        
        # Add contagion effect from neighbors
        neighbor_influence = calculate_neighbor_contagion_influence(env, x, y, cell_params.alpha)
        
        # Total prior probability (clamped to [0, 1])
        # Use a more sophisticated calculation that considers the RSP transition model
        total_prob = min(1.0, initial_prob + neighbor_influence)
        
        # Apply a transformation to make probabilities more meaningful for decision making
        # Higher probabilities should be more attractive for sensing
        prior_map[y, x] = total_prob
    end
    
    println("📊 Prior probability map calculated:")
    println("  Min probability: $(minimum(prior_map))")
    println("  Max probability: $(maximum(prior_map))")
    println("  Mean probability: $(mean(prior_map))")
    
    # Print some sample probabilities for debugging
    println("  Sample probabilities:")
    for y in 1:min(3, height), x in 1:min(3, width)
        println("    Cell ($(x), $(y)): $(round(prior_map[y, x], digits=4))")
    end
    
    return prior_map
end

"""
Calculate contagion influence from neighbors for prior probability
"""
function calculate_neighbor_contagion_influence(env, x::Int, y::Int, alpha::Float64)
    height, width = env.height, env.width
    neighbor_influence = 0.0
    
    # Check all 8 neighbors (including diagonals)
    for dx in -1:1, dy in -1:1
        if dx == 0 && dy == 0
            continue
        end
        
        nx, ny = x + dx, y + dy
        if 1 <= nx <= width && 1 <= ny <= height
            # Get neighbor's ignition probability
            neighbor_params = Types.get_cell_rsp_params(env.rsp_params, ny, nx)
            neighbor_prob = neighbor_params.beta0 + neighbor_params.lambda
            
            # Add weighted influence (alpha controls contagion strength)
            neighbor_influence += alpha * neighbor_prob
        end
    end
    
    return neighbor_influence
end

"""
Generate action sequence based on prior probabilities
"""
function generate_prior_based_sequence(agent, env, C::Int, gs_state, prior_prob_map)
    if C == 0
        return SensingAction[]
    end
    
    action_sequence = SensingAction[]
    
    println("🎯 Generating prior-based actions for agent $(agent.id)")
    println("  Planning horizon: $(C)")
    println("  Agent trajectory period: $(agent.trajectory.period)")
    
    for t in 1:C
        # Get agent's position at this timestep
        global_timestep = gs_state.time_step + t - 1
        agent_pos = get_position_at_time(agent.trajectory, global_timestep, agent.phase_offset)
        
        # Get field of regard at this position
        for_cells = get_field_of_regard_at_position(agent, agent_pos, env)
        
        # Sample action based on prior probabilities in field of regard
        chosen_cell = sample_action_from_prior_probabilities(for_cells, prior_prob_map)
        
        if chosen_cell !== nothing
            # Create sensing action for the chosen cell
            action = SensingAction(agent.id, [chosen_cell], false)
            prob = prior_prob_map[chosen_cell[2], chosen_cell[1]]
            println("  Step $(t): Agent at $(agent_pos), sensing cell $(chosen_cell) (prob: $(round(prob, digits=4)))")
        else
            # No suitable cell found, use wait action
            action = SensingAction(agent.id, Tuple{Int, Int}[], false)
            println("  Step $(t): Agent at $(agent_pos), no suitable cell found, waiting")
        end
        
        push!(action_sequence, action)
    end
    
    return action_sequence
end

"""
Sample action based on prior probabilities in field of regard
"""
function sample_action_from_prior_probabilities(for_cells::Vector{Tuple{Int, Int}}, prior_prob_map::Matrix{Float64})
    if isempty(for_cells)
        return nothing
    end
    
    # Extract probabilities for cells in field of regard
    cell_probs = Float64[]
    for cell in for_cells
        prob = prior_prob_map[cell[2], cell[1]]
        push!(cell_probs, prob)
    end
    
    # Normalize probabilities to sum to 1
    total_prob = sum(cell_probs)
    if total_prob <= 0
        # If all probabilities are zero, choose randomly
        return rand(for_cells)
    end
    
    # Apply a temperature parameter to control exploration vs exploitation
    # Higher temperature = more random, lower temperature = more greedy
    temperature = 0.5  # Adjust this to control randomness
    #scaled_probs = cell_probs .^ (1.0 / temperature)
    scaled_probs = cell_probs
    normalized_probs = cell_probs ./ sum(scaled_probs)
    
    # Sample cell based on probabilities
    chosen_idx = rand(Categorical(normalized_probs))
    return for_cells[chosen_idx]
end

"""
Evaluate a prior-based sequence (simplified - return fixed value)
"""
function evaluate_action_sequence_exact(env, belief₀, agent, seq, other_scripts, C, gs_state, rng::AbstractRNG)
    # For prior-based strategy, return a fixed value since it's deterministic
    return 1.0
end

"""
Calculate reward for prior-based sequence (simplified)
"""
function calculate_macro_script_reward(seq::Vector{SensingAction}, other_scripts, C::Int, env, agent, B_branches, gs_state)
    # For prior-based strategy, return a simple reward based on sequence length
    return length(seq) * 0.1
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
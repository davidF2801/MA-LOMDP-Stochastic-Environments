module MacroPlannerGreedy

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
  – Generate a greedy sequence for the agent
  – At each step, choose the action that maximizes entropy * event_probability
  – Ignore other agents' actions (fully greedy)
  – Return the greedy sequence
"""
function best_script(env, belief::Belief, agent, C::Int, other_scripts, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Start timing
    start_time = time()
    
    # Generate greedy sequence
    greedy_sequence = generate_greedy_sequence(agent, env, C, belief, gs_state)
    
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ Greedy sequence generated in $(round(planning_time, digits=3)) seconds")
    
    return greedy_sequence, planning_time
end

"""
Generate a greedy sequence for the agent with observation branching
"""
function generate_greedy_sequence(agent, env, C::Int, belief::Belief, gs_state)
    if C == 0
        return SensingAction[]
    end
    
    greedy_sequence = SensingAction[]
    
    # Start with the updated ground station belief (after incorporating agent observations)
    current_belief_branches = [(deepcopy(belief), 1.0)]
    
    for t in 1:C
        # Get agent's position at this timestep (t-1 because we're planning for timesteps 1 to C)
        agent_pos = get_position_at_time(agent.trajectory, t-1, agent.phase_offset)
        agent_row = agent_pos[2]
        # Get available cells in the agent's current row
        available_cells = Tuple{Int, Int}[]
        for col in 1:env.width
            push!(available_cells, (col, agent_row))
        end
        
        # Calculate greedy value for each cell by branching over observation outcomes
        best_cell = nothing
        best_value = -Inf
        
        for cell in available_cells
            # Calculate expected entropy for this cell by branching over observation outcomes
            expected_entropy = 0.0
            
            for (belief_branch, prob_branch) in current_belief_branches
                # Calculate entropy for this cell in this belief branch
                entropy = calculate_cell_entropy(belief_branch, cell)
                
                # Weight by the probability of this belief branch
                expected_entropy += prob_branch * entropy
            end
            
            # Calculate expected event probability for this cell
            expected_event_prob = 0.0
            for (belief_branch, prob_branch) in current_belief_branches
                event_prob = get_event_probability(belief_branch, cell)
                expected_event_prob += prob_branch * event_prob
            end
            
            # Greedy value: expected entropy * expected event probability
            greedy_value = expected_entropy * expected_event_prob
            
            if greedy_value > best_value
                best_value = greedy_value
                best_cell = cell
            end
        end
        
        # Create action for the best cell
        if best_cell !== nothing
            action = SensingAction(agent.id, [best_cell], false)
            push!(greedy_sequence, action)
            @infiltrate
            # Branch over observation outcomes for the selected action
            new_belief_branches = Vector{Tuple{Belief, Float64}}()
            
            for (belief_branch, prob_branch) in current_belief_branches
                # Create observation set for this action
                obs_set = [(agent.id, action)]
                
                # Branch over all possible observation outcomes
                for (observation_combo, probability) in enumerate_all_possible_outcomes(belief_branch, obs_set)
                    B_new = deepcopy(belief_branch)
                    
                    # Apply the observation
                    for (obs_cell, observed_state) in observation_combo
                        B_new = collapse_belief_to(B_new, obs_cell, observed_state)
                    end
                    
                    # Evolve belief for next timestep
                    B_next = evolve_no_obs(B_new, env)
                    
                    # Add to new branches with combined probability
                    push!(new_belief_branches, (B_next, prob_branch * probability))
                end
            end
            
            # Merge equivalent beliefs and update current branches
            current_belief_branches = merge_equivalent_beliefs(new_belief_branches)
            
        else
            # Fallback to wait action if no good cell found
            action = SensingAction(agent.id, Tuple{Int, Int}[], false)
            push!(greedy_sequence, action)
            
            # Just evolve all belief branches without observations
            new_belief_branches = Vector{Tuple{Belief, Float64}}()
            for (belief_branch, prob_branch) in current_belief_branches
                B_next = evolve_no_obs(belief_branch, env)
                push!(new_belief_branches, (B_next, prob_branch))
            end
            current_belief_branches = new_belief_branches
            @infiltrate
        end
    end
    
    println("🔄 Generated greedy sequence with observation branching for agent $(agent.id):")
    println("  Agent trajectory: starts in row $(get_position_at_time(agent.trajectory, 0, agent.phase_offset)[2])")
    println("  Sequence: $(length(greedy_sequence)) actions")
    println("  Final belief branches: $(length(current_belief_branches))")
    
    return greedy_sequence
end

"""
Evaluate a greedy sequence (simplified - just return a fixed value since greedy is deterministic)
"""
function evaluate_action_sequence_exact(env, belief₀, agent, seq, other_scripts, C, gs_state, rng::AbstractRNG)
    # For greedy strategy, we don't need complex evaluation
    # Just return a fixed value since the sequence is deterministic
    return 1.0
end

"""
Calculate reward for greedy sequence (simplified)
"""
function calculate_macro_script_reward(seq::Vector{SensingAction}, other_scripts, C::Int, env, agent, B_branches, gs_state)
    # For greedy strategy, return a simple reward based on sequence length
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
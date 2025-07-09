"""
BeliefManagement - Handles local belief state estimation and updates
"""
module BeliefManagement

using POMDPs
using POMDPTools
using Distributions
using Random
using Infiltrator
using ..Types

# Import types from the parent module
import ..Types: SensingAction, GridObservation, EventState, NO_EVENT, EVENT_PRESENT, EVENT_SPREADING, EVENT_DECAYING, EventDynamics
# Import transition probability function

export update_belief_state, initialize_belief, predict_belief_evolution_dbn, 
       calculate_uncertainty_map_from_distributions, calculate_uncertainty_from_distribution,
       update_cell_distribution, get_neighbor_event_probabilities, sample_from_belief,
       predict_belief_rsp, evolve_no_obs, get_neighbor_beliefs, enumerate_joint_states,
       product, normalize_belief_distributions, collapse_belief_to, 
       enumerate_all_possible_outcomes, merge_equivalent_beliefs, calculate_cell_entropy,
       get_event_probability, calculate_entropy_from_distribution

# Belief type is now defined in Types module

# """
# update_belief_state(belief::Belief, action::SensingAction, observation::GridObservation, event_dynamics::EventDynamics)
# Updates belief state using perfect observations and DBN evolution
# """
# function update_belief_state(belief::Belief, action::SensingAction, observation::GridObservation, event_dynamics::EventDynamics)
#     # First, predict belief evolution using DBN for all cells
#     predicted_belief = predict_belief_evolution_dbn(belief, event_dynamics, 1)
    
#     # Then update based on observations (perfect observations collapse belief)
#     updated_distributions = copy(predicted_belief.event_distributions)
    
#     # Create a set of observed cells for efficient lookup
#     observed_cells = Set(observation.sensed_cells)
    
#     # Update observed cells with perfect observations
#     for (i, cell) in enumerate(observation.sensed_cells)
#         x, y = cell
#         observed_state = observation.event_states[i]
        
#         # Perfect observation: belief collapses to certainty
#         updated_distribution = update_cell_distribution(updated_distributions[:, y, x], observed_state)
#         updated_distributions[:, y, x] = updated_distribution
#     end
    
#     # Unobserved cells keep their DBN-evolved distributions (already done in predict_belief_evolution_dbn)
    
#     # Update uncertainty map
#     uncertainty_map = calculate_uncertainty_map_from_distributions(updated_distributions)
    
#     # Update history
#     new_history = copy(belief.history)
#     push!(new_history, (action, observation))
    
#     return Belief(updated_distributions, uncertainty_map, belief.last_update + 1, new_history)
# end

"""
update_cell_distribution(prior_distribution::Vector{Float64}, observed_state::EventState)
Update a cell's probability distribution based on perfect observation
"""
function update_cell_distribution(prior_distribution::Vector{Float64}, observed_state::EventState)
    num_states = length(prior_distribution)
    
    # Perfect observation: belief collapses to certainty
    # P(state|observation) = 1 if state == observed_state, 0 otherwise
    posterior = zeros(num_states)
    observed_idx = Int(observed_state) + 1
    
    if 1 <= observed_idx <= num_states
        posterior[observed_idx] = 1.0
    else
        # If observed state is out of range, keep uniform distribution
        posterior .= 1.0 / num_states
    end
    
    return posterior
end

# """
# predict_belief_evolution_dbn(belief::Belief, event_dynamics::EventDynamics, num_steps::Int)
# Predicts belief evolution using DBN transition model
# """
# function predict_belief_evolution_dbn(belief::Belief, event_dynamics::EventDynamics, num_steps::Int)
#     # For now, use a simplified belief evolution
#     # TODO: Implement proper DBN belief evolution
    
#     current_distributions = copy(belief.event_distributions)
#     num_states, height, width = size(current_distributions)
    
#     for step in 1:num_steps
#         new_distributions = similar(current_distributions)
        
#         for y in 1:height
#             for x in 1:width
#                 # Get neighbor beliefs (simplified - just use event presence probability)
#                 neighbor_event_probs = get_neighbor_event_probabilities(current_distributions, x, y)
                
#                 # Simple belief update for each state
#                 current_cell_dist = current_distributions[:, y, x]
#                 new_cell_dist = similar(current_cell_dist)
                
#                 # Transition probabilities for each state
#                 # NO_EVENT -> EVENT_PRESENT with birth_rate + neighbor_influence
#                 # EVENT_PRESENT -> NO_EVENT with death_rate
#                 # EVENT_PRESENT -> EVENT_SPREADING with spread_rate
#                 # EVENT_SPREADING -> EVENT_DECAYING with decay_rate
#                 # EVENT_DECAYING -> NO_EVENT with decay_rate
                
#                 E_neighbors = sum(neighbor_event_probs)
                
#                 # State transitions based on number of states
#                 if num_states == 2
#                     # Simple 2-state model: NO_EVENT ↔ EVENT_PRESENT
#                     new_cell_dist[0] = current_cell_dist[0] * (1.0 - event_dynamics.birth_rate - event_dynamics.neighbor_influence * E_neighbors) +
#                                        current_cell_dist[1] * event_dynamics.death_rate
                    
#                     new_cell_dist[1] = current_cell_dist[0] * (event_dynamics.birth_rate + event_dynamics.neighbor_influence * E_neighbors) +
#                                        current_cell_dist[1] * (1.0 - event_dynamics.death_rate)
#                 elseif num_states == 4
#                     # 4-state model: NO_EVENT → EVENT_PRESENT → EVENT_SPREADING → EVENT_DECAYING → NO_EVENT
#                     new_cell_dist[0] = current_cell_dist[0] * (1.0 - event_dynamics.birth_rate - event_dynamics.neighbor_influence * E_neighbors) +
#                                        current_cell_dist[1] * event_dynamics.death_rate +
#                                        current_cell_dist[3] * event_dynamics.decay_rate
                    
#                     new_cell_dist[1] = current_cell_dist[0] * (event_dynamics.birth_rate + event_dynamics.neighbor_influence * E_neighbors) +
#                                        current_cell_dist[1] * (1.0 - event_dynamics.death_rate - event_dynamics.spread_rate) +
#                                        current_cell_dist[2] * event_dynamics.decay_rate
                    
#                     new_cell_dist[2] = current_cell_dist[1] * event_dynamics.spread_rate +
#                                        current_cell_dist[2] * (1.0 - event_dynamics.decay_rate)
                    
#                     new_cell_dist[3] = current_cell_dist[2] * event_dynamics.decay_rate +
#                                        current_cell_dist[3] * (1.0 - event_dynamics.decay_rate)
#                 else
#                     # Default: simple 2-state model
#                     new_cell_dist[0] = current_cell_dist[0] * (1.0 - event_dynamics.birth_rate - event_dynamics.neighbor_influence * E_neighbors) +
#                                        current_cell_dist[1] * event_dynamics.death_rate
                    
#                     new_cell_dist[1] = current_cell_dist[0] * (event_dynamics.birth_rate + event_dynamics.neighbor_influence * E_neighbors) +
#                                        current_cell_dist[1] * (1.0 - event_dynamics.death_rate)
#                 end
                
#                 # Normalize
#                 total = sum(new_cell_dist)
#                 if total > 0
#                     new_cell_dist ./= total
#                 end
                
#                 new_distributions[:, y, x] = new_cell_dist
#             end
#         end
        
#         current_distributions = new_distributions
#     end
    
#     # Update uncertainty
#     uncertainty_map = calculate_uncertainty_map_from_distributions(current_distributions)
    
#     return Belief(
#         current_distributions,
#         uncertainty_map,
#         belief.last_update + num_steps,
#         belief.history
#     )
# end
# """
# Get transition probability using environment dynamics
# This is the same function used for both world simulation and belief evolution
# """
# function get_transition_probability(next_state::Int, current_state::Int, neighbor_states::Vector{Int}, env)
#     # Use RSP transition model if available
#     if hasfield(typeof(env), :ignition_prob) && env.ignition_prob !== nothing
#         # RSP transition model - use the same logic as real world simulation
#         # For belief evolution, we don't have the specific cell position, so we'll use a simplified version
#         if next_state == 0  # NO_EVENT
#             if current_state == 0  # Currently NO_EVENT
#                 # Stay NO_EVENT with high probability (no spontaneous birth)
#                 return 0.8  # Reduced from 0.95 to allow more spontaneous birth
#             else  # Currently EVENT_PRESENT
#                 # Death/decay to NO_EVENT
#                 return 0.3  # Increased from 0.1 to allow more death (30% chance of death)
#             end
#         else  # EVENT_PRESENT
#             if current_state == 0  # Currently NO_EVENT
#                 # Birth/spread to EVENT_PRESENT
#                 # Consider neighbor influence
#                 active_neighbors = count(x -> x == 1, neighbor_states)  # Count EVENT_PRESENT neighbors
#                 birth_prob = 0.1 + 0.2 * active_neighbors  # Increased coefficients
#                 return min(0.8, birth_prob)  # Increased cap from 0.3 to 0.8
#             else  # Currently EVENT_PRESENT
#                 # Stay EVENT_PRESENT with high probability
#                 return 0.7  # Reduced from 0.9 to allow more death (30% chance of death)
#             end
#         end
#     else
#         # Use DBN transition model
#         if next_state == 0  # NO_EVENT
#             if current_state == 0  # Currently NO_EVENT
#                 return 1.0 - env.event_dynamics.birth_rate
#             else  # Currently EVENT_PRESENT
#                 return env.event_dynamics.death_rate
#             end
#         else  # EVENT_PRESENT
#             if current_state == 0  # Currently NO_EVENT
#                 active_neighbors = count(x -> x == 1, neighbor_states)
#                 return env.event_dynamics.birth_rate + env.event_dynamics.neighbor_influence * active_neighbors
#             else  # Currently EVENT_PRESENT
#                 return 1.0 - env.event_dynamics.death_rate
#             end
#         end
#     end
# end
"""
get_neighbor_event_probabilities(distributions::Array{Float64, 3}, x::Int, y::Int)
Gets event presence probabilities of neighboring cells
"""
function get_neighbor_event_probabilities(distributions::Array{Float64, 3}, x::Int, y::Int)
    neighbor_probs = Float64[]
    num_states, height, width = size(distributions)
    
    for dx in -1:1
        for dy in -1:1
            if dx == 0 && dy == 0
                continue
            end
            
            nx, ny = x + dx, y + dy
            if 1 <= nx <= width && 1 <= ny <= height
                # Sum probabilities of all event states (states 2 and beyond)
                if num_states >= 2
                    event_prob = sum(distributions[2:end, ny, nx])
                else
                    event_prob = 0.0
                end
                push!(neighbor_probs, event_prob)
            end
        end
    end
    
    return neighbor_probs
end

"""
initialize_belief(grid_width::Int, grid_height::Int, prior_distribution::Vector{Float64}=[0.7, 0.3])
Initializes belief state with distribution over event states
"""
function initialize_belief(grid_width::Int, grid_height::Int, prior_distribution::Vector{Float64}=[0.7, 0.3])
    # Create 3D array: [state, y, x]
    num_states = length(prior_distribution)
    event_distributions = Array{Float64, 3}(undef, num_states, grid_height, grid_width)
    
    # Initialize each cell with the prior distribution
    for y in 1:grid_height
        for x in 1:grid_width
            event_distributions[:, y, x] = prior_distribution
        end
    end
    
    uncertainty_map = calculate_uncertainty_map_from_distributions(event_distributions)
    
    return Belief(event_distributions, uncertainty_map, 0, [])
end

"""
calculate_uncertainty_map_from_distributions(distributions::Array{Float64, 3})
Calculates uncertainty map from distribution array
"""
function calculate_uncertainty_map_from_distributions(distributions::Array{Float64, 3})
    num_states, height, width = size(distributions)
    uncertainty = Matrix{Float64}(undef, height, width)
    
    for y in 1:height
        for x in 1:width
            prob_vector = distributions[:, y, x]
            uncertainty[y, x] = calculate_uncertainty_from_distribution(prob_vector)
        end
    end
    
    return uncertainty
end

"""
calculate_uncertainty_from_distribution(prob_vector::Vector{Float64})
Calculates uncertainty for a probability distribution vector
"""
function calculate_uncertainty_from_distribution(prob_vector::Vector{Float64})
    # Using entropy as uncertainty measure for multi-state distribution
    entropy = 0.0
    for prob in prob_vector
        if prob > 0.0
            entropy -= prob * log(prob)
end
    end
    return entropy
end

function sample_from_belief(belief, rng::AbstractRNG)
    # Assume belief is a Dict with "probabilities" => Matrix
    prob_map = belief["probabilities"]
    height, width = size(prob_map)
    sampled = zeros(Int, height, width)
    for y in 1:height, x in 1:width
        sampled[y, x] = rand(rng) < prob_map[y, x] ? 1 : 0
    end
    return sampled
end

# """
# predict_belief_rsp(B::Belief, λmap, Δt::Int) -> Belief
# Propagate *marginal* probabilities forward Δt steps
# using RSP transition probabilities internally (matrix multiplications).
# Exact – no sampling.
# """
# function predict_belief_rsp(belief::Belief, λmap::Matrix{Float64}, Δt::Int)
#     if Δt <= 0
#         return belief
#     end
    
#     num_states, height, width = size(belief.event_distributions)
    
#     # For small grids, we can use exact enumeration
#     # For larger grids, we'd need approximation
    
#     # Convert belief to event map probabilities
#     event_probs = Matrix{Float64}(undef, height, width)
#     for y in 1:height, x in 1:width
#         if num_states >= 2
#             event_probs[y, x] = belief.event_distributions[2, y, x]  # P(EVENT_PRESENT)
#         else
#             event_probs[y, x] = 0.0
#         end
#     end
    
#     # Propagate for Δt steps
#     for step in 1:Δt
#         new_event_probs = similar(event_probs)
        
#         # For each cell, compute new probability using RSP dynamics
#         for y in 1:height, x in 1:width
#             # Get neighbor event probabilities
#             neighbor_probs = Float64[]
#             for dx in -1:1, dy in -1:1
#                 if dx == 0 && dy == 0
#                     continue
#                 end
#                 nx, ny = x + dx, y + dy
#                 if 1 <= nx <= width && 1 <= ny <= height
#                     push!(neighbor_probs, event_probs[ny, nx])
#                 end
#             end
            
#             # RSP transition: P(event_t+1) = P(event_t) * (1 - death) + P(no_event_t) * (birth + neighbor_influence)
#             current_prob = event_probs[y, x]
#             neighbor_influence = sum(neighbor_probs) * 0.3  # Influence from neighbors
#             birth_rate = λmap[y, x] * 0.1  # Spontaneous birth
#             death_rate = 0.05  # Death rate
            
#             new_prob = current_prob * (1.0 - death_rate) + (1.0 - current_prob) * (birth_rate + neighbor_influence)
#             new_event_probs[y, x] = max(0.0, min(1.0, new_prob))
#         end
        
#         event_probs = new_event_probs
#     end
    
#     # Convert back to belief distributions
#     new_distributions = similar(belief.event_distributions)
#     for y in 1:height, x in 1:width
#         if num_states >= 2
#             new_distributions[0, y, x] = 1.0 - event_probs[y, x]  # P(NO_EVENT)
#             new_distributions[1, y, x] = event_probs[y, x]        # P(EVENT_PRESENT)
#         else
#             new_distributions[0, y, x] = 1.0
#         end
#     end
    
    # Update uncertainty map
#     uncertainty_map = calculate_uncertainty_map_from_distributions(new_distributions)
    
#     return Belief(
#         new_distributions,
#         uncertainty_map,
#         belief.last_update + Δt,
#         belief.history
#     )
# end

# Add helper function for RSP transition probability in belief evolution
function get_rsp_transition_probability_belief(next_state, current_state, neighbor_states, params)
    λ = params.lambda
    β0 = params.beta0
    α = params.alpha
    δ = params.delta
    μ = params.mu
    active_nbrs = count(x -> x == 1, neighbor_states)  # 1 = EVENT_PRESENT
    if current_state == 0  # NO_EVENT
        birth_p = clamp(β0 + λ + α * active_nbrs, 0.0, 1.0)
        return next_state == 1 ? birth_p : 1.0 - birth_p
    else  # current_state == 1 (EVENT_PRESENT)
        return next_state == 1 ? δ : (next_state == 0 ? μ : 0.0)
    end
end

"""
Evolve belief without observations using Díaz-Avalos formula
"""
function evolve_no_obs(B::Belief, env)
    # Create new belief with evolved distributions
    new_distributions = similar(B.event_distributions)
    num_states, height, width = size(new_distributions)
    
    for y in 1:height, x in 1:width
        # Get current cell belief
        current_belief = B.event_distributions[:, y, x]
        
        # Get neighbor beliefs
        neighbor_beliefs = get_neighbor_beliefs(B, x, y)
        
        # Apply Díaz-Avalos evolution for each possible next state
        for next_state in 1:num_states
            prob = 0.0
            for current_state in 1:num_states
                for neighbor_states in enumerate_joint_states(neighbor_beliefs, num_states)
                    params = env.rsp_params
                    p_trans = get_rsp_transition_probability_belief(next_state - 1, current_state - 1, neighbor_states .- 1, params)
                    p_belief = current_belief[current_state] * product([neighbor_beliefs[i][neighbor_states[i]] for i in 1:length(neighbor_beliefs)])
                    prob += p_trans * p_belief
                end
            end
            new_distributions[next_state, y, x] = prob
        end
    end
    
    # Normalize and update uncertainty
    new_distributions = normalize_belief_distributions(new_distributions)
    uncertainty_map = calculate_uncertainty_map_from_distributions(new_distributions)
    
    return Belief(new_distributions, uncertainty_map, B.last_update + 1, B.history)
end

"""
Get neighbor beliefs for a cell
"""
function get_neighbor_beliefs(B::Belief, x::Int, y::Int)
    neighbor_beliefs = Vector{Vector{Float64}}()
    height, width = size(B.event_distributions)[2:3]
    
    for dx in -1:1, dy in -1:1
        if dx == 0 && dy == 0
            continue
        end
        
        nx, ny = x + dx, y + dy
        if 1 <= nx <= width && 1 <= ny <= height
            push!(neighbor_beliefs, B.event_distributions[:, ny, nx])
        end
    end
    
    return neighbor_beliefs
end

"""
Enumerate all possible joint states for neighbors
"""
function enumerate_joint_states(neighbor_beliefs::Vector{Vector{Float64}}, num_states::Int)
    if isempty(neighbor_beliefs)
        return [Int[]]
    end
    
    states = Vector{Vector{Int}}()
    for state in 1:num_states
        for remaining_states in enumerate_joint_states(neighbor_beliefs[2:end], num_states)
            push!(states, [state; remaining_states])
        end
    end
    
    return states
end

"""
Calculate product of probabilities
"""
function product(probs::Vector{Float64})
    return reduce(*, probs, init=1.0)
end

"""
Normalize belief distributions
"""
function normalize_belief_distributions(distributions::Array{Float64, 3})
    num_states, height, width = size(distributions)
    
    for y in 1:height, x in 1:width
        total = sum(distributions[:, y, x])
        if total > 0
            distributions[:, y, x] ./= total
        end
    end
    
    return distributions
end

"""
Collapse belief to a specific state for a cell
"""
function collapse_belief_to(B::Belief, cell::Tuple{Int, Int}, state::EventState)
    x, y = cell
    new_belief = deepcopy(B)
    
    # Collapse to delta distribution
    num_states = size(new_belief.event_distributions, 1)
    new_belief.event_distributions[:, y, x] = zeros(num_states)
    new_belief.event_distributions[Int(state) + 1, y, x] = 1.0
    
    # Update uncertainty map
    new_belief.uncertainty_map = calculate_uncertainty_map_from_distributions(new_belief.event_distributions)
    
    return new_belief
end

"""
Enumerate all possible observation outcomes
"""
function enumerate_all_possible_outcomes(B::Belief, obs_set::Vector{Tuple{Int, SensingAction}})
    outcomes = Vector{Tuple{Tuple{Int, Int}, EventState, Float64}}()
    
    for (agent_id, action) in obs_set
        for cell in action.target_cells
            # Get belief for this cell
            x, y = cell
            cell_belief = B.event_distributions[:, y, x]
            
            # For each possible observation outcome
            for state in 1:length(cell_belief)
                if cell_belief[state] > 0
                    push!(outcomes, (cell, EventState(state-1), cell_belief[state]))
                end
            end
        end
    end
    
    return outcomes
end

"""
Merge equivalent beliefs and sum their probabilities
"""
function merge_equivalent_beliefs(beliefs::Vector{Tuple{Belief, Float64}})
    merged = Dict{String, Tuple{Belief, Float64}}()
    
    for (belief, prob) in beliefs
        # Create a hash of the belief for comparison
        belief_hash = string(belief.event_distributions)
        
        if haskey(merged, belief_hash)
            existing_belief, existing_prob = merged[belief_hash]
            merged[belief_hash] = (existing_belief, existing_prob + prob)
        else
            merged[belief_hash] = (belief, prob)
        end
    end
    
    return collect(values(merged))
end

"""
Calculate entropy for a specific cell
"""
function calculate_cell_entropy(B::Belief, cell::Tuple{Int, Int})
    x, y = cell
    prob_vector = B.event_distributions[:, y, x]
    return calculate_entropy_from_distribution(prob_vector)
end

"""
Get event probability for a cell
"""
function get_event_probability(B::Belief, cell::Tuple{Int, Int})
    x, y = cell
    prob_vector = B.event_distributions[:, y, x]
    
    # Sum probabilities of all event states (states 2 and beyond)
    if length(prob_vector) >= 2
        return sum(prob_vector[2:end])
    else
        return 0.0
    end
end

"""
Calculate entropy for a multi-state belief distribution
H(b_k) = -∑ p_i * log(p_i)
"""
function calculate_entropy_from_distribution(prob_vector::Vector{Float64})
    entropy = 0.0
    for prob in prob_vector
        if prob > 0.0
            entropy -= prob * log(prob)
        end
    end
    return entropy
end

end # module 
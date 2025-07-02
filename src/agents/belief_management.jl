"""
BeliefManagement - Handles local belief state estimation and updates
"""
module BeliefManagement

using POMDPs
using POMDPTools
using Distributions
using Random
using ..Types

# Import types from the parent module
import ..Types: SensingAction, GridObservation, EventState, NO_EVENT, EVENT_PRESENT, EVENT_SPREADING, EVENT_DECAYING, EventDynamics

# Define DBN types and functions locally
struct DBNTransitionModel4
    birth_rate::Float64
    death_rate::Float64
    spread_rate::Float64
    decay_rate::Float64
    neighbor_influence::Float64
end

function DBNTransitionModel4(dynamics::EventDynamics)
    return DBNTransitionModel4(dynamics.birth_rate, dynamics.death_rate, dynamics.spread_rate, dynamics.decay_rate, dynamics.neighbor_influence)
end

function predict_next_belief_dbn(b_k::Float64, b_neighbors::Vector{Float64}, model::DBNTransitionModel4)
    # Expected number of active neighbors
    E_neighbors = sum(b_neighbors)

    # Case 1: x_t = 1 → survives with (1 - death_rate)
    p1 = 1.0 - model.death_rate

    # Case 2: x_t = 0 → activates with birth + influence
    p0 = model.birth_rate + model.neighbor_influence * E_neighbors

    # Total new belief
    return b_k * p1 + (1.0 - b_k) * p0
end

export update_belief_state, initialize_belief, calculate_uncertainty, predict_belief_evolution_dbn, 
       calculate_uncertainty_map_from_distributions, calculate_uncertainty_from_distribution,
       update_cell_distribution, get_neighbor_event_probabilities

# Belief type is now defined in Types module

"""
update_belief_state(belief::Belief, action::SensingAction, observation::GridObservation, event_dynamics::EventDynamics)
Updates belief state using perfect observations and DBN evolution
"""
function update_belief_state(belief::Belief, action::SensingAction, observation::GridObservation, event_dynamics::EventDynamics)
    # First, predict belief evolution using DBN for all cells
    predicted_belief = predict_belief_evolution_dbn(belief, event_dynamics, 1)
    
    # Then update based on observations (perfect observations collapse belief)
    updated_distributions = copy(predicted_belief.event_distributions)
    
    # Create a set of observed cells for efficient lookup
    observed_cells = Set(observation.sensed_cells)
    
    # Update observed cells with perfect observations
    for (i, cell) in enumerate(observation.sensed_cells)
        x, y = cell
        observed_state = observation.event_states[i]
        
        # Perfect observation: belief collapses to certainty
        updated_distribution = update_cell_distribution(updated_distributions[:, y, x], observed_state)
        updated_distributions[:, y, x] = updated_distribution
    end
    
    # Unobserved cells keep their DBN-evolved distributions (already done in predict_belief_evolution_dbn)
    
    # Update uncertainty map
    uncertainty_map = calculate_uncertainty_map_from_distributions(updated_distributions)
    
    # Update history
    new_history = copy(belief.history)
    push!(new_history, (action, observation))
    
    return Belief(updated_distributions, uncertainty_map, belief.last_update + 1, new_history)
end

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

"""
predict_belief_evolution_dbn(belief::Belief, event_dynamics::EventDynamics, num_steps::Int)
Predicts belief evolution using DBN transition model
"""
function predict_belief_evolution_dbn(belief::Belief, event_dynamics::EventDynamics, num_steps::Int)
    # For now, use a simplified belief evolution
    # TODO: Implement proper DBN belief evolution
    
    current_distributions = copy(belief.event_distributions)
    num_states, height, width = size(current_distributions)
    
    for step in 1:num_steps
        new_distributions = similar(current_distributions)
        
        for y in 1:height
            for x in 1:width
                # Get neighbor beliefs (simplified - just use event presence probability)
                neighbor_event_probs = get_neighbor_event_probabilities(current_distributions, x, y)
                
                # Simple belief update for each state
                current_cell_dist = current_distributions[:, y, x]
                new_cell_dist = similar(current_cell_dist)
                
                # Transition probabilities for each state
                # NO_EVENT -> EVENT_PRESENT with birth_rate + neighbor_influence
                # EVENT_PRESENT -> NO_EVENT with death_rate
                # EVENT_PRESENT -> EVENT_SPREADING with spread_rate
                # EVENT_SPREADING -> EVENT_DECAYING with decay_rate
                # EVENT_DECAYING -> NO_EVENT with decay_rate
                
                E_neighbors = sum(neighbor_event_probs)
                
                # State transitions based on number of states
                if num_states == 2
                    # Simple 2-state model: NO_EVENT ↔ EVENT_PRESENT
                    new_cell_dist[1] = current_cell_dist[1] * (1.0 - event_dynamics.birth_rate - event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * event_dynamics.death_rate
                    
                    new_cell_dist[2] = current_cell_dist[1] * (event_dynamics.birth_rate + event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * (1.0 - event_dynamics.death_rate)
                elseif num_states == 4
                    # 4-state model: NO_EVENT → EVENT_PRESENT → EVENT_SPREADING → EVENT_DECAYING → NO_EVENT
                    new_cell_dist[1] = current_cell_dist[1] * (1.0 - event_dynamics.birth_rate - event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * event_dynamics.death_rate +
                                       current_cell_dist[4] * event_dynamics.decay_rate
                    
                    new_cell_dist[2] = current_cell_dist[1] * (event_dynamics.birth_rate + event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * (1.0 - event_dynamics.death_rate - event_dynamics.spread_rate) +
                                       current_cell_dist[3] * event_dynamics.decay_rate
                    
                    new_cell_dist[3] = current_cell_dist[2] * event_dynamics.spread_rate +
                                       current_cell_dist[3] * (1.0 - event_dynamics.decay_rate)
                    
                    new_cell_dist[4] = current_cell_dist[3] * event_dynamics.decay_rate +
                                       current_cell_dist[4] * (1.0 - event_dynamics.decay_rate)
                else
                    # Default: simple 2-state model
                    new_cell_dist[1] = current_cell_dist[1] * (1.0 - event_dynamics.birth_rate - event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * event_dynamics.death_rate
                    
                    new_cell_dist[2] = current_cell_dist[1] * (event_dynamics.birth_rate + event_dynamics.neighbor_influence * E_neighbors) +
                                       current_cell_dist[2] * (1.0 - event_dynamics.death_rate)
                end
                
                # Normalize
                total = sum(new_cell_dist)
                if total > 0
                    new_cell_dist ./= total
                end
                
                new_distributions[:, y, x] = new_cell_dist
            end
        end
        
        current_distributions = new_distributions
    end
    
    # Update uncertainty
    uncertainty_map = calculate_uncertainty_map_from_distributions(current_distributions)
    
    return Belief(
        current_distributions,
        uncertainty_map,
        belief.last_update + num_steps,
        belief.history
    )
end

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
get_neighbor_beliefs(probabilities::Matrix{Float64}, x::Int, y::Int)
Gets beliefs of neighboring cells
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

# Legacy function for backward compatibility
"""
calculate_uncertainty_map(probabilities::Matrix{Float64})
Legacy function - calculates uncertainty map from probability map
"""
function calculate_uncertainty_map(probabilities::Matrix{Float64})
    uncertainty = Matrix{Float64}(undef, size(probabilities))
    
    for i in 1:size(probabilities, 1)
        for j in 1:size(probabilities, 2)
            uncertainty[i, j] = calculate_uncertainty(probabilities[i, j])
        end
    end
    
    return uncertainty
end

"""
calculate_uncertainty(probability::Float64)
Legacy function - calculates uncertainty for a single probability value
"""
function calculate_uncertainty(probability::Float64)
    # Using entropy as uncertainty measure
    if probability <= 0.0 || probability >= 1.0
        return 0.0
    end
    return -(probability * log(probability) + (1 - probability) * log(1 - probability))
end

"""
predict_belief_evolution(belief::Belief, event_dynamics::EventDynamics, num_steps::Int)
Legacy function for backward compatibility
"""
function predict_belief_evolution(belief::Belief, event_dynamics::EventDynamics, num_steps::Int)
    return predict_belief_evolution_dbn(belief, event_dynamics, num_steps)
end

"""
apply_event_dynamics(probabilities::Matrix{Float64}, dynamics::EventDynamics)
Legacy function - now uses DBN
"""
function apply_event_dynamics(probabilities::Matrix{Float64}, dynamics::EventDynamics)
    # Convert to DBN model
    dbn_model = DBNTransitionModel4(dynamics)
    height, width = size(probabilities)
    updated_probabilities = copy(probabilities)
    
    for y in 1:height
        for x in 1:width
            neighbor_probs = get_neighbor_beliefs(probabilities, x, y)
            updated_probabilities[y, x] = predict_next_belief_dbn(
                probabilities[y, x], 
                neighbor_probs, 
                dbn_model
            )
        end
    end
    
    return updated_probabilities
end

"""
get_neighbor_probabilities(probabilities::Matrix{Float64}, x::Int, y::Int)
Legacy function - now uses get_neighbor_beliefs
"""
function get_neighbor_probabilities(probabilities::Matrix{Float64}, x::Int, y::Int)
    return get_neighbor_beliefs(probabilities, x, y)
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

export sample_from_belief

end # module 
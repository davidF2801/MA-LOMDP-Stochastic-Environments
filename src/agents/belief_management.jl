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

export Belief, update_belief_state, initialize_belief, calculate_uncertainty, predict_belief_evolution_dbn

"""
Belief - Represents the local belief state of an agent
"""
struct Belief
    event_probabilities::Matrix{Float64}  # Probability of events at each cell
    uncertainty_map::Matrix{Float64}      # Uncertainty at each cell
    last_update::Int                      # Last update time step
    history::Vector{Tuple{SensingAction, GridObservation}}
end

"""
update_belief_state(belief::Belief, action::SensingAction, observation::GridObservation, event_dynamics::EventDynamics)
Updates belief state using Bayes rule and DBN
"""
function update_belief_state(belief::Belief, action::SensingAction, observation::GridObservation, event_dynamics::EventDynamics)
    # First, predict belief evolution using DBN
    predicted_belief = predict_belief_evolution_dbn(belief, event_dynamics, 1)
    
    # Then update based on observations
    updated_probabilities = copy(predicted_belief.event_probabilities)
    
    # Update based on observations using Bayes rule
    for (i, cell) in enumerate(observation.sensed_cells)
        x, y = cell
        observed_state = observation.event_states[i]
        
        # Get prior probability
        prior_prob = predicted_belief.event_probabilities[y, x]
        
        # Update using Bayes rule
        if observed_state == EVENT_PRESENT
            # P(event|observation) = P(observation|event) * P(event) / P(observation)
            # Assuming P(observation|event) = 0.9 and P(observation|no_event) = 0.1
            likelihood_event = 0.9
            likelihood_no_event = 0.1
            posterior = (likelihood_event * prior_prob) / (likelihood_event * prior_prob + likelihood_no_event * (1 - prior_prob))
            updated_probabilities[y, x] = posterior
        elseif observed_state == NO_EVENT
            # P(no_event|observation) = P(observation|no_event) * P(no_event) / P(observation)
            likelihood_event = 0.1
            likelihood_no_event = 0.9
            posterior = (likelihood_no_event * (1 - prior_prob)) / (likelihood_event * prior_prob + likelihood_no_event * (1 - prior_prob))
            updated_probabilities[y, x] = 1.0 - posterior
        end
        # TODO: Handle other event states
    end
    
    # Update uncertainty map
    uncertainty_map = calculate_uncertainty_map(updated_probabilities)
    
    # Update history
    new_history = copy(belief.history)
    push!(new_history, (action, observation))
    
    return Belief(updated_probabilities, uncertainty_map, belief.last_update + 1, new_history)
end

"""
predict_belief_evolution_dbn(belief::Belief, event_dynamics::EventDynamics, num_steps::Int)
Predicts belief evolution using DBN transition model
"""
function predict_belief_evolution_dbn(belief::Belief, event_dynamics::EventDynamics, num_steps::Int)
    # For now, use a simplified belief evolution
    # TODO: Implement proper DBN belief evolution
    
    current_probabilities = copy(belief.event_probabilities)
    height, width = size(current_probabilities)
    
    for step in 1:num_steps
        new_probabilities = similar(current_probabilities)
        
        for y in 1:height
            for x in 1:width
                # Get neighbor beliefs
                neighbor_beliefs = get_neighbor_beliefs(current_probabilities, x, y)
                
                # Simple belief update (can be replaced with proper DBN)
                E_neighbors = sum(neighbor_beliefs)
                p1 = 1.0 - event_dynamics.death_rate
                p0 = event_dynamics.birth_rate + event_dynamics.neighbor_influence * E_neighbors
                new_probabilities[y, x] = current_probabilities[y, x] * p1 + (1.0 - current_probabilities[y, x]) * p0
            end
        end
        
        current_probabilities = new_probabilities
    end
    
    # Update uncertainty
    uncertainty_map = calculate_uncertainty_map(current_probabilities)
    
    return Belief(
        current_probabilities,
        uncertainty_map,
        belief.last_update + num_steps,
        belief.history
    )
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
initialize_belief(grid_width::Int, grid_height::Int, prior_probability::Float64=0.5)
Initializes belief state with uniform prior
"""
function initialize_belief(grid_width::Int, grid_height::Int, prior_probability::Float64=0.5)
    event_probabilities = fill(prior_probability, grid_height, grid_width)
    uncertainty_map = calculate_uncertainty_map(event_probabilities)
    
    return Belief(event_probabilities, uncertainty_map, 0, [])
end

"""
calculate_uncertainty_map(probabilities::Matrix{Float64})
Calculates uncertainty map from probability map
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
Calculates uncertainty for a single probability value
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
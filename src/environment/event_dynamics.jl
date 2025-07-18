"""
EventDynamicsModule - Models stochastic event evolution in the spatial grid
"""
module EventDynamicsModule

using POMDPs
using POMDPTools
using Distributions
using LinearAlgebra
using Random
using Infiltrator
using ..Types

# Import types from the parent module
import ..Types.EventState, ..Types.EventState2, ..Types.EventState4, ..Types.NO_EVENT_2, ..Types.EVENT_PRESENT_2, ..Types.NO_EVENT_4, ..Types.EVENT_PRESENT_4, ..Types.EVENT_SPREADING_4, ..Types.EVENT_DECAYING_4
import ..Types.EventDynamics, ..Types.TwoStateEventDynamics

export DBNTransitionModel2, DBNTransitionModel4, transition_probability_dbn, predict_next_belief_dbn, update_events!, get_neighbor_states, initialize_random_events,
       transition_rsp!, rsp_transition_probs, get_transition_probability, get_transition_probability_rsp

# """
# Abstract type for different event state enums
# """
# abstract type EventStateEnum end

# # EventState2 and EventState4 are now imported from types.jl

"""
Abstract type for transition models
"""
abstract type TransitionModel end

"""
DBN-based transition model for 2-state events
"""
struct DBNTransitionModel2 <: TransitionModel
    birth_rate::Float64
    death_rate::Float64
    neighbor_influence::Float64
end

"""
DBN-based transition model for 4-state events
"""
struct DBNTransitionModel4 <: TransitionModel
    birth_rate::Float64
    death_rate::Float64
    spread_rate::Float64
    decay_rate::Float64
    neighbor_influence::Float64
end

# Convert legacy EventDynamics to new DBN model
function DBNTransitionModel2(dynamics::EventDynamics)
    return DBNTransitionModel2(dynamics.birth_rate, dynamics.death_rate, dynamics.neighbor_influence)
end

function DBNTransitionModel4(dynamics::EventDynamics)
    return DBNTransitionModel4(dynamics.birth_rate, dynamics.death_rate, dynamics.spread_rate, dynamics.decay_rate, dynamics.neighbor_influence)
end

"""
DBN local conditional probability for 2-state events:
    Pr(x_{t+1} = EVENT_PRESENT | x_t, neighbors)
"""
function transition_probability_dbn(current_state::EventState2, neighbor_states::Vector{EventState2}, model::DBNTransitionModel2)
    num_active_neighbors = count(==(EVENT_PRESENT_2), neighbor_states)

    if current_state == EVENT_PRESENT_2
        # Persistence vs death
        return max(0.0, min(1.0, 1.0 - model.death_rate))
    else
        # Birth from neighbors
        return max(0.0, min(1.0, model.birth_rate + model.neighbor_influence * num_active_neighbors))
    end
end

"""
DBN local conditional probability for 4-state events
"""
function transition_probability_dbn(current_state::EventState4, neighbor_states::Vector{EventState4}, model::DBNTransitionModel4)
    num_active_neighbors = count(x -> x == EVENT_PRESENT_4 || x == EVENT_SPREADING_4, neighbor_states)

    if current_state == NO_EVENT_4
        # Birth from neighbors
        return max(0.0, min(1.0, model.birth_rate + model.neighbor_influence * num_active_neighbors))
    elseif current_state == EVENT_PRESENT_4
        # Can spread or start decaying
        return max(0.0, min(1.0, model.spread_rate))
    elseif current_state == EVENT_SPREADING_4
        # Continue spreading or start decaying
        return max(0.0, min(1.0, model.spread_rate))
    else  # EVENT_DECAYING_4
        # Probability of disappearing
        return max(0.0, min(1.0, model.death_rate))
    end
end

"""
Compute updated belief for a cell using DBN (2-state)
"""
function predict_next_belief_dbn(b_k::Float64, b_neighbors::Vector{Float64}, model::DBNTransitionModel2)
    # Expected number of active neighbors
    E_neighbors = sum(b_neighbors)

    # Case 1: x_t = 1 → survives with (1 - death_rate)
    p1 = 1.0 - model.death_rate

    # Case 2: x_t = 0 → activates with birth + influence
    p0 = model.birth_rate + model.neighbor_influence * E_neighbors

    # Total new belief
    return b_k * p1 + (1.0 - b_k) * p0
end

"""
Compute updated belief for a cell using DBN (4-state)
"""
function predict_next_belief_dbn(b_k::Float64, b_neighbors::Vector{Float64}, model::DBNTransitionModel4)
    # For 4-state, we need to handle the different state transitions
    # This is a simplified version - can be extended for full 4-state belief
    E_neighbors = sum(b_neighbors)

    # Simplified: treat as 2-state for belief update
    p1 = 1.0 - model.death_rate
    p0 = model.birth_rate + model.neighbor_influence * E_neighbors

    return b_k * p1 + (1.0 - b_k) * p0
end

"""
Update events using DBN model (2-state)
"""
function update_events!(model::DBNTransitionModel2, event_map::Matrix{EventState2}, rng::AbstractRNG)
    height, width = size(event_map)
    new_map = similar(event_map)

    for y in 1:height
        for x in 1:width
            current = event_map[y, x]
            neighbors = get_neighbor_states(event_map, x, y)
            p = transition_probability_dbn(current, neighbors, model)
            new_map[y, x] = rand(rng) < p ? EVENT_PRESENT_2 : NO_EVENT_2
        end
    end

    event_map .= new_map
end

"""
Update events using DBN model (4-state)
"""
function update_events!(model::DBNTransitionModel4, event_map::Matrix{EventState4}, rng::AbstractRNG)
    height, width = size(event_map)
    new_map = similar(event_map)

    for y in 1:height
        for x in 1:width
            current = event_map[y, x]
            neighbors = get_neighbor_states(event_map, x, y)
            p = transition_probability_dbn(current, neighbors, model)
            
            # Determine next state based on probability
            if rand(rng) < p
                if current == NO_EVENT_4
                    new_map[y, x] = EVENT_PRESENT_4
                elseif current == EVENT_PRESENT_4
                    new_map[y, x] = rand(rng) < model.decay_rate ? EVENT_DECAYING_4 : EVENT_SPREADING_4
                elseif current == EVENT_SPREADING_4
                    new_map[y, x] = rand(rng) < model.decay_rate ? EVENT_DECAYING_4 : EVENT_SPREADING_4
                else  # EVENT_DECAYING_4
                    new_map[y, x] = rand(rng) < model.death_rate ? NO_EVENT_4 : EVENT_DECAYING_4
                end
            else
                new_map[y, x] = current
            end
        end
    end

    event_map .= new_map
end

"""
Get neighbor states (generic version)
"""
function get_neighbor_states(event_map::Matrix{T}, x::Int, y::Int) where T
    neighbors = T[]
    height, width = size(event_map)
    
    for dx in -1:1
        for dy in -1:1
            if dx == 0 && dy == 0
                continue
            end
            
            nx, ny = x + dx, y + dy
            if 1 <= nx <= width && 1 <= ny <= height
                push!(neighbors, event_map[ny, nx])
            end
        end
    end
    
    return neighbors
end

"""
Initialize random events (generic version)
"""
function initialize_random_events(event_map::Matrix{T}, num_events::Int, rng::AbstractRNG) where T
    height, width = size(event_map)
    
    for _ in 1:num_events
        x = rand(rng, 1:width)
        y = rand(rng, 1:height)
        # Convert to appropriate event state
        if T == EventState2
            event_map[y, x] = EVENT_PRESENT_2
        elseif T == EventState4
            event_map[y, x] = EVENT_PRESENT_4
        else
            # For any other type, we'll need to handle it specifically
            # For now, just skip if we don't know how to handle it
            continue
        end
    end
end

"""
Legacy spread probability calculation for backward compatibility
Note: This requires EventState to be defined elsewhere
"""
function calculate_spread_probability(dynamics::EventDynamics, current_state, neighbor_states)
    # For now, return a simple probability based on dynamics
    # This can be extended when EventState is properly defined
    return dynamics.birth_rate
end

# =============================================================================
# RSP (Random Spread Process) MODEL PARAMETERS
# -----------------------------------------------------------------------------
# These constants control the stochastic event dynamics in the RSP model.
# Each parameter has a specific meaning in the context of event spread and decay:
#
#   RSP_LAMBDA:  Local ignition intensity (λ) - could be λmap[y,x]; 0–1
#                - Additional ignition probability from local conditions
#   RSP_BETA0:   Spontaneous (background) ignition probability when no neighbors burn
#                - Base probability of spontaneous event birth
#   RSP_ALPHA:   Contagion contribution of each active neighbor
#                - How much each neighboring event increases birth probability
#   RSP_DELTA:   Probability the fire persists (EVENT→EVENT)
#                - Probability of event survival/continuation
#   RSP_MU:      Probability the fire dies (EVENT→NO_EVENT)
#                - Probability of event death/extinction
#
# Tune these parameters to explore different behaviors.
# =============================================================================

"""
transition_rsp!(new_map, old_map, λmap, rng)
Sample the Díaz-Avalos Random Spread Process one time step.

old_map, new_map :: EventMap  (values: NO_EVENT, EVENT_PRESENT)
λmap              :: Matrix{Float64}   # ignition intensity already standardised
"""
function transition_rsp!(new_map::EventMap, old_map::EventMap, λmap::Matrix{Float64}, rng::AbstractRNG; rsp_params=nothing)
    height, width = size(old_map)
    if rsp_params === nothing
        error("RSP parameters must be provided via rsp_params keyword argument!")
    end
    
    # Initialize new map
    new_map .= old_map
    
    # For each cell, apply the same transition model used in belief evolution
    for y in 1:height, x in 1:width
        current_state = old_map[y, x]  # 0 for NO_EVENT, 1 for EVENT_PRESENT
        
        # Get neighbor states for this cell
        neighbor_states = Int[]
        for dx in -1:1, dy in -1:1
            if dx == 0 && dy == 0
                continue
            end
            
            nx, ny = x + dx, y + dy
            if 1 <= nx <= width && 1 <= ny <= height
                push!(neighbor_states, Int(old_map[ny, nx]))
            end
        end
        # Calculate transition probabilities for both possible next states using RSP model
        prob_no_event = get_transition_probability_rsp(Int(NO_EVENT), Int(current_state), neighbor_states;
            λ=rsp_params.lambda, β0=rsp_params.beta0, α=rsp_params.alpha, δ=rsp_params.delta, μ=rsp_params.mu)
        prob_event = get_transition_probability_rsp(Int(EVENT_PRESENT), Int(current_state), neighbor_states;
            λ=rsp_params.lambda, β0=rsp_params.beta0, α=rsp_params.alpha, δ=rsp_params.delta, μ=rsp_params.mu)
        
        # Normalize probabilities
        total_prob = prob_no_event + prob_event
        if total_prob > 0
            prob_no_event /= total_prob
            prob_event /= total_prob
        else
            prob_no_event = 0.5
            prob_event = 0.5
        end
        # Sample the next state
        if rand(rng) < prob_event
            new_map[y, x] = EventState(1)
        else
            new_map[y, x] = EventState(0)
        end
    end
end

"""
rsp_transition_probs(old_map, λmap) -> Dict{EventMap,Float64}
Return **all** possible `new_map`s reachable in one step together
with their probabilities P(s'|s) (Sect. 2.3, steps 1–5).
No randomness here – pure enumeration.
"""
function rsp_transition_probs(old_map::EventMap, λmap::Matrix{Float64})
    height, width = size(old_map)
    
    # For small grids, we can enumerate all possible outcomes
    # For larger grids, we'd need to be more clever about pruning
    
    # Identify cells that can change state
    variable_cells = Tuple{Int, Int}[]
    
    for y in 1:height, x in 1:width
        # All cells can potentially change state in RSP
        push!(variable_cells, (x, y))
    end
    
    # Enumerate all possible combinations for all cells
    num_cells = length(variable_cells)
    outcomes = Dict{EventMap, Float64}()
    
    # For small grids, enumerate all 2^num_cells possibilities
    for combination in 0:(2^num_cells - 1)
        new_map = copy(old_map)
        prob = 1.0
        
        # Set cells according to this combination
        for (i, (x, y)) in enumerate(variable_cells)
            bit = (combination >> (i-1)) & 1
            if bit == 1
                new_map[y, x] = 1
                # Calculate probability of this transition based on RSP dynamics
                if old_map[y, x] == 0
                    # Birth probability
                    prob *= λmap[y, x] * 0.1  # Spontaneous birth
                else
                    # Survival probability (no death)
                    prob *= (1.0 - 0.05)  # 1 - death_rate
                end
            else
                new_map[y, x] = 0
                # Calculate probability of this transition
                if old_map[y, x] == 1
                    # Death probability
                    prob *= 0.05  # death_rate
                else
                    # No birth probability
                    prob *= (1.0 - λmap[y, x] * 0.1)  # 1 - spontaneous_birth
                end
            end
        end
        
        if prob > 0.0
            outcomes[new_map] = get(outcomes, new_map, 0.0) + prob
        end
    end
    
    return outcomes
end

"""
Get transition probability using environment dynamics
This is the same function used for both world simulation and belief evolution
"""
function get_transition_probability(next_state::Int, current_state::Int, neighbor_states::Vector{Int}, env)
    # Use RSP transition model if available
    if hasfield(typeof(env), :ignition_prob) && env.ignition_prob !== nothing
        # RSP transition model using ignition probability map
        # For now, we'll use a simplified model that considers neighbor influence
        # In a full implementation, we'd need to pass the cell position to access λmap[y, x]
        
        if next_state == 1  # NO_EVENT
            if current_state == 1  # Currently NO_EVENT
                # Stay NO_EVENT with high probability (no spontaneous birth)
                return 0.95
            else  # Currently EVENT_PRESENT
                # Death/decay to NO_EVENT
                return 0.05
            end
        else  # EVENT_PRESENT
            if current_state == 1  # Currently NO_EVENT
                # Birth/spread to EVENT_PRESENT
                # Consider neighbor influence
                active_neighbors = count(x -> x == 2, neighbor_states)  # Count EVENT_PRESENT neighbors
                birth_prob = 0.05 + 0.1 * active_neighbors  # Base birth + neighbor influence
                return min(0.3, birth_prob)  # Cap the probability
            else  # Currently EVENT_PRESENT
                # Stay EVENT_PRESENT with high probability
                return 0.95
            end
        end
    else
        # Use DBN transition model
        if next_state == 1  # NO_EVENT
            if current_state == 1  # Currently NO_EVENT
                return 1.0 - env.event_dynamics.birth_rate
            else  # Currently EVENT_PRESENT
                return env.event_dynamics.death_rate
            end
        else  # EVENT_PRESENT
            if current_state == 1  # Currently NO_EVENT
                active_neighbors = count(x -> x == 2, neighbor_states)
                return env.event_dynamics.birth_rate + env.event_dynamics.neighbor_influence * active_neighbors
            else  # Currently EVENT_PRESENT
                return 1.0 - env.event_dynamics.death_rate
            end
        end
    end
end
"""
Keyword parameters
------------------
λ   – local ignition intensity (could be λmap[y,x]; 0–1).  
β0  – spontaneous (background) ignition probability when no neighbours burn.  
α   – Contagion contribution of each active neighbour.  
δ   – Probability the fire persists (EVENT→EVENT).  
μ   – Probability the fire dies   (EVENT→NO_EVENT).

Tune these parameters to explore different behaviours.
"""
function get_transition_probability_rsp(next_state::Int, current_state::Int,
                             nbr_states::Vector{Int};
                             λ::Float64,
                             β0::Float64,
                             α::Float64,
                             δ::Float64,
                             μ::Float64)

    active_nbrs = count(x -> x == 1, nbr_states)  # 1 = EVENT_PRESENT
    if current_state == 0  # NO_EVENT
        # ---- Birth / ignition ----
        birth_p = clamp(β0 + λ + α * active_nbrs/8, 0.0, 1.0)
        return next_state == 1 ? birth_p : 1.0 - birth_p
    else  # current_state == 1 (EVENT_PRESENT)
        # ---- Persistence or extinction ----
        return next_state == 1 ? δ : (next_state == 0 ? μ : 0.0)
    end
end

end 
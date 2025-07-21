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
# Import RSP transition probability from Types module
import ..Types.get_transition_probability_rsp
export DBNTransitionModel2, DBNTransitionModel4, update_events!, get_neighbor_states, initialize_random_events,
       transition_rsp!

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
transition_rsp!(new_map, old_map, param_maps, rng)
Sample the Díaz-Avalos Random Spread Process one time step with cell-specific parameters.

old_map, new_map :: EventMap  (values: NO_EVENT, EVENT_PRESENT)
param_maps       :: RSPParameterMaps  # Cell-specific parameter maps
"""
function transition_rsp!(new_map::EventMap, old_map::EventMap, param_maps::Types.RSPParameterMaps, rng::AbstractRNG)
    height, width = size(old_map)
    
    # Initialize new map
    new_map .= old_map
    
    # For each cell, apply cell-specific transition model
    for y in 1:height, x in 1:width
        current_state = old_map[y, x]  # 0 for NO_EVENT, 1 for EVENT_PRESENT
        
        # Get cell-specific parameters
        cell_params = Types.get_cell_rsp_params(param_maps, y, x)
        
        # Get neighbor states for this cell (always 8 neighbors, using 0 for out-of-bounds)
        neighbor_states = Int[]
        for dx in -1:1, dy in -1:1
            if dx == 0 && dy == 0
                continue
            end
            
            nx, ny = x + dx, y + dy
            if 1 <= nx <= width && 1 <= ny <= height
                push!(neighbor_states, Int(old_map[ny, nx]))
            else
                # Out of bounds - treat as NO_EVENT (0)
                push!(neighbor_states, 0)
            end
        end
        
        # Calculate transition probabilities using cell-specific parameters
        prob_event = Types.get_transition_probability_rsp(Int(EVENT_PRESENT), Int(current_state), neighbor_states;
            λ=cell_params.lambda, β0=cell_params.beta0, α=cell_params.alpha, δ=cell_params.delta)
        prob_no_event = 1 - prob_event
        
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
transition_rsp!(new_map, old_map, λmap, rng) - Legacy version for backward compatibility
Sample the Díaz-Avalos Random Spread Process one time step with uniform parameters.

old_map, new_map :: EventMap  (values: NO_EVENT, EVENT_PRESENT)
λmap              :: Matrix{Float64}   # ignition intensity already standardised
"""
function transition_rsp!(new_map::EventMap, old_map::EventMap, λmap::Matrix{Float64}, rng::AbstractRNG; rsp_params=nothing)
    height, width = size(old_map)
    if rsp_params === nothing
        error("RSP parameters must be provided via rsp_params keyword argument!")
    end
    
    # Create uniform parameter maps for backward compatibility
    param_maps = Types.create_uniform_rsp_maps(height, width;
        lambda=rsp_params.lambda, beta0=rsp_params.beta0, 
        alpha=rsp_params.alpha, delta=rsp_params.delta)
    
    # Call the new version
    transition_rsp!(new_map, old_map, param_maps, rng)
end
end 
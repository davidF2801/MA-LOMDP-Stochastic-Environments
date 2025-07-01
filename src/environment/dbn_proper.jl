using POMDPs
using POMDPTools
using Distributions
using Random
using LinearAlgebra
import ..EventState, ..EventState2, ..EventDynamics, ..NO_EVENT_2, ..EVENT_PRESENT_2

"""
Proper DBN implementation for 2-state spatial events
"""

# EventState2 is now imported from types.jl

"""
Proper DBN Transition Model with Conditional Probability Tables (CPTs)
"""
struct ProperDBNModel
    # Standardized field names to match DBNTransitionModel2
    birth_rate::Float64                      # Rate of new events appearing
    death_rate::Float64                      # Rate of events disappearing  
    neighbor_influence::Float64              # Influence of neighboring cells
    
    # Internal CPT parameters (computed from the above)
    p_birth_given_no_neighbors::Float64      # P(event|no_event, 0 neighbors)
    p_birth_given_neighbors::Float64         # P(event|no_event, >0 neighbors) 
    p_survival::Float64                      # P(event|event, any neighbors)
    neighbor_weight::Float64                 # Weight of neighbor influence
end

"""
Create DBN model from simple parameters
"""
function ProperDBNModel(birth_rate::Float64, death_rate::Float64, neighbor_influence::Float64)
    return ProperDBNModel(
        birth_rate,                          # birth_rate
        death_rate,                          # death_rate
        neighbor_influence,                  # neighbor_influence
        birth_rate,                          # p_birth_given_no_neighbors
        birth_rate + neighbor_influence,     # p_birth_given_neighbors
        1.0 - death_rate,                    # p_survival
        neighbor_influence                   # neighbor_weight
    )
end

"""
Proper DBN transition probability using CPT
P(x_{t+1} = 1 | x_t, neighbors_t)
"""
function transition_probability_proper_dbn(current_state::EventState2, neighbor_states::Vector{EventState2}, model::ProperDBNModel)
    num_active_neighbors = count(==(EVENT_PRESENT_2), neighbor_states)
    
    if current_state == EVENT_PRESENT_2
        # P(event_{t+1} | event_t, neighbors_t)
        return model.p_survival
    else
        # P(event_{t+1} | no_event_t, neighbors_t)
        if num_active_neighbors == 0
            return model.p_birth_given_no_neighbors
        else
            # Influence increases with number of neighbors
            influence = model.neighbor_weight * num_active_neighbors
            return min(1.0, model.p_birth_given_neighbors + influence)
        end
    end
end

"""
Proper belief propagation using DBN structure
Updates belief P(x_{t+1} = 1) based on current belief and neighbor beliefs
"""
function belief_propagation_dbn(current_belief::Float64, neighbor_beliefs::Vector{Float64}, model::ProperDBNModel)
    # Expected number of active neighbors
    E_neighbors = sum(neighbor_beliefs)
    
    # Use law of total probability:
    # P(x_{t+1} = 1) = Σ_{x_t, neighbors} P(x_{t+1} = 1 | x_t, neighbors) * P(x_t, neighbors)
    
    # For simplicity, assume independence between current cell and neighbors
    # This is an approximation - full DBN would model the joint distribution
    
    # Case 1: Current cell has event (x_t = 1)
    p_event_given_event = model.p_survival
    
    # Case 2: Current cell has no event (x_t = 0)
    if E_neighbors == 0
        p_event_given_no_event = model.p_birth_given_no_neighbors
    else
        influence = model.neighbor_weight * E_neighbors
        p_event_given_no_event = min(1.0, model.p_birth_given_neighbors + influence)
    end
    
    # Total probability using law of total probability
    new_belief = current_belief * p_event_given_event + (1.0 - current_belief) * p_event_given_no_event
    
    return max(0.0, min(1.0, new_belief))
end

"""
Simulate environment using proper DBN
"""
function simulate_proper_dbn(model::ProperDBNModel, width::Int, height::Int, num_steps::Int, initial_events::Int, rng::AbstractRNG)
    # Initialize event map
    event_map = fill(NO_EVENT_2, height, width)
    
    # Add initial events
    for _ in 1:initial_events
        x = rand(rng, 1:width)
        y = rand(rng, 1:height)
        event_map[y, x] = EVENT_PRESENT_2
    end
    
    # Store evolution
    evolution = [copy(event_map)]
    event_counts = [count(==(EVENT_PRESENT_2), event_map)]
    
    # Simulate evolution
    for step in 1:num_steps
        new_map = similar(event_map)
        
        for y in 1:height
            for x in 1:width
                current = event_map[y, x]
                neighbors = get_neighbor_states(event_map, x, y)
                
                # Use proper DBN transition probability
                p = transition_probability_proper_dbn(current, neighbors, model)
                new_map[y, x] = rand(rng) < p ? EVENT_PRESENT_2 : NO_EVENT_2
            end
        end
        
        event_map = new_map
        push!(evolution, copy(event_map))
        push!(event_counts, count(==(EVENT_PRESENT_2), event_map))
    end
    
    return evolution, event_counts
end

"""
Belief state evolution using proper DBN
"""
function belief_evolution_dbn(model::ProperDBNModel, width::Int, height::Int, num_steps::Int, initial_belief::Matrix{Float64})
    # Initialize belief state
    belief_state = copy(initial_belief)
    evolution = [copy(belief_state)]
    
    # Evolve belief state
    for step in 1:num_steps
        new_belief = similar(belief_state)
        
        for y in 1:height
            for x in 1:width
                # Get neighbor beliefs
                neighbor_beliefs = get_neighbor_beliefs(belief_state, x, y)
                
                # Update belief using proper DBN propagation
                new_belief[y, x] = belief_propagation_dbn(belief_state[y, x], neighbor_beliefs, model)
            end
        end
        
        belief_state = new_belief
        push!(evolution, copy(belief_state))
    end
    
    return evolution
end

"""
Get neighbor states (8-connectivity) for rectangular grid
"""
function get_neighbor_states(event_map::Matrix{EventState2}, x::Int, y::Int)
    neighbors = EventState2[]
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
Get neighbor beliefs (8-connectivity) for rectangular grid
"""
function get_neighbor_beliefs(belief_map::Matrix{Float64}, x::Int, y::Int)
    neighbor_beliefs = Float64[]
    height, width = size(belief_map)
    
    for dx in -1:1
        for dy in -1:1
            if dx == 0 && dy == 0
                continue
            end
            
            nx, ny = x + dx, y + dy
            if 1 <= nx <= width && 1 <= ny <= height
                push!(neighbor_beliefs, belief_map[ny, nx])
            end
        end
    end
    
    return neighbor_beliefs
end

"""
Update events using proper DBN model (for compatibility with existing scripts)
"""
function update_events!(model::ProperDBNModel, event_map::Matrix{EventState2}, rng::AbstractRNG)
    height, width = size(event_map)
    new_map = similar(event_map)
    
    for y in 1:height
        for x in 1:width
            current = event_map[y, x]
            neighbors = get_neighbor_states(event_map, x, y)
            
            # Use proper DBN transition probability
            p = transition_probability_proper_dbn(current, neighbors, model)
            new_map[y, x] = rand(rng) < p ? EVENT_PRESENT_2 : NO_EVENT_2
        end
    end
    
    event_map .= new_map
end

"""
Analyze DBN properties
"""
function analyze_dbn_properties(model::ProperDBNModel)
    println("🔍 DBN Model Analysis")
    println("====================")
    println("Model parameters:")
    println("- Birth rate: $(model.birth_rate)")
    println("- Death rate: $(model.death_rate)")
    println("- Neighbor influence: $(model.neighbor_influence)")
    println()
    println("Internal CPT parameters:")
    println("- P(event|no_event, 0 neighbors): $(model.p_birth_given_no_neighbors)")
    println("- P(event|no_event, >0 neighbors): $(model.p_birth_given_neighbors)")
    println("- P(event|event, any neighbors): $(model.p_survival)")
    println("- Neighbor weight: $(model.neighbor_weight)")
    println()
    
    # Analyze transition probabilities for different scenarios
    scenarios = [
        ("No event, 0 neighbors", NO_EVENT_2, EventState2[]),
        ("No event, 1 neighbor", NO_EVENT_2, [EVENT_PRESENT_2]),
        ("No event, 2 neighbors", NO_EVENT_2, [EVENT_PRESENT_2, EVENT_PRESENT_2]),
        ("Event, 0 neighbors", EVENT_PRESENT_2, EventState2[]),
        ("Event, 1 neighbor", EVENT_PRESENT_2, [EVENT_PRESENT_2])
    ]
    
    println("Transition Probabilities:")
    for (name, current, neighbors) in scenarios
        p = transition_probability_proper_dbn(current, neighbors, model)
        println("- $(name): $(round(p, digits=4))")
    end
    println()
    
    # Analyze belief propagation
    println("Belief Propagation Examples:")
    belief_scenarios = [
        ("Low belief, no neighbors", 0.1, [0.0, 0.0, 0.0]),
        ("High belief, no neighbors", 0.9, [0.0, 0.0, 0.0]),
        ("Low belief, high neighbors", 0.1, [0.8, 0.9, 0.7]),
        ("High belief, high neighbors", 0.9, [0.8, 0.9, 0.7])
    ]
    
    for (name, current_belief, neighbor_beliefs) in belief_scenarios
        new_belief = belief_propagation_dbn(current_belief, neighbor_beliefs, model)
        change = new_belief - current_belief
        println("- $(name): $(round(current_belief, digits=3)) → $(round(new_belief, digits=3)) (Δ=$(round(change, digits=3)))")
    end
end 
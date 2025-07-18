using POMDPs
using POMDPTools
using Distributions
using Random
using ..Types
using LinearAlgebra

"""
SensorModels - Models for range-limited sensors and observations
"""
module SensorModels

using POMDPs
using POMDPTools
using Distributions
using Random
using ..Types
using LinearAlgebra

# Import types from the parent module
import ..Types: EventState, RangeLimitedSensor

export generate_observation, calculate_footprint, is_within_range, calculate_information_gain, calculate_entropy

"""
generate_observation(sensor::RangeLimitedSensor, agent_pos::Tuple{Int, Int}, event_map::Matrix{EventState}, target_cells::Vector{Tuple{Int, Int}})
Generates observations for the specified target cells
"""
function generate_observation(sensor::RangeLimitedSensor, agent_pos::Tuple{Int, Int}, event_map::Matrix{EventState}, target_cells::Vector{Tuple{Int, Int}})
    observed_states = EventState[]
    sensed_cells = Tuple{Int, Int}[]
    for cell in target_cells
        if is_within_range(sensor, agent_pos, cell)
            push!(sensed_cells, cell)
            push!(observed_states, event_map[cell[2], cell[1]])
        end
    end
    return sensed_cells, observed_states
end

"""
calculate_footprint(sensor::RangeLimitedSensor, agent_pos::Tuple{Int, Int}, grid_width::Int, grid_height::Int)
Calculates the sensor footprint (all cells within range)
"""
function calculate_footprint(sensor::RangeLimitedSensor, agent_pos::Tuple{Int, Int}, grid_width::Int, grid_height::Int)
    footprint = Tuple{Int, Int}[]
    ax, ay = agent_pos
    for x in 1:grid_width
        for y in 1:grid_height
            distance = sqrt((x - ax)^2 + (y - ay)^2)
            if distance <= sensor.range
                push!(footprint, (x, y))
            end
        end
    end
    return footprint
end

"""
is_within_range(sensor::RangeLimitedSensor, agent_pos::Tuple{Int, Int}, target_pos::Tuple{Int, Int})
Checks if target position is within sensor range
"""
function is_within_range(sensor::RangeLimitedSensor, agent_pos::Tuple{Int, Int}, target_pos::Tuple{Int, Int})
    distance = sqrt((target_pos[1] - agent_pos[1])^2 + (target_pos[2] - agent_pos[2])^2)
    return distance <= sensor.range
end

"""
calculate_information_gain(belief::Matrix{Float64}, observed_cells::Vector{Tuple{Int, Int}}, observed_states::Vector{EventState})
Calculates information gain from observations
"""
function calculate_information_gain(belief::Matrix{Float64}, observed_cells::Vector{Tuple{Int, Int}}, observed_states::Vector{EventState})
    total_gain = 0.0
    for (i, cell) in enumerate(observed_cells)
        x, y = cell
        prior_entropy = calculate_entropy(belief[y, x])
        posterior_entropy = calculate_entropy(belief[y, x])
        gain = prior_entropy - posterior_entropy
        total_gain += gain
    end
    return total_gain
end

"""
calculate_entropy(probability::Float64)
Calculates entropy for a binary event (event present/not present)
"""
function calculate_entropy(probability::Float64)
    if probability <= 0.0 || probability >= 1.0
        return 0.0
    end
    return -(probability * log2(probability) + (1 - probability) * log2(1 - probability))
end

end # module 
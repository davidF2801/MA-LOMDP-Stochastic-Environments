using POMDPs
using POMDPTools
using Distributions
using Random
using LinearAlgebra

"""
Types - Common types used across the MA-LOMDP project
"""
module Types

using POMDPs
using POMDPTools
using Distributions
using Random
using LinearAlgebra

"""
EventState - State of events in each grid cell
"""
@enum EventState begin
    NO_EVENT = 0
    EVENT_PRESENT = 1
    EVENT_SPREADING = 2
    EVENT_DECAYING = 3
end

"""
EventState2 - Two-state event model (NO_EVENT ↔ EVENT_PRESENT)
"""
@enum EventState2 begin
    NO_EVENT_2 = 0
    EVENT_PRESENT_2 = 1
end

"""
EventState4 - Four-state event model for more complex dynamics
"""
@enum EventState4 begin
    NO_EVENT_4 = 0
    EVENT_PRESENT_4 = 1
    EVENT_SPREADING_4 = 2
    EVENT_DECAYING_4 = 3
end

"""
SensingAction - Actions available to agents (where to sense)
"""
struct SensingAction
    agent_id::Int
    target_cells::Vector{Tuple{Int, Int}}  # Cells to sense within footprint
    communicate::Bool  # Whether to communicate with others
end

"""
GridObservation - What agents can observe (their sensor footprint)
"""
struct GridObservation
    agent_id::Int
    sensed_cells::Vector{Tuple{Int, Int}}
    event_states::Vector{EventState}
    communication_received::Vector{Any}
end

"""
RangeLimitedSensor - Model of a range-limited sensor
"""
struct RangeLimitedSensor
    range::Float64           # Sensing range
    field_of_view::Float64   # Field of view angle (radians)
    noise_level::Float64     # Observation noise
end

"""
Trajectory - Deterministic periodic trajectory for an agent
"""
abstract type Trajectory end

"""
CircularTrajectory - Circular periodic trajectory
"""
struct CircularTrajectory <: Trajectory
    center_x::Int
    center_y::Int
    radius::Float64
    period::Int
    step_size::Float64
end

"""
LinearTrajectory - Linear periodic trajectory
"""
struct LinearTrajectory <: Trajectory
    start_x::Int
    start_y::Int
    end_x::Int
    end_y::Int
    period::Int
    step_size::Float64
end

"""
EventDynamics - Models stochastic event evolution in the spatial grid
"""
struct EventDynamics
    birth_rate::Float64      # Rate of new events appearing
    death_rate::Float64      # Rate of events disappearing
    spread_rate::Float64     # Rate of events spreading to neighbors
    decay_rate::Float64      # Rate of events decaying
    neighbor_influence::Float64  # Influence of neighboring cells
end

"""
TwoStateEventDynamics - Simplified 2-state event dynamics (NO_EVENT ↔ EVENT_PRESENT)
"""
struct TwoStateEventDynamics
    birth_rate::Float64      # Rate of new events appearing
    death_rate::Float64      # Rate of events disappearing
end

"""
Simple deterministic distribution wrapper for POMDP interface
"""
struct DeterministicDistribution{T}
    value::T
end

# Make it work with POMDPs.jl interface
Base.rand(d::DeterministicDistribution) = d.value
Base.rand(rng::AbstractRNG, d::DeterministicDistribution) = d.value

# Alias for convenience
const Deterministic = DeterministicDistribution

"""
Belief - Represents a belief state over the environment
"""
mutable struct Belief
    event_distributions::Array{Float64, 3}  # Distribution over event states at each cell [state, y, x]
    uncertainty_map::Matrix{Float64}        # Uncertainty at each cell
    last_update::Int                        # Last update time step
    history::Vector{Tuple{SensingAction, GridObservation}}
end

# Add copy method for Belief
Base.copy(belief::Belief) = Belief(
    copy(belief.event_distributions),
    copy(belief.uncertainty_map),
    belief.last_update,
    copy(belief.history)
)

# Add deepcopy method for Belief
Base.deepcopy(belief::Belief) = Belief(
    deepcopy(belief.event_distributions),
    deepcopy(belief.uncertainty_map),
    belief.last_update,
    deepcopy(belief.history)
)

# Helper functions for working with belief distributions
"""
get_event_probability(belief::Belief, x::Int, y::Int, state::EventState)
Get probability of a specific event state at a cell
"""
function get_event_probability(belief::Belief, x::Int, y::Int, state::EventState)
    state_idx = Int(state) + 1  # Convert enum to 1-based index
    return belief.event_distributions[state_idx, y, x]
end

"""
set_event_probability!(belief::Belief, x::Int, y::Int, state::EventState, prob::Float64)
Set probability of a specific event state at a cell
"""
function set_event_probability!(belief::Belief, x::Int, y::Int, state::EventState, prob::Float64)
    state_idx = Int(state) + 1  # Convert enum to 1-based index
    belief.event_distributions[state_idx, y, x] = prob
end

"""
get_event_probability_vector(belief::Belief, x::Int, y::Int)
Get the full probability distribution vector for a cell
"""
function get_event_probability_vector(belief::Belief, x::Int, y::Int)
    return belief.event_distributions[:, y, x]
end

"""
set_event_probability_vector!(belief::Belief, x::Int, y::Int, prob_vector::Vector{Float64})
Set the full probability distribution vector for a cell
"""
function set_event_probability_vector!(belief::Belief, x::Int, y::Int, prob_vector::Vector{Float64})
    belief.event_distributions[:, y, x] = prob_vector
end

"""
normalize_cell_distribution!(belief::Belief, x::Int, y::Int)
Normalize the probability distribution for a cell to sum to 1
"""
function normalize_cell_distribution!(belief::Belief, x::Int, y::Int)
    prob_vector = get_event_probability_vector(belief, x, y)
    total = sum(prob_vector)
    if total > 0
        normalized_vector = prob_vector ./ total
        set_event_probability_vector!(belief, x, y, normalized_vector)
    end
end

"""
Agent - Represents an autonomous agent in the multi-agent system
"""
mutable struct Agent
    id::Int                           # Unique agent identifier
    trajectory::Trajectory            # Deterministic periodic trajectory
    sensor::RangeLimitedSensor        # Sensor capabilities
    phase_offset::Int                 # Phase offset for trajectory timing
    belief::Any                       # Local belief state (Belief type from BeliefManagement)
    observation_history::Vector{GridObservation}  # History of observations
    plan_index::Int                   # Index of next action to execute in current plan
end

# Constructor with default observation history and plan index
function Agent(id::Int, trajectory::Trajectory, sensor::RangeLimitedSensor, phase_offset::Int, belief::Any)
    return Agent(id, trajectory, sensor, phase_offset, belief, GridObservation[], 1)
end

# Constructor with default belief, observation history, and plan index
function Agent(id::Int, trajectory::Trajectory, sensor::RangeLimitedSensor, phase_offset::Int)
    return Agent(id, trajectory, sensor, phase_offset, nothing, GridObservation[], 1)
end

# Export all types
export EventState, NO_EVENT, EVENT_PRESENT, EVENT_SPREADING, EVENT_DECAYING
export EventState2, NO_EVENT_2, EVENT_PRESENT_2
export EventState4, NO_EVENT_4, EVENT_PRESENT_4, EVENT_SPREADING_4, EVENT_DECAYING_4
export SensingAction, GridObservation, RangeLimitedSensor
export Trajectory, CircularTrajectory, LinearTrajectory
export EventDynamics, TwoStateEventDynamics
export DeterministicDistribution, Deterministic
export Belief
export Agent

# Export helper functions
export get_event_probability, set_event_probability!, get_event_probability_vector, set_event_probability_vector!, normalize_cell_distribution!

# Add RSP-related types
const EventMap = Matrix{EventState}

@enum DynamicsMode begin
    toy_dbn
    rsp
end

export EventMap, DynamicsMode, toy_dbn, rsp

end # module Types 
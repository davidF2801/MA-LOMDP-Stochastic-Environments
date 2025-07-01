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
    event_probabilities::Matrix{Float64}  # Probability of events at each cell
    uncertainty_map::Matrix{Float64}      # Uncertainty at each cell
    last_update::Int                      # Last update time step
    history::Vector{Tuple{SensingAction, GridObservation}}
end

# Add copy method for Belief
Base.copy(belief::Belief) = Belief(
    copy(belief.event_probabilities),
    copy(belief.uncertainty_map),
    belief.last_update,
    copy(belief.history)
)

# Add deepcopy method for Belief
Base.deepcopy(belief::Belief) = Belief(
    deepcopy(belief.event_probabilities),
    deepcopy(belief.uncertainty_map),
    belief.last_update,
    deepcopy(belief.history)
)

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
end

# Constructor with default observation history
function Agent(id::Int, trajectory::Trajectory, sensor::RangeLimitedSensor, phase_offset::Int, belief::Any)
    return Agent(id, trajectory, sensor, phase_offset, belief, GridObservation[])
end

# Constructor with default belief and observation history
function Agent(id::Int, trajectory::Trajectory, sensor::RangeLimitedSensor, phase_offset::Int)
    return Agent(id, trajectory, sensor, phase_offset, nothing, GridObservation[])
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

end # module Types 
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
using Infiltrator

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
"""
Common utility functions for the MA-LOMDP project
"""

      using Random

        """
        Transition probability for an ignition / extinction cell in a simple fire-spread model.

        Arguments
        ---------
        * `next_state::Int`      : 0 = NO_EVENT, 1 = EVENT_PRESENT  
        * `current_state::Int`   : 0 = NO_EVENT, 1 = EVENT_PRESENT  
        * `neighbor_states::Vector{Int}` (length 8) : Von-Neumann + diagonal neighbours  
        * Keyword parameters  
            * `λ::Float64`   – external (lightning, ember) ignition intensity  
            * `β₀::Float64`  – spontaneous ignition base rate when no neighbour burns  
            * `α::Float64`   – contagion strength (per active-neighbour weight)  
            * `δ₀::Float64`  – baseline persistence probability (EVENT→EVENT)  
            * `wind_weights::Vector{Float64}` – 8-element wind multipliers (default = 1)  
            * `slope_weights::Vector{Float64}` – 8-element slope multipliers (default = 1)  
            * `fuel::Float64` , `fuel_max::Float64` – remaining / initial fuel (0–1)  

        Returns
        -------
        `Float64` – probability of transitioning from `current_state` to `next_state`
        """
        function get_transition_probability_rsp(next_state::Int,
            current_state::Int,
            neighbor_states::Vector{Int};
            λ::Float64,
            β0::Float64,
            α::Float64,
            δ::Float64)

            # ---------------------------------------------------------------------
            # 1. Basic sanity checks (fail fast on invalid input)
            # ---------------------------------------------------------------------
            @assert next_state     in (0, 1) "next_state must be 0 or 1"
            @assert current_state  in (0, 1) "current_state must be 0 or 1"
            @assert 0.0 ≤ λ  ≤ 1.0  "λ must lie in [0,1]"
            @assert 0.0 ≤ β0 ≤ 1.0  "β0 must lie in [0,1]"
            @assert 0.0 ≤ α  ≤ 1.0  "α must lie in [0,1]"
            @assert 0.0 ≤ δ  ≤ 1.0  "δ must lie in [0,1]"

            # ---------------------------------------------------------------------
            # 2. Contagion term (saturating, position-independent)
            #    • Normalise by AVAILABLE neighbours so edge/corner cells aren’t penalised
            #    • Exponential form keeps result strictly < 1 without clamp()
            # ---------------------------------------------------------------------
            active_neighbours = sum(neighbor_states)                       # 0 … n
            norm_active       = isempty(neighbor_states) ? 0.0 :
            active_neighbours / length(neighbor_states)  # 0 … 1
            contagion = 1 - exp(-α * norm_active)          
            # ---------------------------------------------------------------------
            # 3. State-dependent transition probability
            # ---------------------------------------------------------------------
            if current_state == 0           # NO_EVENT  → {NO_EVENT, EVENT}
            p_event = 1 - exp(-(β0 + λ + contagion))  # ignition, ∈ [0,1)
            return next_state == 1 ? p_event : (1.0 - p_event)

            else                            # EVENT     → {NO_EVENT, EVENT}
            μ = 1.0 - δ                                 # extinction
            return next_state == 1 ? δ : μ
        end
    end
"""
Calculate expected lifetime for RSP events

For RSP, the expected lifetime E[L] = 1/μ where μ is the death probability
"""
function calculate_expected_lifetime_rsp(μ::Float64)
    return 1.0 / μ
end

"""
Calculate information gain for a cell: G(b_k) = H(b_k) * P(event)

Parameters:
- prob_vector: Probability distribution over states [P(NO_EVENT), P(EVENT_PRESENT), ...]

Returns: Information gain value
"""
function calculate_cell_information_gain(prob_vector::Vector{Float64})
    # Calculate entropy: H(b_k) = -∑ p_i * log2(p_i)
    entropy = 0.0
    for prob in prob_vector
        if prob > 0.0
            entropy -= prob * log2(prob)
        end
    end
    
    # Weight by event probability: G(b_k) = H(b_k) * P(event)
    # P(event) is the sum of all event state probabilities (states 2 and beyond)
    if length(prob_vector) >= 2
        event_probability = sum(prob_vector[2:end])
    else
        event_probability = 0.0
    end
    
    return entropy * event_probability
end

"""
Calculate entropy for a multi-state belief distribution
H(b_k) = -∑ p_i * log2(p_i)
"""
function calculate_entropy_from_distribution(prob_vector::Vector{Float64})
    entropy = 0.0
    for prob in prob_vector
        if prob > 0.0
            entropy -= prob * log2(prob)
        end
    end
    return entropy
end

"""
Generate combinations of elements

Parameters:
- elements: Vector of elements to combine
- k: Size of each combination

Returns: Vector of combinations
"""
function combinations(elements, k)
    if k == 0
        return [Tuple{}[]]
    elseif k == 1
        return [[element] for element in elements]
    else
        result = []
        for i in 1:length(elements)
            for combo in combinations(elements[i+1:end], k-1)
                push!(result, [elements[i]; combo])
            end
        end
        return result
    end
end

"""
Check if a value is approximately equal to another within tolerance

Parameters:
- a, b: Values to compare
- tol: Tolerance (default: 1e-6)

Returns: true if |a - b| ≤ tol
"""
function isapprox_equal(a::Real, b::Real; tol::Real=1e-6)
    return abs(a - b) ≤ tol
end

"""
Normalize a probability vector to sum to 1.0

Parameters:
- prob_vector: Vector of probabilities

Returns: Normalized probability vector
"""
function normalize_probabilities(prob_vector::Vector{Float64})
    total = sum(prob_vector)
    if total > 0.0
        return prob_vector ./ total
    else
        # If all probabilities are zero, return uniform distribution
        return fill(1.0 / length(prob_vector), length(prob_vector))
    end
end

"""
Clamp a value between min and max

Parameters:
- value: Value to clamp
- min_val: Minimum allowed value
- max_val: Maximum allowed value

Returns: Clamped value
"""
function clamp_value(value::Real, min_val::Real, max_val::Real)
    return max(min_val, min(max_val, value))
end

"""
RSP Parameter Maps for non-uniform environments
"""
struct RSPParameterMaps
    lambda_map::Matrix{Float64}    # Local ignition intensity map
    beta0_map::Matrix{Float64}     # Spontaneous ignition probability map
    alpha_map::Matrix{Float64}     # Contagion strength map
    delta_map::Matrix{Float64}     # Persistence probability map
    mu_map::Matrix{Float64}        # Death probability map (computed as 1 - delta)
end

"""
Cell type definitions for heterogeneous RSP environments
"""
const HETEROGENEOUS_CELL_TYPES = [
    # Immune cells – events almost never start, die immediately
    (name="Immune", lambda=0.0002, beta0=0.0002, alpha=0.03, delta=0.05),

    # Fleeting events – ignite occasionally, burn out fast
    (name="Fleeting", lambda=0.0050, beta0=0.0150, alpha=0.01, delta=0.85),

    # Long-lasting events – rare ignition, but ~10-step lifetime
    (name="Long-lasting", lambda=0.0020, beta0=0.0020, alpha=0.01, delta=0.99),

    # Moderate cells – balanced ignition and lifetime ≈¼ period
    (name="Moderate", lambda=0.0100, beta0=0.0100, alpha=0.01, delta=0.85),

    # High-contagion cells – ignite easily and spread, moderate lifetime
    (name="High-contagion", lambda=0.0200, beta0=0.0100, alpha=0.1, delta=0.85)
]

"""
Create uniform RSP parameter maps
"""
function create_uniform_rsp_maps(height::Int, width::Int; 
                                lambda::Float64=0.0, beta0::Float64=0.0, 
                                alpha::Float64=0.3, delta::Float64=0.8)
    lambda_map = fill(lambda, height, width)
    beta0_map = fill(beta0, height, width)
    alpha_map = fill(alpha, height, width)
    delta_map = fill(delta, height, width)
    mu_map = fill(1.0 - delta, height, width)
    
    return RSPParameterMaps(lambda_map, beta0_map, alpha_map, delta_map, mu_map)
end

"""
Create non-uniform RSP parameter maps with randomly distributed cell types
"""
function create_heterogeneous_rsp_maps(height::Int, width::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    lambda_map = Matrix{Float64}(undef, height, width)
    beta0_map = Matrix{Float64}(undef, height, width)
    alpha_map = Matrix{Float64}(undef, height, width)
    delta_map = Matrix{Float64}(undef, height, width)
    mu_map = Matrix{Float64}(undef, height, width)
    
    for y in 1:height, x in 1:width
        # Randomly select a cell type
        cell_type = rand(rng, HETEROGENEOUS_CELL_TYPES)
        
        lambda_map[y, x] = cell_type.lambda
        beta0_map[y, x] = cell_type.beta0
        alpha_map[y, x] = cell_type.alpha
        delta_map[y, x] = cell_type.delta
        mu_map[y, x] = 1.0 - cell_type.delta
    end
    
    return RSPParameterMaps(lambda_map, beta0_map, alpha_map, delta_map, mu_map)
end



"""
Create RSP parameter maps with reproducible random distribution
"""
function create_heterogeneous_rsp_maps(height::Int, width::Int, seed::Int)
    rng = Random.MersenneTwister(seed)
    return create_heterogeneous_rsp_maps(height, width; rng=rng)
end

"""
Get RSP parameters for a specific cell
"""
function get_cell_rsp_params(param_maps::RSPParameterMaps, y::Int, x::Int)
    return (
        lambda = param_maps.lambda_map[y, x],
        beta0 = param_maps.beta0_map[y, x],
        alpha = param_maps.alpha_map[y, x],
        delta = param_maps.delta_map[y, x],
        mu = param_maps.mu_map[y, x]
    )
end

"""
Analyze cell type distribution in parameter maps
"""
function analyze_cell_type_distribution(param_maps::RSPParameterMaps)
    height, width = size(param_maps.lambda_map)
    total_cells = height * width
    
    # Count cells of each type based on actual parameters
    cell_counts = Dict{String, Int}()
    for cell_type in HETEROGENEOUS_CELL_TYPES
        cell_counts[cell_type.name] = 0
    end
    for y in 1:height, x in 1:width
        lambda = param_maps.lambda_map[y, x]
        beta0 = param_maps.beta0_map[y, x]
        alpha = param_maps.alpha_map[y, x]
        delta = param_maps.delta_map[y, x]
        
        # Find matching cell type (with tolerance for floating point)
        for cell_type in HETEROGENEOUS_CELL_TYPES
            if isapprox(lambda, cell_type.lambda, atol=1e-6) &&
               isapprox(beta0, cell_type.beta0, atol=1e-6) &&
               isapprox(alpha, cell_type.alpha, atol=1e-6) &&
               isapprox(delta, cell_type.delta, atol=1e-6)
                cell_counts[cell_type.name] += 1
                break
            end
        end
    end
    return cell_counts, total_cells
end

"""
Calculate the number of active neighbors in a grid

Parameters:
- grid: 2D grid of states
- x, y: Position to check neighbors around
- event_state: State value that counts as "active" (default: 1 for EVENT_PRESENT)

Returns: Number of active neighbors
"""
function count_active_neighbors(grid::Matrix, x::Int, y::Int; event_state::Int=1)
    height, width = size(grid)
    active_count = 0
    
    for dx in -1:1, dy in -1:1
        if dx == 0 && dy == 0
            continue
        end
        
        nx, ny = x + dx, y + dy
        if 1 <= nx <= width && 1 <= ny <= height
            if grid[ny, nx] == event_state
                active_count += 1
            end
        end
    end
    
    return active_count
end

end # module Types 
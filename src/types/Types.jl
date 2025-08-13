using POMDPs
using POMDPTools
using Distributions
using Random
using LinearAlgebra
using Infiltrator
using DataFrames
using CSV

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
using DataFrames
using CSV
using Plots

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

# Add equality and hash methods for SensingAction to fix duplicate key issues
Base.isequal(a1::SensingAction, a2::SensingAction) = 
    a1.agent_id == a2.agent_id && 
    a1.target_cells == a2.target_cells && 
    a1.communicate == a2.communicate

Base.:(==)(a1::SensingAction, a2::SensingAction) = isequal(a1, a2)

Base.hash(a::SensingAction, h::UInt) = 
    hash(a.agent_id, hash(a.target_cells, hash(a.communicate, h)))

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
RangeLimitedSensor - Model of a range-limited sensor with different patterns
"""
struct RangeLimitedSensor
    range::Float64           # Sensing range
    field_of_view::Float64   # Field of view angle (radians)
    noise_level::Float64     # Observation noise
    pattern::Symbol          # Pattern type: :circular, :row_only, :cross
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
ComplexTrajectory - Complex periodic trajectory with multiple waypoints
Implements the pattern: starts at second column, goes up, then to fourth column, goes up, then repeats
"""
struct ComplexTrajectory <: Trajectory
    waypoints::Vector{Tuple{Int, Int}}  # Sequence of waypoints to visit
    period::Int                         # Total period of the trajectory
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
    battery_level::Float64            # Current battery level
    max_battery::Float64              # Maximum battery capacity
    charging_rate::Float64            # Battery charging rate per timestep
    observation_cost::Float64         # Battery cost per observation
    reactive_policy::Any              # Reactive policy function (for policy tree planner)
end

# Constructor with default observation history and plan index
function Agent(id::Int, trajectory::Trajectory, sensor::RangeLimitedSensor, phase_offset::Int, belief::Any)
    return Agent(id, trajectory, sensor, phase_offset, belief, GridObservation[], 1, 100.0, 100.0, 1.0, 2.0, nothing)
end

# Constructor with default belief, observation history, and plan index
function Agent(id::Int, trajectory::Trajectory, sensor::RangeLimitedSensor, phase_offset::Int)
    return Agent(id, trajectory, sensor, phase_offset, nothing, GridObservation[], 1, 100.0, 100.0, 1.0, 2.0, nothing)
end

# Constructor with custom battery parameters
function Agent(id::Int, trajectory::Trajectory, sensor::RangeLimitedSensor, phase_offset::Int, 
               max_battery::Float64, charging_rate::Float64, observation_cost::Float64)
    return Agent(id, trajectory, sensor, phase_offset, nothing, GridObservation[], 1, max_battery, max_battery, charging_rate, observation_cost, nothing)
end

# Constructor with all parameters
function Agent(id::Int, trajectory::Trajectory, sensor::RangeLimitedSensor, phase_offset::Int, belief::Any,
               max_battery::Float64, charging_rate::Float64, observation_cost::Float64)
    return Agent(id, trajectory, sensor, phase_offset, belief, GridObservation[], 1, max_battery, max_battery, charging_rate, observation_cost, nothing)
end

# Export all types
export EventState, NO_EVENT, EVENT_PRESENT, EVENT_SPREADING, EVENT_DECAYING
export EventState2, NO_EVENT_2, EVENT_PRESENT_2
export EventState4, NO_EVENT_4, EVENT_PRESENT_4, EVENT_SPREADING_4, EVENT_DECAYING_4
export SensingAction, GridObservation, RangeLimitedSensor
export Trajectory, CircularTrajectory, LinearTrajectory, ComplexTrajectory
export EventDynamics, TwoStateEventDynamics
export DeterministicDistribution, Deterministic
export Belief
export Agent

# Export helper functions
export get_event_probability, set_event_probability!, get_event_probability_vector, set_event_probability_vector!, normalize_cell_distribution!

# Battery management functions
"""
update_battery!(agent::Agent, num_observations::Int=0)
Update agent's battery level: charge by charging_rate, discharge by observation_cost * num_observations
"""
function update_battery!(agent::Agent, num_observations::Int=0)
    # Charge the battery
    agent.battery_level = min(agent.max_battery, agent.battery_level + agent.charging_rate)
    
    # Discharge for observations
    total_cost = agent.observation_cost * num_observations
    agent.battery_level = max(0.0, agent.battery_level - total_cost)
end

"""
can_observe(agent::Agent, num_observations::Int=1)
Check if agent has enough battery to make observations
"""
function can_observe(agent::Agent, num_observations::Int=1)
    required_energy = agent.observation_cost * num_observations
    return agent.battery_level >= required_energy
end

"""
get_battery_percentage(agent::Agent)
Get battery level as a percentage
"""
function get_battery_percentage(agent::Agent)
    return (agent.battery_level / agent.max_battery) * 100.0
end

"""
simulate_battery_evolution(agent::Agent, action::SensingAction, current_battery::Float64)
Simulate battery evolution for one timestep: charge by charging_rate, discharge by observation_cost * num_observations
Returns the new battery level
"""
function simulate_battery_evolution(agent::Agent, action::SensingAction, current_battery::Float64)
    # Charge the battery
    new_battery = min(agent.max_battery, current_battery + agent.charging_rate)
    
    # Discharge for observations
    num_observations = length(action.target_cells)
    total_cost = agent.observation_cost * num_observations
    new_battery = max(0.0, new_battery - total_cost+agent.charging_rate)
    new_battery = min(agent.max_battery, new_battery)
    return new_battery
end

"""
check_battery_feasible(agent::Agent, action::SensingAction, current_battery::Float64)
Check if an action is feasible given current battery level
"""
function check_battery_feasible(agent::Agent, action::SensingAction, current_battery::Float64)
    return current_battery >= 0.0
end

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

"""
Enhanced event tracking that also tracks detection times
"""
mutable struct EnhancedEventTracker
    cell_active_event_id::Dict{Tuple{Int,Int}, Union{Nothing,Int}}
    event_registry::Dict{Int, Dict}
    next_event_id::Int
end

"""
Initialize enhanced event tracker
"""
function initialize_enhanced_event_tracker()
    return EnhancedEventTracker(
        Dict{Tuple{Int,Int}, Union{Nothing,Int}}(),
        Dict{Int, Dict}(),
        1
    )
end

"""
Update enhanced event tracking for a timestep
"""
function update_enhanced_event_tracking!(tracker::EnhancedEventTracker, prev_environment::Matrix{EventState}, 
                                        curr_environment::Matrix{EventState}, timestep::Int)
    height, width = size(curr_environment)
    
    for y in 1:height, x in 1:width
        cell = (x, y)
        prev_state = prev_environment[y, x]
        curr_state = curr_environment[y, x]
        
        if prev_state == NO_EVENT && curr_state == EVENT_PRESENT
            # New event started
            event_id = tracker.next_event_id
            tracker.cell_active_event_id[cell] = event_id
            tracker.event_registry[event_id] = Dict(
                :cell => cell,
                :start_time => timestep,
                :observed => false,
                :detection_time => nothing,  # Track when event was first detected
                :end_time => nothing
            )
            tracker.next_event_id += 1
            
        elseif prev_state == EVENT_PRESENT && curr_state == NO_EVENT
            # Event ended
            event_id = tracker.cell_active_event_id[cell]
            if event_id !== nothing
                tracker.event_registry[event_id][:end_time] = timestep
                tracker.cell_active_event_id[cell] = nothing
            end
        end
    end
end

"""
Mark events as observed with detection time tracking
"""
function mark_observed_events_with_time!(tracker::EnhancedEventTracker, agent_observations::Vector{Tuple{Int, Vector{Tuple{Tuple{Int,Int}, EventState}}}}, current_timestep::Int)
    for (agent_id, observations) in agent_observations
        for (cell, observed_state) in observations
            if observed_state == EVENT_PRESENT
                event_id = tracker.cell_active_event_id[cell]
                if event_id !== nothing
                    event_info = tracker.event_registry[event_id]
                    if !event_info[:observed]
                        # First time this event is observed
                        event_info[:observed] = true
                        event_info[:detection_time] = current_timestep
                    end
                end
            end
        end
    end
end

"""
Get event statistics from enhanced tracker
"""
function get_event_statistics(tracker::EnhancedEventTracker)
    total_events = length(tracker.event_registry)
    observed_events = count(e -> e[:observed], collect(values(tracker.event_registry)))
    return total_events, observed_events
end

"""
Save agent actions to CSV file

Parameters:
- action_history: Vector of Vector{SensingAction} for each timestep
- results_dir: Base directory for results
- run_number: Current run number
- planning_mode: Planning mode used
- num_steps: Total number of simulation steps

Returns: Path to the saved CSV file
"""
function save_agent_actions_to_csv(action_history::Vector{Vector{SensingAction}}, 
                                  results_dir::String, 
                                  run_number::Int, 
                                  planning_mode::Symbol,
                                  num_steps::Int)
    # Create the metrics directory path
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "metrics")
    if !isdir(metrics_dir)
        mkpath(metrics_dir)
    end
    
    filename = "agent_actions_$(planning_mode)_run$(run_number).csv"
    filepath = joinpath(metrics_dir, filename)
    
    # Prepare data for CSV
    rows = []
    for timestep in 1:length(action_history)
        actions = action_history[timestep]
        for action in actions
            # Convert target cells to string representation
            target_cells_str = join([string(cell) for cell in action.target_cells], ";")
            
            row = Dict(
                :timestep => timestep - 1,  # Convert to 0-based indexing
                :agent_id => action.agent_id,
                :target_cells => target_cells_str,
                :num_targets => length(action.target_cells),
                :communicate => action.communicate
            )
            push!(rows, row)
        end
    end
    
    # Create DataFrame and save
    df = DataFrame(rows)
    CSV.write(filepath, df)
    
    println("📁 Agent actions saved to: $(filepath)")
    return filepath
end

"""
Calculate NDD with expected lifetime (using cell-specific RSP parameters)

Parameters:
- event_tracker: EnhancedEventTracker with event information
- env: Environment with RSP parameters
- simulation_end_time: End time of simulation (for ongoing events)

Returns: NDD value using expected lifetime
"""
function calculate_ndd_expected_lifetime(event_tracker::EnhancedEventTracker, 
                                       env, 
                                       simulation_end_time::Int)
    detected_events = filter(e -> e[:observed], collect(values(event_tracker.event_registry)))
    
    if isempty(detected_events)
        return 0.0  # No detected events
    end
    
    total_ndd = 0.0
    
    for event in detected_events
        # Calculate detection delay
        t_start = event[:start_time]
        t_detect = event[:detection_time]
        
        # Get the cell where this event occurred
        cell = event[:cell]
        x, y = cell
        
        # Get cell-specific parameters for this event
        cell_params = get_cell_rsp_params(env.rsp_params, y, x)
        
        # Calculate expected lifetime E[L_e] for RSP events using cell-specific parameters
        # For RSP, E[L] = 1/μ where μ is the death probability
        cell_mu = 1.0 - cell_params.delta
        expected_lifetime = 1.0 / cell_mu
        
        # Calculate normalized delay for this event
        detection_delay = t_detect - t_start
        normalized_delay = detection_delay / expected_lifetime
        
        total_ndd += normalized_delay
    end
    
    # Average over all detected events
    ndd_expected = total_ndd / length(detected_events)
    
    return ndd_expected
end

"""
Calculate NDD with actual recorded lifetime (using simulation timeline)

Parameters:
- event_tracker: EnhancedEventTracker with event information
- simulation_end_time: End time of simulation (for ongoing events)

Returns: NDD value using actual lifetime
"""
function calculate_ndd_actual_lifetime(event_tracker::EnhancedEventTracker, 
                                     simulation_end_time::Int)
    detected_events = filter(e -> e[:observed], collect(values(event_tracker.event_registry)))
    
    if isempty(detected_events)
        return 0.0  # No detected events
    end
    
    total_ndd = 0.0
    
    for event in detected_events
        # Calculate detection delay
        t_start = event[:start_time]
        t_detect = event[:detection_time]
        
        # Calculate actual lifetime
        if event[:end_time] !== nothing
            # Event ended during simulation
            actual_lifetime = event[:end_time] - event[:start_time]
        else
            # Event is ongoing at end of simulation, use simulation end time
            actual_lifetime = simulation_end_time - event[:start_time]
        end
        
        # Avoid division by zero
        if actual_lifetime > 0
            # Calculate normalized delay for this event
            detection_delay = t_detect - t_start
            normalized_delay = detection_delay / actual_lifetime
            total_ndd += normalized_delay
        end
    end
    
    # Average over all detected events
    ndd_actual = total_ndd / length(detected_events)
    
    return ndd_actual
end

"""
Calculate both NDD metrics and save to CSV

Parameters:
- event_tracker: EnhancedEventTracker with event information
- env: Environment with RSP parameters
- simulation_end_time: End time of simulation
- results_dir: Base directory for results
- run_number: Current run number
- planning_mode: Planning mode used

Returns: Tuple of (ndd_expected, ndd_actual, csv_filepath)
"""
function calculate_and_save_ndd_metrics(event_tracker::EnhancedEventTracker,
                                       env,
                                       simulation_end_time::Int,
                                       results_dir::String,
                                       run_number::Int,
                                       planning_mode::Symbol)
    # Calculate both NDD metrics
    ndd_expected = calculate_ndd_expected_lifetime(event_tracker, env, simulation_end_time)
    ndd_actual = calculate_ndd_actual_lifetime(event_tracker, simulation_end_time)
    
    # Create the metrics directory path
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "metrics")
    if !isdir(metrics_dir)
        mkpath(metrics_dir)
    end
    
    filename = "ndd_metrics_$(planning_mode)_run$(run_number).csv"
    filepath = joinpath(metrics_dir, filename)
    
    # Prepare data for CSV
    rows = []
    for (event_id, event_info) in event_tracker.event_registry
        if event_info[:observed]
            # Get the cell where this event occurred
            cell = event_info[:cell]
            x, y = cell
            
            # Get cell-specific parameters for this event
            cell_params = get_cell_rsp_params(env.rsp_params, y, x)
            
            # Calculate expected lifetime
            cell_mu = 1.0 - cell_params.delta
            expected_lifetime = 1.0 / cell_mu
            
            # Calculate actual lifetime
            if event_info[:end_time] !== nothing
                actual_lifetime = event_info[:end_time] - event_info[:start_time]
            else
                actual_lifetime = simulation_end_time - event_info[:start_time]
            end
            
            # Calculate detection delays
            detection_delay = event_info[:detection_time] - event_info[:start_time]
            ndd_expected_event = detection_delay / expected_lifetime
            ndd_actual_event = actual_lifetime > 0 ? detection_delay / actual_lifetime : 0.0
            
            row = Dict(
                :event_id => event_id,
                :cell_x => cell[1],
                :cell_y => cell[2],
                :start_time => event_info[:start_time],
                :detection_time => event_info[:detection_time],
                :end_time => event_info[:end_time] !== nothing ? event_info[:end_time] : simulation_end_time,
                :detection_delay => detection_delay,
                :expected_lifetime => expected_lifetime,
                :actual_lifetime => actual_lifetime,
                :ndd_expected => ndd_expected_event,
                :ndd_actual => ndd_actual_event
            )
            push!(rows, row)
        end
    end
    
    # Add summary row
    summary_row = Dict(
        :event_id => "SUMMARY",
        :cell_x => -1,
        :cell_y => -1,
        :start_time => -1,
        :detection_time => -1,
        :end_time => -1,
        :detection_delay => -1,
        :expected_lifetime => -1,
        :actual_lifetime => -1,
        :ndd_expected => ndd_expected,
        :ndd_actual => ndd_actual
    )
    push!(rows, summary_row)
    
    # Create DataFrame and save
    df = DataFrame(rows)
    CSV.write(filepath, df)
    
    println("📁 NDD metrics saved to: $(filepath)")
    println("  NDD (expected lifetime): $(round(ndd_expected, digits=3))")
    println("  NDD (actual lifetime): $(round(ndd_actual, digits=3))")
    
    return ndd_expected, ndd_actual, filepath
end

# Export the new functions
export save_agent_actions_to_csv, calculate_ndd_expected_lifetime, calculate_ndd_actual_lifetime, calculate_and_save_ndd_metrics
export EnhancedEventTracker, initialize_enhanced_event_tracker, update_enhanced_event_tracking!, mark_observed_events_with_time!, get_event_statistics
export save_event_tracking_data, save_uncertainty_evolution_data, save_sync_event_data, create_observation_heatmap

"""
Save detailed event tracking data to CSV
"""
function save_event_tracking_data(event_tracker::EnhancedEventTracker, results_dir, run_number, planning_mode)
    # Create the metrics directory path
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "metrics")
    if !isdir(metrics_dir)
        mkpath(metrics_dir)
    end
    
    filename = "event_tracking_$(planning_mode)_run$(run_number).csv"
    filepath = joinpath(metrics_dir, filename)
    
    # Prepare data for CSV
    rows = []
    for (event_id, event_info) in event_tracker.event_registry
        row = Dict(
            :event_id => event_id,
            :cell_x => event_info[:cell][1],
            :cell_y => event_info[:cell][2],
            :start_time => event_info[:start_time],
            :end_time => event_info[:end_time] !== nothing ? event_info[:end_time] : -1,
            :observed => event_info[:observed],
            :detection_time => event_info[:detection_time] !== nothing ? event_info[:detection_time] : -1,
            :lifetime => event_info[:end_time] !== nothing ? event_info[:end_time] - event_info[:start_time] : -1,
            :detection_delay => event_info[:observed] && event_info[:detection_time] !== nothing ? 
                               event_info[:detection_time] - event_info[:start_time] : -1
        )
        push!(rows, row)
    end
    
    # Create DataFrame and save
    df = DataFrame(rows)
    CSV.write(filepath, df)
    
    println("📁 Event tracking data saved to: $(filepath)")
    return filepath
end

"""
Save uncertainty evolution data to CSV
"""
function save_uncertainty_evolution_data(uncertainty_evolution, avg_uncertainty, results_dir, run_number, planning_mode)
    # Create the metrics directory path
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "metrics")
    if !isdir(metrics_dir)
        mkpath(metrics_dir)
    end
    
    filename = "uncertainty_evolution_$(planning_mode)_run$(run_number).csv"
    filepath = joinpath(metrics_dir, filename)
    
    # Prepare data for CSV
    rows = []
    for (timestep, avg_unc) in enumerate(avg_uncertainty)
        row = Dict(
            :timestep => timestep - 1,  # Convert to 0-based indexing
            :average_uncertainty => avg_unc
        )
        push!(rows, row)
    end
    
    # Create DataFrame and save
    df = DataFrame(rows)
    CSV.write(filepath, df)
    
    println("📁 Uncertainty evolution data saved to: $(filepath)")
    return filepath
end

"""
Save sync event data to CSV
"""
function save_sync_event_data(sync_events, results_dir, run_number, planning_mode)
    # Create the metrics directory path
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "metrics")
    if !isdir(metrics_dir)
        mkpath(metrics_dir)
    end
    
    filename = "sync_events_$(planning_mode)_run$(run_number).csv"
    filepath = joinpath(metrics_dir, filename)
    
    # Prepare data for CSV
    rows = []
    for (timestep, agent_id) in sync_events
        row = Dict(
            :timestep => timestep,
            :agent_id => agent_id
        )
        push!(rows, row)
    end
    
    # Create DataFrame and save
    df = DataFrame(rows)
    CSV.write(filepath, df)
    
    println("📁 Sync event data saved to: $(filepath)")
    return filepath
end

"""
Create and save a heatmap showing the number of observations per cell
"""
function create_observation_heatmap(action_history, grid_width, grid_height, results_dir, run_number, planning_mode)
    # Create the plots directory path
    plots_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "plots")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end
    
    # Initialize observation count matrix
    observation_counts = zeros(Int, grid_height, grid_width)
    
    # Count observations for each cell
    for timestep_actions in action_history
        for action in timestep_actions
            for cell in action.target_cells
                x, y = cell
                if 1 <= x <= grid_width && 1 <= y <= grid_height
                    observation_counts[y, x] += 1
                end
            end
        end
    end
    
    # Create heatmap
    p = heatmap(
        observation_counts,
        title="Cell Observation Frequency - $(planning_mode), Run $(run_number)",
        xlabel="X Coordinate",
        ylabel="Y Coordinate",
        colorbar_title="Number of Observations",
        colormap=:plasma,
        aspect_ratio=:equal,
        size=(600, 500)
    )
    
    # Save the plot
    plot_filename = joinpath(plots_dir, "observation_heatmap_$(planning_mode)_run$(run_number).png")
    savefig(p, plot_filename)
    println("📁 Observation heatmap saved to: $(basename(plot_filename))")
    
    # Also save the raw data as CSV in metrics directory
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "metrics")
    if !isdir(metrics_dir)
        mkpath(metrics_dir)
    end
    csv_filename = joinpath(metrics_dir, "observation_counts_$(planning_mode)_run$(run_number).csv")
    df = DataFrame(
        x = repeat(1:grid_width, outer=grid_height),
        y = repeat(1:grid_height, inner=grid_width),
        observations = vec(observation_counts)
    )
    CSV.write(csv_filename, df)
    println("📁 Observation counts data saved to: $(basename(csv_filename))")
    
    return plot_filename, csv_filename
end

# Export the additional saving functions
export save_event_tracking_data, save_uncertainty_evolution_data, save_sync_event_data

end # module Types 
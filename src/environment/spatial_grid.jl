using POMDPs
using POMDPTools
using Distributions
using Random
using LinearAlgebra

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
Trajectory - Deterministic periodic trajectory for an agent
"""
abstract type Trajectory end

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
GridState - Represents the state of the spatial grid
"""
struct GridState
    event_map::Matrix{EventState}  # Event states at each cell
    agent_positions::Vector{Tuple{Int, Int}}
    agent_trajectories::Vector{Trajectory}
    time_step::Int
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
RangeLimitedSensor - Model of a range-limited sensor
"""
struct RangeLimitedSensor
    range::Float64           # Sensing range
    field_of_view::Float64   # Field of view angle (radians)
    noise_level::Float64     # Observation noise
end

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
Agent - Represents an agent with a deterministic periodic trajectory
"""
struct Agent
    id::Int
    trajectory::Trajectory
    sensor::RangeLimitedSensor
    current_time::Int
end

"""
SpatialGrid - A 2D discretized environment for multi-agent information gathering
"""
struct SpatialGrid <: POMDP{GridState, SensingAction, GridObservation}
    width::Int
    height::Int
    event_dynamics::EventDynamics
    agents::Vector{Agent}  # Added agents field
    sensor_range::Float64
    discount::Float64
    initial_events::Int    # Number of initial events
    max_sensing_targets::Int  # Maximum cells an agent can sense per step
end

# POMDP interface functions

"""
POMDPs.initialstate(pomdp::SpatialGrid)
Creates initial state distribution for the spatial grid
"""
function POMDPs.initialstate(pomdp::SpatialGrid)
    # Create initial event map with random events
    event_map = fill(NO_EVENT, pomdp.height, pomdp.width)
    
    # Initialize random events
    rng = Random.GLOBAL_RNG
    initialize_random_events(event_map, pomdp.initial_events, rng)
    
    # Get initial agent positions from trajectories
    agent_positions = Vector{Tuple{Int, Int}}()
    agent_trajectories = Vector{Trajectory}()
    
    for agent in pomdp.agents
        initial_pos = get_position_at_time(agent.trajectory, 0)
        push!(agent_positions, initial_pos)
        push!(agent_trajectories, agent.trajectory)
    end
    
    initial_state = GridState(event_map, agent_positions, agent_trajectories, 0)
    
    # Return deterministic distribution
    return Deterministic(initial_state)
end

"""
POMDPs.transition(pomdp::SpatialGrid, s::GridState, a::SensingAction)
Implements state transition function
"""
function POMDPs.transition(pomdp::SpatialGrid, s::GridState, a::SensingAction)
    # Create new state
    new_event_map = copy(s.event_map)
    new_time_step = s.time_step + 1
    
    # Update event dynamics
    rng = Random.GLOBAL_RNG
    update_events!(pomdp.event_dynamics, new_event_map, rng)
    
    # Update agent positions based on trajectories
    new_agent_positions = Vector{Tuple{Int, Int}}()
    for (i, trajectory) in enumerate(s.agent_trajectories)
        new_pos = get_position_at_time(trajectory, new_time_step)
        push!(new_agent_positions, new_pos)
    end
    
    new_state = GridState(new_event_map, new_agent_positions, s.agent_trajectories, new_time_step)
    
    # Return deterministic transition
    return Deterministic(new_state)
end

"""
POMDPs.observation(pomdp::SpatialGrid, a::SensingAction, sp::GridState)
Generates observations for the sensing action
"""
function POMDPs.observation(pomdp::SpatialGrid, a::SensingAction, sp::GridState)
    # Get agent and sensor
    agent = pomdp.agents[a.agent_id]
    agent_pos = sp.agent_positions[a.agent_id]
    
    # Generate observation using sensor model
    sensed_cells, event_states = generate_observation(
        agent.sensor, 
        agent_pos, 
        sp.event_map, 
        a.target_cells
    )
    
    # Create observation
    obs = GridObservation(
        a.agent_id,
        sensed_cells,
        event_states,
        []  # No communication received for now
    )
    
    return Deterministic(obs)
end

"""
POMDPs.reward(pomdp::SpatialGrid, s::GridState, a::SensingAction, sp::GridState)
Calculates reward based on information gain
"""
function POMDPs.reward(pomdp::SpatialGrid, s::GridState, a::SensingAction, sp::GridState)
    # Get agent
    agent = pomdp.agents[a.agent_id]
    agent_pos = s.agent_positions[a.agent_id]
    
    # Calculate information gain from sensing
    # For now, use a simple belief matrix (uniform prior)
    belief = fill(0.5, pomdp.height, pomdp.width)
    
    # Get observation
    sensed_cells, event_states = generate_observation(
        agent.sensor, 
        agent_pos, 
        sp.event_map, 
        a.target_cells
    )
    
    # Calculate information gain
    info_gain = calculate_information_gain(belief, sensed_cells, event_states)
    
    # Apply communication cost if communicating
    communication_cost = a.communicate ? -0.1 : 0.0
    
    # Apply sensing cost based on number of targets
    sensing_cost = -0.01 * length(a.target_cells)
    
    # Total reward
    total_reward = info_gain + communication_cost + sensing_cost
    
    return total_reward
end

"""
POMDPs.discount(pomdp::SpatialGrid)
Returns discount factor
"""
function POMDPs.discount(pomdp::SpatialGrid)
    return pomdp.discount
end

"""
POMDPs.isterminal(pomdp::SpatialGrid, s::GridState)
Checks if state is terminal
"""
function POMDPs.isterminal(pomdp::SpatialGrid, s::GridState)
    # Terminal if maximum time reached or all events disappeared
    max_time = 1000  # Arbitrary maximum time
    
    if s.time_step >= max_time
        return true
    end
    
    # Check if all events have disappeared
    if all(event -> event == NO_EVENT, s.event_map)
        return true
    end
    
    return false
end

"""
POMDPs.actions(pomdp::SpatialGrid)
Returns action space for a given state
"""
function POMDPs.actions(pomdp::SpatialGrid, s::GridState = nothing)
    # For now, return a simple action space
    # In practice, this would depend on the current state and agent positions
    return [SensingAction(1, [], false)]  # Placeholder
end

"""
POMDPs.states(pomdp::SpatialGrid)
Returns state space
"""
function POMDPs.states(pomdp::SpatialGrid)
    # State space is too large to enumerate explicitly
    # Return a placeholder for now
    return nothing
end

"""
POMDPs.observations(pomdp::SpatialGrid)
Returns observation space
"""
function POMDPs.observations(pomdp::SpatialGrid)
    # Observation space is too large to enumerate explicitly
    # Return a placeholder for now
    return nothing
end

# Helper functions

"""
get_position_at_time(trajectory::CircularTrajectory, time::Int)
Gets agent position at a specific time for circular trajectory
"""
function get_position_at_time(trajectory::CircularTrajectory, time::Int)
    angle = 2π * (time % trajectory.period) / trajectory.period
    x = trajectory.center_x + round(Int, trajectory.radius * cos(angle))
    y = trajectory.center_y + round(Int, trajectory.radius * sin(angle))
    return (x, y)
end

"""
get_position_at_time(trajectory::LinearTrajectory, time::Int)
Gets agent position at a specific time for linear trajectory
"""
function get_position_at_time(trajectory::LinearTrajectory, time::Int)
    t = (time % trajectory.period) / trajectory.period
    x = round(Int, trajectory.start_x + t * (trajectory.end_x - trajectory.start_x))
    y = round(Int, trajectory.start_y + t * (trajectory.end_y - trajectory.start_y))
    return (x, y)
end

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
        
        # Simple belief update based on observation
        if observed_states[i] == EVENT_PRESENT
            belief[y, x] = 0.9  # High probability of event
        elseif observed_states[i] == NO_EVENT
            belief[y, x] = 0.1  # Low probability of event
        end
        
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
    return -(probability * log(probability) + (1 - probability) * log(1 - probability))
end

"""
initialize_random_events(event_map::Matrix{EventState}, num_events::Int, rng::AbstractRNG)
Initializes random events in the grid
"""
function initialize_random_events(event_map::Matrix{EventState}, num_events::Int, rng::AbstractRNG)
    height, width = size(event_map)
    
    for _ in 1:num_events
        x = rand(rng, 1:width)
        y = rand(rng, 1:height)
        event_map[y, x] = EVENT_PRESENT
    end
end

"""
update_events!(dynamics::EventDynamics, event_map::Matrix{EventState}, rng::AbstractRNG)
Updates event states based on stochastic dynamics
"""
function update_events!(dynamics::EventDynamics, event_map::Matrix{EventState}, rng::AbstractRNG)
    height, width = size(event_map)
    new_event_map = copy(event_map)
    
    for y in 1:height
        for x in 1:width
            current_state = event_map[y, x]
            neighbor_states = get_neighbor_states(event_map, x, y)
            
            # Calculate transition probabilities
            if current_state == NO_EVENT
                # Probability of new event
                prob = calculate_spread_probability(dynamics, current_state, neighbor_states)
                if rand(rng) < prob
                    new_event_map[y, x] = EVENT_PRESENT
                end
            elseif current_state == EVENT_PRESENT
                # Probability of spreading or decaying
                if rand(rng) < dynamics.spread_rate
                    new_event_map[y, x] = EVENT_SPREADING
                elseif rand(rng) < dynamics.decay_rate
                    new_event_map[y, x] = EVENT_DECAYING
                end
            elseif current_state == EVENT_SPREADING
                # Continue spreading or start decaying
                if rand(rng) < dynamics.decay_rate
                    new_event_map[y, x] = EVENT_DECAYING
                end
            elseif current_state == EVENT_DECAYING
                # Probability of disappearing
                if rand(rng) < dynamics.death_rate
                    new_event_map[y, x] = NO_EVENT
                end
            end
        end
    end
    
    # Update the original event map
    event_map .= new_event_map
end

"""
calculate_spread_probability(dynamics::EventDynamics, current_state::EventState, neighbor_states::Vector{EventState})
Calculates probability of event spreading
"""
function calculate_spread_probability(dynamics::EventDynamics, current_state::EventState, neighbor_states::Vector{EventState})
    if current_state == NO_EVENT
        # Probability of new event based on neighbor influence
        neighbor_events = count(x -> x != NO_EVENT, neighbor_states)
        return dynamics.birth_rate * (1 + dynamics.neighbor_influence * neighbor_events)
    elseif current_state == EVENT_PRESENT
        return dynamics.spread_rate
    elseif current_state == EVENT_SPREADING
        return dynamics.spread_rate
    else  # EVENT_DECAYING
        return dynamics.death_rate
    end
end

"""
get_neighbor_states(event_map::Matrix{EventState}, x::Int, y::Int)
Gets the states of neighboring cells (8-connectivity)
"""
function get_neighbor_states(event_map::Matrix{EventState}, x::Int, y::Int)
    neighbors = EventState[]
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
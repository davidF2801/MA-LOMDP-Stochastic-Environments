using POMDPs
using POMDPTools
using Distributions
using Random
using LinearAlgebra
using ..Types

# Import types from the parent module
import ..Types.EventState, ..Types.NO_EVENT, ..Types.EVENT_PRESENT, ..Types.EVENT_SPREADING, ..Types.EVENT_DECAYING
import ..Types.SensingAction, ..Types.GridObservation, ..Types.Agent
import ..Types.Trajectory, ..Types.CircularTrajectory, ..Types.LinearTrajectory, ..Types.RangeLimitedSensor
import ..Types.EventDynamics, ..Types.TwoStateEventDynamics, ..Types.Deterministic

# Import belief management types from Agents module
import ..Agents.BeliefManagement: initialize_belief, Belief

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
LocalityFunction - Represents the footprint function g_i for each agent
Maps agent phase τ to set of observable regions (cells)
"""
struct LocalityFunction
    agent_id::Int
    trajectory::Trajectory
    sensor::RangeLimitedSensor
    grid_width::Int
    grid_height::Int
end

"""
Get observable cells for agent at a specific phase/time
g_i(τ) → set of observable regions

This returns the FOR (Field of Regard) - all cells within sensor range.
The FOV (Field of View) is then a subset of these cells chosen as actions.
"""
function get_observable_cells(locality::LocalityFunction, phase::Int)
    # Get agent position at this phase
    agent_pos = get_position_at_time(locality.trajectory, phase)
    
    # Get all cells within sensor range (FOR - Field of Regard)
    observable_cells = Tuple{Int, Int}[]
    
    for y in 1:locality.grid_height
        for x in 1:locality.grid_width
            if is_within_range(locality.sensor, agent_pos, (x, y))
                push!(observable_cells, (x, y))
            end
        end
    end
    
    return observable_cells
end

"""
Get locality set for agent i and cell k
L_i(k) = {τ ∈ {0, ..., T_i-1} | k ∈ g_i(τ)}
"""
function get_locality_set(locality::LocalityFunction, cell::Tuple{Int, Int})
    locality_set = Int[]
    period = get_trajectory_period(locality.trajectory)
    
    for phase in 0:(period-1)
        observable_cells = get_observable_cells(locality, phase)
        if cell in observable_cells
            push!(locality_set, phase)
        end
    end
    
    return locality_set
end

"""
Get trajectory period
"""
function get_trajectory_period(trajectory::CircularTrajectory)
    return trajectory.period
end

function get_trajectory_period(trajectory::LinearTrajectory)
    return trajectory.period
end

"""
SpatialGrid - A 2D discretized environment for multi-agent information gathering
Supports rectangular grids with separate width and height
"""
struct SpatialGrid <: POMDP{GridState, SensingAction, GridObservation}
    width::Int
    height::Int
    event_dynamics::EventDynamics
    agents::Vector{Agent}  # Added agents field
    locality_functions::Vector{LocalityFunction}  # Locality functions for each agent
    sensor_range::Float64
    discount::Float64
    initial_events::Int    # Number of initial events
    max_sensing_targets::Int  # Maximum cells an agent can sense per step
    ground_station_pos::Tuple{Int, Int}  # Position of the ground station
end

"""
Constructor for SpatialGrid with rectangular dimensions
"""
function SpatialGrid(width::Int, height::Int, event_dynamics::EventDynamics, agents::Vector{Agent}, 
                    sensor_range::Float64, discount::Float64, initial_events::Int, max_sensing_targets::Int, ground_station_pos::Tuple{Int,Int})
    # Create locality functions for each agent
    locality_functions = Vector{LocalityFunction}()
    for agent in agents
        locality = LocalityFunction(agent.id, agent.trajectory, agent.sensor, width, height)
        push!(locality_functions, locality)
    end
    
    return SpatialGrid(width, height, event_dynamics, agents, locality_functions, sensor_range, discount, initial_events, max_sensing_targets, ground_station_pos)
end

"""
Constructor for SpatialGrid with TwoStateEventDynamics (converts to EventDynamics)
"""
function SpatialGrid(width::Int, height::Int, agents::Vector{Agent}, 
                    two_state_dynamics::TwoStateEventDynamics, locality_functions::Vector{LocalityFunction}, ground_station_pos::Tuple{Int,Int})
    # Convert TwoStateEventDynamics to EventDynamics
    event_dynamics = EventDynamics(
        two_state_dynamics.birth_rate,
        two_state_dynamics.death_rate,
        0.0,  # spread_rate (not used in 2-state system)
        0.0,  # decay_rate (not used in 2-state system)
        0.0   # neighbor_influence (not used in 2-state system)
    )
    
    # Default values for other parameters
    sensor_range = 2.0
    discount = 0.95
    initial_events = 1
    max_sensing_targets = 1
    
    return SpatialGrid(width, height, event_dynamics, agents, locality_functions, sensor_range, discount, initial_events, max_sensing_targets, ground_station_pos)
end

"""
Convenience constructor for square grids
"""
function SpatialGrid(grid_size::Int, event_dynamics::EventDynamics, agents::Vector{Agent}, 
                    sensor_range::Float64, discount::Float64, initial_events::Int, max_sensing_targets::Int, ground_station_pos::Tuple{Int,Int})
    return SpatialGrid(grid_size, grid_size, event_dynamics, agents, sensor_range, discount, initial_events, max_sensing_targets, ground_station_pos)
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
    
    # Get initial agent positions from trajectories and initialize beliefs
    agent_positions = Vector{Tuple{Int, Int}}()
    agent_trajectories = Vector{Trajectory}()
    
    for agent in pomdp.agents
        initial_pos = get_position_at_time(agent.trajectory, 0)
        push!(agent_positions, initial_pos)
        push!(agent_trajectories, agent.trajectory)
        
        # Initialize agent's belief if not already initialized
        if !isdefined(agent, :belief) || (agent.belief isa Array && isempty(agent.belief)) || (agent.belief isa Belief && agent.belief.last_update == -1)
            # Initialize belief with uniform prior
            agent.belief = initialize_belief(pomdp.width, pomdp.height, 0.5)
        end
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
update_agent_belief!(agent::Agent, action::SensingAction, observation::GridObservation, env::SpatialGrid)
Updates the agent's belief state after an observation
"""
function update_agent_belief!(agent::Agent, action::SensingAction, observation::GridObservation, env::SpatialGrid)
    # Update agent's belief using the belief management module
    agent.belief = update_belief_state(agent.belief, action, observation, env.event_dynamics)
end

"""
POMDPs.observation(pomdp::SpatialGrid, a::SensingAction, sp::GridState)
Generates observations for the sensing action using locality functions
"""
function POMDPs.observation(pomdp::SpatialGrid, a::SensingAction, sp::GridState)
    # Get agent and sensor
    agent = pomdp.agents[a.agent_id]
    agent_pos = sp.agent_positions[a.agent_id]
    
    # Get observable cells using locality function
    locality = pomdp.locality_functions[a.agent_id]
    period = get_trajectory_period(agent.trajectory)
    current_phase = sp.time_step % period
    observable_cells = get_observable_cells(locality, current_phase)
    
    # Filter target cells to only include observable ones
    valid_targets = filter(cell -> cell in observable_cells, a.target_cells)
    
    # Generate observation using sensor model
    sensed_cells, event_states = generate_observation(
        agent.sensor, 
        agent_pos, 
        sp.event_map, 
        valid_targets
    )
    
    # Create observation
    obs = GridObservation(
        a.agent_id,
        sensed_cells,
        event_states,
        []  # No communication received for now
    )
    
    # Update agent's belief with the new observation
    update_agent_belief!(agent, a, obs, pomdp)
    
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
    
    # Use agent's belief instead of hardcoded uniform prior
    belief = agent.belief.event_probabilities
    
    # Get observation
    sensed_cells, event_states = generate_observation(
        agent.sensor, 
        agent_pos, 
        sp.event_map, 
        a.target_cells
    )
    
    # Calculate information gain using agent's belief
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
Generate all combinations of size k from a collection
"""
function combinations(collection, k)
    if k == 0
        return [[]]
    elseif k > length(collection)
        return []
    elseif k == length(collection)
        return [collect(collection)]
    else
        result = []
        for i in 1:(length(collection)-k+1)
            for combo in combinations(collection[i+1:end], k-1)
                push!(result, [collection[i]; combo])
            end
        end
        return result
    end
end

"""
POMDPs.actions(pomdp::SpatialGrid, s::GridState)
Returns action space for a given state using locality functions
"""
function POMDPs.actions(pomdp::SpatialGrid, s::GridState)
    actions = Vector{SensingAction}()
    
    for (i, agent) in enumerate(pomdp.agents)
        # Get current phase (time step modulo period)
        period = get_trajectory_period(agent.trajectory)
        current_phase = s.time_step % period
        
        # Get observable cells using locality function
        locality = pomdp.locality_functions[i]
        observable_cells = get_observable_cells(locality, current_phase)
        
        # Generate all possible sensing actions (subsets of observable cells)
        # FOV is always exactly 1 cell for consistent sensor footprint
        if !isempty(observable_cells)
            # Individual cell sensing actions (FOV = 1 cell)
            for cell in observable_cells
                push!(actions, SensingAction(i, [cell], false))
                push!(actions, SensingAction(i, [cell], true))  # With communication
            end
        else
            # Wait action when no cells are observable
            push!(actions, SensingAction(i, [], false))
        end
    end
    
    return actions
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
Calculates information gain from observations without modifying the belief
"""
function calculate_information_gain(belief::Matrix{Float64}, observed_cells::Vector{Tuple{Int, Int}}, observed_states::Vector{EventState})
    total_gain = 0.0
    
    for (i, cell) in enumerate(observed_cells)
        x, y = cell
        prior_entropy = calculate_entropy(belief[y, x])
        
        # Calculate posterior probability based on observation (without modifying belief)
        posterior_prob = calculate_posterior_probability(belief[y, x], observed_states[i])
        
        posterior_entropy = calculate_entropy(posterior_prob)
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
calculate_posterior_probability(prior_prob::Float64, observed_state::EventState)
Calculates posterior probability given observation without modifying the original belief
"""
function calculate_posterior_probability(prior_prob::Float64, observed_state::EventState)
    if observed_state == EVENT_PRESENT
        return 0.9  # High probability of event given observation
    elseif observed_state == NO_EVENT
        return 0.1  # Low probability of event given observation
    else
        return prior_prob  # No change for other states
    end
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

"""
create_agent(id::Int, trajectory::Trajectory, sensor::RangeLimitedSensor, grid_width::Int, grid_height::Int)
Creates an agent with properly initialized belief state
"""
function create_agent(id::Int, trajectory::Trajectory, sensor::RangeLimitedSensor, grid_width::Int, grid_height::Int)
    # Initialize belief with uniform prior
    belief = initialize_belief(grid_width, grid_height, 0.5)
    
    return Agent(id, trajectory, sensor, 0, belief)
end 
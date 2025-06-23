# Code Structure Diagram

## Project Architecture Overview

```
MA-LOMDP-Stochastic-Environments/
│
├── Project.toml                    # Dependencies & metadata
├── Manifest.toml                   # Exact dependency versions
├── README.md                       # Project documentation
│
├── src/
│   └── MyProject.jl               # Main module entry point
│       ├── Environment/           # Environment simulation
│       └── Agents/                # Agent behavior & coordination
│
├── scripts/
│   ├── run_simulation.jl          # Main simulation script
│   └── plot_results.jl            # Visualization script
│
└── plots/                         # Output directory
```

## Detailed Module Structure

```
src/
│
├── MyProject.jl                   # Main Module
│   ├── exports: Environment, Agents
│   ├── exports: simulate_environment, update_belief, plan_sensing, evaluate_policy
│   └── includes: Environment.jl, Agents.jl
│
├── environment/
│   ├── Environment.jl             # Environment Module
│   │   ├── exports: SpatialGrid, EventDynamics, SensorModels
│   │   ├── exports: simulate_environment()
│   │   └── includes: spatial_grid.jl, event_dynamics.jl, sensor_models.jl
│   │
│   ├── spatial_grid.jl            # Core POMDP Implementation
│   │   ├── SpatialGrid <: POMDP{GridState, SensingAction, GridObservation}
│   │   ├── GridState
│   │   │   ├── event_map::Matrix{EventState}
│   │   │   ├── agent_positions::Vector{Tuple{Int, Int}}
│   │   │   ├── agent_trajectories::Vector{Trajectory}
│   │   │   └── time_step::Int
│   │   ├── EventState (enum)
│   │   │   ├── NO_EVENT
│   │   │   ├── EVENT_PRESENT
│   │   │   ├── EVENT_SPREADING
│   │   │   └── EVENT_DECAYING
│   │   ├── SensingAction
│   │   │   ├── agent_id::Int
│   │   │   ├── target_cells::Vector{Tuple{Int, Int}}
│   │   │   └── communicate::Bool
│   │   ├── GridObservation
│   │   │   ├── agent_id::Int
│   │   │   ├── sensed_cells::Vector{Tuple{Int, Int}}
│   │   │   ├── event_states::Vector{EventState}
│   │   │   └── communication_received::Vector{Any}
│   │   ├── Trajectory (abstract)
│   │   │   ├── CircularTrajectory
│   │   │   │   ├── center_x::Int, center_y::Int
│   │   │   │   ├── radius::Float64
│   │   │   │   └── period::Int
│   │   │   └── LinearTrajectory
│   │   │       ├── start_x::Int, start_y::Int
│   │   │       ├── end_x::Int, end_y::Int
│   │   │       └── period::Int
│   │   └── POMDP Interface Functions
│   │       ├── initialstate(pomdp::SpatialGrid)
│   │       ├── transition(pomdp::SpatialGrid, s::GridState, a::SensingAction)
│   │       ├── observation(pomdp::SpatialGrid, a::SensingAction, sp::GridState)
│   │       ├── reward(pomdp::SpatialGrid, s::GridState, a::SensingAction, sp::GridState)
│   │       ├── discount(pomdp::SpatialGrid)
│   │       ├── isterminal(pomdp::SpatialGrid, s::GridState)
│   │       ├── actions(pomdp::SpatialGrid)
│   │       ├── states(pomdp::SpatialGrid)
│   │       └── observations(pomdp::SpatialGrid)
│   │
│   ├── event_dynamics.jl          # Stochastic Event Evolution
│   │   ├── EventDynamics
│   │   │   ├── birth_rate::Float64
│   │   │   ├── death_rate::Float64
│   │   │   ├── spread_rate::Float64
│   │   │   ├── decay_rate::Float64
│   │   │   └── neighbor_influence::Float64
│   │   ├── update_events!(dynamics, event_map, rng)
│   │   ├── calculate_spread_probability(dynamics, current_state, neighbor_states)
│   │   ├── get_neighbor_states(event_map, x, y)
│   │   └── initialize_random_events(event_map, num_events, rng)
│   │
│   └── sensor_models.jl           # Range-Limited Sensor Models
│       ├── SensorModels (module)
│       ├── RangeLimitedSensor
│       │   ├── range::Float64
│       │   ├── field_of_view::Float64
│       │   └── noise_level::Float64
│       ├── generate_observation(sensor, agent_pos, event_map, target_cells)
│       ├── calculate_footprint(sensor, agent_pos, grid_width, grid_height)
│       ├── is_within_range(sensor, agent_pos, target_pos)
│       ├── calculate_information_gain(belief, observed_cells, observed_states)
│       └── calculate_entropy(probability)
│
└── agents/
    ├── Agents.jl                  # Agents Module
    │   ├── exports: TrajectoryPlanner, SensingPolicy, Communication, BeliefManagement
    │   ├── exports: update_belief(), plan_sensing(), evaluate_policy()
    │   └── includes: trajectory_planner.jl, sensing_policy.jl, communication.jl, belief_management.jl
    │
    ├── trajectory_planner.jl      # Deterministic Periodic Trajectories
    │   ├── TrajectoryPlanner (module)
    │   ├── Agent
    │   │   ├── id::Int
    │   │   ├── trajectory::Trajectory
    │   │   ├── sensor::RangeLimitedSensor
    │   │   └── current_time::Int
    │   ├── get_position_at_time(trajectory::CircularTrajectory, time::Int)
    │   ├── get_position_at_time(trajectory::LinearTrajectory, time::Int)
    │   ├── calculate_trajectory_period(trajectory)
    │   ├── update_agent_position!(agent, time)
    │   ├── get_trajectory_waypoints(trajectory, num_points)
    │   ├── create_circular_trajectory(center_x, center_y, radius, period)
    │   └── create_linear_trajectory(start_x, start_y, end_x, end_y, period)
    │
    ├── sensing_policy.jl          # Sensing Decision Strategies
    │   ├── SensingPolicy (module)
    │   ├── InformationGainPolicy <: Policy
    │   │   └── exploration_weight::Float64
    │   ├── RandomSensingPolicy <: Policy
    │   │   └── rng::AbstractRNG
    │   ├── GreedySensingPolicy <: Policy
    │   │   └── uncertainty_threshold::Float64
    │   ├── select_sensing_targets(policy::InformationGainPolicy, belief, footprint, max_targets)
    │   ├── select_sensing_targets(policy::RandomSensingPolicy, belief, footprint, max_targets)
    │   ├── select_sensing_targets(policy::GreedySensingPolicy, belief, footprint, max_targets)
    │   ├── calculate_entropy(probability)
    │   ├── create_information_gain_policy(exploration_weight)
    │   ├── create_random_sensing_policy(rng)
    │   └── create_greedy_sensing_policy(uncertainty_threshold)
    │
    ├── communication.jl           # Multi-Agent Communication Protocols
    │   ├── Communication (module)
    │   ├── CentralizedProtocol
    │   │   ├── communication_frequency::Int
    │   │   └── bandwidth_limit::Int
    │   ├── DecentralizedProtocol
    │   │   ├── neighbor_radius::Float64
    │   │   └── fusion_method::String
    │   ├── HybridProtocol
    │   │   ├── sync_frequency::Int
    │   │   └── ground_station_id::Int
    │   ├── communicate_beliefs(protocol::CentralizedProtocol, agents, beliefs, time_step)
    │   ├── communicate_beliefs(protocol::DecentralizedProtocol, agents, beliefs, positions)
    │   ├── communicate_beliefs(protocol::HybridProtocol, agents, beliefs, time_step)
    │   ├── find_neighbors(agent_id, positions, radius)
    │   ├── fuse_beliefs(main_belief, neighbor_beliefs, method)
    │   ├── create_centralized_protocol(frequency, bandwidth)
    │   ├── create_decentralized_protocol(radius, fusion_method)
    │   └── create_hybrid_protocol(frequency, ground_station_id)
    │
    └── belief_management.jl       # Local Belief State Management
        ├── BeliefManagement (module)
        ├── Belief
        │   ├── event_probabilities::Matrix{Float64}
        │   ├── uncertainty_map::Matrix{Float64}
        │   ├── last_update::Int
        │   └── history::Vector{Tuple{SensingAction, GridObservation}}
        ├── update_belief_state(belief, action, observation, env)
        ├── initialize_belief(grid_width, grid_height, prior_probability)
        ├── calculate_uncertainty_map(probabilities)
        ├── calculate_uncertainty(probability)
        ├── predict_belief_evolution(belief, env, num_steps)
        ├── apply_event_dynamics(probabilities, dynamics)
        └── get_neighbor_probabilities(probabilities, x, y)
```

## Data Flow Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Environment   │    │     Agents      │    │   Simulation    │
│                 │    │                 │    │                 │
│ SpatialGrid     │◄──►│ Agent           │◄──►│ run_simulation  │
│ EventDynamics   │    │ Trajectory      │    │                 │
│ SensorModels    │    │ SensingPolicy   │    │                 │
└─────────────────┘    │ Communication   │    └─────────────────┘
         │              │ BeliefManagement│              │
         │              └─────────────────┘              │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   State/Event   │    │   Actions/      │    │   Results/      │
│   Evolution     │    │   Observations  │    │   Analysis      │
│                 │    │                 │    │                 │
│ GridState       │    │ SensingAction   │    │ History         │
│ EventState      │    │ GridObservation │    │ Performance     │
│                 │    │ Belief          │    │ Metrics         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Relationships

### 1. **POMDP Interface Implementation**
- `SpatialGrid` implements the full POMDP interface
- `GridState`, `SensingAction`, `GridObservation` define the core types
- All POMDP functions are defined as skeletons for implementation

### 2. **Agent Architecture**
- `Agent` combines trajectory, sensor, and policy components
- `TrajectoryPlanner` manages deterministic periodic motion
- `SensingPolicy` decides where to sense within footprint
- `Communication` handles multi-agent coordination
- `BeliefManagement` maintains local state estimates

### 3. **Environment Dynamics**
- `EventDynamics` models stochastic event evolution
- `SensorModels` handles range-limited observations
- `SpatialGrid` coordinates all environment components

### 4. **Information Flow**
- Agents follow trajectories → get positions
- Positions determine sensor footprints → select sensing targets
- Sensing generates observations → update beliefs
- Beliefs inform next sensing decisions → coordination through communication

## Module Dependencies

```
MyProject.jl
├── Environment/
│   ├── spatial_grid.jl (core POMDP)
│   ├── event_dynamics.jl (stochastic evolution)
│   └── sensor_models.jl (observation generation)
└── Agents/
    ├── trajectory_planner.jl (motion control)
    ├── sensing_policy.jl (decision making)
    ├── communication.jl (coordination)
    └── belief_management.jl (state estimation)
```

This structure provides a clean separation of concerns while maintaining the flexibility to implement various multi-agent information gathering strategies. 
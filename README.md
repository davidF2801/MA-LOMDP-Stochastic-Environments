# Multi-Agent Locally Observable Markov Decision Process (MA-LOMDP)

A Julia project implementing multi-agent information-gathering strategies in spatially distributed, partially observable, and stochastically evolving environments using POMDPs.jl.

## Problem Description

We study a **multi-agent information-gathering problem** in a spatially distributed, partially observable, and stochastically evolving environment. A team of autonomous agents—such as UAVs or satellites—is deployed to monitor dynamic events (e.g., wildfires, anomalies, or environmental changes) that evolve over a discretized 2D map.

Each agent is constrained to a **deterministic periodic trajectory** and is equipped with a **range-limited sensor** capable of observing a subset of regions at each timestep. The agents do not control their motion: instead, they must decide **where to sense** (within their footprint) and, optionally, **when to communicate** information to others or a central planner. Observations are assumed to be **noise-free**, but the environment is only **partially observable** due to the limited field of view.

The environment evolves according to a **stochastic dynamics model**, such as a cell-wise Markov chain or a **spatial contagion process**, where the state of each cell may depend on its past value and the values of its neighbors. This gives rise to complex event dynamics like spread, decay, or persistence, which agents must learn to track.

We formalize the environment as a **Multi-Agent Locally Observable Markov Decision Process (MA-LOMDP)**. Each agent has partial access to the global state (its visible footprint) and may synchronize with others or a ground station at periodic intervals. Between synchronizations, agents maintain **local beliefs** over the environment and act independently. The reward is based on **information gain**, measuring the reduction in uncertainty (e.g., via entropy or KL divergence) across the map.

### Key Features

* **Multi-Agent Coordination**: Agents must coordinate their sensing and communication strategies under bandwidth constraints.
* **Periodic Trajectories**: Agents follow deterministic, recurring routes; actions are restricted to footprint regions.
* **Spatiotemporal Dynamics**: Events evolve stochastically in time and space, possibly with local contagion models.
* **Partial Observability**: Sensing is limited to local footprints; full state is never known.
* **Advanced Information Gain**: Sophisticated expected information gain calculation considering other agents' future observations and proper timing.

### Goal

The primary objective is to design **planning and coordination strategies** that maximize the long-term value of the information gathered. This includes maximizing the number of events detected, minimizing detection latency, and promoting persistent tracking. The framework supports **asynchronous and synchronous** centralized planning modes, allowing trade-offs between performance, scalability, and communication cost.

### Applications

This framework is applicable to:

* Satellite constellations monitoring environmental or human-made phenomena.
* UAV swarms tracking dynamic targets or hazards (e.g., fires, floods).
* Sensor networks for persistent surveillance and anomaly detection.

## Project Structure

```
│
├── Project.toml          # Project dependencies and metadata
├── Manifest.toml         # Exact dependency versions (auto-generated)
│
├── src/
│   ├── MyProject.jl      # Main module
│   ├── environment/      # Environment simulation components
│   │   ├── Environment.jl
│   │   ├── spatial_grid.jl
│   │   ├── event_dynamics.jl
│   │   ├── dbn_proper.jl
│   │   └── sensor_models.jl
│   │
│   ├── agents/
│   │   ├── Agents.jl
│   │   ├── trajectory_planner.jl
│   │   ├── sensing_policy.jl
│   │   ├── communication.jl
│   │   └── belief_management.jl
│   │
│   ├── planners/
│   │   ├── Planners.jl
│   │   ├── macro_planner_async.jl
│   │   ├── macro_planner_sync.jl
│   │   ├── policy_tree_planner.jl
│   │   └── ground_station.jl
│   │
│   └── types/
│       └── Types.jl
│
├── scripts/              # Execution scripts
│   ├── test_async_centralized_planner.jl
│   ├── test_centralized_planner.jl
│   ├── test_modular_system.jl
│   └── plot_results.jl
│
└── visualizations/       # Output directory for visualizations
    └── *.png, *.gif, etc.
```

## Key Components

### Environment
- **SpatialGrid**: 2D discretized environment with stochastic event dynamics
- **EventDynamics**: Markov chain and spatial contagion models for event evolution
- **DBNTransitionModel**: Dynamic Bayesian Network for proper belief evolution
- **SensorModels**: Range-limited sensor footprints

### Agents
- **TrajectoryPlanner**: Deterministic periodic trajectory management with phase offsets
- **SensingPolicy**: Decision-making for where to sense within footprint
- **Communication**: In case we want to implement some communication realism in the future
- **BeliefManagement**: Local belief state estimation and fusion

### Planners
- **MacroPlannerAsync**: Asynchronous centralized planning with decentralized execution, open loop
- **MacroPlannerSync**: Synchronous centralized planning
- **PolicyTreePlanner**: Closed-loop policy tree planning
- **GroundStation**: Central coordination and plan management

## Current Implementation Status

### ✅ Completed
- **Project Structure**: Complete modular architecture with proper separation of concerns
- **Core Types**: All major data structures and interfaces defined
- **POMDP Interface**: Full POMDPs.jl interface implementation
- **Trajectory Management**: Circular and linear periodic trajectory implementations with phase offsets
- **Sensor Models**: Range-limited sensor footprint calculations
- **Event Dynamics**: Stochastic event evolution framework with spatial contagion
- **Belief Management**: 3D belief state structure and DBN-based evolution
- **Communication Protocols**: Centralized, decentralized, and hybrid communication models
- **Sensing Policies**: Information gain, random, and greedy sensing strategies
- **Asynchronous Planning**: Ground station coordination with periodic synchronization
- **Visualization**: Agent trajectories, environment evolution, and action statistics

### 🔄 In Progress
- **Policy Tree Planning**: Closed-loop policy implementation
- **Performance Optimization**: Computational efficiency improvements
- **Advanced Coordination**: More sophisticated multi-agent coordination strategies

### 📋 TODO
- **Comprehensive Testing**: Unit and integration tests
- **Performance Benchmarks**: Comparison with baseline methods
- **Documentation**: API documentation and usage examples
- **Real-world Validation**: Testing with realistic scenarios

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MA-LOMDP-Stochastic-Environments
```

2. Start Julia and activate the project:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```


### Configuration

The simulation can be configured by modifying parameters in `scripts/test_async_centralized_planner.jl`:

```julia
# Main simulation parameters
const NUM_STEPS = 20                  # Total simulation steps
const PLANNING_MODE = :script         # :script or :policy

# Environment parameters
const GRID_WIDTH = 5                  # Grid width
const GRID_HEIGHT = 5                 # Grid height
const INITIAL_EVENTS = 2              # Number of initial events

# Agent parameters
const NUM_AGENTS = 2                  # Number of agents
const AGENT1_PHASE_OFFSET = 0         # Phase offset for agent 1
const AGENT2_PHASE_OFFSET = 3         # Phase offset for agent 2
```

## Advanced Features

### Expected Information Gain Calculation

The system implements a sophisticated expected information gain calculation that:

1. **Identifies Future Observations**: Determines which other agents will observe a cell before the current agent's evaluation time
2. **Global Timeline Conversion**: Properly converts agent timesteps to global simulation time using phase offsets
3. **Belief Evolution Simulation**: Simulates belief evolution between observations using DBN models
4. **Observation Outcome Weighting**: Averages information gains weighted by observation outcome probabilities

### Asynchronous Multi-Agent Coordination

- **Ground Station Coordination**: Central planning with periodic synchronization
- **Phase Offset Handling**: Proper timing coordination between agents with different phase offsets
- **Plan Management**: Ground station maintains and distributes agent plans
- **Synchronization Events**: Agents sync with ground station at periodic intervals

### Belief State Management

- **3D Belief Distributions**: Belief states represented as 3D arrays [states, height, width]
- **DBN Evolution**: Dynamic Bayesian Network for proper belief evolution over time
- **Multi-State Events**: Support for multiple event states (NO_EVENT, EVENT_PRESENT, etc.)
- **Uncertainty Tracking**: Entropy-based uncertainty measurement

## POMDPs.jl Integration

This project demonstrates several key POMDPs.jl concepts:

1. **MA-LOMDP Interface**: Multi-agent locally observable MDP implementation
2. **Belief States**: Local belief management for partial observability
3. **Information Gain Rewards**: Reward functions based on uncertainty reduction and wiugthed by the expected probability of an event happening
4. **Multi-Agent Simulation**: Coordinated agent behavior simulation

### Key POMDPs.jl Functions Implemented

- `initialstate(pomdp)`: Initial environment state distribution
- `transition(pomdp, s, a)`: Stochastic event dynamics with DBN models
- `observation(pomdp, a, sp)`: Range-limited sensor observations
- `reward(pomdp, s, a, sp)`: Information gain-based rewards
- `discount(pomdp)`: Discount factor for long-term planning
- `isterminal(pomdp, s)`: Terminal state conditions

## Dependencies

- **POMDPs.jl**: Core POMDP interface
- **POMDPTools**: Standard library components
- **POMDPPolicies**: Policy implementations
- **POMDPSimulators**: Simulation tools
- **Distributions**: Probability distributions
- **Random**: Random number generation
- **LinearAlgebra**: Linear algebra operations
- **Statistics**: Statistical functions
- **Plots**: Visualization
- **Infiltrator**: Debugging


## References

- [POMDPs.jl Documentation](https://juliapomdp.github.io/POMDPs.jl/latest/)
- [JuliaPOMDP Ecosystem](https://github.com/JuliaPOMDP)
- [Multi-Agent POMDPs](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process#Multi-agent_POMDPs)

## Development Roadmap

### Phase 1: Core Implementation ✅
- [x] Project structure and architecture
- [x] Basic types and interfaces
- [x] Complete POMDP interface implementation
- [x] Basic simulation framework
- [x] Asynchronous multi-agent coordination
- [x] Advanced information gain calculation

### Phase 2: Advanced Features 🔄
- [x] Belief state algorithms (DBN-based evolution)
- [x] Advanced sensing policies with multi-agent coordination
- [x] Communication protocol implementations
- [ ] Performance optimization
- [ ] Policy tree planning improvements

### Phase 3: Analysis and Visualization ✅
- [x] Policy evaluation tools
- [x] Visualization framework
- [ ] Performance benchmarks
- [ ] Documentation and examples

### Phase 4: Extensions 📋
- [ ] Additional trajectory types
- [ ] More complex event dynamics
- [ ] Real-world data integration
- [ ] Deployment tools 
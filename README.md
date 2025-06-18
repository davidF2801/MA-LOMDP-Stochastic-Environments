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
* **Communication Models**:
  * Centralized: Continuous or periodic uplink to a global planner.
  * Decentralized: Peer-to-peer sharing with belief fusion.
  * Hybrid: Agents sync asynchronously with a ground station to update their policy.

### Goal

The primary objective is to design **planning and coordination strategies** that maximize the long-term value of the information gathered. This includes maximizing the number of events detected, minimizing detection latency, and promoting persistent tracking. The framework supports **centralized**, **decentralized**, and **asynchronous hybrid** planning modes, allowing trade-offs between performance, scalability, and communication cost.

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
│   │   └── sensor_models.jl
│   │
│   └── agents/
│       ├── Agents.jl
│       ├── trajectory_planner.jl
│       ├── sensing_policy.jl
│       ├── communication.jl
│       └── belief_management.jl
│
├── scripts/              # Execution scripts
│   ├── run_simulation.jl
│   └── plot_results.jl
│
└── plots/                # Output directory for visualizations
    └── *.png, *.mp4, etc.
```

## Key Components

### Environment
- **SpatialGrid**: 2D discretized environment with stochastic event dynamics
- **EventDynamics**: Markov chain and spatial contagion models for event evolution
- **SensorModels**: Range-limited sensor footprints and observation models

### Agents
- **TrajectoryPlanner**: Deterministic periodic trajectory management
- **SensingPolicy**: Decision-making for where to sense within footprint
- **Communication**: Coordination strategies (centralized/decentralized/hybrid)
- **BeliefManagement**: Local belief state estimation and fusion

## Current Implementation Status

### ✅ Completed
- **Project Structure**: Complete modular architecture with proper separation of concerns
- **Core Types**: All major data structures and interfaces defined
- **POMDP Interface**: Full POMDPs.jl interface implementation skeleton
- **Trajectory Management**: Circular and linear periodic trajectory implementations
- **Sensor Models**: Range-limited sensor footprint calculations
- **Event Dynamics**: Stochastic event evolution framework
- **Belief Management**: Local belief state structure and update framework
- **Communication Protocols**: Centralized, decentralized, and hybrid communication models
- **Sensing Policies**: Information gain, random, and greedy sensing strategies

### 🔄 In Progress
- **State Transitions**: Event dynamics implementation
- **Observation Generation**: Sensor observation models
- **Reward Functions**: Information gain-based reward calculations
- **Belief Updates**: Bayesian belief update algorithms
- **Communication Logic**: Actual message passing and belief fusion

### 📋 TODO
- **Full POMDP Implementation**: Complete all interface functions
- **Simulation Engine**: Robust multi-agent simulation framework
- **Policy Evaluation**: Performance metrics and comparison tools
- **Visualization**: Trajectory and event visualization tools
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: API documentation and usage examples

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

## Usage

### Running the Simulation

```julia
# Run the main simulation
include("scripts/run_simulation.jl")
```

### Creating Visualizations

```julia
# Plot results (requires plotting packages)
include("scripts/plot_results.jl")
```

## POMDPs.jl Integration

This project demonstrates several key POMDPs.jl concepts:

1. **MA-LOMDP Interface**: Multi-agent locally observable MDP implementation
2. **Belief States**: Local belief management for partial observability
3. **Information Gain Rewards**: Reward functions based on uncertainty reduction
4. **Multi-Agent Simulation**: Coordinated agent behavior simulation

### Key POMDPs.jl Functions Implemented

- `initialstate(pomdp)`: Initial environment state distribution
- `transition(pomdp, s, a)`: Stochastic event dynamics
- `observation(pomdp, a, sp)`: Range-limited sensor observations
- `reward(pomdp, s, a, sp)`: Information gain-based rewards
- `discount(pomdp)`: Discount factor for long-term planning
- `isterminal(pomdp, s)`: Terminal state conditions

## Example: Creating a MA-LOMDP

```julia
using MyProject
using POMDPs
using POMDPTools

# Create environment with stochastic event dynamics
env = SpatialGrid(20, 20, EventDynamics())

# Create agents with periodic trajectories
agents = [
    Agent(1, CircularTrajectory(10, 10, 5.0, 20), RangeLimitedSensor(3.0, π/2, 0.0), 0),
    Agent(2, LinearTrajectory(5, 5, 15, 15, 15), RangeLimitedSensor(2.5, π/3, 0.0), 0)
]

# Create sensing policy
policy = InformationGainPolicy(0.1)

# Run simulation
sim = HistoryRecorder(max_steps=100)
hist = simulate(sim, env, agents, policy)
```

## Dependencies

- **POMDPs.jl**: Core POMDP interface
- **POMDPTools**: Standard library components
- **POMDPPolicies**: Policy implementations
- **POMDPSimulators**: Simulation tools
- **Distributions**: Probability distributions
- **Random**: Random number generation
- **LinearAlgebra**: Linear algebra operations
- **Statistics**: Statistical functions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [POMDPs.jl Documentation](https://juliapomdp.github.io/POMDPs.jl/latest/)
- [JuliaPOMDP Ecosystem](https://github.com/JuliaPOMDP)
- [Multi-Agent POMDPs](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process#Multi-agent_POMDPs)

## Development Roadmap

### Phase 1: Core Implementation (Current)
- [x] Project structure and architecture
- [x] Basic types and interfaces
- [ ] Complete POMDP interface implementation
- [ ] Basic simulation framework

### Phase 2: Advanced Features
- [ ] Belief state algorithms (particle filters, Kalman filters)
- [ ] Advanced sensing policies
- [ ] Communication protocol implementations
- [ ] Performance optimization

### Phase 3: Analysis and Visualization
- [ ] Policy evaluation tools
- [ ] Visualization framework
- [ ] Performance benchmarks
- [ ] Documentation and examples

### Phase 4: Extensions
- [ ] Additional trajectory types
- [ ] More complex event dynamics
- [ ] Real-world data integration
- [ ] Deployment tools 
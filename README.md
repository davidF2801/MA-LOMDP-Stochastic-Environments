# Multi-Agent Local Observable MDP in Stochastic Environments

A Julia project demonstrating the use of POMDPs.jl for multi-agent decision making in stochastic environments with local observability.

## Overview

This project implements a multi-agent system where agents operate in a grid world environment with partial observability. The agents must coordinate to manage contagion spread while maintaining their own health and safety.

## Project Structure

```
│
├── Project.toml          # Project dependencies and metadata
├── Manifest.toml         # Exact dependency versions (auto-generated)
│
├── src/
│   ├── MyProject.jl      # Main module
│   ├── world_simulator/  # Environment simulation components
│   │   ├── WorldSimulator.jl
│   │   ├── grid_environment.jl
│   │   ├── contagion_model.jl
│   │   └── transition_models.jl
│   │
│   └── agents/           # Agent-related components
│       ├── Agents.jl
│       ├── policy_interfaces.jl
│       ├── centralized_planner.jl
│       ├── decentralized_planner.jl
│       └── belief_update.jl
│
├── scripts/              # Execution scripts
│   ├── run_simulation.jl
│   └── plot_results.jl
│
└── plots/                # Output directory for visualizations
    └── *.png, *.mp4, etc.
```

## Key Components

### World Simulator
- **GridEnvironment**: 2D grid world with obstacles and agents
- **ContagionModel**: Disease spread dynamics and transmission
- **TransitionModels**: State transition logic and action application

### Agents
- **PolicyInterfaces**: Abstract policy types and interfaces
- **CentralizedPlanner**: Centralized coordination algorithms
- **DecentralizedPlanner**: Decentralized decision making
- **BeliefUpdate**: Belief state estimation and filtering

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

1. **POMDP Interface**: The `GridEnvironment` implements the full POMDP interface
2. **Belief States**: Agents maintain belief states about the world
3. **Policy Types**: Multiple policy types for different decision-making approaches
4. **Simulation**: Standard POMDPs.jl simulation framework

### Key POMDPs.jl Functions Implemented

- `initialstate(pomdp)`: Initial state distribution
- `transition(pomdp, s, a)`: State transition function
- `observation(pomdp, a, sp)`: Observation function
- `reward(pomdp, s, a, sp)`: Reward function
- `discount(pomdp)`: Discount factor
- `isterminal(pomdp, s)`: Terminal state check

## Example: Creating a Custom POMDP

```julia
using MyProject
using POMDPs
using POMDPTools

# Create environment
env = GridEnvironment(10, 10, Set(), [], 0.95)

# Create policy
policy = RandomPolicy()

# Run simulation
sim = HistoryRecorder(max_steps=100)
hist = simulate(sim, env, policy)
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
- [POMDP Theory](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process)

## TODO

- [ ] Implement full POMDP interface functions
- [ ] Add belief state update algorithms
- [ ] Implement centralized planning algorithms
- [ ] Add decentralized coordination protocols
- [ ] Create visualization tools
- [ ] Add comprehensive tests
- [ ] Performance optimization
- [ ] Documentation improvements 
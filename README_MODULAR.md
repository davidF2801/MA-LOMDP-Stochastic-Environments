# Modular MA-LOMDP System for Stochastic Environments

A highly modular and extensible implementation of Multi-Agent Locally Observable Markov Decision Processes (MA-LOMDP) for information gathering in spatially distributed, partially observable, and stochastically evolving environments.

## 🎯 Key Features

- **Modular Event State Systems**: Easily switch between 2-state, 4-state, or custom event state systems
- **DBN-based Transition Models**: Dynamic Bayesian Network models for realistic event evolution
- **Multiple Planning Strategies**: Information gain, uncertainty reduction, coverage, and multi-objective planning
- **Agent Belief Management**: Sophisticated belief state estimation and updates using DBNs
- **Easy Extensibility**: Add new event states, transition models, or planning strategies with minimal code changes
- **Backward Compatibility**: Legacy code continues to work seamlessly

## 🏗️ System Architecture

```
src/
├── environment/
│   ├── spatial_grid.jl          # Main POMDP environment
│   ├── event_dynamics.jl        # Modular event state systems and DBN models
│   ├── sensor_models.jl         # Sensor and observation models
│   └── Environment.jl           # Environment module exports
├── agents/
│   ├── belief_management.jl     # DBN-based belief state management
│   ├── planning_strategies.jl   # Modular planning strategies
│   ├── trajectory_planner.jl    # Agent trajectory planning
│   ├── sensing_policy.jl        # Sensing policy implementation
│   ├── communication.jl         # Inter-agent communication
│   └── Agents.jl                # Agents module exports
└── MyProject.jl                 # Main project module
```

## 🧠 Event State Systems

### 2-State System (Simple)
```julia
@enum EventState2 begin
    NO_EVENT_2 = 0
    EVENT_PRESENT_2 = 1
end
```

### 4-State System (Extended)
```julia
@enum EventState4 begin
    NO_EVENT_4 = 0
    EVENT_PRESENT_4 = 1
    EVENT_SPREADING_4 = 2
    EVENT_DECAYING_4 = 3
end
```

### Custom Event States
You can easily add new event state systems:

```julia
@enum EventState3 begin
    NO_FIRE_3 = 0
    SMOLDERING_3 = 1
    BURNING_3 = 2
end
```

## 🔄 DBN Transition Models

### 2-State DBN Model
```julia
struct DBNTransitionModel2 <: TransitionModel
    birth_rate::Float64
    death_rate::Float64
    neighbor_influence::Float64
end
```

### 4-State DBN Model
```julia
struct DBNTransitionModel4 <: TransitionModel
    birth_rate::Float64
    death_rate::Float64
    spread_rate::Float64
    decay_rate::Float64
    neighbor_influence::Float64
end
```

### Custom DBN Models
```julia
struct WildfireDBNModel <: TransitionModel
    ignition_rate::Float64
    spread_rate::Float64
    extinction_rate::Float64
    wind_influence::Float64
end
```

## 🎯 Planning Strategies

### Information Gain Planning
```julia
strategy = InformationGainPlanning(horizon=3, exploration_weight=0.1)
```

### Uncertainty Reduction Planning
```julia
strategy = UncertaintyReductionPlanning(horizon=3, uncertainty_threshold=0.5)
```

### Coverage Planning
```julia
strategy = CoveragePlanning(horizon=3, coverage_weight=1.0)
```

### Multi-Objective Planning
```julia
strategy = MultiObjectivePlanning(3, 0.4, 0.3, 0.3)  # info, uncertainty, coverage weights
```

### Custom Planning Strategies
```julia
struct RiskAwarePlanning <: PlanningStrategy
    horizon::Int
    risk_threshold::Float64
    risk_weight::Float64
end
```

## 🚀 Quick Start

### Basic Usage
```julia
using POMDPs
using POMDPTools

# Include the system
include("src/environment/spatial_grid.jl")
include("src/agents/belief_management.jl")
include("src/agents/planning_strategies.jl")

# Create environment
dynamics = EventDynamics(0.1, 0.05, 0.2, 0.1, 0.3)
agent = create_agent(1, CircularTrajectory(5, 5, 3.0, 10), RangeLimitedSensor(2.0, π/2, 0.0), 10, 10)
env = SpatialGrid(10, 10, dynamics, [agent], 3.0, 0.95, 3, 5)

# Get initial state
initial_dist = POMDPs.initialstate(env)
state = rand(initial_dist)

# Plan action
strategy = InformationGainPlanning(3, 0.1)
action = plan_action(strategy, agent, env, state)
```

### Using Different Event States
```julia
# 2-state system
dbn_2 = DBNTransitionModel2(0.1, 0.05, 0.3)
event_map_2 = fill(NO_EVENT_2, 5, 5)
update_events!(dbn_2, event_map_2, rng)

# 4-state system
dbn_4 = DBNTransitionModel4(0.1, 0.05, 0.2, 0.1, 0.3)
event_map_4 = fill(NO_EVENT_4, 5, 5)
update_events!(dbn_4, event_map_4, rng)
```

### Belief Evolution
```julia
# Predict belief evolution using DBN
predicted_belief = predict_belief_evolution_dbn(agent.belief, env, 5)
println("Belief mean: $(mean(predicted_belief.event_probabilities))")
```

## 🔧 Extending the System

### Adding New Event States

1. Define the new event state enum:
```julia
@enum MyEventState begin
    STATE_1 = 0
    STATE_2 = 1
    STATE_3 = 2
end
```

2. Create a DBN transition model:
```julia
struct MyDBNModel <: TransitionModel
    param1::Float64
    param2::Float64
end
```

3. Implement transition probability function:
```julia
function transition_probability_dbn(current_state::MyEventState, neighbor_states::Vector{MyEventState}, model::MyDBNModel)
    # Your transition logic here
end
```

4. Implement update function:
```julia
function update_events!(model::MyDBNModel, event_map::Matrix{MyEventState}, rng::AbstractRNG)
    # Your update logic here
end
```

### Adding New Planning Strategies

1. Define the strategy struct:
```julia
struct MyPlanningStrategy <: PlanningStrategy
    horizon::Int
    my_param::Float64
end
```

2. Implement the planning function:
```julia
function plan_action(strategy::MyPlanningStrategy, agent::Agent, env::SpatialGrid, current_state::GridState)
    # Your planning logic here
end
```

## 🧪 Testing

### Run Basic Tests
```bash
julia scripts/test_environment.jl
```

### Run Modular System Tests
```bash
julia scripts/test_modular_system.jl
```

### Run Extensibility Example
```bash
julia examples/extensibility_example.jl
```

## 📊 System Benefits

### Modularity
- **Event States**: Easy to add new event state systems
- **Transition Models**: Flexible DBN-based transition models
- **Planning Strategies**: Pluggable planning algorithms
- **Sensor Models**: Extensible sensor and observation models

### Performance
- **DBN Efficiency**: Fast belief updates using DBN structure
- **Parallel Planning**: Independent agent planning
- **Memory Efficient**: Belief state compression

### Extensibility
- **New Domains**: Easy adaptation to different problem domains
- **New Objectives**: Simple addition of new planning objectives
- **New Sensors**: Flexible sensor model integration

## 🔬 Research Applications

This system is designed for research in:
- **Multi-Agent Information Gathering**
- **Spatial Event Monitoring**
- **Stochastic Environment Modeling**
- **Belief State Estimation**
- **Planning Under Uncertainty**

## 📝 Dependencies

- **POMDPs.jl**: POMDP interface
- **POMDPTools.jl**: POMDP utilities
- **POMDPPolicies.jl**: Policy implementations
- **Distributions.jl**: Probability distributions
- **LinearAlgebra.jl**: Linear algebra operations

## 🤝 Contributing

The system is designed to be easily extensible. To contribute:

1. **Add New Event States**: Follow the pattern in `event_dynamics.jl`
2. **Add New Planning Strategies**: Follow the pattern in `planning_strategies.jl`
3. **Add New Sensor Models**: Extend the sensor model interface
4. **Improve Belief Management**: Enhance the DBN-based belief updates

## 📄 License

This project is open source and available under the MIT License.

---

**🎯 The modular MA-LOMDP system provides a flexible foundation for research in multi-agent information gathering, with easy extensibility for new domains and applications.** 
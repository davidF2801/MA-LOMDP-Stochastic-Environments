# Graph-Based Case Study

This directory contains a graph-based case study for the ABBA-PLUS project, where the environment is represented as a 2D graph instead of a grid.

## Overview

In this case study:
- **Environment**: Represented as a graph where nodes are cells and edges represent connectivity
- **Agents**: UAVs (Unmanned Aerial Vehicles) that move between connected nodes
- **Actions**: Agents can move to any neighbor node (or stay at current node)
- **Observations**: Agents observe the exact state value (0 or 1) of the node they're currently at
- **Dynamics**: Same local kernel contagion dynamics as the grid-based case (RSP or DBN-2)
- **Communication**: No communication between agents

## Files

### Core Components

- **`graph_environment.py`**: Graph-based environment implementation
  - `GraphEnvironment`: Main environment class with graph structure
  - Nodes represent cells, edges represent connectivity
  - Supports same spatial dynamics (RSP, DBN-2, VIIRS table mode)

- **`graph_agents/`**: Directory containing graph-based agents
  - **`graph_belief.py`**: Belief representation over graph nodes
  - **`graph_agent.py`**: Abstract base class for graph agents
  - **`graph_monte_carlo_agent.py`**: Monte Carlo planning agent for graphs

- **`visualize_graph.py`**: Visualization module for graph-based simulation
  - `GraphAnimator`: Animate graph simulation with agent movement
  - `plot_graph_state`: Plot a single snapshot of the graph state

- **`main_graph_agents.py`**: Main simulation runner with visualization
- **`example_graph_simulation.py`**: Example script showing how to use the simulation

## Usage

### Basic Usage

```python
from environment_simulation.main_graph_agents import main

# Run simulation with visualization
sim, env, agents = main(
    num_nodes=100,           # Number of nodes (e.g., 10x10 grid)
    num_agents=3,             # Number of UAV agents
    num_steps=50,             # Number of simulation steps
    planning_horizon=5,       # Planning horizon for agents
    num_rollouts=50,          # Monte Carlo rollouts per action
    mode="rsp",               # Environment mode: "rsp" or "dbn2"
    seed=42,                  # Random seed
    save_animation=True,      # Save animation GIF
    animation_path=None,      # Auto-generate path if None
    animation_interval=200,   # Milliseconds between frames
)
```

### Running the Example

```bash
python environment_simulation/example_graph_simulation.py
```

### Visualization

The visualization shows:
- **Nodes**: Colored based on state (red = event present, gray = no event)
- **Edges**: Connections between nodes (gray lines)
- **Agents**: Star markers showing current positions, with colored trajectory lines
- **Time**: Current time step and number of active events displayed in title

The animation saves to `results/YYYYMMDD_HHMMSS/animation_graph.gif` by default.

## Key Differences from Grid-Based Case

1. **Graph Structure**: Environment is a graph (nodes + edges) instead of a 2D grid
2. **Agent Movement**: Agents move between connected nodes (graph traversal)
3. **Observations**: Agents observe exact state of current node (no field of regard)
4. **Actions**: Movement actions (which neighbor node to move to) instead of observation region selection

## Dependencies

- `numpy`: Core numerical operations
- `matplotlib`: Visualization
- `networkx`: Graph visualization and operations

Install with:
```bash
pip install numpy matplotlib networkx
```

## Example Output

The simulation will:
1. Create a graph environment (e.g., 10x10 grid with 8-connected neighbors)
2. Initialize agents at random starting nodes
3. Run simulation where agents:
   - Plan movements using Monte Carlo rollouts
   - Move to neighbor nodes
   - Observe exact state of current node
   - Update beliefs based on observations
4. Generate an animated visualization showing:
   - Graph structure
   - Event propagation (node state changes)
   - Agent movement trajectories

Statistics are printed at the end showing:
- Per-agent metrics (events observed, rewards, nodes visited)
- Multi-agent metrics (unique events, reobservations)
- Environment statistics


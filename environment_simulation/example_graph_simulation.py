"""Example script demonstrating graph-based simulation with visualization."""

from __future__ import annotations

from environment_simulation.main_graph_agents import main

if __name__ == "__main__":
    # Run a simple example simulation
    print("="*80)
    print("Graph-Based UAV Simulation Example")
    print("="*80)
    print()
    
    # Run simulation with visualization
    sim, env, agents = main(
        num_nodes=100,           # 10x10 grid
        num_agents=3,             # 3 UAVs
        num_steps=30,             # 30 simulation steps
        planning_horizon=5,       # 5-step planning horizon
        num_rollouts=20,          # 20 Monte Carlo rollouts per action
        mode="rsp",               # Use RSP dynamics
        seed=42,                  # Random seed
        save_animation=True,      # Save animation
        animation_path=None,      # Auto-generate path
        animation_interval=300,   # 300ms between frames
        animation_fps=None,       # Auto-calculate from interval
    )
    
    print("\nSimulation complete!")
    print("Check the results/ directory for the animation and statistics.")


"""
Debug script to check if kernel is working correctly.
"""

import sys
import os
import numpy as np

FIRE_SIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'fire-simulation')
if FIRE_SIM_PATH not in sys.path:
    sys.path.insert(0, FIRE_SIM_PATH)

try:
    from .kernel_learning import TransitionKernelLearner
    from .kernel_simulation import KernelBasedEnvironment, create_comparison_environment
    from .test_fire_physics_graph import GraphEnvironmentFire
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from kernel_learning import TransitionKernelLearner
    from kernel_simulation import KernelBasedEnvironment, create_comparison_environment
    from test_fire_physics_graph import GraphEnvironmentFire

def main():
    kernel_path = "learned_kernel.pkl"
    
    print("Loading kernel...")
    kernel = TransitionKernelLearner()
    kernel.load_kernel(kernel_path)
    
    print(f"Total transitions: {kernel.total_transitions}")
    print(f"Configurations: {len(kernel.transition_probs)}")
    
    # Check some key transitions
    print("\nChecking key transitions:")
    
    # Unburned WOOD with different neighbor counts
    from src.material import Material
    print("\nTransition probabilities for UNBURNED WOOD:")
    for k_prime in range(9):
        key = (0, "WOOD", k_prime)
        if key in kernel.transition_probs:
            probs = kernel.transition_probs[key]
            print(f"  k'={k_prime}: UNBURNED={probs.get(0, 0.0):.4f}, BURNING={probs.get(1, 0.0):.4f}, BURNED={probs.get(2, 0.0):.4f}")
    
    # Check what transitions were actually observed
    print("\nObserved transitions for UNBURNED WOOD, k'=1:")
    counts = {}
    for (s, m, k, s_next), count in kernel.count_dict.items():
        if s == 0 and m == "WOOD" and k == 1:
            counts[s_next] = count
    print(f"  Counts: {counts}")
    total = sum(counts.values())
    if total > 0:
        print(f"  Proportions: 0→0={counts.get(0, 0)/total:.4f}, 0→1={counts.get(1, 0)/total:.4f}, 0→2={counts.get(2, 0)/total:.4f}")
    
    # Burning WOOD
    key = (1, "WOOD", 0)
    if key in kernel.transition_probs:
        probs = kernel.transition_probs[key]
        print(f"BURNING WOOD, 0 neighbors: {probs}")
        print(f"  -> Probability of burning out (1→2): {probs.get(2, 0.0):.4f}")
        print(f"  -> Probability of staying burning (1→1): {probs.get(1, 0.0):.4f}")
    
    # Create a simple test
    print("\n" + "="*70)
    print("Testing kernel simulation on simple case...")
    print("="*70)
    
    # Create a small test environment
    num_nodes = 100
    from main_graph_agents import create_graph_environment
    
    base_env = create_graph_environment(num_nodes=num_nodes, mode="rsp", seed=42)
    
    # Create physical env with one burning node
    physics_env = GraphEnvironmentFire(
        num_nodes=num_nodes,
        mode="fire",
        edges=base_env.edges,
        width=base_env.width,
        height=base_env.height,
        seed=42,
        initial_burning_nodes=[50],  # Start with one burning node
    )
    
    # Create kernel env with same initial conditions
    kernel_env = create_comparison_environment(physics_env, kernel)
    
    print(f"Initial state - Physics: {np.sum(physics_env.get_state() == 1)} burning, "
          f"Kernel: {np.sum(kernel_env.get_state() == 1)} burning")
    
    # Run 20 steps and see what happens
    for step in range(20):
        physics_env.step()
        kernel_env.step()
        
        physics_state = physics_env.get_state()
        kernel_state = kernel_env.get_state()
        
        physics_burning = np.sum(physics_state == 1)
        kernel_burning = np.sum(kernel_state == 1)
        
        print(f"Step {step+1}: Physics={physics_burning} burning, Kernel={kernel_burning} burning")
        
        if kernel_burning == 0 and step > 5:
            print("WARNING: Kernel simulation stopped evolving!")
            break

if __name__ == "__main__":
    main()


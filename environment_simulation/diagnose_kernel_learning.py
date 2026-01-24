"""
Diagnose kernel learning to see what transitions are being collected.
"""

import sys
import os
import numpy as np

FIRE_SIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'fire-simulation')
if FIRE_SIM_PATH not in sys.path:
    sys.path.insert(0, FIRE_SIM_PATH)

try:
    from .kernel_learning import TransitionKernelLearner, learn_kernel_from_simulation
    from .test_fire_physics_graph import GraphEnvironmentFire
    from .main_graph_agents import create_graph_environment
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from kernel_learning import TransitionKernelLearner, learn_kernel_from_simulation
    from test_fire_physics_graph import GraphEnvironmentFire
    from main_graph_agents import create_graph_environment

def analyze_collected_transitions(learner: TransitionKernelLearner):
    """Analyze what transitions were actually collected."""
    print("\n" + "="*70)
    print("ANALYZING COLLECTED TRANSITIONS")
    print("="*70)
    
    # Group by (current_state, material, k_prime)
    transitions_by_config = {}
    for (s, m, k, s_next), count in learner.count_dict.items():
        key = (s, m, k)
        if key not in transitions_by_config:
            transitions_by_config[key] = {}
        transitions_by_config[key][s_next] = count
    
    print(f"\nTotal unique configurations: {len(transitions_by_config)}")
    print(f"Total transitions collected: {learner.total_transitions}")
    
    # Analyze unburned → burning transitions (the key ones!)
    print("\n" + "-"*70)
    print("UNBURNED → BURNING TRANSITIONS (Key for fire spread)")
    print("-"*70)
    
    ignition_transitions = {}
    for (s, m, k), transitions in transitions_by_config.items():
        if s == 0:  # UNBURNED
            ignited_count = transitions.get(1, 0)  # 0 → 1
            stayed_unburned_count = transitions.get(0, 0)  # 0 → 0
            total = ignited_count + stayed_unburned_count
            if total > 0:
                ignition_transitions[(m, k)] = {
                    'ignited': ignited_count,
                    'stayed': stayed_unburned_count,
                    'total': total,
                    'p_ignite': ignited_count / total if total > 0 else 0.0
                }
    
    if ignition_transitions:
        print("\nMaterial | k' | Ignited | Stayed | Total | P(ignite)")
        print("-"*70)
        for (m, k) in sorted(ignition_transitions.keys(), key=lambda x: (x[0], x[1])):
            data = ignition_transitions[(m, k)]
            print(f"{m:8s} | {k:2d} | {data['ignited']:7d} | {data['stayed']:6d} | {data['total']:5d} | {data['p_ignite']:.4f}")
    else:
        print("NO IGNITION TRANSITIONS FOUND!")
    
    # Check learned probabilities
    print("\n" + "-"*70)
    print("LEARNED PROBABILITIES FOR UNBURNED → BURNING")
    print("-"*70)
    
    from src.material import Material
    for material_name in ["WOOD", "GRASS"]:  # Only WOOD and GRASS (no GASOLINE)
        print(f"\n{material_name}:")
        for k in range(9):
            key = (0, material_name, k)
            if key in learner.transition_probs:
                probs = learner.transition_probs[key]
                p_ignite = probs.get(1, 0.0)
                total_transitions = sum(
                    learner.count_dict.get((0, material_name, k, s_next), 0)
                    for s_next in range(3)
                )
                print(f"  k'={k}: P(0→1)={p_ignite:.6f} (from {total_transitions} transitions)")
    
    # Check if we're seeing any fire spread at all
    print("\n" + "-"*70)
    print("OVERALL FIRE SPREAD STATISTICS")
    print("-"*70)
    
    total_ignitions = sum(
        count for (s, m, k, s_next), count in learner.count_dict.items()
        if s == 0 and s_next == 1
    )
    total_unburned_stayed = sum(
        count for (s, m, k, s_next), count in learner.count_dict.items()
        if s == 0 and s_next == 0
    )
    
    print(f"Total unburned→burning transitions: {total_ignitions}")
    print(f"Total unburned→unburned transitions: {total_unburned_stayed}")
    if total_ignitions + total_unburned_stayed > 0:
        p_overall_ignite = total_ignitions / (total_ignitions + total_unburned_stayed)
        print(f"Overall ignition rate: {p_overall_ignite:.6f} ({p_overall_ignite*100:.4f}%)")
    
    # Check burning → burned transitions
    total_burnouts = sum(
        count for (s, m, k, s_next), count in learner.count_dict.items()
        if s == 1 and s_next == 2
    )
    total_burning_stayed = sum(
        count for (s, m, k, s_next), count in learner.count_dict.items()
        if s == 1 and s_next == 1
    )
    
    print(f"\nTotal burning→burned transitions: {total_burnouts}")
    print(f"Total burning→burning transitions: {total_burning_stayed}")
    if total_burnouts + total_burning_stayed > 0:
        p_overall_burnout = total_burnouts / (total_burnouts + total_burning_stayed)
        print(f"Overall burnout rate: {p_overall_burnout:.6f} ({p_overall_burnout*100:.4f}%)")

def test_learning_process():
    """Test the learning process with detailed diagnostics."""
    print("="*70)
    print("TESTING KERNEL LEARNING PROCESS")
    print("="*70)
    
    num_nodes = 1000
    num_steps = 100  # Start small for testing
    
    # Create environment
    print(f"\nCreating fire environment with {num_nodes} nodes...")
    base_env = create_graph_environment(num_nodes=num_nodes, mode="rsp", seed=42)
    
    # Create initial burning nodes (small clusters)
    rng = np.random.default_rng(42)
    num_clusters = 3
    initial_burning_nodes = []
    available_nodes = list(range(num_nodes))
    
    for cluster_idx in range(num_clusters):
        if not available_nodes:
            break
        start_node = rng.choice(available_nodes)
        neighbors = list(base_env.get_neighbors(start_node))
        neighbors = [n for n in neighbors if n not in initial_burning_nodes and n in available_nodes]
        cluster_size = min(3, len(neighbors) + 1)
        cluster_nodes = [start_node] + neighbors[:cluster_size-1]
        initial_burning_nodes.extend(cluster_nodes)
        for node in cluster_nodes:
            if node in available_nodes:
                available_nodes.remove(node)
    
    print(f"Initial burning nodes: {len(initial_burning_nodes)} across {num_clusters} clusters")
    
    # Handle blocked_mask
    blocked_mask_to_pass = None
    if base_env.blocked_mask is not None:
        blocked_mask = base_env.blocked_mask
        if blocked_mask.ndim == 2:
            blocked_flat = blocked_mask.flatten()
        else:
            blocked_flat = blocked_mask.copy()
        if len(blocked_flat) >= num_nodes:
            blocked_mask_to_pass = blocked_flat[:num_nodes].copy()
        else:
            blocked_mask_to_pass = np.zeros(num_nodes, dtype=bool)
            blocked_mask_to_pass[:len(blocked_flat)] = blocked_flat
    
    # Create fire environment
    fire_env = GraphEnvironmentFire(
        num_nodes=num_nodes,
        mode="fire",
        edges=base_env.edges,
        width=base_env.width,
        height=base_env.height,
        blocked_mask=blocked_mask_to_pass,
        seed=42,
        initial_burning_nodes=initial_burning_nodes,
    )
    
    # Monitor fire spread during learning
    print(f"\nRunning {num_steps} steps of fire simulation...")
    burning_counts = []
    for step in range(num_steps):
        fire_env.step()
        state = fire_env.get_state()
        burning = np.sum(state == 1)
        burning_counts.append(burning)
        if step % 10 == 0:
            burned = np.sum(state == 2)
            unburned = np.sum(state == 0)
            print(f"  Step {step}: Burning={burning}, Burned={burned}, Unburned={unburned}")
    
    print(f"\nFire spread observed: max burning = {max(burning_counts)}, final burning = {burning_counts[-1]}")
    
    if max(burning_counts) <= len(initial_burning_nodes):
        print("WARNING: Fire did not spread! All collected transitions will be (0,m,0,0) or (1,m,0,1/2)")
        print("This means the kernel will not learn how fires spread.")
    
    # Reset environment and learn kernel from scratch
    print(f"\nResetting environment and learning kernel from scratch...")
    
    # Create fresh environment for learning
    fire_env_fresh = GraphEnvironmentFire(
        num_nodes=num_nodes,
        mode="fire",
        edges=base_env.edges,
        width=base_env.width,
        height=base_env.height,
        blocked_mask=blocked_mask_to_pass,
        seed=42,
        initial_burning_nodes=initial_burning_nodes,
    )
    
    learner = learn_kernel_from_simulation(
        env=fire_env_fresh,
        num_steps=num_steps,
        collect_every=1,
    )
    
    # Also check what happened during learning
    final_state = fire_env_fresh.get_state()
    final_burning = np.sum(final_state == 1)
    print(f"\nAfter learning: {final_burning} nodes burning")
    
    # Analyze
    analyze_collected_transitions(learner)
    
    return learner

def analyze_existing_kernel(kernel_path: str):
    """Analyze an existing kernel file."""
    print("="*70)
    print(f"ANALYZING EXISTING KERNEL: {kernel_path}")
    print("="*70)
    
    learner = TransitionKernelLearner()
    learner.load_kernel(kernel_path)
    
    analyze_collected_transitions(learner)
    
    return learner

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default=None, help="Path to existing kernel to analyze")
    parser.add_argument("--test_learning", action="store_true", help="Test the learning process")
    args = parser.parse_args()
    
    if args.kernel_path:
        analyze_existing_kernel(args.kernel_path)
    else:
        learner = test_learning_process()


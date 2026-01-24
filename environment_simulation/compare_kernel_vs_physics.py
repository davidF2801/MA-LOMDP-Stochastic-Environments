"""
Compare kernel-based simulation vs physical model side-by-side.

Creates a visualization showing both simulations evolving from the same initial conditions.
"""

import sys
import os
import numpy as np
import pygame

# Add fire-simulation to path
FIRE_SIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'fire-simulation')
if FIRE_SIM_PATH not in sys.path:
    sys.path.insert(0, FIRE_SIM_PATH)

# Mock window_option before importing physics
import types
mock_window_option = types.ModuleType('src.window_option')
mock_window_option.CELL_WIDTH = 1.0
mock_window_option.CELL_HEIGHT = 1.0
sys.modules['src.window_option'] = mock_window_option

from src.material import Material

# Import our modules
try:
    from .test_fire_physics_graph import GraphEnvironmentFire
    from .kernel_learning import TransitionKernelLearner
    from .kernel_simulation import KernelBasedEnvironment, create_comparison_environment
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_fire_physics_graph import GraphEnvironmentFire
    from kernel_learning import TransitionKernelLearner
    from kernel_simulation import KernelBasedEnvironment, create_comparison_environment


def visualize_comparison(
    physics_env: GraphEnvironmentFire,
    kernel_env: KernelBasedEnvironment,
    num_steps: int = 500,
    fps: int = 30,
    width: int = 1600,
    height: int = 800,
):
    """
    Visualize side-by-side comparison of physical vs kernel-based simulation.
    
    Args:
        physics_env: GraphEnvironmentFire instance (physical model)
        kernel_env: KernelBasedEnvironment instance (kernel-based)
        num_steps: Number of simulation steps
        fps: Frames per second for animation
        width: Window width
        height: Window height
    """
    pygame.init()
    
    # Create window
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Kernel vs Physics Simulation Comparison")
    
    clock = pygame.time.Clock()
    
    # Color definitions
    # Note: Material-specific colors are used for unburned cells (wood vs grass)
    COLORS = {
        'unburned': (34, 139, 34),      # Forest green (default, used if material color not available)
        'wood': (164, 114, 11),          # Brown (from Material.WOOD.color)
        'grass': (34, 139, 34),          # Forest green (from Material.GRASS.color)
        'burning': (255, 69, 0),         # Orange red
        'burned': (64, 64, 64),          # Dark gray
        'water': (70, 130, 180),         # Steel blue
        'background': (20, 20, 20),      # Dark background
        'text': (255, 255, 255),         # White text
        'separator': (100, 100, 100),    # Gray separator line
    }
    
    # Calculate layout
    panel_width = width // 2
    panel_height = height - 100  # Leave space for stats
    
    # Try to arrange nodes in a grid for visualization
    # Assume roughly square grid
    grid_size = int(np.sqrt(physics_env.num_nodes))
    if grid_size * grid_size < physics_env.num_nodes:
        grid_size += 1
    
    cell_size = min(panel_width // grid_size, panel_height // grid_size, 10)
    
    # Calculate offsets for centering
    grid_width = grid_size * cell_size
    grid_height = grid_size * cell_size
    offset_x_physics = (panel_width - grid_width) // 2
    offset_x_kernel = panel_width + (panel_width - grid_width) // 2
    offset_y = (panel_height - grid_height) // 2
    
    running = True
    step = 0
    paused = False
    
    print(f"Starting comparison visualization ({num_steps} steps)...")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  ESC: Exit")
    
    while running and step < num_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if not paused:
            # Step both environments
            physics_state = physics_env.step()
            kernel_state = kernel_env.step()
            step += 1
            
            # Debug: print progress every 10 steps
            if step % 10 == 0:
                physics_stats = _compute_stats(physics_state)
                kernel_stats = _compute_stats(kernel_state)
                print(f"Step {step}: Physics - Burning={physics_stats['burning']}, Burned={physics_stats['burned']} | "
                      f"Kernel - Burning={kernel_stats['burning']}, Burned={kernel_stats['burned']}")
        
        # Clear screen
        screen.fill(COLORS['background'])
        
        # Draw separator line
        pygame.draw.line(screen, COLORS['separator'], 
                        (panel_width, 0), (panel_width, panel_height), 2)
        
        # Draw physics simulation (left panel)
        _draw_grid(screen, physics_env, physics_state, physics_env.state,
                  offset_x_physics, offset_y, cell_size, grid_size, COLORS, "PHYSICS MODEL")
        
        # Draw kernel simulation (right panel)
        _draw_grid(screen, kernel_env, kernel_state, kernel_env.state,
                  offset_x_kernel, offset_y, cell_size, grid_size, COLORS, "KERNEL MODEL")
        
        # Draw statistics
        stats_y = panel_height + 10
        font = pygame.font.Font(None, 24)
        
        # Physics stats
        physics_stats = _compute_stats(physics_state)
        stats_text = [
            f"PHYSICS: Step {step}/{num_steps} | Unburned: {physics_stats['unburned']} | "
            f"Burning: {physics_stats['burning']} | Burned: {physics_stats['burned']}"
        ]
        for i, text in enumerate(stats_text):
            surface = font.render(text, True, COLORS['text'])
            screen.blit(surface, (10, stats_y + i * 25))
        
        # Kernel stats
        kernel_stats = _compute_stats(kernel_state)
        stats_text = [
            f"KERNEL:  Step {step}/{num_steps} | Unburned: {kernel_stats['unburned']} | "
            f"Burning: {kernel_stats['burning']} | Burned: {kernel_stats['burned']}"
        ]
        for i, text in enumerate(stats_text):
            surface = font.render(text, True, COLORS['text'])
            screen.blit(surface, (panel_width + 10, stats_y + i * 25))
        
        # Pause indicator
        if paused:
            pause_font = pygame.font.Font(None, 48)
            pause_text = pause_font.render("PAUSED", True, (255, 0, 0))
            text_rect = pause_text.get_rect(center=(width // 2, 30))
            screen.blit(pause_text, text_rect)
        
        pygame.display.flip()
        clock.tick(fps)
    
    pygame.quit()
    print(f"\nSimulation complete! ({step} steps)")


def _draw_grid(screen, env, state, state_array, offset_x, offset_y, cell_size, grid_size, colors, title):
    """Draw a grid visualization of the environment state."""
    # Draw title
    font = pygame.font.Font(None, 32)
    title_surface = font.render(title, True, colors['text'])
    title_rect = title_surface.get_rect(center=(offset_x + (grid_size * cell_size) // 2, 20))
    screen.blit(title_surface, title_rect)
    
    # Draw cells
    # Map nodes to grid positions (simple mapping)
    for node in range(env.num_nodes):
        row = node // grid_size
        col = node % grid_size
        
        x = offset_x + col * cell_size
        y = offset_y + row * cell_size
        
        # Get state
        cell_state = int(state_array[node])
        material = env.material_map[node]
        
        # Choose color based on state and material
        if material == Material.WATER:
            color = colors['water']
        elif cell_state == 0:  # Unburned - use material-specific color
            # Different colors for wood and grass when unburned
            if material == Material.WOOD:
                color = colors['wood']
            elif material == Material.GRASS:
                color = colors['grass']
            else:
                color = colors['unburned']  # Fallback
        elif cell_state == 1:  # Burning
            color = colors['burning']
        else:  # Burned
            color = colors['burned']
        
        # Draw cell
        pygame.draw.rect(screen, color, (x, y, cell_size - 1, cell_size - 1))


def _compute_stats(state: np.ndarray) -> dict:
    """Compute statistics from state array."""
    return {
        'unburned': int(np.sum(state == 0)),
        'burning': int(np.sum(state == 1)),
        'burned': int(np.sum(state == 2)),
    }


def main():
    """Main function to run comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare kernel-based vs physics simulation")
    parser.add_argument("--kernel_path", type=str, required=True, 
                       help="Path to learned kernel file (.pkl)")
    parser.add_argument("--num_nodes", type=int, default=1000, 
                       help="Number of nodes in graph")
    parser.add_argument("--num_steps", type=int, default=500, 
                       help="Number of simulation steps")
    parser.add_argument("--fps", type=int, default=30, 
                       help="Animation frames per second")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Random seed for initial conditions")
    
    args = parser.parse_args()
    
    print("="*70)
    print("KERNEL vs PHYSICS SIMULATION COMPARISON")
    print("="*70)
    
    # Load kernel
    print(f"\nLoading kernel from {args.kernel_path}...")
    kernel_learner = TransitionKernelLearner()
    try:
        kernel_learner.load_kernel(args.kernel_path)
        print("✓ Kernel loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading kernel: {e}")
        return
    
    # Create RNG for reproducible initial conditions
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()
    
    # Create base environment structure first (to ensure same graph)
    print("\nCreating base environment structure...")
    from main_graph_agents import create_graph_environment
    
    base_env = create_graph_environment(
        num_nodes=args.num_nodes,
        mode="rsp",  # Just for structure
        seed=args.seed,
    )
    
    # Create initial burning nodes (same for both)
    num_clusters = 3
    initial_burning_nodes = []
    available_nodes = list(range(args.num_nodes))
    
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
    
    print(f"Initial burning clusters: {len(initial_burning_nodes)} nodes across {num_clusters} clusters")
    
    # Handle blocked_mask
    blocked_mask_to_pass = None
    if base_env.blocked_mask is not None:
        blocked_mask = base_env.blocked_mask
        if blocked_mask.ndim == 2:
            blocked_flat = blocked_mask.flatten()
        else:
            blocked_flat = blocked_mask.copy()
        if len(blocked_flat) >= args.num_nodes:
            blocked_mask_to_pass = blocked_flat[:args.num_nodes].copy()
        else:
            blocked_mask_to_pass = np.zeros(args.num_nodes, dtype=bool)
            blocked_mask_to_pass[:len(blocked_flat)] = blocked_flat
    
    # Create physical environment
    print("\nCreating physical environment...")
    physics_env = GraphEnvironmentFire(
        num_nodes=args.num_nodes,
        mode="fire",
        edges=base_env.edges,
        width=base_env.width,
        height=base_env.height,
        blocked_mask=blocked_mask_to_pass,
        seed=args.seed,
        initial_burning_nodes=initial_burning_nodes,
    )
    
    # Get initial state and material map (these will be the same for kernel env)
    initial_state = physics_env.get_state().copy()
    material_map = physics_env.material_map.copy()
    print(f"Initial state: {_compute_stats(initial_state)}")
    
    # Create kernel-based environment with same initial conditions
    print("\nCreating kernel-based environment with same initial conditions...")
    kernel_env = create_comparison_environment(
        base_env=physics_env,
        kernel_learner=kernel_learner,
        rng=rng,
    )
    
    print("✓ Both environments initialized with same conditions")
    print(f"  - Same material map")
    print(f"  - Same initial state")
    print(f"  - Same graph structure")
    
    # Run visualization
    print("\nStarting side-by-side comparison...")
    visualize_comparison(
        physics_env=physics_env,
        kernel_env=kernel_env,
        num_steps=args.num_steps,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()


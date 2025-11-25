"""
Visualize fire spreading using the learned 512-pattern transition model from VIIRS data.

This script:
1. Loads the transition probability table computed from VIIRS data
2. Creates an Environment using the learned model
3. Animates fire spreading on a 3D globe using GlobeFireAnimator
"""

from __future__ import annotations

import os
import sys
import numpy as np

# Add parent directory to path to access environment_simulation module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment_simulation.environment import Environment
from auxiliary_scripts.visualize_globe import GlobeFireAnimator


def load_transition_table(table_path: str | None = None) -> np.ndarray:
    """
    Load the 512-pattern transition probability table.
    
    Args:
        table_path: Path to local_transition_p_fire.npy. If None, looks in current directory.
    
    Returns:
        Array of shape (512,) with transition probabilities
    """
    if table_path is None:
        # Look for the file in the project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        table_path = os.path.join(project_root, "local_transition_p_fire.npy")
    
    if not os.path.exists(table_path):
        raise FileNotFoundError(
            f"Transition table not found at {table_path}\n"
            "Please run p_table_compute_from_data.py first to generate the transition table."
        )
    
    p_fire = np.load(table_path)
    
    if p_fire.shape != (512,):
        raise ValueError(
            f"Expected transition table shape (512,), got {p_fire.shape}\n"
            "This might not be the correct file. Expected file: local_transition_p_fire.npy"
        )
    
    print(f"Loaded transition table from {table_path}")
    print(f"  Shape: {p_fire.shape}")
    print(f"  Min probability: {p_fire.min():.6f}")
    print(f"  Max probability: {p_fire.max():.6f}")
    print(f"  Mean probability: {p_fire.mean():.6f}")
    
    return p_fire


def create_environment_from_transition_table(
    transition_table: np.ndarray,
    height: int = 30,
    width: int = 60,
    blocked_mask: np.ndarray | None = None,
    seed: int = 42,
) -> Environment:
    """
    Create an Environment that uses the learned transition table.
    
    Args:
        transition_table: Array of shape (512,) with transition probabilities
        height: Grid height (latitude cells)
        width: Grid width (longitude cells)
        blocked_mask: Optional boolean mask for cells that cannot ignite
        seed: Random seed
    
    Returns:
        Environment instance using the viirs_table mode
    """
    env = Environment(
        width=width,
        height=height,
        mode="viirs_table",
        topology="sphere",
        blocked_mask=blocked_mask,
        seed=seed,
        transition_table=transition_table,
    )
    
    return env


def create_blocked_mask(height: int, width: int, ocean_fraction: float = 0.05) -> np.ndarray:
    """
    Create a simple blocked mask (e.g., ocean regions that cannot have fires).
    
    Args:
        height: Grid height
        width: Grid width
        ocean_fraction: Fraction of cells to block (random)
    
    Returns:
        Boolean mask where True indicates blocked cells
    """
    blocked = np.zeros((height, width), dtype=bool)
    
    # Block a permanent ocean strip (e.g., first few columns)
    blocked[:, :width // 18] = True
    
    # Randomly block additional cells (ocean/water bodies)
    rng = np.random.default_rng(7)  # Fixed seed for reproducibility
    blocked |= rng.random((height, width)) < ocean_fraction
    
    return blocked


def main(
    transition_table_path: str | None = None,
    height: int = 30,
    width: int = 60,
    initial_fires: int = 8,
    num_steps: int = 200,
    animation_interval: int = 150,
    save_path: str | None = None,
    seed: int = 42,
    ocean_fraction: float = 0.05,
):
    """
    Main function to visualize fire spreading with the learned transition model.
    
    Args:
        transition_table_path: Path to local_transition_p_fire.npy
        height: Grid height (latitude cells)
        width: Grid width (longitude cells)
        initial_fires: Number of initial fire cells to seed
        num_steps: Number of time steps to simulate
        animation_interval: Milliseconds between frames
        save_path: Optional path to save animation (e.g., 'animation.mp4')
        seed: Random seed for reproducibility
        ocean_fraction: Fraction of cells to block (ocean/water)
    """
    print("=" * 70)
    print("Fire Spreading Visualization using Learned VIIRS Transition Model")
    print("=" * 70)
    
    # Load transition table
    print("\n1. Loading transition probability table...")
    transition_table = load_transition_table(transition_table_path)
    
    # Create blocked mask (optional)
    print("\n2. Creating environment...")
    blocked_mask = create_blocked_mask(height, width, ocean_fraction=ocean_fraction)
    
    # Create environment with learned transition model
    env = create_environment_from_transition_table(
        transition_table=transition_table,
        height=height,
        width=width,
        blocked_mask=blocked_mask,
        seed=seed,
    )
    
    # Initialize with some fires
    print(f"\n3. Initializing with {initial_fires} initial fire cells...")
    env.reset(initial_events=initial_fires)
    print(f"   Initial active fires: {env.count_active()}")
    
    # Create animator
    print("\n4. Creating globe animator...")
    animator = GlobeFireAnimator(env)
    
    # Animate
    print(f"\n5. Animating {num_steps} steps...")
    print("   (Close the window or interrupt to stop)")
    
    animator.animate(
        frames=num_steps,
        interval=animation_interval,
        repeat=False,
        save_path=save_path,
        dpi=120,
    )
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize fire spreading using learned VIIRS transition model"
    )
    parser.add_argument(
        "--table-path",
        type=str,
        default=None,
        help="Path to local_transition_p_fire.npy (default: looks in project root)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=30,
        help="Grid height (latitude cells, default: 30)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=60,
        help="Grid width (longitude cells, default: 60)",
    )
    parser.add_argument(
        "--initial-fires",
        type=int,
        default=8,
        help="Number of initial fire cells (default: 8)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of time steps to simulate (default: 200)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=150,
        help="Animation interval in milliseconds (default: 150)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save animation (e.g., 'animation.mp4' or 'animation.gif')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--ocean-fraction",
        type=float,
        default=0.05,
        help="Fraction of cells to block as ocean (default: 0.05)",
    )
    
    args = parser.parse_args()
    
    main(
        transition_table_path=args.table_path,
        height=args.height,
        width=args.width,
        initial_fires=args.initial_fires,
        num_steps=args.steps,
        animation_interval=args.interval,
        save_path=args.save,
        seed=args.seed,
        ocean_fraction=args.ocean_fraction,
    )


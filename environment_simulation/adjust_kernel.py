"""
Script to adjust learned kernel transition probabilities.

Fixes:
1. Enforces physical constraints (no impossible transitions)
2. Reduces BURNING → BURNED transition probabilities to slow down burnout
3. Renormalizes probabilities after adjustments
"""

import argparse
import pickle
from typing import Dict, Tuple
from collections import defaultdict
import numpy as np
import sys
import os

# Mock window_option before importing kernel_learning to avoid favicon error
FIRE_SIM_PATH = os.path.join(os.path.dirname(__file__), '..', 'fire-simulation')
if FIRE_SIM_PATH not in sys.path:
    sys.path.insert(0, FIRE_SIM_PATH)

# Mock the window_option module to prevent favicon loading error
import types
mock_window_option = types.ModuleType('src.window_option')
mock_window_option.CELL_WIDTH = 10
mock_window_option.CELL_HEIGHT = 10
sys.modules['src.window_option'] = mock_window_option

# Now import TransitionKernelLearner
sys.path.insert(0, os.path.dirname(__file__))
from kernel_learning import TransitionKernelLearner


def enforce_physical_constraints(
    transition_probs: Dict[Tuple[int, str, int], Dict[int, float]]
) -> Dict[Tuple[int, str, int], Dict[int, float]]:
    """
    Enforce physical constraints on transition probabilities.
    
    Physical constraints:
    - UNBURNED (0) → BURNED (2): Must be 0 (must go through BURNING)
    - BURNING (1) → UNBURNED (0): Must be 0 (must go through BURNED)
    - BURNED (2) → BURNING (1): Must be 0 (recovery goes to UNBURNED)
    
    Args:
        transition_probs: Original transition probabilities
        
    Returns:
        Adjusted transition probabilities with constraints enforced
    """
    adjusted = {}
    
    for key, probs in transition_probs.items():
        current_state, material, k_prime = key
        adjusted_probs = probs.copy()
        
        # Enforce constraints based on current state
        if current_state == 0:  # UNBURNED
            adjusted_probs[2] = 0.0  # Cannot go directly to BURNED
        elif current_state == 1:  # BURNING
            adjusted_probs[0] = 0.0  # Cannot go directly to UNBURNED
        elif current_state == 2:  # BURNED
            adjusted_probs[1] = 0.0  # Cannot go to BURNING (recovery goes to UNBURNED)
        
        # Normalize probabilities over valid states
        valid_next_states = []
        if current_state == 0:  # UNBURNED → {0, 1}
            valid_next_states = [0, 1]
        elif current_state == 1:  # BURNING → {1, 2}
            valid_next_states = [1, 2]
        else:  # BURNED → {0, 2}
            valid_next_states = [0, 2]
        
        # Renormalize
        total_prob = sum(adjusted_probs.get(s, 0.0) for s in valid_next_states)
        if total_prob > 0:
            for s in valid_next_states:
                adjusted_probs[s] = adjusted_probs.get(s, 0.0) / total_prob
        else:
            # If all probabilities are zero (shouldn't happen), set uniform distribution
            for s in valid_next_states:
                adjusted_probs[s] = 1.0 / len(valid_next_states)
        
        # Ensure probabilities for invalid states are explicitly 0
        for s in range(3):
            if s not in valid_next_states:
                adjusted_probs[s] = 0.0
        
        adjusted[key] = adjusted_probs
    
    return adjusted


def reduce_burning_to_burned(
    transition_probs: Dict[Tuple[int, str, int], Dict[int, float]],
    reduction_factor: float = 0.5,
) -> Dict[Tuple[int, str, int], Dict[int, float]]:
    """
    Reduce transition probabilities from BURNING (1) to BURNED (2).
    
    This slows down the burnout rate, making fires burn longer.
    
    Args:
        transition_probs: Transition probabilities
        reduction_factor: Factor to multiply P(1→2) by (default 0.5 = halve it)
        
    Returns:
        Adjusted transition probabilities
    """
    adjusted = {}
    
    for key, probs in transition_probs.items():
        current_state, material, k_prime = key
        adjusted_probs = probs.copy()
        
        if current_state == 1:  # BURNING state
            # Reduce P(BURNING → BURNED)
            p_burned = adjusted_probs.get(2, 0.0) * reduction_factor
            # Increase P(BURNING → BURNING) to compensate
            p_burning = adjusted_probs.get(1, 0.0) + adjusted_probs.get(2, 0.0) * (1 - reduction_factor)
            
            adjusted_probs[2] = p_burned
            adjusted_probs[1] = p_burning
            
            # Ensure P(BURNING → UNBURNED) is 0 (physical constraint)
            adjusted_probs[0] = 0.0
            
            # Renormalize (should already sum to 1, but ensure it)
            total = adjusted_probs[1] + adjusted_probs[2]
            if total > 0:
                adjusted_probs[1] /= total
                adjusted_probs[2] /= total
        
        adjusted[key] = adjusted_probs
    
    return adjusted


def adjust_kernel(
    input_path: str,
    output_path: str,
    reduce_burnout_factor: float = 0.5,
    enforce_constraints: bool = True,
    verbose: bool = True,
):
    """
    Load kernel, apply adjustments, and save.
    
    Args:
        input_path: Path to input kernel file (.pkl)
        output_path: Path to save adjusted kernel (.pkl)
        reduce_burnout_factor: Factor to reduce P(BURNING→BURNED) by
        enforce_constraints: Whether to enforce physical constraints
        verbose: Print progress messages
    """
    if verbose:
        print("="*70)
        print("ADJUSTING LEARNED KERNEL")
        print("="*70)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Reduce burnout factor: {reduce_burnout_factor}")
        print(f"Enforce constraints: {enforce_constraints}")
        print()
    
    # Load kernel
    if verbose:
        print("Loading kernel...")
    learner = TransitionKernelLearner()
    try:
        learner.load_kernel(input_path)
        if verbose:
            print(f"✓ Kernel loaded successfully")
            print(f"  Total transitions: {learner.total_transitions}")
            print(f"  Configurations: {len(learner.transition_probs)}")
    except FileNotFoundError:
        print(f"✗ Error: File not found: {input_path}")
        return
    except Exception as e:
        print(f"✗ Error loading kernel: {e}")
        return
    
    # Get original transition probabilities
    original_probs = learner.transition_probs.copy()
    
    # Apply adjustments
    adjusted_probs = original_probs
    
    if enforce_constraints:
        if verbose:
            print("\nEnforcing physical constraints...")
        
        # Count violations before
        violations_before = 0
        for key, probs in original_probs.items():
            current_state, _, _ = key
            if current_state == 0 and probs.get(2, 0.0) > 1e-6:  # UNBURNED → BURNED
                violations_before += 1
            elif current_state == 1 and probs.get(0, 0.0) > 1e-6:  # BURNING → UNBURNED
                violations_before += 1
            elif current_state == 2 and probs.get(1, 0.0) > 1e-6:  # BURNED → BURNING
                violations_before += 1
        
        adjusted_probs = enforce_physical_constraints(adjusted_probs)
        
        # Count violations after
        violations_after = 0
        for key, probs in adjusted_probs.items():
            current_state, _, _ = key
            if current_state == 0 and probs.get(2, 0.0) > 1e-6:
                violations_after += 1
            elif current_state == 1 and probs.get(0, 0.0) > 1e-6:
                violations_after += 1
            elif current_state == 2 and probs.get(1, 0.0) > 1e-6:
                violations_after += 1
        
        if verbose:
            print(f"  Violations before: {violations_before}")
            print(f"  Violations after: {violations_after}")
            if violations_before > 0:
                print(f"  ✓ Fixed {violations_before} impossible transitions")
    
    if reduce_burnout_factor < 1.0:
        if verbose:
            print(f"\nReducing BURNING → BURNED transitions by factor {reduce_burnout_factor}...")
        
        # Count BURNING states and show before/after
        burning_configs = [k for k in adjusted_probs.keys() if k[0] == 1]
        if verbose and burning_configs:
            sample_key = burning_configs[0]
            before_p_burned = adjusted_probs[sample_key].get(2, 0.0)
            print(f"  Sample config {sample_key}:")
            print(f"    Before: P(BURNING→BURNED) = {before_p_burned:.4f}")
        
        adjusted_probs = reduce_burning_to_burned(adjusted_probs, reduction_factor=reduce_burnout_factor)
        
        if verbose and burning_configs:
            after_p_burned = adjusted_probs[sample_key].get(2, 0.0)
            after_p_burning = adjusted_probs[sample_key].get(1, 0.0)
            print(f"    After: P(BURNING→BURNED) = {after_p_burned:.4f}")
            print(f"           P(BURNING→BURNING) = {after_p_burning:.4f}")
    
    # Update learner with adjusted probabilities
    learner.transition_probs = adjusted_probs
    
    # Final verification: check for any remaining impossible transitions
    if verbose:
        print(f"\nVerifying physical constraints...")
    violations_remaining = 0
    for key, probs in adjusted_probs.items():
        current_state, _, _ = key
        if current_state == 0 and probs.get(2, 0.0) > 1e-6:  # UNBURNED → BURNED
            violations_remaining += 1
            if verbose and violations_remaining <= 5:  # Show first 5
                print(f"  WARNING: {key} still has P(UNBURNED→BURNED)={probs.get(2, 0.0):.6f}")
        elif current_state == 1 and probs.get(0, 0.0) > 1e-6:  # BURNING → UNBURNED
            violations_remaining += 1
            if verbose and violations_remaining <= 5:
                print(f"  WARNING: {key} still has P(BURNING→UNBURNED)={probs.get(0, 0.0):.6f}")
        elif current_state == 2 and probs.get(1, 0.0) > 1e-6:  # BURNED → BURNING
            violations_remaining += 1
            if verbose and violations_remaining <= 5:
                print(f"  WARNING: {key} still has P(BURNED→BURNING)={probs.get(1, 0.0):.6f}")
    
    if violations_remaining > 0:
        print(f"\n  ⚠ WARNING: {violations_remaining} configurations still have impossible transitions!")
        print(f"    The simulation code will enforce constraints at runtime, but you may want to fix the kernel.")
    elif verbose:
        print(f"  ✓ All physical constraints satisfied")
    
    # Save adjusted kernel
    if verbose:
        print(f"\nSaving adjusted kernel to {output_path}...")
    try:
        learner.save_kernel(output_path)
        if verbose:
            print(f"✓ Kernel saved successfully")
            
            # Print summary of adjustments
            print("\n" + "="*70)
            print("SUMMARY OF ADJUSTMENTS")
            print("="*70)
            print(f"Total configurations: {len(adjusted_probs)}")
            
            # Count by state
            by_state = {0: 0, 1: 0, 2: 0}
            for key in adjusted_probs.keys():
                by_state[key[0]] += 1
            
            print(f"UNBURNED configurations: {by_state[0]}")
            print(f"BURNING configurations: {by_state[1]}")
            print(f"BURNED configurations: {by_state[2]}")
            
            # Show some example transitions
            print("\nExample transitions (BURNING state):")
            burning_keys = [k for k in adjusted_probs.keys() if k[0] == 1][:5]
            for key in burning_keys:
                probs = adjusted_probs[key]
                print(f"  {key}: P(BURNING→BURNING)={probs.get(1, 0.0):.4f}, "
                      f"P(BURNING→BURNED)={probs.get(2, 0.0):.4f}")
            
            print("\nExample transitions (UNBURNED state):")
            unburned_keys = [k for k in adjusted_probs.keys() if k[0] == 0][:5]
            for key in unburned_keys:
                probs = adjusted_probs[key]
                print(f"  {key}: P(UNBURNED→UNBURNED)={probs.get(0, 0.0):.4f}, "
                      f"P(UNBURNED→BURNING)={probs.get(1, 0.0):.4f}, "
                      f"P(UNBURNED→BURNED)={probs.get(2, 0.0):.4f}")
            
    except Exception as e:
        print(f"✗ Error saving kernel: {e}")
        return


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Adjust learned kernel transition probabilities"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input kernel file (.pkl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save adjusted kernel (.pkl)"
    )
    parser.add_argument(
        "--reduce_burnout",
        type=float,
        default=0.5,
        help="Factor to reduce P(BURNING→BURNED) by (default: 0.5, meaning halve it)"
    )
    parser.add_argument(
        "--no_constraints",
        action="store_true",
        help="Skip enforcing physical constraints"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    adjust_kernel(
        input_path=args.input,
        output_path=args.output,
        reduce_burnout_factor=args.reduce_burnout,
        enforce_constraints=not args.no_constraints,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()


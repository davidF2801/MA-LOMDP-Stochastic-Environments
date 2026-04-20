#!/usr/bin/env julia

"""
Wrapper script to regenerate animations with labels from the project root directory.

Usage:
    julia regenerate_animations.jl <results_folder_path>

Example:
    julia regenerate_animations.jl "E:\\MA-LOMDP-Stochastic-Environments\\results\\run_2025-09-21T18-28-33-550\\Run 1"
"""

# Run the regeneration script from the scripts directory
include("scripts/regenerate_animations_with_labels.jl")

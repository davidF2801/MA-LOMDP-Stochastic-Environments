# Animation Labels Usage Guide

## Quick Start

To regenerate animations with labels from your existing results, use one of these methods:

### Method 1: Minimal Script (Most Reliable)
```bash
# From the project root directory
julia scripts/minimal_regenerate_animations.jl "E:\MA-LOMDP-Stochastic-Environments\results\run_2025-09-21T18-28-33-550\Run 1"
```

### Method 2: Simple Script (Auto-installs packages)
```bash
# From the project root directory
julia scripts/simple_regenerate_animations.jl "E:\MA-LOMDP-Stochastic-Environments\results\run_2025-09-21T18-28-33-550\Run 1"
```

### Method 3: Wrapper Script
```bash
# From the project root directory
julia regenerate_animations.jl "E:\MA-LOMDP-Stochastic-Environments\results\run_2025-09-21T18-28-33-550\Run 1"
```

### Method 4: Original Script (if MyProject module works)
```bash
# From the project root directory
julia scripts/regenerate_animations_with_labels.jl "E:\MA-LOMDP-Stochastic-Environments\results\run_2025-09-21T18-28-33-550\Run 1"
```

## What the Script Does

1. **Loads Data**: Reads CSV files from your results folder:
   - `agent_actions_*.csv` - Agent actions and detected events
   - `uncertainty_evolution_*.csv` - Uncertainty data per timestep
   - `event_tracking_*.csv` - Event tracking information

2. **Extracts Labels**: 
   - Counts events detected per timestep
   - Calculates average uncertainty per timestep

3. **Reconstructs Simulation**: 
   - Rebuilds environment evolution
   - Reconstructs action history
   - Creates simplified agent trajectories

4. **Generates Animations**: Creates new GIF files with labels showing:
   - Number of events detected by agents
   - Average uncertainty across all cells

## Output Files

The script creates new animation files with `_with_labels` suffix:
- `rsp_3x4_script_run1_with_labels.gif` - Main simulation animation
- `belief_event_present_3x4_script_run1_with_labels.gif` - Belief animation (if available)

## Troubleshooting

### Common Issues

1. **"MyProject not defined"**: Use the simple script instead:
   ```bash
   julia scripts/simple_regenerate_animations.jl "your_results_path"
   ```

2. **"CSV file not found"**: Check that your results folder contains:
   - `script/metrics/agent_actions_script.csv`
   - `script/metrics/uncertainty_evolution_script.csv`
   - `script/metrics/event_tracking_script.csv`

3. **"No planning mode folders found"**: Make sure you're pointing to a specific run folder (e.g., "Run 1"), not the main results folder.

### Example Directory Structure
```
results/
└── run_2025-09-21T18-28-33-550/
    └── Run 1/
        ├── script/
        │   ├── animations/
        │   └── metrics/
        │       ├── agent_actions_script.csv
        │       ├── uncertainty_evolution_script.csv
        │       └── event_tracking_script.csv
        └── environment/
```

## What the Labels Show

The animation titles now display:
```
RSP 3x4 - Time Step 5 - γ=0.95, Events: 2 | Agents: 2 | Detected: 1 | Avg Uncertainty: 0.623
```

Where:
- **Events: 2** - Events present in the environment
- **Agents: 2** - Number of agents
- **Detected: 1** - Events detected by agents this timestep
- **Avg Uncertainty: 0.623** - Average uncertainty across all cells

## For New Simulations

If you want labels in new simulations, the main scripts have been updated:
- `scripts/main.jl` (3x4 grid)
- `scripts/main_5x5_complex.jl` (5x5 complex trajectories)
- `scripts/main_9x9_circular.jl` (9x9 circular trajectories)

Just run them normally and the labels will be included automatically.

## Need Help?

If you encounter issues:
1. Try the simple script first: `julia scripts/simple_regenerate_animations.jl`
2. Check that your results folder has the required CSV files
3. Make sure you're running from the project root directory
4. Verify the path format uses double backslashes: `"E:\\path\\to\\results"`

# Animation Labels Feature

This document describes the new animation labels feature that adds real-time detected events and average uncertainty information to the saved animations.

## Overview

The animation labels feature enhances the visualization of RSP (Reactive Search and Planning) simulations by displaying:
- **Events Detected**: Number of events detected by agents in each timestep
- **Average Uncertainty**: Average uncertainty across all cells in each timestep

These labels appear in both:
1. Main RSP simulation animations (`rsp_*.gif`)
2. Belief event present animations (`belief_event_present_*.gif`)

## Files Modified

### Core Animation Functions
- `scripts/main.jl` - Main simulation script with 3x4 grid
- `scripts/main_5x5_complex.jl` - 5x5 complex trajectory simulation
- `scripts/main_9x9_circular.jl` - 9x9 circular trajectory simulation

### New Files
- `scripts/regenerate_animations_with_labels.jl` - Script to regenerate animations from existing results
- `scripts/test_animation_labels.jl` - Test script to verify the feature works
- `ANIMATION_LABELS_README.md` - This documentation

## Changes Made

### 1. Updated `visualize_rsp_state` Function
```julia
function visualize_rsp_state(
    time_step::Int,
    agents::Vector{Agent},
    environment_state::Matrix{EventState}=fill(NO_EVENT, GRID_HEIGHT, GRID_WIDTH),
    actions::Vector{SensingAction}=SensingAction[],
    ground_station_pos::Tuple{Int, Int}=(GROUND_STATION_X, GROUND_STATION_Y);
    events_detected::Int=0,
    avg_uncertainty::Float64=0.0
)
```

**Changes:**
- Added optional parameters `events_detected` and `avg_uncertainty`
- Updated plot title to include the new information:
  ```
  "RSP $(width)x$(height) - Time Step $(time_step) - γ=$(DISCOUNT_FACTOR), Events: $(count(==(EVENT_PRESENT), environment_state)) | Agents: $(length(agents)) | Detected: $(events_detected) | Avg Uncertainty: $(round(avg_uncertainty, digits=3))"
  ```

### 2. Updated `create_rsp_animation` Function
```julia
function create_rsp_animation(
    agents::Vector{Agent},
    num_steps::Int,
    environment_evolution::Vector{Matrix{EventState}}=Vector{Matrix{EventState}}(),
    action_history::Vector{Vector{SensingAction}}=Vector{Vector{SensingAction}}(),
    ground_station_pos::Tuple{Int, Int}=(GROUND_STATION_X, GROUND_STATION_Y),
    results_dir::String="",
    run_number::Int=1,
    planning_mode::Symbol=:script;
    events_detected_per_timestep::Vector{Int}=Int[],
    avg_uncertainty_per_timestep::Vector{Float64}=Float64[]
)
```

**Changes:**
- Added optional parameters for events detected and uncertainty data
- Updated frame creation loop to pass the data to `visualize_rsp_state`

### 3. Updated `create_belief_event_animation` Function
```julia
function create_belief_event_animation(
    belief_event_present_evolution::Vector{Matrix{Float64}},
    results_dir::String="",
    run_number::Int=1,
    planning_mode::Symbol=:script;
    events_detected_per_timestep::Vector{Int}=Int[],
    avg_uncertainty_per_timestep::Vector{Float64}=Float64[]
)
```

**Changes:**
- Added optional parameters for events detected and uncertainty data
- Updated heatmap title to include the new information:
  ```
  "Belief P(Event) — t=$(t - 1) | Detected: $(events_detected) | Avg Uncertainty: $(round(avg_uncertainty, digits=3))"
  ```

### 4. Updated Simulation Functions
All simulation functions now:
- Track `events_detected_per_timestep` vector
- Count events detected in each timestep
- Return the events detected data
- Pass the data to animation functions

### 5. Created Regeneration Script
`scripts/regenerate_animations_with_labels.jl` allows you to:
- Load existing simulation results from CSV files
- Extract events detected and uncertainty data
- Reconstruct environment evolution and action history
- Generate new animations with labels

## Usage

### For New Simulations
The labels are automatically included when running any of the main simulation scripts:
```bash
julia scripts/main.jl
julia scripts/main_5x5_complex.jl
julia scripts/main_9x9_circular.jl
```

### For Existing Results
Use the regeneration script to add labels to existing animations:
```bash
julia scripts/regenerate_animations_with_labels.jl "E:\MA-LOMDP-Stochastic-Environments\results\run_2025-09-21T18-28-33-550\Run 1"
```

### Testing
Run the test script to verify everything works:
```bash
julia scripts/test_animation_labels.jl
```

## Data Sources

The labels use data from the following sources:

### Events Detected
- Counted from agent observations in each timestep
- Based on `EVENT_PRESENT` states observed by agents
- Stored in `events_detected_per_timestep` vector

### Average Uncertainty
- Calculated from the global belief uncertainty map
- Mean of all cell uncertainties in each timestep
- Stored in `average_uncertainty_per_timestep` vector

## Output Files

The enhanced animations are saved with the same naming convention as before:
- `rsp_$(GRID_WIDTH)x$(GRID_HEIGHT)_$(planning_mode)_run$(run_number).gif`
- `belief_event_present_$(GRID_WIDTH)x$(GRID_HEIGHT)_$(planning_mode)_run$(run_number).gif`

When using the regeneration script, files are saved with `_with_labels` suffix:
- `rsp_$(GRID_WIDTH)x$(GRID_HEIGHT)_$(planning_mode)_run$(run_number)_with_labels.gif`
- `belief_event_present_$(GRID_WIDTH)x$(GRID_HEIGHT)_$(planning_mode)_run$(run_number)_with_labels.gif`

## Example Output

The animation titles now look like:
```
RSP 3x4 - Time Step 5 - γ=0.95, Events: 2 | Agents: 2 | Detected: 1 | Avg Uncertainty: 0.623
```

This shows:
- Grid size: 3x4
- Current timestep: 5
- Discount factor: 0.95
- Events present in environment: 2
- Number of agents: 2
- Events detected by agents: 1
- Average uncertainty: 0.623

## Technical Details

### Backward Compatibility
- All changes are backward compatible
- Optional parameters have default values
- Existing code will continue to work without modification

### Performance Impact
- Minimal performance impact
- Only adds simple counting and data passing
- No significant memory overhead

### Error Handling
- Graceful handling of missing data
- Default values when data is not available
- Robust CSV parsing in regeneration script

## Troubleshooting

### Common Issues

1. **Missing CSV files**: Ensure the results folder contains the required CSV files
2. **Data format errors**: Check that CSV files have the expected column names
3. **Memory issues**: For large simulations, consider processing in batches

### Debug Mode
Enable debug output by setting environment variables:
```bash
export JULIA_DEBUG=1
julia scripts/regenerate_animations_with_labels.jl "path/to/results"
```

## Future Enhancements

Potential improvements for the animation labels feature:
1. **Additional metrics**: Detection delay, agent efficiency
2. **Interactive labels**: Clickable elements with detailed information
3. **Custom formatting**: User-defined label formats and positions
4. **Export options**: Save label data to separate files
5. **Real-time updates**: Live updating during simulation

## Contributing

When modifying the animation labels feature:
1. Update all three main script files consistently
2. Test with the provided test script
3. Verify backward compatibility
4. Update this documentation
5. Test the regeneration script with various result formats

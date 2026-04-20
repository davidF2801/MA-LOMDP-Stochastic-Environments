#!/usr/bin/env julia

"""
Regenerate MP4 animations from existing CSV results data
Creates high-quality MP4 videos with real-time labels showing:
- Number of detected events per timestep
- Average uncertainty per timestep
- Agent positions and actions
- Event locations

Usage:

The script creates a new folder in the same directory as the original results
with a descriptive name: {planning_mode}_{grid_width}x{grid_height}_run{run_number}_mp4_animations

This script works with results from:
- main.jl (3x4 grid, row-only visibility)
- main_5x5_complex.jl (5x5 grid, cross-shaped sensors)
- main_9x9_circular.jl (9x9 grid, circular trajectories)
"""

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Plots
using Dates
using DataFrames
using CSV
using Statistics
using DelimitedFiles

# Use plotlyjs backend for high-quality output
Plots.plotlyjs()

# Constants
const GROUND_STATION_X = 2
const GROUND_STATION_Y = 1
const DISCOUNT_FACTOR = 0.95

# Event states
const NO_EVENT = 0
const EVENT_PRESENT = 1

# Agent colors for visualization
const AGENT_COLORS = [:red, :blue, :green, :orange, :purple, :brown, :pink, :gray]

println("🎬 MP4 Animation Regenerator with Real-time Labels")
println("==================================================")

"""
Load simulation data from CSV files in the results folder
"""
function load_simulation_data(results_folder::String)
    println("📁 Loading simulation data from: $(results_folder)")
    
    # Find the planning mode folder (script, prior_based, pbvi_*, etc.)
    planning_mode_folders = filter(x -> isdir(joinpath(results_folder, x)) && x != "environment", readdir(results_folder))
    
    if isempty(planning_mode_folders)
        error("No planning mode folders found in $(results_folder)")
    end
    
    # Use the first planning mode folder found
    planning_mode = planning_mode_folders[1]
    planning_mode_path = joinpath(results_folder, planning_mode)
    
    println("  Using planning mode: $(planning_mode)")
    
    # Extract run number from folder path
    run_number = parse(Int, split(basename(results_folder), " ")[2])
    
    # Load agent actions CSV
    actions_csv_path = joinpath(planning_mode_path, "metrics", "agent_actions_$(planning_mode)_run$(run_number).csv")
    if !isfile(actions_csv_path)
        error("Agent actions CSV not found: $(actions_csv_path)")
    end
    
    actions_df = CSV.read(actions_csv_path, DataFrame)
    println("  Loaded $(nrow(actions_df)) action records")
    
    # Load uncertainty evolution data
    uncertainty_csv_path = joinpath(planning_mode_path, "metrics", "uncertainty_evolution_$(planning_mode)_run$(run_number).csv")
    if !isfile(uncertainty_csv_path)
        error("Uncertainty evolution CSV not found: $(uncertainty_csv_path)")
    end
    
    uncertainty_df = CSV.read(uncertainty_csv_path, DataFrame)
    println("  Loaded $(nrow(uncertainty_df)) uncertainty records")
    
    # Load event tracking data
    event_tracking_csv_path = joinpath(planning_mode_path, "metrics", "event_tracking_$(planning_mode)_run$(run_number).csv")
    if !isfile(event_tracking_csv_path)
        error("Event tracking CSV not found: $(event_tracking_csv_path)")
    end
    
    event_tracking_df = CSV.read(event_tracking_csv_path, DataFrame)
    println("  Loaded $(nrow(event_tracking_df)) event tracking records")
    
    # Load RSP parameters from environment if available
    rsp_params = nothing
    env_params_path = joinpath(results_folder, "environment", "environment_parameters_run$(run_number).png")
    if isfile(env_params_path)
        println("  Found environment parameters file")
        # We can't easily load the RSP parameters from the PNG, but we know they exist
        rsp_params = "available"
    end
    
    return planning_mode, actions_df, uncertainty_df, event_tracking_df, rsp_params
end

"""
Extract events detected per timestep from agent actions data
"""
function extract_events_detected_per_timestep(actions_df::DataFrame, event_tracking_df::DataFrame)
    println("📊 Extracting events detected per timestep...")
    
    # Get all timesteps from actions
    timesteps = sort(unique(actions_df.timestep))
    events_detected_per_timestep = Int[]
    
    for t in timesteps
        # Count events detected at this timestep
        events_detected_this_step = 0
        
        # Get actions at this timestep
        timestep_actions = filter(row -> row.timestep == t, actions_df)
        
        for action_row in eachrow(timestep_actions)
            if !ismissing(action_row.target_cells) && action_row.target_cells != ""
                # Parse target cell coordinates
                cell_str = string(action_row.target_cells)
                # Remove parentheses and parse coordinates
                cell_str = replace(cell_str, "(" => "", ")" => "")
                coords = split(cell_str, ", ")
                if length(coords) == 2
                    cell_x = parse(Int, coords[1])
                    cell_y = parse(Int, coords[2])
                    
                    # Check if there was an event at this cell at this timestep
                    # Look for events that were detected at exactly this timestep
                    detected_events = filter(row -> 
                        row.cell_x == cell_x && 
                        row.cell_y == cell_y &&
                        row.observed == true &&
                        row.detection_time == t,
                        event_tracking_df
                    )
                    
                    events_detected_this_step += nrow(detected_events)
                end
            end
        end
        
        push!(events_detected_per_timestep, events_detected_this_step)
    end
    
    println("  Extracted events detected for $(length(events_detected_per_timestep)) timesteps")
    return events_detected_per_timestep
end

"""
Extract average uncertainty per timestep
"""
function extract_average_uncertainty_per_timestep(uncertainty_df::DataFrame)
    println("📊 Extracting average uncertainty per timestep...")
    
    # Sort by timestep to ensure correct order
    sorted_df = sort(uncertainty_df, :timestep)
    
    avg_uncertainty_per_timestep = sorted_df.average_uncertainty
    
    println("  Extracted uncertainty for $(length(avg_uncertainty_per_timestep)) timesteps")
    return avg_uncertainty_per_timestep
end

"""
Determine grid size and sensor type from the data
"""
function determine_grid_configuration(actions_df::DataFrame, event_tracking_df::DataFrame)
    println("🔍 Determining grid configuration...")
    
    # Get grid size from event tracking data
    max_x = maximum(event_tracking_df.cell_x)
    max_y = maximum(event_tracking_df.cell_y)
    grid_width = max_x
    grid_height = max_y
    
    # Determine sensor type based on grid size and agent behavior
    if grid_width == 3 && grid_height == 4
        sensor_type = :row_only
        println("  Detected 3x4 grid with row-only visibility")
    elseif grid_width == 5 && grid_height == 5
        sensor_type = :cross
        println("  Detected 5x5 grid with cross-shaped sensors")
    elseif grid_width == 9 && grid_height == 9
        sensor_type = :circular
        println("  Detected 9x9 grid with circular trajectories")
    else
        sensor_type = :unknown
        println("  Unknown grid configuration: $(grid_width)x$(grid_height)")
    end
    
    return grid_width, grid_height, sensor_type
end

"""
Create agent positions based on trajectory type and grid size
"""
function create_agent_positions(grid_width::Int, grid_height::Int, sensor_type::Symbol, timestep::Int)
    positions = Tuple{Int,Int}[]
    
    if sensor_type == :row_only
        # Row-only visibility: agents move up and down in specific columns
        # Agent 1: column 2, starts at row 1
        agent1_row = ((timestep + 0) % 4) + 1
        push!(positions, (2, agent1_row))
        
        # Agent 2: column 2, starts at row 3
        agent2_row = ((timestep + 2) % 4) + 1
        push!(positions, (2, agent2_row))
        
    elseif sensor_type == :cross
        # Cross-shaped sensors: agents move in complex trajectories
        # Agent 1: column 2, moving up and down
        agent1_row = ((timestep + 0) % 5) + 1
        push!(positions, (2, agent1_row))
        
        # Agent 2: column 4, moving up and down with offset
        agent2_row = ((timestep + 3) % 5) + 1
        push!(positions, (4, agent2_row))
        
    elseif sensor_type == :circular
        # Circular trajectories: agents move in circles
        center_x = div(grid_width, 2)
        center_y = div(grid_height, 2)
        radius = min(div(grid_width, 4), div(grid_height, 4))
        
        # Agent 1: clockwise circle
        angle1 = (timestep * 0.3) % (2π)
        x1 = center_x + radius * cos(angle1)
        y1 = center_y + radius * sin(angle1)
        push!(positions, (round(Int, x1), round(Int, y1)))
        
        # Agent 2: counter-clockwise circle with offset
        angle2 = (timestep * -0.3 + π) % (2π)
        x2 = center_x + radius * cos(angle2)
        y2 = center_y + radius * sin(angle2)
        push!(positions, (round(Int, x2), round(Int, y2)))
    end
    
    return positions
end

"""
Get field of regard based on sensor type
"""
function get_field_of_regard(position::Tuple{Int,Int}, sensor_type::Symbol, grid_width::Int, grid_height::Int)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    if sensor_type == :row_only
        # Row-only visibility: agent can only see cells in its own row
        for col in 1:grid_width
            if col != x  # Don't include current position
                push!(fov_cells, (col, y))
            end
        end
        
    elseif sensor_type == :cross
        # Cross-shaped sensor: agent's position and adjacent cells in cardinal directions
        for dx in -1:1, dy in -1:1
            nx, ny = x + dx, y + dy
            if 1 <= nx <= grid_width && 1 <= ny <= grid_height
                # Only include cross pattern (not diagonal)
                if (dx == 0 && dy == 0) || (dx == 0 && dy != 0) || (dx != 0 && dy == 0)
                    push!(fov_cells, (nx, ny))
                end
            end
        end
        
    elseif sensor_type == :circular
        # Circular sensor: agent can see cells within a certain radius
        radius = 2
        for dx in -radius:radius, dy in -radius:radius
            nx, ny = x + dx, y + dy
            if 1 <= nx <= grid_width && 1 <= ny <= grid_height
                distance = sqrt(dx^2 + dy^2)
                if distance <= radius
                    push!(fov_cells, (nx, ny))
                end
            end
        end
    end
    
    return fov_cells
end

"""
Reconstruct environment state from event tracking data
"""
function reconstruct_environment_state(event_tracking_df::DataFrame, grid_width::Int, grid_height::Int, timestep::Int)
    env_state = fill(NO_EVENT, grid_height, grid_width)
    
    # Find all events that are active at this timestep
    # Handle missing end_time values (represented as -1 in CSV)
    active_events = filter(row -> 
        row.start_time <= timestep &&
        (ismissing(row.end_time) || row.end_time == -1 || row.end_time > timestep),
        event_tracking_df
    )
    
    # Place events in the environment
    for event in eachrow(active_events)
        if 1 <= event.cell_x <= grid_width && 1 <= event.cell_y <= grid_height
            env_state[event.cell_y, event.cell_x] = EVENT_PRESENT
        end
    end
    
    return env_state
end

"""
Get agent actions for a specific timestep
"""
function get_agent_actions(actions_df::DataFrame, timestep::Int)
    timestep_actions = filter(row -> row.timestep == timestep, actions_df)
    actions = []
    
    for action_row in eachrow(timestep_actions)
        target_cells = Tuple{Int,Int}[]
        
        if !ismissing(action_row.target_cells) && action_row.target_cells != ""
            # Parse target cell coordinates
            cell_str = string(action_row.target_cells)
            cell_str = replace(cell_str, "(" => "", ")" => "")
            coords = split(cell_str, ", ")
            if length(coords) == 2
                cell_x = parse(Int, coords[1])
                cell_y = parse(Int, coords[2])
                push!(target_cells, (cell_x, cell_y))
            end
        end
        
        action = (
            agent_id = action_row.agent_id,
            target_cells = target_cells,
            communicate = action_row.communicate
        )
        push!(actions, action)
    end
    
    return actions
end

"""
Visualize RSP state with real-time labels
"""
function visualize_rsp_state_with_labels(
    time_step::Int,
    grid_width::Int,
    grid_height::Int,
    sensor_type::Symbol,
    environment_state::Matrix{Int},
    agent_positions::Vector{Tuple{Int,Int}},
    actions::Vector{Any},
    ground_station_pos::Tuple{Int, Int};
    events_detected::Int=0,
    avg_uncertainty::Float64=0.0
)
    # Create a blank plot with correct limits and aspect
    p = plot(; xlim=(0.5, grid_width+0.5), ylim=(0.5, grid_height+0.5),
        aspect_ratio=:equal, size=(800, 800), legend=false,
        xlabel="X Coordinate", ylabel="Y Coordinate",
        grid=false,
        title="RSP $(grid_width)x$(grid_height) - Time Step $(time_step) - γ=$(DISCOUNT_FACTOR), Events: $(count(==(EVENT_PRESENT), environment_state)) | Agents: $(length(agent_positions)) | Detected: $(events_detected) | Avg Uncertainty: $(round(avg_uncertainty, digits=3))",
        titlefontsize=14,
        background_color=:white
    )

    # Draw grid cells as white squares with black borders
    for x in 1:grid_width, y in 1:grid_height
        xs = [x-0.5, x+0.5, x+0.5, x-0.5]
        ys = [y-0.5, y-0.5, y+0.5, y+0.5]
        plot!(p, xs, ys, seriestype=:shape, fillcolor=:white, linecolor=:black, linewidth=1, alpha=1, label=false)
    end

    # Overlay: FOR and action for each agent
    for (i, position) in enumerate(agent_positions)
        color = AGENT_COLORS[i]
        
        # Get FOR cells
        for_cells = get_field_of_regard(position, sensor_type, grid_width, grid_height)
        
        # Highlight FOR (light color)
        for (x, y) in for_cells
            xs = [x-0.5, x+0.5, x+0.5, x-0.5]
            ys = [y-0.5, y-0.5, y+0.5, y+0.5]
            plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.18, label=false)
        end
        
        # Highlight agent's own position in FOR (light color, but slightly more opaque)
        x, y = position
        xs = [x-0.5, x+0.5, x+0.5, x-0.5]
        ys = [y-0.5, y-0.5, y+0.5, y+0.5]
        plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.28, label=false)
        
        # Highlight action (dark color), matching by agent id
        action_idx = findfirst(a -> a.agent_id == i, actions)
        if action_idx !== nothing && !isempty(actions[action_idx].target_cells)
            for (x, y) in actions[action_idx].target_cells
                xs = [x-0.5, x+0.5, x+0.5, x-0.5]
                ys = [y-0.5, y-0.5, y+0.5, y+0.5]
                plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.5, label=false)
            end
        end
    end

    # Overlay: Ground station as green star
    scatter!(p, [ground_station_pos[1]], [ground_station_pos[2]]; marker=:star, markersize=14, color=:green, alpha=0.9, label=false)

    # Overlay: Agents as small circles
    for (i, position) in enumerate(agent_positions)
        agent_color = AGENT_COLORS[i]
        scatter!(p, [position[1]], [position[2]]; marker=:circle, markersize=10, color=agent_color, alpha=0.9, label=false)
    end

    # Overlay: Fire emoji for events
    for y in 1:grid_height, x in 1:grid_width
        if environment_state[y, x] == EVENT_PRESENT
            annotate!(p, x, y, text("🔥", :center, 18))
        end
    end

    return p
end

"""
Create MP4 animation from simulation data
"""
function create_mp4_animation(
    actions_df::DataFrame,
    uncertainty_df::DataFrame,
    event_tracking_df::DataFrame,
    grid_width::Int,
    grid_height::Int,
    sensor_type::Symbol,
    planning_mode::String,
    run_number::Int,
    output_dir::String
)
    println("\n🎬 Creating MP4 Animation with Real-time Labels")
    println("==============================================")
    
    # Extract data
    events_detected_per_timestep = extract_events_detected_per_timestep(actions_df, event_tracking_df)
    avg_uncertainty_per_timestep = extract_average_uncertainty_per_timestep(uncertainty_df)
    
    # Get timesteps
    timesteps = sort(unique(actions_df.timestep))
    num_steps = length(timesteps)
    
    println("  Creating animation with $(num_steps) frames...")
    
    # Create frames for animation
    frames = []
    
    for (frame_idx, timestep) in enumerate(timesteps)
        # Get environment state for this timestep
        env_state = reconstruct_environment_state(event_tracking_df, grid_width, grid_height, timestep)
        
        # Get agent positions
        agent_positions = create_agent_positions(grid_width, grid_height, sensor_type, timestep)
        
        # Get actions for this timestep
        actions = get_agent_actions(actions_df, timestep)
        
        # Get events detected and uncertainty for this timestep
        events_detected = if frame_idx <= length(events_detected_per_timestep)
            events_detected_per_timestep[frame_idx]
        else
            0
        end
        
        avg_uncertainty = if frame_idx <= length(avg_uncertainty_per_timestep)
            avg_uncertainty_per_timestep[frame_idx]
        else
            0.0
        end
        
        # Create frame
        frame = visualize_rsp_state_with_labels(
            timestep, grid_width, grid_height, sensor_type, env_state, 
            agent_positions, actions, (GROUND_STATION_X, GROUND_STATION_Y);
            events_detected=events_detected, avg_uncertainty=avg_uncertainty
        )
        push!(frames, frame)
    end
    
    # Create animation
    anim = @animate for frame in frames
        plot(frame, size=(800, 800))
    end
    
    # Save as MP4 (if supported) or GIF
    animation_filename = joinpath(output_dir, "rsp_$(grid_width)x$(grid_height)_$(planning_mode)_run$(run_number)_with_labels.mp4")
    
    try
        # Try to save as MP4 first
        mp4(anim, animation_filename, fps=2.0)
        println("✓ Saved MP4 animation: $(basename(animation_filename))")
    catch
        # Fallback to GIF if MP4 is not supported
        animation_filename = replace(animation_filename, ".mp4" => ".gif")
        gif(anim, animation_filename, fps=2.0)
        println("✓ Saved GIF animation: $(basename(animation_filename))")
    end
    
    return anim
end

"""
Create belief animation (heatmap of P(Event Present)) using actual uncertainty data
"""
function create_belief_mp4_animation(
    uncertainty_df::DataFrame,
    event_tracking_df::DataFrame,
    actions_df::DataFrame,
    grid_width::Int,
    grid_height::Int,
    planning_mode::String,
    run_number::Int,
    output_dir::String
)
    println("\n🎬 Creating Belief MP4 Animation (P(Event Present))")
    println("=================================================")

    # Get timesteps and uncertainty data
    timesteps = sort(unique(uncertainty_df.timestep))
    sorted_uncertainty_df = sort(uncertainty_df, :timestep)
    avg_uncertainty_per_timestep = sorted_uncertainty_df.average_uncertainty

    # Build frames as heatmaps with fixed color limits [0, 1]
    anim = @animate for (t_idx, timestep) in enumerate(timesteps)
        # Create a probability map based on actual environment state
        prob_map = zeros(Float64, grid_height, grid_width)
        
        # Find all events that are active at this timestep
        active_events = filter(row -> 
            row.start_time <= timestep &&
            (ismissing(row.end_time) || row.end_time == -1 || row.end_time > timestep),
            event_tracking_df
        )
        
        # Set probability to 1.0 for cells with active events
        for event in eachrow(active_events)
            if 1 <= event.cell_x <= grid_width && 1 <= event.cell_y <= grid_height
                prob_map[event.cell_y, event.cell_x] = 1.0
            end
        end
        
        # Calculate events detected at this timestep from actions
        events_detected = 0
        timestep_actions = filter(row -> row.timestep == timestep, actions_df)
        
        for action_row in eachrow(timestep_actions)
            if !ismissing(action_row.target_cells) && action_row.target_cells != ""
                # Parse target cell coordinates
                cell_str = string(action_row.target_cells)
                cell_str = replace(cell_str, "(" => "", ")" => "")
                coords = split(cell_str, ", ")
                if length(coords) == 2
                    cell_x = parse(Int, coords[1])
                    cell_y = parse(Int, coords[2])
                    
                    # Check if there was an event at this cell at this timestep
                    active_events_at_cell = filter(row -> 
                        row.cell_x == cell_x && 
                        row.cell_y == cell_y &&
                        row.start_time <= timestep &&
                        (ismissing(row.end_time) || row.end_time == -1 || row.end_time > timestep),
                        event_tracking_df
                    )
                    
                    # If there was an active event and it was observed, count it
                    if !isempty(active_events_at_cell) && any(row -> row.observed && row.detection_time <= timestep, active_events_at_cell)
                        events_detected += 1
                    end
                end
            end
        end
        
        avg_uncertainty = if t_idx <= length(avg_uncertainty_per_timestep)
            avg_uncertainty_per_timestep[t_idx]
        else
            0.0
        end
        
        heatmap(
            prob_map;
            title = "Belief P(Event) — t=$(timestep) | Detected: $(events_detected) | Avg Uncertainty: $(round(avg_uncertainty, digits=3))",
            xlabel = "X Coordinate",
            ylabel = "Y Coordinate",
            aspect_ratio = :equal,
            colorbar_title = "P(Event)",
            c = :viridis,
            clims = (0.0, 1.0),
            size = (800, 800)
        )
    end

    # Save as MP4 (if supported) or GIF
    gif_filename = joinpath(output_dir, "belief_event_present_$(grid_width)x$(grid_height)_$(planning_mode)_run$(run_number)_with_labels.mp4")
    
    try
        # Try to save as MP4 first
        mp4(anim, gif_filename, fps=2.0)
        println("✓ Saved belief MP4 animation: $(basename(gif_filename))")
    catch
        # Fallback to GIF if MP4 is not supported
        gif_filename = replace(gif_filename, ".mp4" => ".gif")
        gif(anim, gif_filename, fps=2.0)
        println("✓ Saved belief GIF animation: $(basename(gif_filename))")
    end

    return anim
end

"""
Main function to regenerate animations
"""
function main(results_folder::String)
    println("🎬 Starting MP4 Animation Regeneration")
    println("=====================================")
    
    # Load simulation data
    planning_mode, actions_df, uncertainty_df, event_tracking_df, rsp_params = load_simulation_data(results_folder)
    
    # Determine grid configuration
    grid_width, grid_height, sensor_type = determine_grid_configuration(actions_df, event_tracking_df)
    
    # Extract run number from folder path (already extracted in load_simulation_data)
    run_number = parse(Int, split(basename(results_folder), " ")[2])
    
    # Create output directory in the same folder as the original results
    # Format: {planning_mode}_{grid_width}x{grid_height}_run{run_number}_mp4_animations
    output_dir_name = "$(planning_mode)_$(grid_width)x$(grid_height)_run$(run_number)_mp4_animations"
    output_dir = joinpath(results_folder, output_dir_name)
    
    if !isdir(output_dir)
        mkpath(output_dir)
        println("📁 Created output directory: $(output_dir)")
    end
    
    println("\n📊 Configuration:")
    println("  Grid: $(grid_width)x$(grid_height)")
    println("  Sensor type: $(sensor_type)")
    println("  Planning mode: $(planning_mode)")
    println("  Run number: $(run_number)")
    println("  Output directory: $(output_dir)")
    
    # Create main RSP animation
    create_mp4_animation(
        actions_df, uncertainty_df, event_tracking_df,
        grid_width, grid_height, sensor_type,
        planning_mode, run_number, output_dir
    )
    
    # Create belief animation
    create_belief_mp4_animation(
        uncertainty_df, event_tracking_df, actions_df,
        grid_width, grid_height,
        planning_mode, run_number, output_dir
    )
    
    println("\n✅ MP4 Animation Regeneration Complete!")
    println("📁 Check the output directory for the new videos with real-time labels:")
    println("   $(output_dir)")
    println("   - Main simulation animation")
    println("   - Belief probability animation")
end

# Run the script
if length(ARGS) >= 1
    # Use command line argument if provided
    results_folder = ARGS[1]
else
    # Default to the test folder if no argument provided
    results_folder = "E:\\MA-LOMDP-Stochastic-Environments\\results\\run_2025-09-22T10-36-50-358\\Run 1"
    println("No arguments provided, using default folder: $(results_folder)")
    println("Usage: julia scripts/regenerate_mp4_animations.jl \"<results_folder_path>\"")
    println("Example: julia scripts/regenerate_mp4_animations.jl \"E:\\MA-LOMDP-Stochastic-Environments\\results\\run_2025-09-22T10-36-50-358\\Run 1\"")
end
main(results_folder)

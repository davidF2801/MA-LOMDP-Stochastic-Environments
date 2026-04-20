#!/usr/bin/env julia

"""
Script to regenerate animations with real-time detected events and uncertainty labels
from existing simulation results.

Usage:
    julia regenerate_animations_with_labels.jl <results_folder_path>

Example:
    julia regenerate_animations_with_labels.jl "E:\\MA-LOMDP-Stochastic-Environments\\results\\run_2025-09-21T18-28-33-550\\Run 1"
"""

using Pkg

# Change to project root directory
cd(@__DIR__)
cd("..")
Pkg.activate(".")

# Import required modules
using CSV
using DataFrames
using Plots
using Plots.PlotMeasures
using Random
using Dates
using Statistics

# Import project modules
using .MyProject
using .MyProject.Types
using .MyProject.Environment
using .MyProject.Planners.GroundStation

# Import specific functions
using .Environment: GridState, EventState, EVENT_PRESENT, NO_EVENT, SensingAction, GridObservation
using .Planners.GroundStation: get_position_at_time
using .MyProject.Types: get_cross_field_of_regard, get_nine_cell_field_of_regard, get_row_field_of_regard

# Constants
const GROUND_STATION_X = 2
const GROUND_STATION_Y = 1
const DISCOUNT_FACTOR = 0.95

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
    
    # Load agent actions CSV
    actions_csv_path = joinpath(planning_mode_path, "metrics", "agent_actions_$(planning_mode).csv")
    if !isfile(actions_csv_path)
        error("Agent actions CSV not found: $(actions_csv_path)")
    end
    
    actions_df = CSV.read(actions_csv_path, DataFrame)
    println("  Loaded $(nrow(actions_df)) action records")
    
    # Load uncertainty evolution data
    uncertainty_csv_path = joinpath(planning_mode_path, "metrics", "uncertainty_evolution_$(planning_mode).csv")
    if !isfile(uncertainty_csv_path)
        error("Uncertainty evolution CSV not found: $(uncertainty_csv_path)")
    end
    
    uncertainty_df = CSV.read(uncertainty_csv_path, DataFrame)
    println("  Loaded $(nrow(uncertainty_df)) uncertainty records")
    
    # Load event tracking data
    event_tracking_csv_path = joinpath(planning_mode_path, "metrics", "event_tracking_$(planning_mode).csv")
    if !isfile(event_tracking_csv_path)
        error("Event tracking CSV not found: $(event_tracking_csv_path)")
    end
    
    event_tracking_df = CSV.read(event_tracking_csv_path, DataFrame)
    println("  Loaded $(nrow(event_tracking_df)) event tracking records")
    
    return planning_mode, actions_df, uncertainty_df, event_tracking_df
end

"""
Extract events detected per timestep from agent actions data
"""
function extract_events_detected_per_timestep(actions_df::DataFrame)
    events_detected_per_timestep = Int[]
    
    # Group by timestep
    timesteps = sort(unique(actions_df.timestep))
    
    for t in timesteps
        timestep_actions = actions_df[actions_df.timestep .== t, :]
        events_detected_this_step = 0
        
        for row in eachrow(timestep_actions)
            if !ismissing(row.event_detected) && row.event_detected
                events_detected_this_step += 1
            end
        end
        
        push!(events_detected_per_timestep, events_detected_this_step)
    end
    
    return events_detected_per_timestep
end

"""
Extract average uncertainty per timestep from uncertainty data
"""
function extract_average_uncertainty_per_timestep(uncertainty_df::DataFrame)
    avg_uncertainty_per_timestep = Float64[]
    
    # Group by timestep
    timesteps = sort(unique(uncertainty_df.timestep))
    
    for t in timesteps
        timestep_uncertainty = uncertainty_df[uncertainty_df.timestep .== t, :]
        avg_uncertainty = mean(timestep_uncertainty.uncertainty)
        push!(avg_uncertainty_per_timestep, avg_uncertainty)
    end
    
    return avg_uncertainty_per_timestep
end

"""
Reconstruct environment evolution from event tracking data
"""
function reconstruct_environment_evolution(event_tracking_df::DataFrame, num_timesteps::Int, grid_width::Int, grid_height::Int)
    environment_evolution = Matrix{EventState}[]
    
    for t in 0:(num_timesteps-1)
        env_state = fill(NO_EVENT, grid_height, grid_width)
        
        # Find events active at this timestep
        active_events = event_tracking_df[
            (event_tracking_df.start_time .<= t) .& 
            ((event_tracking_df.end_time .>= t) .| (event_tracking_df.end_time .== -1)), 
            :
        ]
        
        for row in eachrow(active_events)
            if 1 <= row.cell_x <= grid_width && 1 <= row.cell_y <= grid_height
                env_state[row.cell_y, row.cell_x] = EVENT_PRESENT
            end
        end
        
        push!(environment_evolution, env_state)
    end
    
    return environment_evolution
end

"""
Reconstruct action history from agent actions data
"""
function reconstruct_action_history(actions_df::DataFrame, num_timesteps::Int)
    action_history = Vector{Vector{SensingAction}}()
    
    for t in 0:(num_timesteps-1)
        timestep_actions = actions_df[actions_df.timestep .== t, :]
        actions = SensingAction[]
        
        for row in eachrow(timestep_actions)
            if !ismissing(row.target_cells) && !isempty(row.target_cells)
                # Parse target cells (assuming they're stored as string representation)
                target_cells = eval(Meta.parse(row.target_cells))
                action = SensingAction(row.agent_id, target_cells)
                push!(actions, action)
            end
        end
        
        push!(action_history, actions)
    end
    
    return action_history
end

"""
Create agents from the simulation data (simplified version)
"""
function create_agents_from_data(actions_df::DataFrame, grid_width::Int, grid_height::Int)
    # This is a simplified version - in practice, you'd need to store agent trajectories
    # For now, we'll create dummy agents with basic trajectories
    
    agent_ids = sort(unique(actions_df.agent_id))
    agents = Agent[]
    
    for (i, agent_id) in enumerate(agent_ids)
        # Create a simple circular trajectory
        center_x = div(grid_width, 2)
        center_y = div(grid_height, 2)
        radius = min(div(grid_width, 4), div(grid_height, 4))
        
        trajectory = CircularTrajectory(center_x, center_y, radius, 20, 0.5)
        
        agent = Agent(
            agent_id,
            trajectory,
            i,  # phase_offset
            100.0,  # max_battery
            100.0,  # battery_level
            1.0,    # charging_rate
            RangeLimitedSensor(2.0, π/2, 0.0, :circular),
            []
        )
        
        push!(agents, agent)
    end
    
    return agents
end

"""
Visualize RSP state with labels (updated version)
"""
function visualize_rsp_state_with_labels(
    time_step::Int,
    agents::Vector{Agent},
    environment_state::Matrix{EventState},
    actions::Vector{SensingAction},
    ground_station_pos::Tuple{Int, Int},
    events_detected::Int,
    avg_uncertainty::Float64,
    grid_width::Int,
    grid_height::Int
)
    height, width = size(environment_state)
    agent_colors = [:red, :blue, :green, :orange]
    
    # Create a blank plot with correct limits and aspect
    p = plot(; xlim=(0.5, width+0.5), ylim=(0.5, height+0.5),
        aspect_ratio=:equal, size=(600, 800), legend=false,
        xlabel="X Coordinate", ylabel="Y Coordinate",
        grid=false,
        title="RSP $(width)x$(height) - Time Step $(time_step) - γ=$(DISCOUNT_FACTOR), Events: $(count(==(EVENT_PRESENT), environment_state)) | Agents: $(length(agents)) | Detected: $(events_detected) | Avg Uncertainty: $(round(avg_uncertainty, digits=3))",
        titlefontsize=12,
        background_color=:white
    )
    
    # Draw grid cells as white squares with black borders
    for x in 1:width, y in 1:height
        xs = [x-0.5, x+0.5, x+0.5, x-0.5]
        ys = [y-0.5, y-0.5, y+0.5, y+0.5]
        plot!(p, xs, ys, seriestype=:shape, fillcolor=:white, linecolor=:black, linewidth=1, alpha=1, label=false)
    end
    
    # Overlay: FOR and action for each agent
    for (i, agent) in enumerate(agents)
        color = agent_colors[i]
        pos = get_position_at_time(agent.trajectory, time_step, agent.phase_offset)
        
        # Get FOR cells (simplified - using row pattern)
        for_cells = get_row_field_of_regard(agent, pos, (width=width, height=height))
        
        # Highlight FOR (light color)
        for (x, y) in for_cells
            xs = [x-0.5, x+0.5, x+0.5, x-0.5]
            ys = [y-0.5, y-0.5, y+0.5, y+0.5]
            plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.18, label=false)
        end
        
        # Highlight agent's own position in FOR
        x, y = pos
        xs = [x-0.5, x+0.5, x+0.5, x-0.5]
        ys = [y-0.5, y-0.5, y+0.5, y+0.5]
        plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.28, label=false)
        
        # Highlight action (dark color)
        action_idx = findfirst(a -> a.agent_id == agent.id, actions)
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
    for (i, agent) in enumerate(agents)
        pos = get_position_at_time(agent.trajectory, time_step, agent.phase_offset)
        agent_color = agent_colors[i]
        scatter!(p, [pos[1]], [pos[2]]; marker=:circle, markersize=10, color=agent_color, alpha=0.9, label=false)
    end
    
    # Overlay: Fire emoji for events
    for y in 1:height, x in 1:width
        if environment_state[y, x] == EVENT_PRESENT
            annotate!(p, x, y, text("🔥", :center, 18))
        end
    end
    
    return p
end

"""
Create RSP animation with labels
"""
function create_rsp_animation_with_labels(
    agents::Vector{Agent},
    num_steps::Int,
    environment_evolution::Vector{Matrix{EventState}},
    action_history::Vector{Vector{SensingAction}},
    events_detected_per_timestep::Vector{Int},
    avg_uncertainty_per_timestep::Vector{Float64},
    ground_station_pos::Tuple{Int, Int},
    results_dir::String,
    run_number::Int,
    planning_mode::String,
    grid_width::Int,
    grid_height::Int
)
    println("\n🎬 Creating RSP Simulation Animation with Labels")
    println("===============================================")
    
    # Create animations directory path
    animations_dir = joinpath(results_dir, "Run $(run_number)", planning_mode, "animations")
    if !isdir(animations_dir)
        mkpath(animations_dir)
    end
    
    # Create frames for animation
    frames = []
    
    for step in 0:(num_steps-1)
        # Get environment state for this step
        env_state = if step < length(environment_evolution)
            environment_evolution[step + 1]
        else
            fill(NO_EVENT, grid_height, grid_width)
        end
        
        # Get actions for this step
        actions = if step < length(action_history)
            action_history[step + 1]
        else
            SensingAction[]
        end
        
        # Get events detected and uncertainty for this step
        events_detected = if step < length(events_detected_per_timestep)
            events_detected_per_timestep[step + 1]
        else
            0
        end
        
        avg_uncertainty = if step < length(avg_uncertainty_per_timestep)
            avg_uncertainty_per_timestep[step + 1]
        else
            0.0
        end
        
        # Create frame
        frame = visualize_rsp_state_with_labels(step, agents, env_state, actions, ground_station_pos, 
                                              events_detected, avg_uncertainty, grid_width, grid_height)
        push!(frames, frame)
    end
    
    # Create animation
    anim = @animate for frame in frames
        plot(frame, size=(600, 800))
    end
    
    # Save animation with new naming convention
    animation_filename = joinpath(animations_dir, "rsp_$(grid_width)x$(grid_height)_$(planning_mode)_run$(run_number)_with_labels.gif")
    gif(anim, animation_filename, fps=1.0)
    println("✓ Saved animation with labels: $(basename(animation_filename))")
    
    return anim
end

"""
Create belief animation with labels
"""
function create_belief_animation_with_labels(
    belief_event_present_evolution::Vector{Matrix{Float64}},
    events_detected_per_timestep::Vector{Int},
    avg_uncertainty_per_timestep::Vector{Float64},
    results_dir::String,
    run_number::Int,
    planning_mode::String,
    grid_width::Int,
    grid_height::Int
)
    println("\n🎬 Creating Belief Animation with Labels")
    println("=======================================")
    
    # Create animations directory path
    animations_dir = joinpath(results_dir, "Run $(run_number)", planning_mode, "animations")
    if !isdir(animations_dir)
        mkpath(animations_dir)
    end
    
    # Build frames as heatmaps with fixed color limits [0, 1]
    anim = @animate for (t, prob_map) in enumerate(belief_event_present_evolution)
        # Get events detected and uncertainty for this timestep
        events_detected = if t <= length(events_detected_per_timestep)
            events_detected_per_timestep[t]
        else
            0
        end
        
        avg_uncertainty = if t <= length(avg_uncertainty_per_timestep)
            avg_uncertainty_per_timestep[t]
        else
            0.0
        end
        
        heatmap(
            prob_map;
            title = "Belief P(Event) — t=$(t - 1) | Detected: $(events_detected) | Avg Uncertainty: $(round(avg_uncertainty, digits=3))",
            xlabel = "X Coordinate",
            ylabel = "Y Coordinate",
            aspect_ratio = :equal,
            colorbar_title = "P(Event)",
            c = :viridis,
            clims = (0.0, 1.0),
            size = (600, 600)
        )
    end
    
    gif_filename = joinpath(animations_dir, "belief_event_present_$(grid_width)x$(grid_height)_$(planning_mode)_run$(run_number)_with_labels.gif")
    gif(anim, gif_filename, fps=3.0)
    println("✓ Saved belief animation with labels: $(basename(gif_filename))")
    
    return anim
end

"""
Main function to regenerate animations with labels
"""
function regenerate_animations_with_labels(results_folder::String)
    println("🔄 Regenerating animations with labels...")
    println("=========================================")
    println("Results folder: $(results_folder)")
    
    # Load simulation data
    planning_mode, actions_df, uncertainty_df, event_tracking_df = load_simulation_data(results_folder)
    
    # Extract data
    events_detected_per_timestep = extract_events_detected_per_timestep(actions_df)
    avg_uncertainty_per_timestep = extract_average_uncertainty_per_timestep(uncertainty_df)
    
    # Determine grid dimensions from data
    grid_width = maximum(event_tracking_df.cell_x)
    grid_height = maximum(event_tracking_df.cell_y)
    num_timesteps = maximum(actions_df.timestep) + 1
    
    println("  Grid dimensions: $(grid_width)x$(grid_height)")
    println("  Number of timesteps: $(num_timesteps)")
    println("  Events detected per timestep: $(length(events_detected_per_timestep))")
    println("  Average uncertainty per timestep: $(length(avg_uncertainty_per_timestep))")
    
    # Reconstruct simulation data
    environment_evolution = reconstruct_environment_evolution(event_tracking_df, num_timesteps, grid_width, grid_height)
    action_history = reconstruct_action_history(actions_df, num_timesteps)
    agents = create_agents_from_data(actions_df, grid_width, grid_height)
    
    # Extract run number from folder path
    run_number = parse(Int, split(basename(results_folder), " ")[2])
    
    # Create animations with labels
    create_rsp_animation_with_labels(
        agents, num_timesteps, environment_evolution, action_history,
        events_detected_per_timestep, avg_uncertainty_per_timestep,
        (GROUND_STATION_X, GROUND_STATION_Y), dirname(results_folder), run_number,
        planning_mode, grid_width, grid_height
    )
    
    # For belief animation, we need to reconstruct belief data
    # This is more complex and would require additional data storage
    # For now, we'll skip the belief animation or create a simplified version
    
    println("\n✅ Animation regeneration completed!")
    println("📁 Check the animations folder for updated GIFs with labels")
end

# Main execution
if length(ARGS) != 1
    println("Usage: julia regenerate_animations_with_labels.jl <results_folder_path>")
    println("Example: julia regenerate_animations_with_labels.jl \"E:\\\\MA-LOMDP-Stochastic-Environments\\\\results\\\\run_2025-09-21T18-28-33-550\\\\Run 1\"")
    exit(1)
end

results_folder = ARGS[1]

if !isdir(results_folder)
    println("Error: Results folder does not exist: $(results_folder)")
    exit(1)
end

regenerate_animations_with_labels(results_folder)

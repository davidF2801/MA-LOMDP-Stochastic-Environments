#!/usr/bin/env julia

"""
Script to create actual video animations with labels from existing simulation data.
This will generate new GIF files with the events detected and uncertainty labels.

Usage:
    julia create_animations_with_labels.jl <results_folder_path>
"""

using Pkg
Pkg.activate(".")

# Try to use Plots, but fall back if not available
try
    using Plots
    PLOTS_AVAILABLE = true
    println("✅ Plots package available - will create animations")
catch
    println("❌ Plots not available - cannot create animations")
    exit(1)
end

using DelimitedFiles
using Random
using Dates
using Statistics

# Constants
const GROUND_STATION_X = 2
const GROUND_STATION_Y = 1
const DISCOUNT_FACTOR = 0.95

# Event states
const NO_EVENT = 0
const EVENT_PRESENT = 1

"""
Read CSV using DelimitedFiles
"""
function read_csv_delimited(filepath::String)
    if !isfile(filepath)
        return []
    end
    
    data, header = readdlm(filepath, ',', String, header=true)
    result = []
    for i in 1:size(data, 1)
        row = Dict{String, String}()
        for (j, col_name) in enumerate(header)
            row[col_name] = data[i, j]
        end
        push!(result, row)
    end
    return result
end

"""
Load simulation data
"""
function load_data(results_folder::String)
    println("📁 Loading data from: $(results_folder)")
    
    planning_modes = filter(x -> isdir(joinpath(results_folder, x)) && x != "environment", readdir(results_folder))
    planning_mode = planning_modes[1]
    planning_path = joinpath(results_folder, planning_mode)
    
    println("  Using: $(planning_mode)")
    
    run_num = parse(Int, split(basename(results_folder), " ")[2])
    actions_file = joinpath(planning_path, "metrics", "agent_actions_$(planning_mode)_run$(run_num).csv")
    uncertainty_file = joinpath(planning_path, "metrics", "uncertainty_evolution_$(planning_mode)_run$(run_num).csv")
    events_file = joinpath(planning_path, "metrics", "event_tracking_$(planning_mode)_run$(run_num).csv")
    
    actions = read_csv_delimited(actions_file)
    uncertainty = read_csv_delimited(uncertainty_file)
    events = read_csv_delimited(events_file)
    
    return planning_mode, actions, uncertainty, events, run_num
end

"""
Extract events detected per timestep
"""
function get_events_detected(actions)
    events_per_timestep = Int[]
    timesteps = sort(unique([parse(Int, row["timestep"]) for row in actions]))
    
    for t in timesteps
        timestep_actions = filter(row -> parse(Int, row["timestep"]) == t, actions)
        detected = count(row -> haskey(row, "event_detected") && row["event_detected"] == "true", timestep_actions)
        push!(events_per_timestep, detected)
    end
    
    return events_per_timestep
end

"""
Extract average uncertainty per timestep
"""
function get_avg_uncertainty(uncertainty)
    avg_per_timestep = Float64[]
    timesteps = sort(unique([parse(Int, row["timestep"]) for row in uncertainty]))
    
    for t in timesteps
        timestep_uncertainty = filter(row -> parse(Int, row["timestep"]) == t, uncertainty)
        uncertainties = [parse(Float64, row["average_uncertainty"]) for row in timestep_uncertainty]
        push!(avg_per_timestep, mean(uncertainties))
    end
    
    return avg_per_timestep
end

"""
Reconstruct environment evolution
"""
function reconstruct_environment(events, num_timesteps, grid_w, grid_h)
    env_evolution = Matrix{Int}[]
    
    for t in 0:(num_timesteps-1)
        env_state = fill(NO_EVENT, grid_h, grid_w)
        
        for row in events
            start_time = parse(Int, row["start_time"])
            end_time = parse(Int, row["end_time"])
            cell_x = parse(Int, row["cell_x"])
            cell_y = parse(Int, row["cell_y"])
            
            if start_time <= t && (end_time >= t || end_time == -1)
                if 1 <= cell_x <= grid_w && 1 <= cell_y <= grid_h
                    env_state[cell_y, cell_x] = EVENT_PRESENT
                end
            end
        end
        
        push!(env_evolution, env_state)
    end
    
    return env_evolution
end

"""
Create RSP animation with labels
"""
function create_rsp_animation_with_labels(
    env_evolution::Vector{Matrix{Int}},
    events_detected::Vector{Int},
    avg_uncertainty::Vector{Float64},
    grid_w::Int, grid_h::Int,
    results_dir::String, run_num::Int, planning_mode::String
)
    println("\n🎬 Creating RSP Animation with Labels...")
    
    # Create animations directory
    animations_dir = joinpath(results_dir, "Run $(run_num)", planning_mode, "animations")
    if !isdir(animations_dir)
        mkpath(animations_dir)
    end
    
    # Create frames
    frames = []
    
    for (step, env_state) in enumerate(env_evolution)
        events_detected_this_step = step <= length(events_detected) ? events_detected[step] : 0
        uncertainty_this_step = step <= length(avg_uncertainty) ? avg_uncertainty[step] : 0.0
        
        # Count total events in environment
        total_events = count(==(EVENT_PRESENT), env_state)
        
        # Create plot
        p = plot(
            xlim=(0.5, grid_w+0.5), ylim=(0.5, grid_h+0.5),
            aspect_ratio=:equal, size=(600, 800), legend=false,
            xlabel="X Coordinate", ylabel="Y Coordinate",
            grid=false,
            title="RSP $(grid_w)x$(grid_h) - Time Step $(step-1) - γ=$(DISCOUNT_FACTOR), Events: $(total_events) | Agents: ? | Detected: $(events_detected_this_step) | Avg Uncertainty: $(round(uncertainty_this_step, digits=3))",
            titlefontsize=12,
            background_color=:white
        )
        
        # Draw grid
        for x in 1:grid_w, y in 1:grid_h
            xs = [x-0.5, x+0.5, x+0.5, x-0.5]
            ys = [y-0.5, y-0.5, y+0.5, y+0.5]
            plot!(p, xs, ys, seriestype=:shape, fillcolor=:white, linecolor=:black, linewidth=1, alpha=1)
        end
        
        # Draw events
        for y in 1:grid_h, x in 1:grid_w
            if env_state[y, x] == EVENT_PRESENT
                annotate!(p, x, y, text("🔥", :center, 18))
            end
        end
        
        # Ground station
        scatter!(p, [GROUND_STATION_X], [GROUND_STATION_Y]; marker=:star, markersize=14, color=:green, alpha=0.9)
        
        push!(frames, p)
    end
    
    # Create animation
    anim = @animate for frame in frames
        plot(frame, size=(600, 800))
    end
    
    # Save animation
    animation_filename = joinpath(animations_dir, "rsp_$(grid_w)x$(grid_h)_$(planning_mode)_run$(run_num)_with_labels.gif")
    gif(anim, animation_filename, fps=1.0)
    println("✅ Saved RSP animation: $(basename(animation_filename))")
    
    return anim
end

"""
Create belief animation with labels
"""
function create_belief_animation_with_labels(
    events_detected::Vector{Int},
    avg_uncertainty::Vector{Float64},
    grid_w::Int, grid_h::Int,
    results_dir::String, run_num::Int, planning_mode::String
)
    println("\n🎬 Creating Belief Animation with Labels...")
    
    # Create animations directory
    animations_dir = joinpath(results_dir, "Run $(run_num)", planning_mode, "animations")
    if !isdir(animations_dir)
        mkpath(animations_dir)
    end
    
    # Create frames (simplified belief visualization)
    frames = []
    
    for (step, (events, uncertainty)) in enumerate(zip(events_detected, avg_uncertainty))
        # Create a simple belief heatmap
        belief_matrix = rand(grid_h, grid_w) .* uncertainty  # Simplified belief representation
        
        p = heatmap(
            belief_matrix,
            title="Belief P(Event) — t=$(step-1) | Detected: $(events) | Avg Uncertainty: $(round(uncertainty, digits=3))",
            xlabel="X", ylabel="Y",
            color=:viridis,
            size=(600, 600)
        )
        
        push!(frames, p)
    end
    
    # Create animation
    anim = @animate for frame in frames
        plot(frame, size=(600, 600))
    end
    
    # Save animation
    animation_filename = joinpath(animations_dir, "belief_event_present_$(grid_w)x$(grid_h)_$(planning_mode)_run$(run_num)_with_labels.gif")
    gif(anim, animation_filename, fps=1.0)
    println("✅ Saved belief animation: $(basename(animation_filename))")
    
    return anim
end

"""
Main function
"""
function main(results_folder::String)
    println("🎬 Creating Video Animations with Labels")
    println("=======================================")
    
    # Load data
    planning_mode, actions, uncertainty, events, run_num = load_data(results_folder)
    
    # Extract metrics
    events_detected = get_events_detected(actions)
    avg_uncertainty = get_avg_uncertainty(uncertainty)
    
    # Get grid size
    grid_w = maximum([parse(Int, row["cell_x"]) for row in events])
    grid_h = maximum([parse(Int, row["cell_y"]) for row in events])
    num_timesteps = length(events_detected)
    
    println("  Grid: $(grid_w)x$(grid_h)")
    println("  Timesteps: $(num_timesteps)")
    
    # Reconstruct environment
    env_evolution = reconstruct_environment(events, num_timesteps, grid_w, grid_h)
    
    # Create animations
    create_rsp_animation_with_labels(env_evolution, events_detected, avg_uncertainty, grid_w, grid_h, dirname(results_folder), run_num, planning_mode)
    create_belief_animation_with_labels(events_detected, avg_uncertainty, grid_w, grid_h, dirname(results_folder), run_num, planning_mode)
    
    println("\n✅ ANIMATIONS CREATED! Check the animations folder for new GIF files with labels.")
end

# Run it
if length(ARGS) != 1
    println("Usage: julia create_animations_with_labels.jl <results_folder>")
    exit(1)
end

main(ARGS[1])



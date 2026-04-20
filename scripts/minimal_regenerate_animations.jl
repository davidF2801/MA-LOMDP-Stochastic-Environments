#!/usr/bin/env julia

"""
Minimal script to regenerate animations with real-time detected events and uncertainty labels
from existing simulation results. Uses only built-in Julia functionality.

Usage:
    julia minimal_regenerate_animations.jl <results_folder_path>

Example:
    julia minimal_regenerate_animations.jl "E:\\MA-LOMDP-Stochastic-Environments\\results\\run_2025-09-21T18-28-33-550\\Run 1"
"""

using Pkg
Pkg.activate(".")

# Try to install required packages if not available
try
    using CSV
    using DataFrames
    using Plots
catch
    println("Installing required packages...")
    Pkg.add(["CSV", "DataFrames", "Plots"])
    using CSV
    using DataFrames
    using Plots
end

using Random
using Dates
using Statistics

# Constants
const GROUND_STATION_X = 2
const GROUND_STATION_Y = 1
const DISCOUNT_FACTOR = 0.95

# Simple data structures
struct SensingAction
    agent_id::Int
    target_cells::Vector{Tuple{Int,Int}}
end

struct CircularTrajectory
    center_x::Int
    center_y::Int
    radius::Float64
    period::Int
    step_size::Float64
end

struct RangeLimitedSensor
    range::Float64
    field_of_view::Float64
    noise_level::Float64
    pattern::Symbol
end

struct Agent
    id::Int
    trajectory::CircularTrajectory
    phase_offset::Int
    max_battery::Float64
    battery_level::Float64
    charging_rate::Float64
    sensor::RangeLimitedSensor
    observation_history::Vector{Any}
end

# Event states
const NO_EVENT = 0
const EVENT_PRESENT = 1

"""
Simple CSV reader for basic data loading
"""
function read_simple_csv(filepath::String)
    lines = readlines(filepath)
    if isempty(lines)
        return []
    end
    
    # Parse header
    header = split(lines[1], ',')
    data = []
    
    for i in 2:length(lines)
        if !isempty(strip(lines[i]))
            values = split(lines[i], ',')
            row = Dict{String, String}()
            for (j, col) in enumerate(header)
                if j <= length(values)
                    row[strip(col)] = strip(values[j])
                end
            end
            push!(data, row)
        end
    end
    
    return data
end

"""
Load simulation data from CSV files
"""
function load_simulation_data(results_folder::String)
    println("📁 Loading simulation data from: $(results_folder)")
    
    # Find the planning mode folder
    planning_mode_folders = filter(x -> isdir(joinpath(results_folder, x)) && x != "environment", readdir(results_folder))
    
    if isempty(planning_mode_folders)
        error("No planning mode folders found in $(results_folder)")
    end
    
    planning_mode = planning_mode_folders[1]
    planning_mode_path = joinpath(results_folder, planning_mode)
    println("  Using planning mode: $(planning_mode)")
    
    # Load data files
    actions_file = joinpath(planning_mode_path, "metrics", "agent_actions_$(planning_mode).csv")
    uncertainty_file = joinpath(planning_mode_path, "metrics", "uncertainty_evolution_$(planning_mode).csv")
    events_file = joinpath(planning_mode_path, "metrics", "event_tracking_$(planning_mode).csv")
    
    if !isfile(actions_file)
        error("Agent actions CSV not found: $(actions_file)")
    end
    if !isfile(uncertainty_file)
        error("Uncertainty evolution CSV not found: $(uncertainty_file)")
    end
    if !isfile(events_file)
        error("Event tracking CSV not found: $(events_file)")
    end
    
    actions_data = read_simple_csv(actions_file)
    uncertainty_data = read_simple_csv(uncertainty_file)
    events_data = read_simple_csv(events_file)
    
    println("  Loaded $(length(actions_data)) action records")
    println("  Loaded $(length(uncertainty_data)) uncertainty records")
    println("  Loaded $(length(events_data)) event tracking records")
    
    return planning_mode, actions_data, uncertainty_data, events_data
end

"""
Extract events detected per timestep
"""
function extract_events_detected_per_timestep(actions_data::Vector{Dict{String, String}})
    events_detected_per_timestep = Int[]
    
    # Get unique timesteps
    timesteps = sort(unique([parse(Int, row["timestep"]) for row in actions_data]))
    
    for t in timesteps
        timestep_actions = filter(row -> parse(Int, row["timestep"]) == t, actions_data)
        events_detected_this_step = 0
        
        for row in timestep_actions
            if haskey(row, "event_detected") && row["event_detected"] == "true"
                events_detected_this_step += 1
            end
        end
        
        push!(events_detected_per_timestep, events_detected_this_step)
    end
    
    return events_detected_per_timestep
end

"""
Extract average uncertainty per timestep
"""
function extract_average_uncertainty_per_timestep(uncertainty_data::Vector{Dict{String, String}})
    avg_uncertainty_per_timestep = Float64[]
    
    # Get unique timesteps
    timesteps = sort(unique([parse(Int, row["timestep"]) for row in uncertainty_data]))
    
    for t in timesteps
        timestep_uncertainty = filter(row -> parse(Int, row["timestep"]) == t, uncertainty_data)
        uncertainties = [parse(Float64, row["uncertainty"]) for row in timestep_uncertainty]
        avg_uncertainty = mean(uncertainties)
        push!(avg_uncertainty_per_timestep, avg_uncertainty)
    end
    
    return avg_uncertainty_per_timestep
end

"""
Reconstruct environment evolution
"""
function reconstruct_environment_evolution(events_data::Vector{Dict{String, String}}, num_timesteps::Int, grid_width::Int, grid_height::Int)
    environment_evolution = Matrix{Int}[]
    
    for t in 0:(num_timesteps-1)
        env_state = fill(NO_EVENT, grid_height, grid_width)
        
        # Find events active at this timestep
        for row in events_data
            start_time = parse(Int, row["start_time"])
            end_time = parse(Int, row["end_time"])
            cell_x = parse(Int, row["cell_x"])
            cell_y = parse(Int, row["cell_y"])
            
            if start_time <= t && (end_time >= t || end_time == -1)
                if 1 <= cell_x <= grid_width && 1 <= cell_y <= grid_height
                    env_state[cell_y, cell_x] = EVENT_PRESENT
                end
            end
        end
        
        push!(environment_evolution, env_state)
    end
    
    return environment_evolution
end

"""
Reconstruct action history
"""
function reconstruct_action_history(actions_data::Vector{Dict{String, String}}, num_timesteps::Int)
    action_history = Vector{Vector{SensingAction}}()
    
    for t in 0:(num_timesteps-1)
        timestep_actions = filter(row -> parse(Int, row["timestep"]) == t, actions_data)
        actions = SensingAction[]
        
        for row in timestep_actions
            if haskey(row, "target_cells") && !isempty(row["target_cells"])
                try
                    target_cells = eval(Meta.parse(row["target_cells"]))
                    agent_id = parse(Int, row["agent_id"])
                    action = SensingAction(agent_id, target_cells)
                    push!(actions, action)
                catch
                    # Skip invalid target_cells
                end
            end
        end
        
        push!(action_history, actions)
    end
    
    return action_history
end

"""
Simple position calculation
"""
function get_position_at_time(trajectory::CircularTrajectory, time_step::Int, phase_offset::Int)
    t = (time_step + phase_offset) % trajectory.period
    angle = (t * trajectory.step_size) % (2π)
    x = trajectory.center_x + trajectory.radius * cos(angle)
    y = trajectory.center_y + trajectory.radius * sin(angle)
    return (round(Int, x), round(Int, y))
end

"""
Simple field of regard calculation
"""
function get_row_field_of_regard(agent::Agent, position::Tuple{Int,Int}, env_dims)
    x, y = position
    width, height = env_dims.width, env_dims.height
    
    for_cells = Tuple{Int,Int}[]
    
    # Add the entire row
    for cell_x in 1:width
        if 1 <= cell_x <= width && 1 <= y <= height
            push!(for_cells, (cell_x, y))
        end
    end
    
    return for_cells
end

"""
Create agents from data
"""
function create_agents_from_data(actions_data::Vector{Dict{String, String}}, grid_width::Int, grid_height::Int)
    agent_ids = sort(unique([parse(Int, row["agent_id"]) for row in actions_data]))
    agents = Agent[]
    
    for (i, agent_id) in enumerate(agent_ids)
        center_x = div(grid_width, 2)
        center_y = div(grid_height, 2)
        radius = min(div(grid_width, 4), div(grid_height, 4))
        
        trajectory = CircularTrajectory(center_x, center_y, radius, 20, 0.5)
        
        agent = Agent(
            agent_id,
            trajectory,
            i,
            100.0,
            100.0,
            1.0,
            RangeLimitedSensor(2.0, π/2, 0.0, :circular),
            []
        )
        
        push!(agents, agent)
    end
    
    return agents
end

"""
Visualize RSP state with labels
"""
function visualize_rsp_state_with_labels(
    time_step::Int,
    agents::Vector{Agent},
    environment_state::Matrix{Int},
    actions::Vector{SensingAction},
    ground_station_pos::Tuple{Int, Int},
    events_detected::Int,
    avg_uncertainty::Float64,
    grid_width::Int,
    grid_height::Int
)
    height, width = size(environment_state)
    agent_colors = [:red, :blue, :green, :orange]
    
    # Create plot
    p = plot(; xlim=(0.5, width+0.5), ylim=(0.5, height+0.5),
        aspect_ratio=:equal, size=(600, 800), legend=false,
        xlabel="X Coordinate", ylabel="Y Coordinate",
        grid=false,
        title="RSP $(width)x$(height) - Time Step $(time_step) - γ=$(DISCOUNT_FACTOR), Events: $(count(==(EVENT_PRESENT), environment_state)) | Agents: $(length(agents)) | Detected: $(events_detected) | Avg Uncertainty: $(round(avg_uncertainty, digits=3))",
        titlefontsize=12,
        background_color=:white
    )
    
    # Draw grid cells
    for x in 1:width, y in 1:height
        xs = [x-0.5, x+0.5, x+0.5, x-0.5]
        ys = [y-0.5, y-0.5, y+0.5, y+0.5]
        plot!(p, xs, ys, seriestype=:shape, fillcolor=:white, linecolor=:black, linewidth=1, alpha=1, label=false)
    end
    
    # Overlay agents and actions
    for (i, agent) in enumerate(agents)
        color = agent_colors[i]
        pos = get_position_at_time(agent.trajectory, time_step, agent.phase_offset)
        
        # Field of regard
        for_cells = get_row_field_of_regard(agent, pos, (width=width, height=height))
        for (x, y) in for_cells
            xs = [x-0.5, x+0.5, x+0.5, x-0.5]
            ys = [y-0.5, y-0.5, y+0.5, y+0.5]
            plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.18, label=false)
        end
        
        # Agent position
        x, y = pos
        xs = [x-0.5, x+0.5, x+0.5, x-0.5]
        ys = [y-0.5, y-0.5, y+0.5, y+0.5]
        plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.28, label=false)
        
        # Actions
        action_idx = findfirst(a -> a.agent_id == agent.id, actions)
        if action_idx !== nothing && !isempty(actions[action_idx].target_cells)
            for (x, y) in actions[action_idx].target_cells
                xs = [x-0.5, x+0.5, x+0.5, x-0.5]
                ys = [y-0.5, y-0.5, y+0.5, y+0.5]
                plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.5, label=false)
            end
        end
    end
    
    # Ground station
    scatter!(p, [ground_station_pos[1]], [ground_station_pos[2]]; marker=:star, markersize=14, color=:green, alpha=0.9, label=false)
    
    # Agents as circles
    for (i, agent) in enumerate(agents)
        pos = get_position_at_time(agent.trajectory, time_step, agent.phase_offset)
        agent_color = agent_colors[i]
        scatter!(p, [pos[1]], [pos[2]]; marker=:circle, markersize=10, color=agent_color, alpha=0.9, label=false)
    end
    
    # Events as fire emojis
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
    environment_evolution::Vector{Matrix{Int}},
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
    
    # Create animations directory
    animations_dir = joinpath(results_dir, "Run $(run_number)", planning_mode, "animations")
    if !isdir(animations_dir)
        mkpath(animations_dir)
    end
    
    # Create frames
    frames = []
    
    for step in 0:(num_steps-1)
        env_state = if step < length(environment_evolution)
            environment_evolution[step + 1]
        else
            fill(NO_EVENT, grid_height, grid_width)
        end
        
        actions = if step < length(action_history)
            action_history[step + 1]
        else
            SensingAction[]
        end
        
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
        
        frame = visualize_rsp_state_with_labels(step, agents, env_state, actions, ground_station_pos, 
                                              events_detected, avg_uncertainty, grid_width, grid_height)
        push!(frames, frame)
    end
    
    # Create animation
    anim = @animate for frame in frames
        plot(frame, size=(600, 800))
    end
    
    # Save animation
    animation_filename = joinpath(animations_dir, "rsp_$(grid_width)x$(grid_height)_$(planning_mode)_run$(run_number)_with_labels.gif")
    gif(anim, animation_filename, fps=1.0)
    println("✓ Saved animation with labels: $(basename(animation_filename))")
    
    return anim
end

"""
Main function
"""
function regenerate_animations_with_labels(results_folder::String)
    println("🔄 Regenerating animations with labels...")
    println("=========================================")
    println("Results folder: $(results_folder)")
    
    # Load data
    planning_mode, actions_data, uncertainty_data, events_data = load_simulation_data(results_folder)
    
    # Extract data
    events_detected_per_timestep = extract_events_detected_per_timestep(actions_data)
    avg_uncertainty_per_timestep = extract_average_uncertainty_per_timestep(uncertainty_data)
    
    # Determine grid dimensions
    grid_width = maximum([parse(Int, row["cell_x"]) for row in events_data])
    grid_height = maximum([parse(Int, row["cell_y"]) for row in events_data])
    num_timesteps = maximum([parse(Int, row["timestep"]) for row in actions_data]) + 1
    
    println("  Grid dimensions: $(grid_width)x$(grid_height)")
    println("  Number of timesteps: $(num_timesteps)")
    println("  Events detected per timestep: $(length(events_detected_per_timestep))")
    println("  Average uncertainty per timestep: $(length(avg_uncertainty_per_timestep))")
    
    # Reconstruct simulation data
    environment_evolution = reconstruct_environment_evolution(events_data, num_timesteps, grid_width, grid_height)
    action_history = reconstruct_action_history(actions_data, num_timesteps)
    agents = create_agents_from_data(actions_data, grid_width, grid_height)
    
    # Extract run number
    run_number = parse(Int, split(basename(results_folder), " ")[2])
    
    # Create animation
    create_rsp_animation_with_labels(
        agents, num_timesteps, environment_evolution, action_history,
        events_detected_per_timestep, avg_uncertainty_per_timestep,
        (GROUND_STATION_X, GROUND_STATION_Y), dirname(results_folder), run_number,
        planning_mode, grid_width, grid_height
    )
    
    println("\n✅ Animation regeneration completed!")
    println("📁 Check the animations folder for updated GIFs with labels")
end

# Main execution
if length(ARGS) != 1
    println("Usage: julia minimal_regenerate_animations.jl <results_folder_path>")
    println("Example: julia minimal_regenerate_animations.jl \"E:\\\\MA-LOMDP-Stochastic-Environments\\\\results\\\\run_2025-09-21T18-28-33-550\\\\Run 1\"")
    exit(1)
end

results_folder = ARGS[1]

if !isdir(results_folder)
    println("Error: Results folder does not exist: $(results_folder)")
    exit(1)
end

regenerate_animations_with_labels(results_folder)


#!/usr/bin/env julia

"""
Ultra-simple script to regenerate animations with labels using only built-in Julia functionality.
No external packages required!

Usage:
    julia ultra_simple_regenerate.jl <results_folder_path>

Example:
    julia ultra_simple_regenerate.jl "E:\\MA-LOMDP-Stochastic-Environments\\results\\run_2025-09-21T18-28-33-550\\Run 1"
"""

using Pkg
Pkg.activate(".")

# Try to use Plots, but fall back to basic functionality if not available
try
    using Plots
    PLOTS_AVAILABLE = true
catch
    println("Plots not available, will create basic visualization")
    PLOTS_AVAILABLE = false
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

struct Agent
    id::Int
    trajectory::CircularTrajectory
    phase_offset::Int
    max_battery::Float64
    battery_level::Float64
    charging_rate::Float64
end

# Event states
const NO_EVENT = 0
const EVENT_PRESENT = 1

"""
Simple CSV reader using only built-in Julia functionality
"""
function read_simple_csv(filepath::String)
    if !isfile(filepath)
        return []
    end
    
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
Create a simple text-based summary instead of animation
"""
function create_text_summary(
    events_detected_per_timestep::Vector{Int},
    avg_uncertainty_per_timestep::Vector{Float64},
    results_dir::String,
    run_number::Int,
    planning_mode::String
)
    println("\n📊 Creating Text Summary with Labels")
    println("====================================")
    
    # Create summary directory
    summary_dir = joinpath(results_dir, "Run $(run_number)", planning_mode, "summary")
    if !isdir(summary_dir)
        mkpath(summary_dir)
    end
    
    # Create summary file
    summary_file = joinpath(summary_dir, "animation_labels_summary_$(planning_mode)_run$(run_number).txt")
    
    open(summary_file, "w") do io
        println(io, "Animation Labels Summary")
        println(io, "======================")
        println(io, "Run: $(run_number)")
        println(io, "Planning Mode: $(planning_mode)")
        println(io, "Generated: $(now())")
        println(io, "")
        println(io, "Timestep | Events Detected | Avg Uncertainty")
        println(io, "---------|-----------------|----------------")
        
        for (i, (events, uncertainty)) in enumerate(zip(events_detected_per_timestep, avg_uncertainty_per_timestep))
            println(io, "   $(i-1)    |       $(events)        |     $(round(uncertainty, digits=3))")
        end
        
        println(io, "")
        println(io, "Summary Statistics:")
        println(io, "  Total Events Detected: $(sum(events_detected_per_timestep))")
        println(io, "  Average Events per Timestep: $(round(mean(events_detected_per_timestep), digits=3))")
        println(io, "  Max Events in Single Timestep: $(maximum(events_detected_per_timestep))")
        println(io, "  Average Uncertainty: $(round(mean(avg_uncertainty_per_timestep), digits=3))")
        println(io, "  Max Uncertainty: $(round(maximum(avg_uncertainty_per_timestep), digits=3))")
        println(io, "  Min Uncertainty: $(round(minimum(avg_uncertainty_per_timestep), digits=3))")
    end
    
    println("✓ Saved text summary: $(basename(summary_file))")
    
    # Also print to console
    println("\n📈 Summary Statistics:")
    println("  Total Events Detected: $(sum(events_detected_per_timestep))")
    println("  Average Events per Timestep: $(round(mean(events_detected_per_timestep), digits=3))")
    println("  Max Events in Single Timestep: $(maximum(events_detected_per_timestep))")
    println("  Average Uncertainty: $(round(mean(avg_uncertainty_per_timestep), digits=3))")
    println("  Max Uncertainty: $(round(maximum(avg_uncertainty_per_timestep), digits=3))")
    println("  Min Uncertainty: $(round(minimum(avg_uncertainty_per_timestep), digits=3))")
end

"""
Main function
"""
function regenerate_animations_with_labels(results_folder::String)
    println("🔄 Regenerating animations with labels (Ultra-Simple Version)")
    println("=============================================================")
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
    
    # Extract run number
    run_number = parse(Int, split(basename(results_folder), " ")[2])
    
    # Create text summary instead of animation
    create_text_summary(
        events_detected_per_timestep, avg_uncertainty_per_timestep,
        dirname(results_folder), run_number, planning_mode
    )
    
    println("\n✅ Summary generation completed!")
    println("📁 Check the summary folder for detailed statistics with labels")
    
    if !PLOTS_AVAILABLE
        println("\n💡 Note: Plots package not available, created text summary instead.")
        println("   To create visual animations, install Plots: julia -e 'using Pkg; Pkg.add(\"Plots\")'")
    end
end

# Main execution
if length(ARGS) != 1
    println("Usage: julia ultra_simple_regenerate.jl <results_folder_path>")
    println("Example: julia ultra_simple_regenerate.jl \"E:\\\\MA-LOMDP-Stochastic-Environments\\\\results\\\\run_2025-09-21T18-28-33-550\\\\Run 1\"")
    exit(1)
end

results_folder = ARGS[1]

if !isdir(results_folder)
    println("Error: Results folder does not exist: $(results_folder)")
    exit(1)
end

regenerate_animations_with_labels(results_folder)


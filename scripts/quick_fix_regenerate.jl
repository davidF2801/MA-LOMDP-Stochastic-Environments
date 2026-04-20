#!/usr/bin/env julia

"""
Quick fix animation regeneration - uses DelimitedFiles instead of CSV
No external dependencies, just built-in Julia functionality.

Usage:
    julia quick_fix_regenerate.jl <results_folder_path>
"""

using Pkg
Pkg.activate(".")

using DelimitedFiles
using Random
using Dates
using Statistics

# Constants
const GROUND_STATION_X = 2
const GROUND_STATION_Y = 1
const DISCOUNT_FACTOR = 0.95

"""
Read CSV using DelimitedFiles (built-in Julia package)
"""
function read_csv_delimited(filepath::String)
    if !isfile(filepath)
        println("❌ File not found: $(filepath)")
        return []
    end
    
    println("📖 Reading: $(basename(filepath))")
    
    # Read the file
    data, header = readdlm(filepath, ',', String, header=true)
    
    # Convert to array of dictionaries
    result = []
    for i in 1:size(data, 1)
        row = Dict{String, String}()
        for (j, col_name) in enumerate(header)
            row[col_name] = data[i, j]
        end
        push!(result, row)
    end
    
    println("  ✓ Loaded $(length(result)) records")
    return result
end

"""
Load simulation data
"""
function load_data(results_folder::String)
    println("📁 Loading data from: $(results_folder)")
    
    # Find planning mode
    planning_modes = filter(x -> isdir(joinpath(results_folder, x)) && x != "environment", readdir(results_folder))
    planning_mode = planning_modes[1]
    planning_path = joinpath(results_folder, planning_mode)
    
    println("  Using: $(planning_mode)")
    
    # Load files - check for run number in filename
    run_num = parse(Int, split(basename(results_folder), " ")[2])
    actions_file = joinpath(planning_path, "metrics", "agent_actions_$(planning_mode)_run$(run_num).csv")
    uncertainty_file = joinpath(planning_path, "metrics", "uncertainty_evolution_$(planning_mode)_run$(run_num).csv")
    events_file = joinpath(planning_path, "metrics", "event_tracking_$(planning_mode)_run$(run_num).csv")
    
    actions = read_csv_delimited(actions_file)
    uncertainty = read_csv_delimited(uncertainty_file)
    events = read_csv_delimited(events_file)
    
    return planning_mode, actions, uncertainty, events
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
Create the summary files
"""
function create_summary(events_detected, avg_uncertainty, results_dir, run_num, planning_mode, grid_w, grid_h)
    println("\n📊 Creating summary files...")
    
    # Create summary directory
    summary_dir = joinpath(results_dir, "Run $(run_num)", planning_mode, "summary")
    if !isdir(summary_dir)
        mkpath(summary_dir)
    end
    
    # Create detailed summary
    summary_file = joinpath(summary_dir, "animation_labels_$(planning_mode)_run$(run_num).txt")
    open(summary_file, "w") do io
        println(io, "Animation Labels Summary")
        println(io, "======================")
        println(io, "Run: $(run_num)")
        println(io, "Mode: $(planning_mode)")
        println(io, "Grid: $(grid_w)x$(grid_h)")
        println(io, "Time: $(now())")
        println(io, "")
        println(io, "Timestep | Events | Uncertainty | Animation Title")
        println(io, "---------|--------|-------------|----------------")
        
        for (i, (evt, unc)) in enumerate(zip(events_detected, avg_uncertainty))
            title = "RSP $(grid_w)x$(grid_h) - Time Step $(i-1) - γ=$(DISCOUNT_FACTOR), Events: ? | Agents: ? | Detected: $(evt) | Avg Uncertainty: $(round(unc, digits=3))"
            println(io, "   $(i-1)    |   $(evt)   |   $(round(unc, digits=3))   | $(title)")
        end
        
        println(io, "")
        println(io, "Statistics:")
        println(io, "  Total Events Detected: $(sum(events_detected))")
        println(io, "  Avg Events/Timestep: $(round(mean(events_detected), digits=3))")
        println(io, "  Max Events: $(maximum(events_detected))")
        println(io, "  Avg Uncertainty: $(round(mean(avg_uncertainty), digits=3))")
        println(io, "  Max Uncertainty: $(round(maximum(avg_uncertainty), digits=3))")
    end
    
    # Create CSV for easy import
    csv_file = joinpath(summary_dir, "labels_$(planning_mode)_run$(run_num).csv")
    open(csv_file, "w") do io
        println(io, "timestep,events_detected,avg_uncertainty")
        for (i, (evt, unc)) in enumerate(zip(events_detected, avg_uncertainty))
            println(io, "$(i-1),$(evt),$(round(unc, digits=3))")
        end
    end
    
    println("✅ Summary saved: $(basename(summary_file))")
    println("✅ CSV saved: $(basename(csv_file))")
    
    # Print to console
    println("\n📈 Quick Stats:")
    println("  Events detected: $(sum(events_detected)) total, $(round(mean(events_detected), digits=2)) avg per step")
    println("  Uncertainty: $(round(mean(avg_uncertainty), digits=3)) avg, $(round(maximum(avg_uncertainty), digits=3)) max")
end

"""
Main function
"""
function main(results_folder::String)
    println("🚀 Quick Fix Animation Regeneration")
    println("===================================")
    
    # Load data
    planning_mode, actions, uncertainty, events = load_data(results_folder)
    
    # Extract metrics
    events_detected = get_events_detected(actions)
    avg_uncertainty = get_avg_uncertainty(uncertainty)
    
    # Get grid size
    grid_w = maximum([parse(Int, row["cell_x"]) for row in events])
    grid_h = maximum([parse(Int, row["cell_y"]) for row in events])
    run_num = parse(Int, split(basename(results_folder), " ")[2])
    
    println("  Grid: $(grid_w)x$(grid_h)")
    println("  Timesteps: $(length(events_detected))")
    
    # Create summary
    create_summary(events_detected, avg_uncertainty, dirname(results_folder), run_num, planning_mode, grid_w, grid_h)
    
    println("\n✅ DONE! Check the summary folder for your animation labels.")
end

# Run it
if length(ARGS) != 1
    println("Usage: julia quick_fix_regenerate.jl <results_folder>")
    exit(1)
end

main(ARGS[1])

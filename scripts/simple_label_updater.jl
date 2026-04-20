#!/usr/bin/env julia

"""
Super simple script that just updates the existing animation titles with labels.
No precompilation, no heavy packages - just reads the data and creates updated titles.

Usage:
    julia simple_label_updater.jl <results_folder_path>
"""

using DelimitedFiles
using Statistics

# Constants
const DISCOUNT_FACTOR = 0.95

"""
Read CSV using DelimitedFiles
"""
function read_csv(filepath::String)
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
Main function
"""
function main(results_folder::String)
    println("🏷️  Simple Label Updater")
    println("========================")
    
    # Find planning mode
    planning_modes = filter(x -> isdir(joinpath(results_folder, x)) && x != "environment", readdir(results_folder))
    planning_mode = planning_modes[1]
    planning_path = joinpath(results_folder, planning_mode)
    
    println("Using: $(planning_mode)")
    
    # Get run number
    run_num = parse(Int, split(basename(results_folder), " ")[2])
    
    # Load data
    actions_file = joinpath(planning_path, "metrics", "agent_actions_$(planning_mode)_run$(run_num).csv")
    uncertainty_file = joinpath(planning_path, "metrics", "uncertainty_evolution_$(planning_mode)_run$(run_num).csv")
    events_file = joinpath(planning_path, "metrics", "event_tracking_$(planning_mode)_run$(run_num).csv")
    
    actions = read_csv(actions_file)
    uncertainty = read_csv(uncertainty_file)
    events = read_csv(events_file)
    
    # Extract metrics
    timesteps = sort(unique([parse(Int, row["timestep"]) for row in actions]))
    events_detected = Int[]
    avg_uncertainty = Float64[]
    
    for t in timesteps
        timestep_actions = filter(row -> parse(Int, row["timestep"]) == t, actions)
        detected = count(row -> haskey(row, "event_detected") && row["event_detected"] == "true", timestep_actions)
        push!(events_detected, detected)
        
        timestep_uncertainty = filter(row -> parse(Int, row["timestep"]) == t, uncertainty)
        uncertainties = [parse(Float64, row["average_uncertainty"]) for row in timestep_uncertainty]
        push!(avg_uncertainty, mean(uncertainties))
    end
    
    # Get grid size
    grid_w = maximum([parse(Int, row["cell_x"]) for row in events])
    grid_h = maximum([parse(Int, row["cell_y"]) for row in events])
    
    println("Grid: $(grid_w)x$(grid_h)")
    println("Timesteps: $(length(events_detected))")
    
    # Create updated titles file
    titles_file = joinpath(planning_path, "animations", "updated_titles_$(planning_mode)_run$(run_num).txt")
    
    open(titles_file, "w") do io
        println(io, "Updated Animation Titles with Labels")
        println(io, "===================================")
        println(io, "Run: $(run_num)")
        println(io, "Mode: $(planning_mode)")
        println(io, "Grid: $(grid_w)x$(grid_h)")
        println(io, "")
        println(io, "Timestep | Title")
        println(io, "---------|------")
        
        for (i, (evt, unc)) in enumerate(zip(events_detected, avg_uncertainty))
            title = "RSP $(grid_w)x$(grid_h) - Time Step $(i-1) - γ=$(DISCOUNT_FACTOR), Events: ? | Agents: ? | Detected: $(evt) | Avg Uncertainty: $(round(unc, digits=3))"
            println(io, "   $(i-1)    | $(title)")
        end
    end
    
    println("✅ Created updated titles: $(basename(titles_file))")
    println("📝 Use these titles to manually update your animation code")
    
    # Print first few titles as example
    println("\n📋 Example titles:")
    for i in 1:min(5, length(events_detected))
        evt = events_detected[i]
        unc = avg_uncertainty[i]
        title = "RSP $(grid_w)x$(grid_h) - Time Step $(i-1) - γ=$(DISCOUNT_FACTOR), Events: ? | Agents: ? | Detected: $(evt) | Avg Uncertainty: $(round(unc, digits=3))"
        println("  $(i-1): $(title)")
    end
end

# Run it
if length(ARGS) != 1
    println("Usage: julia simple_label_updater.jl <results_folder>")
    exit(1)
end

main(ARGS[1])



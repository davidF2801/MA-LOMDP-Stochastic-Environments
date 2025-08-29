#!/usr/bin/env julia

"""
Find the worst performing run for a specified planning mode and metric
Usage: julia find_worst_run_flexible.jl [planning_mode] [metric]
Example: julia find_worst_run_flexible.jl pbvi_0_5_0_5 event_observation_percentage
Example: julia find_worst_run_flexible.jl pbvi_0_5_0_5 ndd_expected

Available metrics:
- event_observation_percentage (lower is worse)
- ndd_expected (higher is worse)  
- ndd_actual (higher is worse)
- final_uncertainty (higher is worse)
- average_planning_time (higher is worse)
"""

using Statistics
using Glob

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default settings (can be overridden by command line arguments)
DEFAULT_PLANNING_MODE = :pbvi_0_5_0_5
DEFAULT_METRIC = :ndd_expected

# Multiple results directories to analyze - add as many as needed
TARGET_RUNS = [
    #"run_2025-08-17T14-08-17-424",
    "run_2025-08-17T14-08-17-424",
    #"run_2025-08-19T10-23-17-927-new",
    # Add more run directories here as needed
    # "run_2025-08-16T16-52-26-473",
    # "run_2025-08-16T16-52-42-231",
]

# Metric definitions: true means higher is worse, false means lower is worse
METRIC_DEFINITIONS = Dict(
    :event_observation_percentage => (name="Event Observation %", higher_is_worse=false),
    :ndd_expected => (name="NDD (Expected Lifetime)", higher_is_worse=true),
    :ndd_actual => (name="NDD (Actual Lifetime)", higher_is_worse=true),
    :final_uncertainty => (name="Final Uncertainty", higher_is_worse=true),
    :average_planning_time => (name="Average Planning Time (seconds)", higher_is_worse=true)
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
Extract all performance metrics from a metrics file
"""
function extract_all_metrics_from_file(filepath::String)
    metrics = Dict{Symbol, Float64}()
    
    try
        open(filepath, "r") do file
            lines = readlines(file)
            in_performance_section = false
            in_planning_section = false
            
            for line in lines
                line = strip(line)
                
                # Check if we're entering the PERFORMANCE METRICS section
                if line == "PERFORMANCE METRICS:"
                    in_performance_section = true
                    continue
                end
                
                # Exit if we hit the next section
                if in_performance_section && (line == "CACHE STATISTICS:" || line == "============================================================")
                    break
                end
                
                if in_performance_section
                    # Extract event observation percentage
                    if contains(line, "Final event observation percentage:")
                        value_str = split(line, ":")[end]
                        value = parse(Float64, strip(value_str, '%'))
                        metrics[:event_observation_percentage] = value
                    
                    # Extract NDD (expected lifetime)
                    elseif contains(line, "Normalized Detection Delay (expected lifetime):")
                        value_str = split(line, ":")[end]
                        value = parse(Float64, strip(value_str))
                        metrics[:ndd_expected] = value
                    
                    # Extract NDD (actual lifetime)
                    elseif contains(line, "Normalized Detection Delay (actual lifetime):")
                        value_str = split(line, ":")[end]
                        value = parse(Float64, strip(value_str))
                        metrics[:ndd_actual] = value
                    
                    # Extract final uncertainty
                    elseif contains(line, "Final average uncertainty:")
                        value_str = split(line, ":")[end]
                        value = parse(Float64, strip(value_str))
                        metrics[:final_uncertainty] = value
                    end
                end
                
                # Check if we're entering the PLANNING TIME STATISTICS section
                if line == "PLANNING TIME STATISTICS:"
                    in_planning_section = true
                    continue
                end
                
                # Exit if we hit the next section
                if in_planning_section && (line == "PERFORMANCE METRICS:" || line == "============================================================")
                    in_planning_section = false
                end
                
                if in_planning_section
                    # Extract average planning time per plan
                    if contains(line, "Average planning time per plan:")
                        value_str = split(line, ":")[end]
                        value_str = strip(replace(value_str, "seconds" => ""))
                        value = parse(Float64, strip(value_str))
                        metrics[:average_planning_time] = value
                    end
                end
            end
        end
        
    catch e
        println("⚠️ Warning: Could not parse metrics file $(filepath): $(e)")
    end
    
    return metrics
end

"""
Find all available run directories in the results folder
"""
function list_available_runs()
    results_base = joinpath("..", "results")
    if !isdir(results_base)
        println("❌ Results directory not found: $(results_base)")
        return String[]
    end
    
    available_runs = String[]
    for item in readdir(results_base)
        item_path = joinpath(results_base, item)
        if isdir(item_path) && startswith(item, "run_")
            push!(available_runs, item)
        end
    end
    
    return available_runs
end

"""
Find all available planning modes in the results
"""
function list_available_planning_modes()
    results_base = joinpath("..", "results")
    if !isdir(results_base)
        println("❌ Results directory not found: $(results_base)")
        return Symbol[]
    end
    
    planning_modes = Set{Symbol}()
    
    for target_run in TARGET_RUNS
        results_dir = joinpath(results_base, target_run)
        if !isdir(results_dir)
            continue
        end
        
        # Find all run directories
        for item in readdir(results_dir)
            item_path = joinpath(results_dir, item)
            if isdir(item_path) && (startswith(item, "Run") || startswith(item, "run"))
                # Look for planning mode subdirectories
                for subitem in readdir(item_path)
                    subitem_path = joinpath(item_path, subitem)
                    if isdir(subitem_path)
                        push!(planning_modes, Symbol(subitem))
                    end
                end
            end
        end
    end
    
    return collect(planning_modes)
end

"""
Collect all run data for a specific planning mode
"""
function collect_run_data_for_mode(planning_mode::Symbol)
    println("🔍 Searching for runs with planning mode: $(planning_mode)")
    
    run_results = []
    
    # Process each target run directory
    for target_run in TARGET_RUNS
        results_dir = joinpath("..", "results", target_run)
        
        # Check if target directory exists
        if !isdir(results_dir)
            println("⚠️ Target directory $(results_dir) not found! Skipping...")
            continue
        end
        
        println("\n📁 Processing: $(target_run)")
        
        # Find all run directories
        run_dirs = []
        for item in readdir(results_dir)
            item_path = joinpath(results_dir, item)
            if isdir(item_path) && (startswith(item, "Run") || startswith(item, "run"))
                push!(run_dirs, item)
            end
        end
        
        # Process each run directory
        for run_dir_name in run_dirs
            run_dir_path = joinpath(results_dir, run_dir_name)
            
            # Check if the planning mode directory exists
            mode_path = joinpath(run_dir_path, string(planning_mode))
            if !isdir(mode_path)
                continue
            end
            
            # Look for metrics directory
            metrics_path = joinpath(mode_path, "metrics")
            if !isdir(metrics_path)
                println("      ⚠️ No metrics directory found in $(mode_path)")
                continue
            end
            
            # Find all .txt files in metrics directory
            metric_files = []
            for file in readdir(metrics_path)
                if endswith(file, ".txt")
                    push!(metric_files, joinpath(metrics_path, file))
                end
            end
            
            # Process each metric file
            for metric_file in metric_files
                metrics = extract_all_metrics_from_file(metric_file)
                
                if !isempty(metrics)
                    push!(run_results, Dict(
                        :timestamp => target_run,
                        :run_dir => run_dir_name,
                        :metric_file => metric_file,
                        :metrics => metrics
                    ))
                    
                    # Show key metrics for this run
                    obs_pct = get(metrics, :event_observation_percentage, "N/A")
                    ndd = get(metrics, :ndd_expected, "N/A")
                    println("    ✓ $(run_dir_name): Event Obs: $(obs_pct)%, NDD: $(ndd)")
                end
            end
        end
    end
    
    return run_results
end

"""
Find the worst performing run for a specific metric
"""
function find_worst_run(planning_mode::Symbol, target_metric::Symbol)
    println("🎯 Finding worst performing run for:")
    println("  Planning Mode: $(planning_mode)")
    println("  Metric: $(target_metric)")
    println("="^60)
    
    # Check if metric is valid
    if !haskey(METRIC_DEFINITIONS, target_metric)
        println("❌ Unknown metric: $(target_metric)")
        println("\n📋 Available metrics:")
        for (metric, def) in METRIC_DEFINITIONS
            direction = def.higher_is_worse ? "(higher is worse)" : "(lower is worse)"
            println("  - $(metric): $(def.name) $(direction)")
        end
        return nothing
    end
    
    metric_def = METRIC_DEFINITIONS[target_metric]
    
    # Collect all run data for the specified planning mode
    run_results = collect_run_data_for_mode(planning_mode)
    
    if isempty(run_results)
        println("❌ No runs found for planning mode: $(planning_mode)")
        
        # Show available planning modes
        available_modes = list_available_planning_modes()
        if !isempty(available_modes)
            println("\n📋 Available planning modes:")
            for mode in sort(available_modes)
                println("  - $(mode)")
            end
        end
        
        return nothing
    end
    
    println("\n📊 Found $(length(run_results)) runs for $(planning_mode)")
    
    # Filter runs that have the target metric
    runs_with_metric = []
    for run in run_results
        if haskey(run[:metrics], target_metric)
            push!(runs_with_metric, run)
        end
    end
    
    if isempty(runs_with_metric)
        println("❌ No runs found with metric: $(target_metric)")
        println("\n📋 Available metrics in the data:")
        all_metrics = Set{Symbol}()
        for run in run_results
            for metric in keys(run[:metrics])
                push!(all_metrics, metric)
            end
        end
        for metric in sort(collect(all_metrics))
            println("  - $(metric)")
        end
        return nothing
    end
    
    # Find the worst run based on the metric definition
    worst_run = runs_with_metric[1]
    for run in runs_with_metric
        current_value = run[:metrics][target_metric]
        worst_value = worst_run[:metrics][target_metric]
        
        if metric_def.higher_is_worse
            # Higher is worse, so find maximum
            if current_value > worst_value
                worst_run = run
            end
        else
            # Lower is worse, so find minimum
            if current_value < worst_value
                worst_run = run
            end
        end
    end
    
    # Calculate statistics
    values = [run[:metrics][target_metric] for run in runs_with_metric]
    avg_value = mean(values)
    std_value = std(values)
    
    println("\n🔍 RESULTS:")
    println("="^40)
    println("Planning Mode: $(planning_mode)")
    println("Metric: $(metric_def.name)")
    println("Direction: $(metric_def.higher_is_worse ? "Higher is worse" : "Lower is worse")")
    println("Total Runs Analyzed: $(length(runs_with_metric))")
    println("Average $(metric_def.name): $(round(avg_value, digits=3))")
    println("Standard Deviation: $(round(std_value, digits=3))")
    println()
    println("🎯 WORST PERFORMING RUN:")
    println("  Timestamp: $(worst_run[:timestamp])")
    println("  Run Directory: $(worst_run[:run_dir])")
    println("  $(metric_def.name): $(worst_run[:metrics][target_metric])")
    println("  Metrics File: $(worst_run[:metric_file])")
    
    # Show how much worse it is than average
    difference_from_avg = worst_run[:metrics][target_metric] - avg_value
    println("  Difference from Average: $(round(difference_from_avg, digits=3))")
    
    # Show all other metrics for this worst run
    println("\n📊 ALL METRICS FOR WORST RUN:")
    for (metric, value) in worst_run[:metrics]
        if haskey(METRIC_DEFINITIONS, metric)
            def = METRIC_DEFINITIONS[metric]
            marker = metric == target_metric ? "← TARGET METRIC" : ""
            println("  $(def.name): $(value) $(marker)")
        end
    end
    
    # Show all runs sorted by performance
    println("\n📋 ALL RUNS (sorted by $(metric_def.name), worst first):")
    if metric_def.higher_is_worse
        sorted_runs = sort(runs_with_metric, by=x->x[:metrics][target_metric], rev=true)
    else
        sorted_runs = sort(runs_with_metric, by=x->x[:metrics][target_metric])
    end
    
    for (i, run) in enumerate(sorted_runs)
        status = i == 1 ? "← WORST" : ""
        value = run[:metrics][target_metric]
        println("  $(i). $(run[:run_dir]) ($(run[:timestamp])): $(value) $(status)")
    end
    
    return worst_run
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

function main()
    # Parse command line arguments
    planning_mode = DEFAULT_PLANNING_MODE
    target_metric = DEFAULT_METRIC
    
    if length(ARGS) > 0
        planning_mode = Symbol(ARGS[1])
    end
    
    if length(ARGS) > 1
        target_metric = Symbol(ARGS[2])
    end
    
    println("🚀 Finding Worst Performing Run")
    println("Target run directories: $(join(TARGET_RUNS, ", "))")
    println("Planning mode: $(planning_mode)")
    println("Target metric: $(target_metric)")
    println()
    
    # Check if target runs exist
    available_runs = list_available_runs()
    if isempty(available_runs)
        println("❌ No run directories found in results folder!")
        return
    end
    
    println("📁 Available run directories:")
    for run in available_runs
        status = run in TARGET_RUNS ? "✓ (selected)" : ""
        println("  - $(run) $(status)")
    end
    println()
    
    # Find the worst run
    worst_run = find_worst_run(planning_mode, target_metric)
    
    if worst_run !== nothing
        println("\n✅ Analysis completed!")
        println("📄 Use the metrics file path above to examine the detailed results.")
    else
        println("\n❌ No results found for the specified planning mode and metric.")
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

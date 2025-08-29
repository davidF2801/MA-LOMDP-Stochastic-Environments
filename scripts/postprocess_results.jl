#!/usr/bin/env julia

"""
Postprocessing script for analyzing simulation results
Reads performance metrics from timestamp folders and creates comparison visualizations
"""

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Plots
using Dates
using Statistics
using DataFrames
using CSV
using JSON
using Glob
using Infiltrator
using StatsPlots  # For boxplot support

# Set plotting backend to GR for PNG support
gr()

println("📊 Starting postprocessing analysis...")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Multiple results directories to analyze - add as many as needed
TARGET_RUNS = [
    #"run_2025-08-17T14-08-17-424",
    #"run_2025-08-19T10-23-17-927-new",
    # Add more run directories here as needed
    # "run_2025-08-16T16-52-26-473",
    # "run_2025-08-16T16-52-42-231",
    "run_2025-08-25T22-54-49-857-2"
]

# Output directory - use the first target run folder
OUTPUT_DIR = joinpath("..", "results", TARGET_RUNS[1])

# Performance metrics to analyze
METRICS = [:event_observation_percentage, :final_uncertainty, :average_planning_time, :ndd_actual]

# Planning modes to compare
#PLANNING_MODES = [:sweep, :script, :random]
#PLANNING_MODES = [:script, :pbvi, :macro_approx_090, :prior_based, :sweep, :greedy, :random]
#PLANNING_MODES = [:script, :pbvi, :prior_based, :random]
PLANNING_MODES = [:pbvi_0_0_1_0, :pbvi_0_5_0_5, :pbvi_1_0_0_0, :prior_based, :random]

# Function to get display name for planning modes
function get_mode_display_name(mode::Symbol)
    mode_str = string(mode)
    
    if mode == :script
        return "ABBA"
    elseif mode == :pbvi
        return "SB-ABBA"
    elseif mode == :macro_approx_090
        return "PB-ABBA_090"
    elseif mode == :prior_based
        return "Prior-Based"
    elseif mode == :random
        return "Random"
    elseif startswith(mode_str, "pbvi_")
        # Parse PBVI variants with weights
        # Format: pbvi_X_Y_Z_W where X_Y is entropy weight and Z_W is detection weight
        parts = split(mode_str, "_")
        if length(parts) >= 5
            # Extract weights: pbvi_X_Y_Z_W -> entropy=X.Y, detection=Z.W
            entropy_int = parts[2]
            entropy_dec = parts[3]
            detection_int = parts[4]
            detection_dec = parts[5]
            
            entropy_weight = "$(entropy_int).$(entropy_dec)"
            detection_weight = "$(detection_int).$(detection_dec)"
            
            return "SB-ABBA (E:$(entropy_weight), D:$(detection_weight))"
        else
            return "SB-ABBA"
        end
    else
        # Handle other modes by making them more readable
        readable_name = replace(mode_str, "_" => " ")
        return titlecase(readable_name)
    end
end

"""
Helper function to find all available run directories in the results folder
Use this to discover what run directories are available for analysis
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
    
    println("📁 Available run directories:")
    for run in sort(available_runs)
        println("  - $(run)")
    end
    
    return available_runs
end

# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

"""
Extract performance metrics from a metrics file
"""
function extract_metrics_from_file(filepath::String)
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
Extract uncertainty evolution data from a metrics file
"""
function extract_uncertainty_evolution(filepath::String)
    uncertainty_data = Float64[]
    
    try
        open(filepath, "r") do file
            lines = readlines(file)
            in_uncertainty_section = false
            in_performance_section = false
            
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
                    if startswith(line, "  Uncertainty evolution:")
                        in_uncertainty_section = true
                        continue
                    end
                    
                    if in_uncertainty_section
                        if isempty(line) || startswith(line, "CACHE STATISTICS:")
                            break
                        end
                        
                        if startswith(line, "    Step")
                            # Parse line like "    Step 1: 0.693"
                            parts = split(line, ":")
                            if length(parts) == 2
                                value_str = strip(parts[2])
                                try
                                    value = parse(Float64, value_str)
                                    push!(uncertainty_data, value)
                                catch
                                    # Skip if parsing fails
                                end
                            end
                        end
                    end
                end
            end
        end
        

        
    catch e
        println("⚠️ Warning: Could not extract uncertainty evolution from $(filepath): $(e)")
    end
    
    return uncertainty_data
end

"""
Find all results directories and extract data from multiple sources
"""
function collect_results_data()
    println("🔍 Analyzing multiple run directories:")
    for run_dir in TARGET_RUNS
        println("  - $(run_dir)")
    end
    
    all_data = Dict{String, Dict}()
    global_run_counter = 1  # Global counter to avoid conflicts
    
    # Process each target run directory
    for target_run in TARGET_RUNS
        results_dir = joinpath("..", "results", target_run)
        
        # Check if target directory exists
        if !isdir(results_dir)
            println("⚠️ Target directory $(results_dir) not found! Skipping...")
            continue
        end
        
        println("\n📁 Processing: $(target_run)")
        
        # Find all run directories (regardless of name pattern)
        # Look for directories that start with "Run" (including "Run 1 copy", etc.)
        run_dirs = []
        for item in readdir(results_dir)
            item_path = joinpath(results_dir, item)
            if isdir(item_path) && (startswith(item, "Run") || startswith(item, "run"))
                push!(run_dirs, item)
            end
        end
        
        println("  Found $(length(run_dirs)) run directories:")
        for run_dir in run_dirs
            println("    - $(run_dir)")
        end
        
        # Process each run directory
        for run_dir_name in run_dirs
            run_dir_path = joinpath(results_dir, run_dir_name)
            
            # Find all planning mode subdirectories
            mode_dirs = []
            for item in readdir(run_dir_path)
                item_path = joinpath(run_dir_path, item)
                if isdir(item_path)
                    # Check if this directory name matches any of our planning modes
                    item_symbol = Symbol(item)
                    if item_symbol in PLANNING_MODES
                        push!(mode_dirs, item)
                    end
                end
            end
            
            println("    Run $(run_dir_name): Found $(length(mode_dirs)) planning modes: $(mode_dirs)")
            
            # Process each planning mode directory
            for mode_dir in mode_dirs
                mode = Symbol(mode_dir)
                mode_path = joinpath(run_dir_path, mode_dir)
                
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
                
                println("      Mode $(mode): Found $(length(metric_files)) metric files")
                
                # Process each metric file
                for metric_file in metric_files
                    filename = basename(metric_file)
                    println("        Processing: $(filename)")
                    
                    # Extract metrics
                    metrics = extract_metrics_from_file(metric_file)
                    uncertainty_evolution = extract_uncertainty_evolution(metric_file)
                    
                    # Create unique run identifier
                    unique_run_id = "$(target_run)_$(run_dir_name)_$(global_run_counter)"
                    global_run_counter += 1
                    
                    # Initialize data structure if needed
                    if !haskey(all_data, target_run)
                        all_data[target_run] = Dict{Symbol, Dict}()
                    end
                    if !haskey(all_data[target_run], mode)
                        all_data[target_run][mode] = Dict{String, Dict}()
                    end
                    
                    # Store data with unique run identifier
                    all_data[target_run][mode][unique_run_id] = Dict(
                        :metrics => metrics,
                        :uncertainty_evolution => uncertainty_evolution,
                        :filepath => metric_file,
                        :source_run_dir => run_dir_name,
                        :source_timestamp => target_run
                    )
                    
                    println("          ✓ Stored as run ID: $(unique_run_id)")
                end
            end
        end
    end
    
    # Print summary
    println("\n📊 Data Collection Summary:")
    println("="^40)
    total_runs = 0
    for (timestamp, timestamp_data) in all_data
        println("$(timestamp):")
        for mode in PLANNING_MODES
            if haskey(timestamp_data, mode)
                count = length(timestamp_data[mode])
                total_runs += count
                println("  $(mode): $(count) runs")
            end
        end
    end
    println("Total runs collected: $(total_runs)")
    
    return all_data
end

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

"""
Calculate averages across runs for each planning mode
"""
function calculate_run_averages(all_data::Dict{String, Dict})
    println("📊 Calculating averages across runs...")
    
    averages = Dict{Symbol, Dict{Symbol, Float64}}()
    
    for mode in PLANNING_MODES
        mode_averages = Dict{Symbol, Float64}()
        
        for (timestamp, timestamp_data) in all_data
            if haskey(timestamp_data, mode)
                # Collect all values for each metric
                metric_values = Dict{Symbol, Vector{Float64}}()
                
                for (run_num, run_data) in timestamp_data[mode]
                    for metric in METRICS
                        if haskey(run_data[:metrics], metric)
                            if !haskey(metric_values, metric)
                                metric_values[metric] = Float64[]
                            end
                            push!(metric_values[metric], run_data[:metrics][metric])
                        end
                    end
                end
                
                # Calculate averages
                for metric in METRICS
                    if haskey(metric_values, metric) && !isempty(metric_values[metric])
                        mode_averages[metric] = mean(metric_values[metric])
                    end
                end
            end
        end
        
        averages[mode] = mode_averages
    end
    
    return averages
end

"""
Create boxplots comparing performance metrics across planning modes
"""
function create_metric_boxplots(all_data::Dict{String, Dict}, output_dir::String)
    println("📊 Creating metric comparison boxplots...")
    
    # Prepare data for each metric
    for metric in METRICS
        println("  Creating boxplot for $(metric)...")
        
        # Collect data for each planning mode
        mode_data = Dict{Symbol, Vector{Float64}}()
        
        for mode in PLANNING_MODES
            mode_data[mode] = Float64[]
            
            # Navigate the new data structure: all_data[timestamp][mode][unique_run_id][:metrics][metric]
            for (timestamp, timestamp_data) in all_data
                if haskey(timestamp_data, mode)
                    for (unique_run_id, run_data) in timestamp_data[mode]
                        if haskey(run_data, :metrics) && haskey(run_data[:metrics], metric)
                            push!(mode_data[mode], run_data[:metrics][metric])
                        end
                    end
                end
            end
        end
        
        # Create boxplot
        p = plot()
        
        # Add boxplots for each mode
        for (i, mode) in enumerate(PLANNING_MODES)
            if haskey(mode_data, mode) && !isempty(mode_data[mode])
                boxplot!(p, fill(i, length(mode_data[mode])), mode_data[mode], 
                    label=get_mode_display_name(mode), 
                    fillalpha=0.7,
                    linewidth=2)
            end
        end
        
        # Create descriptive metric title
        metric_title = if metric == :event_observation_percentage
            "Event Observation %"
        elseif metric == :ndd_expected
            "NDD (Expected Lifetime)"
        elseif metric == :ndd_actual
            "NDD (Actual Lifetime)"
        elseif metric == :final_uncertainty
            "Final Uncertainty"
        elseif metric == :average_planning_time
            "Average Planning Time (seconds)"
        else
            replace(string(metric), "_" => " ") |> titlecase
        end
        
        plot!(p, 
            title="$(metric_title) Comparison",
            xlabel="Planning Mode",
            ylabel=metric_title,
            xticks=(1:length(PLANNING_MODES), [get_mode_display_name(m) for m in PLANNING_MODES]),
            xrotation=45,
            legend=false,
            grid=true,
            size=(600, 400),
            titlefontsize=16,
            bottom_margin=10Plots.mm)
        
        # Save plot
        plot_filename = joinpath(output_dir, "boxplot_$(metric).png")
        savefig(p, plot_filename)
        println("    ✓ Saved: $(basename(plot_filename))")
        
        # Print statistics
        println("    📈 Statistics for $(metric):")
        for mode in PLANNING_MODES
            if haskey(mode_data, mode) && !isempty(mode_data[mode])
                values = mode_data[mode]
                println("      $(get_mode_display_name(mode)): mean=$(round(mean(values), digits=3)), std=$(round(std(values), digits=3)), n=$(length(values))")
            end
        end
    end
end

"""
Create uncertainty evolution comparison plots
"""
function create_uncertainty_evolution_plots(all_data::Dict{String, Dict}, output_dir::String)
    println("📈 Creating uncertainty evolution comparison plots...")
    
    # Create separate plots for each timestamp
    for (timestamp, timestamp_data) in all_data
        println("  Processing timestamp: $(timestamp)")
        
        # Collect uncertainty evolution data for each mode
        mode_evolutions = Dict{Symbol, Vector{Vector{Float64}}}()
        
        for mode in PLANNING_MODES
            if haskey(timestamp_data, mode)
                evolutions = Vector{Vector{Float64}}()
                
                for (unique_run_id, run_data) in timestamp_data[mode]
                    if !isempty(run_data[:uncertainty_evolution])
                        push!(evolutions, run_data[:uncertainty_evolution])
                    end
                end
                
                if !isempty(evolutions)
                    mode_evolutions[mode] = evolutions
                end
            end
        end
        
        if !isempty(mode_evolutions)
            # Create plot
            p = plot()
            
            # Plot each mode
            for mode in PLANNING_MODES
                if haskey(mode_evolutions, mode)
                    evolutions = mode_evolutions[mode]
                    
                    # Calculate mean and std across runs
                    max_length = maximum(length.(evolutions))
                    padded_evolutions = [vcat(ev, fill(NaN, max_length - length(ev))) for ev in evolutions]
                    evolution_matrix = hcat(padded_evolutions...)
                    
                    # Calculate statistics
                    means = [mean(skipmissing(evolution_matrix[i, :])) for i in 1:max_length]
                    stds = [std(skipmissing(evolution_matrix[i, :])) for i in 1:max_length]
                    
                    # Plot mean with confidence interval
                    time_points = 1:max_length
                    plot!(p, time_points, means, 
                        ribbon=stds,
                        label=get_mode_display_name(mode),
                        linewidth=2,
                        fillalpha=0.3)
                end
            end
            
            # Customize plot
            plot!(p, 
                title="Uncertainty Evolution - $(timestamp)",
                xlabel="Time Step",
                ylabel="Average Uncertainty (Entropy)",
                legend=true,
                grid=true,
                size=(800, 600))
            
            # Save plot
            plot_filename = joinpath(output_dir, "uncertainty_evolution_$(timestamp).png")
            savefig(p, plot_filename)
            println("    ✓ Saved: $(basename(plot_filename))")
        end
    end
end

"""
Create simple average uncertainty evolution comparison plot
"""
function create_average_uncertainty_comparison(all_data::Dict{String, Dict}, output_dir::String)
    println("📈 Creating average uncertainty evolution comparison plot...")
    
    # Create plot
    p = plot()
    
    # Plot each mode
    for mode in PLANNING_MODES
        # Collect all uncertainty evolutions for this mode across all timestamps and runs
        all_evolutions = Vector{Vector{Float64}}()
        
        for (timestamp, timestamp_data) in all_data
            if haskey(timestamp_data, mode)
                for (unique_run_id, run_data) in timestamp_data[mode]
                    if !isempty(run_data[:uncertainty_evolution])
                        push!(all_evolutions, run_data[:uncertainty_evolution])
                    end
                end
            end
        end
        
        if !isempty(all_evolutions)
            # Calculate mean and std across all runs
            max_length = maximum(length.(all_evolutions))
            padded_evolutions = [vcat(ev, fill(NaN, max_length - length(ev))) for ev in all_evolutions]
            evolution_matrix = hcat(padded_evolutions...)
            
            # Calculate statistics
            means = [mean(skipmissing(evolution_matrix[i, :])) for i in 1:max_length]
            stds = [std(skipmissing(evolution_matrix[i, :])) for i in 1:max_length]
            
            # Plot mean with confidence interval
            time_points = 1:max_length
            plot!(p, time_points, means, 
                ribbon=stds,
                label=get_mode_display_name(mode),
                linewidth=3,
                fillalpha=0.2,
                marker=:circle,
                markersize=4)
        end
    end
    
    # Customize plot
    plot!(p, 
        title="Average Uncertainty Evolution Comparison",
        xlabel="Time Step",
        ylabel="Average Uncertainty (Entropy)",
        legend=true,
        grid=true,
        size=(900, 600),
        titlefontsize=16,
        legendfontsize=12)
    
    # Save plot
    plot_filename = joinpath(output_dir, "average_uncertainty_comparison.png")
    savefig(p, plot_filename)
    println("    ✓ Saved: $(basename(plot_filename))")
    
    return p
end

"""
Create summary statistics table
"""
function create_summary_table(all_data::Dict{String, Dict}, output_dir::String)
    println("📋 Creating summary statistics table...")
    
    # Prepare data for DataFrame
    rows = []
    
    run_counter = 1
    for (timestamp, timestamp_data) in all_data
        for mode in PLANNING_MODES
            if haskey(timestamp_data, mode)
                for (unique_run_id, run_data) in timestamp_data[mode]
                    row = Dict(
                        :timestamp => timestamp,
                        :planning_mode => get_mode_display_name(mode),
                        :run_number => run_counter,
                        :unique_run_id => unique_run_id,
                        :source_run_dir => get(run_data, :source_run_dir, "unknown"),
                        :source_timestamp => get(run_data, :source_timestamp, timestamp)
                    )
                    
                    # Add metrics
                    for metric in METRICS
                        if haskey(run_data[:metrics], metric)
                            row[metric] = run_data[:metrics][metric]
                        else
                            row[metric] = missing
                        end
                    end
                    
                    push!(rows, row)
                    run_counter += 1
                end
            end
        end
    end
    
    # Create DataFrame
    df = DataFrame(rows)
    
    # Save as CSV
    csv_filename = joinpath(output_dir, "summary_statistics.csv")
    CSV.write(csv_filename, df)
    println("    ✓ Saved: $(basename(csv_filename))")
    
    # Print summary statistics
    println("\n📊 Summary Statistics:")
    println("=====================")
    
    for mode in PLANNING_MODES
        mode_data = filter(row -> row.planning_mode == get_mode_display_name(mode), df)
        
        if !isempty(mode_data)
            println("\n$(get_mode_display_name(mode)):")
            for metric in METRICS
                if haskey(mode_data[1, :], metric)
                    values = collect(skipmissing(mode_data[!, metric]))
                    if !isempty(values)
                        println("  $(metric): mean=$(round(mean(values), digits=3)), std=$(round(std(values), digits=3)), n=$(length(values))")
                    end
                end
            end
        end
    end
    
    return df
end

"""
Create bar plot showing averages across runs
"""
function create_averages_bar_plot(averages::Dict{Symbol, Dict{Symbol, Float64}}, output_dir::String)
    println("📊 Creating averages bar plot...")
    
    # Check if we have any data
    if isempty(averages)
        println("    ⚠️ No data available for bar plot")
        return nothing
    end
    
    # Prepare data for plotting - use same order as PLANNING_MODES for consistency
    modes = PLANNING_MODES
    metrics = collect(METRICS)
    
    # Create subplots for each metric
    plots = []
    
    for (i, metric) in enumerate(metrics)
        values = Float64[]
        mode_labels = String[]
        
        for mode in modes
            if haskey(averages, mode) && haskey(averages[mode], metric)
                push!(values, averages[mode][metric])
                push!(mode_labels, get_mode_display_name(mode))
            end
        end
        
        if !isempty(values)
            # Create descriptive metric title
            metric_title = if metric == :event_observation_percentage
                "Event Observation %"
            elseif metric == :ndd_expected
                "NDD (Expected Lifetime)"
            elseif metric == :ndd_actual
                "NDD (Actual Lifetime)"
            elseif metric == :final_uncertainty
                "Final Uncertainty"
            elseif metric == :average_planning_time
                "Average Planning Time (seconds)"
            else
                replace(string(metric), "_" => " ") |> titlecase
            end
            
            # Create bar plot with better formatting
            p = bar(mode_labels, values,
                title=metric_title,
                ylabel=metric_title,
                color=:steelblue,
                alpha=0.7,
                legend=false,
                grid=true,
                size=(400, 300),
                titlefontsize=16,
                xrotation=45,
                bottom_margin=10Plots.mm)
            
            # Add value labels on bars with better positioning
            for (j, val) in enumerate(values)
                # Position text above the bar with more offset to avoid overlap
                y_pos = val + 0.05 * maximum(values)
                # Use smaller font size and better positioning
                annotate!(p, j, y_pos, text(round(val, digits=3), 8, :center, :black))
            end
            
            push!(plots, p)
        end
    end
    
    # Check if we have any plots to combine
    if isempty(plots)
        println("    ⚠️ No valid plots to create")
        return nothing
    end
    
    # Combine plots with better layout
    if length(plots) == 4
        combined_plot = plot(plots[1], plots[2], plots[3], plots[4],
            layout=(2,2),
            size=(1200, 800),
            margin=5Plots.mm)
    elseif length(plots) == 3
        combined_plot = plot(plots[1], plots[2], plots[3],
            layout=(1,3),
            size=(1200, 400),
            margin=5Plots.mm)
    elseif length(plots) == 2
        combined_plot = plot(plots[1], plots[2],
            layout=(1,2),
            size=(800, 400),
            margin=5Plots.mm)
    else
        combined_plot = plot(plots[1],
            size=(400, 400),
            margin=5Plots.mm)
    end
    
    # Save plot
    plot_filename = joinpath(output_dir, "averages_bar_plot.png")
    savefig(combined_plot, plot_filename)
    println("    ✓ Saved: $(basename(plot_filename))")
    
    return combined_plot
end

"""
Create combined comparison plot
"""
function create_combined_comparison(all_data::Dict{String, Dict}, output_dir::String)
    println("📊 Creating combined comparison plot...")
    
    # Prepare data for plotting
    metric_data = Dict{Symbol, Dict{Symbol, Vector{Float64}}}()
    
    for metric in METRICS
        metric_data[metric] = Dict{Symbol, Vector{Float64}}()
        
        for mode in PLANNING_MODES
            values = Float64[]
            
            # Navigate the new data structure: all_data[timestamp][mode][unique_run_id][:metrics][metric]
            for (timestamp, timestamp_data) in all_data
                if haskey(timestamp_data, mode)
                    for (unique_run_id, run_data) in timestamp_data[mode]
                        if haskey(run_data, :metrics) && haskey(run_data[:metrics], metric)
                            push!(values, run_data[:metrics][metric])
                        end
                    end
                end
            end
            
            metric_data[metric][mode] = values
        end
    end
    
    # Create dynamic number of subplots based on number of metrics
    plots = [plot() for _ in 1:length(METRICS)]
    
    # Add boxplots for each metric
    for (i, metric) in enumerate(METRICS)
        p = plots[i]
        
        # Add boxplots for each mode
        for (j, mode) in enumerate(PLANNING_MODES)
            if haskey(metric_data[metric], mode) && !isempty(metric_data[metric][mode])
                boxplot!(p, fill(j, length(metric_data[metric][mode])), metric_data[metric][mode], 
                    label=get_mode_display_name(mode), 
                    fillalpha=0.7,
                    linewidth=2)
            end
        end
        
        # Create descriptive metric title
        metric_title = if metric == :event_observation_percentage
            "Event Observation %"
        elseif metric == :ndd_expected
            "NDD (Expected Lifetime)"
        elseif metric == :ndd_actual
            "NDD (Actual Lifetime)"
        elseif metric == :final_uncertainty
            "Final Uncertainty"
        elseif metric == :average_planning_time
            "Average Planning Time (seconds)"
        else
            replace(string(metric), "_" => " ") |> titlecase
        end
        
        plot!(p, 
            title=metric_title,
            xlabel="Planning Mode",
            ylabel=metric_title,
            xticks=(1:length(PLANNING_MODES), [get_mode_display_name(m) for m in PLANNING_MODES]),
            xrotation=45,
            legend=false,
            grid=true,
            titlefontsize=12,
            bottom_margin=10Plots.mm)
    end
    
    # Combine plots with dynamic layout
    num_metrics = length(METRICS)
    
    # Determine optimal layout based on number of metrics
    if num_metrics == 1
        layout = (1, 1)
        plot_size = (600, 400)
    elseif num_metrics == 2
        layout = (1, 2)
        plot_size = (1200, 400)
    elseif num_metrics == 3
        layout = (1, 3)
        plot_size = (1800, 400)
    elseif num_metrics == 4
        layout = (2, 2)
        plot_size = (1200, 800)
    elseif num_metrics <= 6
        layout = (2, 3)
        plot_size = (1800, 800)
    else
        # For more than 6 metrics, use a grid layout
        cols = ceil(Int, sqrt(num_metrics))
        rows = ceil(Int, num_metrics / cols)
        layout = (rows, cols)
        plot_size = (300 * cols, 300 * rows)
    end
    
    combined_plot = plot(plots..., 
        layout=layout, 
        size=plot_size)
    
    # Save plot
    plot_filename = joinpath(output_dir, "combined_comparison.png")
    savefig(combined_plot, plot_filename)
    println("    ✓ Saved: $(basename(plot_filename))")
    
    return combined_plot
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

function main()
    println("🚀 Starting postprocessing analysis for multiple run directories...")
    println("Target runs: $(join(TARGET_RUNS, ", "))")
    println("="^60)
    
    # Collect all results data
    all_data = collect_results_data()
    
    if isempty(all_data)
        println("❌ No results data found!")
        return
    end
    
    println("\n✅ Data collection completed!")
    
    # Calculate averages across runs
    averages = calculate_run_averages(all_data)
    
    # Print averages summary
    println("\n📊 Averages Across Runs:")
    println("="^30)
    if isempty(averages)
        println("❌ No data found to calculate averages!")
        println("Check that the target directory exists and contains metric files.")
        return
    end
    
    for mode in PLANNING_MODES
        if haskey(averages, mode)
            println("$(get_mode_display_name(mode)):")
            for metric in METRICS
                if haskey(averages[mode], metric)
                    println("  $(metric): $(round(averages[mode][metric], digits=3))")
                end
            end
        end
    end
    
    # Create output directory in the same folder we're reading from
    output_dir = joinpath(OUTPUT_DIR, "postprocessing_analysis")
    mkpath(output_dir)
    
    println("\n📁 Output will be saved in: $(output_dir)")
    
    # Create visualizations
    println("\n🎨 Creating visualizations...")
    println("="^30)
    
    # Create averages bar plot
    create_averages_bar_plot(averages, output_dir)
    
    # Create boxplots
    create_metric_boxplots(all_data, output_dir)
    
    # Create uncertainty evolution plots
    create_uncertainty_evolution_plots(all_data, output_dir)
    
    # Create average uncertainty evolution comparison plot
    create_average_uncertainty_comparison(all_data, output_dir)
    
    # Create combined comparison
    create_combined_comparison(all_data, output_dir)
    
    # Create summary table
    df = create_summary_table(all_data, output_dir)
    
    # Save averages to file
    averages_filename = joinpath(output_dir, "run_averages.txt")
    open(averages_filename, "w") do file
        println(file, "Averages Across Runs - Combined Analysis")
        println(file, "Source directories: $(join(TARGET_RUNS, ", "))")
        println(file, "="^50)
        println(file, "Generated: $(now())")
        println(file, "")
        
        for mode in PLANNING_MODES
            if haskey(averages, mode)
                println(file, "$(get_mode_display_name(mode)):")
                for metric in METRICS
                    if haskey(averages[mode], metric)
                        println(file, "  $(metric): $(round(averages[mode][metric], digits=3))")
                    end
                end
                println(file, "")
            end
        end
    end
    println("    ✓ Saved: $(basename(averages_filename))")
    
    println("\n✅ Postprocessing completed!")
    println("📁 Results saved in: $(output_dir)")
    println("\n📊 Summary:")
    println("  - Averages bar plot: 1 overview of run averages")
    println("  - Bar plots: $(length(METRICS)) metric comparisons")
    println("  - Uncertainty evolution: $(length(all_data)) timestamp plots")
    println("  - Average uncertainty comparison: 1 combined plot")
    println("  - Combined comparison: 1 overview plot")
    println("  - Summary statistics: CSV table")
    println("  - Run averages: Text file with averages")
    
    return all_data, df, averages
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 
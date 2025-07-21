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

# Specific results directory to analyze
TARGET_RUN = "run_2025-07-19T00-53-36-739"
RESULTS_DIR = joinpath("..", "results", TARGET_RUN)
OUTPUT_DIR = RESULTS_DIR  # Save results in the same folder we're reading from

# Performance metrics to analyze
METRICS = [:event_observation_percentage, :ndd_life, :final_uncertainty]

# Planning modes to compare
PLANNING_MODES = [:sweep, :script, :random]

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
                    
                    # Extract NDD
                    elseif contains(line, "Normalized Detection Delay (lifetime):")
                        value_str = split(line, ":")[end]
                        value = parse(Float64, strip(value_str))
                        metrics[:ndd_life] = value
                    
                    # Extract final uncertainty
                    elseif contains(line, "Final average uncertainty:")
                        value_str = split(line, ":")[end]
                        value = parse(Float64, strip(value_str))
                        metrics[:final_uncertainty] = value
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
Find all results directories and extract data
"""
function collect_results_data()
    println("🔍 Analyzing specific run directory: $(TARGET_RUN)")
    
    # Check if target directory exists
    if !isdir(RESULTS_DIR)
        println("❌ Target directory $(RESULTS_DIR) not found!")
        return Dict{String, Dict}()
    end
    
    all_data = Dict{String, Dict}()
    timestamp_data = Dict{Symbol, Dict}()
    
    # Process each planning mode
    for mode in PLANNING_MODES
        mode_data = Dict{String, Dict}()
        
        # Find all run directories for this mode
        # The files are in: Run X/mode/metrics/performance_metrics_mode_runX.txt
        run_pattern = replace(joinpath(RESULTS_DIR, "Run *", string(mode), "metrics", "performance_metrics_$(mode)_run*.txt"), "\\" => "/")
        metric_files = glob(run_pattern)
        
        # If no files found, try alternative pattern
        if isempty(metric_files)
            run_pattern = replace(joinpath(RESULTS_DIR, "Run *", string(mode), "metrics", "*.txt"), "\\" => "/")
            metric_files = glob(run_pattern)
        end
        
        # Debug: print the pattern being searched
        println("    Searching pattern: $(run_pattern)")
        
        println("  Mode $(mode): Found $(length(metric_files)) metric files")
        if !isempty(metric_files)
            println("    Files found:")
            for file in metric_files
                println("      $(basename(file))")
            end
        end
        
        for metric_file in metric_files
            # Extract run number from filename
            filename = basename(metric_file)
            # Handle both formats: performance_metrics_mode_run1.txt and performance_metrics_mode_run1.txt
            run_match = match(r"run(\d+)\.txt$", filename)
            
            if run_match !== nothing
                run_number = run_match[1]
                println("    Processing: $(filename) -> Run $(run_number)")
                
                # Extract metrics
                metrics = extract_metrics_from_file(metric_file)
                uncertainty_evolution = extract_uncertainty_evolution(metric_file)
                
                mode_data[run_number] = Dict(
                    :metrics => metrics,
                    :uncertainty_evolution => uncertainty_evolution,
                    :filepath => metric_file
                )
            else
                println("    ⚠️ Could not extract run number from: $(filename)")
            end
        end
        
        timestamp_data[mode] = mode_data
    end
    
    all_data[TARGET_RUN] = timestamp_data
    
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
            
            # Navigate the correct data structure: all_data[TARGET_RUN][mode][run_number][:metrics][metric]
            if haskey(all_data, TARGET_RUN) && haskey(all_data[TARGET_RUN], mode)
                for (run_number, run_data) in all_data[TARGET_RUN][mode]
                    if haskey(run_data, :metrics) && haskey(run_data[:metrics], metric)
                        push!(mode_data[mode], run_data[:metrics][metric])
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
                    label=string(mode), 
                    fillalpha=0.7,
                    linewidth=2)
            end
        end
        
        # Create descriptive metric title
        metric_title = if metric == :event_observation_percentage
            "Event Observation %"
        elseif metric == :ndd_life
            "Normalized Detection Delay"
        elseif metric == :final_uncertainty
            "Final Uncertainty"
        else
            replace(string(metric), "_" => " ") |> titlecase
        end
        
        plot!(p, 
            title="$(metric_title) Comparison",
            xlabel="Planning Mode",
            ylabel=metric_title,
            xticks=(1:length(PLANNING_MODES), [string(m) for m in PLANNING_MODES]),
            legend=false,
            grid=true,
            size=(600, 400),
            titlefontsize=16)
        
        # Save plot
        plot_filename = joinpath(output_dir, "boxplot_$(metric).png")
        savefig(p, plot_filename)
        println("    ✓ Saved: $(basename(plot_filename))")
        
        # Print statistics
        println("    📈 Statistics for $(metric):")
        for mode in PLANNING_MODES
            if haskey(mode_data, mode) && !isempty(mode_data[mode])
                values = mode_data[mode]
                println("      $(mode): mean=$(round(mean(values), digits=3)), std=$(round(std(values), digits=3)), n=$(length(values))")
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
                
                for (run_num, run_data) in timestamp_data[mode]
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
                        label=string(mode),
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
Create summary statistics table
"""
function create_summary_table(all_data::Dict{String, Dict}, output_dir::String)
    println("📋 Creating summary statistics table...")
    
    # Prepare data for DataFrame
    rows = []
    
    for (timestamp, timestamp_data) in all_data
        for mode in PLANNING_MODES
            if haskey(timestamp_data, mode)
                for (run_num, run_data) in timestamp_data[mode]
                    row = Dict(
                        :timestamp => timestamp,
                        :planning_mode => string(mode),
                        :run_number => parse(Int, run_num)
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
        mode_data = filter(row -> row.planning_mode == string(mode), df)
        
        if !isempty(mode_data)
            println("\n$(mode):")
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
    
    # Prepare data for plotting
    modes = collect(keys(averages))
    metrics = collect(METRICS)
    
    # Create subplots for each metric
    plots = []
    
    for (i, metric) in enumerate(metrics)
        values = Float64[]
        mode_labels = String[]
        
        for mode in modes
            if haskey(averages[mode], metric)
                push!(values, averages[mode][metric])
                push!(mode_labels, string(mode))
            end
        end
        
        if !isempty(values)
            # Create descriptive metric title
            metric_title = if metric == :event_observation_percentage
                "Event Observation %"
            elseif metric == :ndd_life
                "Normalized Detection Delay"
            elseif metric == :final_uncertainty
                "Final Uncertainty"
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
                titlefontsize=16)
            
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
    if length(plots) == 3
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
            
            # Navigate the correct data structure: all_data[TARGET_RUN][mode][run_number][:metrics][metric]
            if haskey(all_data, TARGET_RUN) && haskey(all_data[TARGET_RUN], mode)
                for (run_number, run_data) in all_data[TARGET_RUN][mode]
                    if haskey(run_data, :metrics) && haskey(run_data[:metrics], metric)
                        push!(values, run_data[:metrics][metric])
                    end
                end
            end
            
            metric_data[metric][mode] = values
        end
    end
    
    # Create subplots
    p1 = plot()
    p2 = plot()
    p3 = plot()
    
    # Add boxplots for each metric
    for (i, metric) in enumerate(METRICS)
        p = i == 1 ? p1 : i == 2 ? p2 : p3
        
        # Add boxplots for each mode
        for (j, mode) in enumerate(PLANNING_MODES)
            if haskey(metric_data[metric], mode) && !isempty(metric_data[metric][mode])
                boxplot!(p, fill(j, length(metric_data[metric][mode])), metric_data[metric][mode], 
                    label=string(mode), 
                    fillalpha=0.7,
                    linewidth=2)
            end
        end
        
        # Create descriptive metric title
        metric_title = if metric == :event_observation_percentage
            "Event Observation %"
        elseif metric == :ndd_life
            "Normalized Detection Delay"
        elseif metric == :final_uncertainty
            "Final Uncertainty"
        else
            replace(string(metric), "_" => " ") |> titlecase
        end
        
        plot!(p, 
            title=metric_title,
            xlabel="Planning Mode",
            ylabel=metric_title,
            xticks=(1:length(PLANNING_MODES), [string(m) for m in PLANNING_MODES]),
            legend=false,
            grid=true,
            titlefontsize=12)
    end
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, 
        layout=(1,3), 
        size=(1200, 400))
    
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
    println("🚀 Starting postprocessing analysis for $(TARGET_RUN)...")
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
            println("$(mode):")
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
    
    # Create combined comparison
    create_combined_comparison(all_data, output_dir)
    
    # Create summary table
    df = create_summary_table(all_data, output_dir)
    
    # Save averages to file
    averages_filename = joinpath(output_dir, "run_averages.txt")
    open(averages_filename, "w") do file
        println(file, "Averages Across Runs - $(TARGET_RUN)")
        println(file, "="^50)
        println(file, "Generated: $(now())")
        println(file, "")
        
        for mode in PLANNING_MODES
            if haskey(averages, mode)
                println(file, "$(mode):")
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
    println("  - Combined comparison: 1 overview plot")
    println("  - Summary statistics: CSV table")
    println("  - Run averages: Text file with averages")
    
    return all_data, df, averages
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 
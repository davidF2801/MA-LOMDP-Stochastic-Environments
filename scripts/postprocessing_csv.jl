#!/usr/bin/env julia

"""
Postprocessing script for analyzing simulation results from CSV files
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

println("📊 Starting CSV-based postprocessing analysis...")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Specific results directory to analyze
TARGET_RUN = "run_2025-08-14T15-17-32-555"
RESULTS_DIR = joinpath("..", "results", TARGET_RUN)
OUTPUT_DIR = RESULTS_DIR  # Save results in the same folder we're reading from

# Performance metrics to analyze
METRICS = [:event_observation_percentage, :ndd_expected, :ndd_actual]

# Planning modes to compare
PLANNING_MODES = [:pbvi_0_0_1_0, :pbvi_0_5_0_5, :pbvi_1_0_0_0, :prior_based, :random]

# =============================================================================
# CSV DATA EXTRACTION FUNCTIONS
# =============================================================================

"""
Extract performance metrics from NDD metrics CSV file
"""
function extract_metrics_from_ndd_csv(filepath::String)
    metrics = Dict{Symbol, Float64}()
    
    try
        # Read the CSV file
        df = CSV.read(filepath, DataFrame)
        
        # Find the summary row (last row with event_id == "SUMMARY")
        summary_row = filter(row -> row.event_id == "SUMMARY", df)
        
        if !isempty(summary_row)
            # Extract metrics from summary row
            metrics[:ndd_expected] = summary_row[1, :ndd_expected]
            metrics[:ndd_actual] = summary_row[1, :ndd_actual]
        else
            # If no summary row, calculate from individual events
            if !isempty(df)
                # Filter out invalid rows (where event_id != "SUMMARY")
                valid_rows = filter(row -> row.event_id != "SUMMARY", df)
                if !isempty(valid_rows)
                    metrics[:ndd_expected] = mean(valid_rows.ndd_expected)
                    metrics[:ndd_actual] = mean(valid_rows.ndd_actual)
                end
            end
        end
        
    catch e
        println("⚠️ Warning: Could not parse NDD CSV file $(filepath): $(e)")
    end
    
    return metrics
end

"""
Extract event observation percentage from event tracking CSV file
"""
function extract_event_observation_from_csv(filepath::String)
    metrics = Dict{Symbol, Float64}()
    
    try
        # Read the CSV file
        df = CSV.read(filepath, DataFrame)
        
        # Filter out invalid rows (where event_id != "SUMMARY" and end_time != -1)
        # Use proper DataFrame filtering syntax
        valid_rows = df[df.event_id .!= "SUMMARY" .&& df.end_time .!= -1, :]
        
        if !isempty(valid_rows)
            # Calculate observation percentage
            total_events = nrow(valid_rows)
            observed_events = count(valid_rows.observed)
            observation_percentage = (observed_events / total_events) * 100.0
            
            metrics[:event_observation_percentage] = observation_percentage
        end
        
    catch e
        println("⚠️ Warning: Could not parse event tracking CSV file $(filepath): $(e)")
    end
    
    return metrics
end

"""
Extract uncertainty evolution data from CSV file
"""
function extract_uncertainty_evolution_from_csv(filepath::String)
    uncertainty_data = Float64[]
    
    try
        # Read the CSV file
        df = CSV.read(filepath, DataFrame)
        
        if !isempty(df)
            # Extract average uncertainty values
            uncertainty_data = collect(df.average_uncertainty)
        end
        
    catch e
        println("⚠️ Warning: Could not extract uncertainty evolution from $(filepath): $(e)")
    end
    
    return uncertainty_data
end

"""
Combine metrics from multiple CSV files
"""
function combine_metrics_from_csv_files(ndd_file::String, event_file::String)
    combined_metrics = Dict{Symbol, Float64}()
    
    # Extract NDD metrics
    ndd_metrics = extract_metrics_from_ndd_csv(ndd_file)
    merge!(combined_metrics, ndd_metrics)
    
    # Extract event observation metrics
    event_metrics = extract_event_observation_from_csv(event_file)
    merge!(combined_metrics, event_metrics)
    
    return combined_metrics
end

"""
Find all results directories and extract data from CSV files
"""
function collect_results_data_from_csv()
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
        run_pattern = replace(joinpath(RESULTS_DIR, "Run *", string(mode), "metrics"), "\\" => "/")
        metric_dirs = glob(run_pattern)
        
        println("  Mode $(mode): Found $(length(metric_dirs)) metric directories")
        
        for metric_dir in metric_dirs
            # Extract run number from directory path
            dir_path = splitpath(metric_dir)
            # The path structure is: .../Run X/mode/metrics, so Run X is at dir_path[end-2]
            run_match = match(r"Run (\d+)", dir_path[end-2])
            
            if run_match !== nothing
                run_number = run_match[1]
                println("    Processing: Run $(run_number)")
                
                # Look for specific CSV files
                ndd_file = joinpath(metric_dir, "ndd_metrics_$(mode)_run$(run_number).csv")
                event_file = joinpath(metric_dir, "event_tracking_$(mode)_run$(run_number).csv")
                uncertainty_file = joinpath(metric_dir, "uncertainty_evolution_$(mode)_run$(run_number).csv")
                
                # Check if files exist
                if isfile(ndd_file) && isfile(event_file)
                    # Extract metrics from CSV files
                    metrics = combine_metrics_from_csv_files(ndd_file, event_file)
                    
                    # Extract uncertainty evolution
                    uncertainty_evolution = Float64[]
                    if isfile(uncertainty_file)
                        uncertainty_evolution = extract_uncertainty_evolution_from_csv(uncertainty_file)
                    end
                    
                    mode_data[run_number] = Dict(
                        :metrics => metrics,
                        :uncertainty_evolution => uncertainty_evolution,
                        :ndd_file => ndd_file,
                        :event_file => event_file,
                        :uncertainty_file => uncertainty_file
                    )
                    
                    println("      ✓ Extracted metrics: $(keys(metrics))")
                    println("      ✓ Uncertainty evolution: $(length(uncertainty_evolution)) steps")
                else
                    println("      ⚠️ Missing required CSV files")
                    if !isfile(ndd_file)
                        println("        Missing: $(basename(ndd_file))")
                    end
                    if !isfile(event_file)
                        println("        Missing: $(basename(event_file))")
                    end
                end
            else
                println("    ⚠️ Could not extract run number from: $(metric_dir)")
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
        elseif metric == :ndd_expected
            "NDD (Expected Lifetime)"
        elseif metric == :ndd_actual
            "NDD (Actual Lifetime)"
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
                for (run_num, run_data) in timestamp_data[mode]
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
                label=string(mode),
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
            elseif metric == :ndd_expected
                "NDD (Expected Lifetime)"
            elseif metric == :ndd_actual
                "NDD (Actual Lifetime)"
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
    p4 = plot()
    
    # Add boxplots for each metric
    for (i, metric) in enumerate(METRICS)
        p = i == 1 ? p1 : i == 2 ? p2 : i == 3 ? p3 : p4
        
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
        elseif metric == :ndd_expected
            "NDD (Expected Lifetime)"
        elseif metric == :ndd_actual
            "NDD (Actual Lifetime)"
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
            xrotation=45,
            legend=false,
            grid=true,
            titlefontsize=12,
            bottom_margin=10Plots.mm)
    end
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, 
        layout=(2,2), 
        size=(1200, 800))
    
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
    println("🚀 Starting CSV-based postprocessing analysis for $(TARGET_RUN)...")
    println("="^60)
    
    # Collect all results data from CSV files
    all_data = collect_results_data_from_csv()
    
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
        println("Check that the target directory exists and contains CSV metric files.")
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
    output_dir = joinpath(OUTPUT_DIR, "postprocessing_csv_analysis")
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
        println(file, "Averages Across Runs - $(TARGET_RUN)")
        println(file, "="^50)
        println(file, "Generated: $(now())")
        println(file, "Data Source: CSV files")
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
    
    println("\n✅ CSV-based postprocessing completed!")
    println("📁 Results saved in: $(output_dir)")
    println("\n📊 Summary:")
    println("  - Averages bar plot: 1 overview of run averages")
    println("  - Bar plots: $(length(METRICS)) metric comparisons")
    println("  - Uncertainty evolution: $(length(all_data)) timestamp plots")
    println("  - Average uncertainty comparison: 1 combined plot")
    println("  - Combined comparison: 1 overview plot")
    println("  - Summary statistics: CSV table")
    println("  - Run averages: Text file with averages")
    println("\n💡 Data Source: CSV files (cleaner and more efficient)")
    
    return all_data, df, averages
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 
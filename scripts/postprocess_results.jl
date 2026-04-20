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

# Set plotting backend to GR for high-quality PDF support
gr()
# Set high DPI for better PDF quality
ENV["GKSwstype"] = "nul"
# Ensure consistent color rendering between PNG and PDF
ENV["GR_COLORSPACE"] = "sRGB"
# Set consistent color palette for both PNG and PDF
default(palette=:default)

println("📊 Starting postprocessing analysis...")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Multiple results directories to analyze - add as many as needed
TARGET_RUNS = [
    #"run_2025-08-17T14-08-17-424",
    #"run_2025-08-16T16-52-42-231"
    #"run_2025-08-19T10-23-17-927-new",
    # Add more run directories here as needed
    # "run_2025-08-16T16-52-26-473",
    # "run_2025-08-16T16-52-42-231", #4x3 original submission
    #"run_2026-01-29T16-25-34-037"
    #"run_2026-01-29T16-55-16-714"
    #"run_2026-02-04T11-36-43-894"
    #"run_2026-02-04T18-45-16-865"
    #"run_2026-02-06T16-06-11-775"
    #"run_2026-02-13T09-40-14-099"
    #"run_2026-02-14T17-12-46-276",
    #"run_2026-02-14T17-12-46-276"
    #"run_2026-02-26T12-13-17-457"
    #"run_2026-02-13T09-40-14-099"
    #"run_2026-02-20T17-38-12-398"
    #"run_2026-02-27T19-06-36-217"
    #"run_2026-02-28T23-42-26-769"
    #  "run_2026-02-27T19-06-36-217", #4x3 klolop
    #  "run_2025-08-16T16-52-42-231", #4x3 original submission
    #  "run_2026-02-15T22-34-22-346", #4x3 New batch
    # "run_2026-02-28T23-42-19-465",  #5x5 klolop
    # "run_2025-08-17T14-08-17-424", #5x5 original submission
    # "run_2026-03-02T12-58-22-025", #5x5 oracle
    # "run_2026-03-01T09-19-46-709", #9x9 klolop two cells
    # "run_2026-02-28T23-42-26-769", #9x9 klolop one cell
    # "run_2025-09-08T09-36-38-974", #9x9 original submission
    # "run_2026-03-02T12-59-00-637", #9x9 oracle
    # "run_2026-02-20T17-38-12-398" #9x9 New batch two cells
    #"run_2026-03-26T14-03-37-467", #4x3 joint abba
    #"run_2026-04-08T23-44-11-840"
    #"run_2026-04-08T23-44-11-864" #4x3 with rewards
    #"run_2026-04-10T13-21-43-246" #5x5 with rewards
    #"run_2026-04-18T12-18-53-687"  #5x5 with rewards
    "run_2026-04-18T12-19-01-687", #9x9 with rewards
    "run_2026-04-18T15-40-16-095" #9x9 with rewards

    
]

# Results path: relative to script location so it works regardless of cwd
RESULTS_BASE = joinpath(@__DIR__, "..", "results")

# Save plot robustly in both PDF and PNG (with verification/retry)
function _savefig_checked(p, outpath::String; attempts::Int=2, min_bytes::Int=128)
    local last_err = nothing
    for attempt in 1:attempts
        try
            savefig(p, outpath)
            if isfile(outpath)
                sz = stat(outpath).size
                if sz >= min_bytes
                    return true, sz
                else
                    # Remove clearly corrupt output (e.g., 0-byte files)
                    rm(outpath; force=true)
                    last_err = "file too small ($(sz) bytes)"
                end
            else
                last_err = "file was not created"
            end
        catch e
            last_err = e
        end
        sleep(0.15)
    end
    return false, last_err
end

function savefig_both(p, basepath_with_ext::String)
    mkpath(dirname(basepath_with_ext))
    pngpath = replace(basepath_with_ext, r"\.pdf$" => ".png")

    ok_pdf, info_pdf = _savefig_checked(p, basepath_with_ext; attempts=3, min_bytes=128)
    ok_png, info_png = if pngpath != basepath_with_ext
        _savefig_checked(p, pngpath; attempts=3, min_bytes=128)
    else
        (true, "same path")
    end

    if ok_pdf && ok_png
        println("    ✓ Saved: $(basename(basepath_with_ext)) and $(basename(pngpath))")
    else
        println("    ❌ Plot save failed or produced corrupt file(s):")
        println("      - PDF: $(ok_pdf ? "ok" : "failed ($(info_pdf))")")
        if pngpath != basepath_with_ext
            println("      - PNG: $(ok_png ? "ok" : "failed ($(info_png))")")
        end
    end

    return (pdf_ok=ok_pdf, png_ok=ok_png, pdf_path=basepath_with_ext, png_path=pngpath)
end
# Output directory - use the first target run folder
OUTPUT_DIR = joinpath(RESULTS_BASE, TARGET_RUNS[1])

# Performance metrics to analyze
METRICS = [:event_observation_percentage, :final_uncertainty, :average_planning_time, :ndd_actual]

# If true, for :average_planning_time we only use data from the LAST
# TARGET_RUNS folder (typically the newest batch). Other metrics still
# use all TARGET_RUNS as usual.
const USE_LAST_RUN_FOR_PLANNING_TIME = true

# Planning modes to compare
#PLANNING_MODES = [:joint_abba]
#PLANNING_MODES = [:oracle, :script, :mpomdp_openloop, :pbvi, :klolop, :greedy, :sweep, :prior_based,:random] #4x3
#PLANNING_MODES = [:oracle, :script, :mpomdp_openloop, :pbvi] #4x3

# PLANNING_MODES = [:oracle, :pbvi_1_0_0_0, :pbvi_0_5_0_5, :pbvi_0_0_1_0,:klolop, :prior_based, :random, :greedy] #5x5 and 9x9
PLANNING_MODES = [:oracle, :pbvi_1_0_0_0,:klolop, :prior_based, :random, :greedy, :pbvi_rollout] #5x5 and 9x9

#PLANNING_MODES = [:sweep, :script, :random]
#PLANNING_MODES = [:pomcp]
#PLANNING_MODES = [:script, :pbvi, :prior_based, :random]
#PLANNING_MODES = [:pbvi_mis, :oracle, :pbvi_0_5_0_5]
#PLANNING_MODES = [:klolop]

# Function to get display name for planning modes
function get_mode_display_name(mode::Symbol)
    mode_str = string(mode)
    
    if mode == :script
        return "ABBA"
    elseif mode == :pbvi
        return "SB-ABBA"
    elseif mode == :pbvi_rollout
        return "SB-ABBA (rollout)"
    elseif mode == :macro_approx_090
        return "PB-ABBA_090"
    elseif mode == :prior_based
        return "Prior-Based"
    elseif mode == :random
        return "Random"
    elseif mode == :klolop
        # Special capitalization for KL-OLOP planner
        return "KL-OLOP"
    elseif mode == :posts
        return "POSTS"
    elseif mode == :joint_abba
        return "Joint ABBA"
    elseif startswith(mode_str, "pbvi_")
        # Parse PBVI variants with weights
        # Format: pbvi_X_Y_Z_W where X_Y is entropy weight (wh) and Z_W is detection weight (wv)
        parts = split(mode_str, "_")
        if length(parts) >= 5
            # Extract weights: pbvi_X_Y_Z_W -> entropy=X.Y (wh), detection=Z.W (wv)
            entropy_int = parts[2]
            entropy_dec = parts[3]
            detection_int = parts[4]
            detection_dec = parts[5]
            
            entropy_weight = "$(entropy_int).$(entropy_dec)"
            detection_weight = "$(detection_int).$(detection_dec)"
            
            return "SB-ABBA\n(wh:$(entropy_weight), wv:$(detection_weight))"
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
    results_base = RESULTS_BASE
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
Extract per-timestep reward evolution from action_reward_log CSV.
Returns a dict with:
- :team_step_reward::Vector{Float64}
- :cumulative_discounted_team::Vector{Float64}
"""
function extract_reward_log(filepath::String)
    reward_data = Dict{Symbol, Vector{Float64}}(
        :team_step_reward => Float64[],
        :cumulative_discounted_team => Float64[]
    )

    try
        df = CSV.read(filepath, DataFrame)
        # CSV.jl uses String column names; `(:timestep in names(df))` is false for Symbol checks
        colset = Set(String.(names(df)))
        required_names = ("timestep", "team_step_reward", "cumulative_discounted_team")
        if !all(n -> n in colset, required_names)
            println("⚠️ Warning: Reward log missing required columns in $(filepath); have $(names(df))")
            return reward_data
        end

        sort!(df, "timestep")
        grouped = groupby(df, "timestep")

        for g in grouped
            push!(reward_data[:team_step_reward], first(skipmissing(g[!, "team_step_reward"])))
            push!(reward_data[:cumulative_discounted_team], first(skipmissing(g[!, "cumulative_discounted_team"])))
        end
    catch e
        println("⚠️ Warning: Could not parse reward log $(filepath): $(e)")
    end

    return reward_data
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
        results_dir = joinpath(RESULTS_BASE, target_run)
        
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

                # Reward logs are stored separately from the performance text file
                reward_log_files = String[]
                for file in readdir(metrics_path)
                    if startswith(file, "action_reward_log_") && endswith(file, ".csv")
                        push!(reward_log_files, joinpath(metrics_path, file))
                    end
                end
                if length(reward_log_files) > 1
                    println("      ⚠️ Found multiple reward logs in $(metrics_path), using first one")
                end
                reward_log = isempty(reward_log_files) ? Dict{Symbol, Vector{Float64}}(
                    :team_step_reward => Float64[],
                    :cumulative_discounted_team => Float64[]
                ) : extract_reward_log(reward_log_files[1])
                
                # Process each metric file (.txt performance summary)
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
                        :reward_log => reward_log,
                        :filepath => metric_file,
                        :source_run_dir => run_dir_name,
                        :source_timestamp => target_run
                    )
                    
                    println("          ✓ Stored as run ID: $(unique_run_id)")
                end

                # If there are reward logs but no .txt metrics file, still record reward for plotting
                if isempty(metric_files) && !isempty(reward_log_files) &&
                   any(!isempty, values(reward_log))
                    if !haskey(all_data, target_run)
                        all_data[target_run] = Dict{Symbol, Dict}()
                    end
                    if !haskey(all_data[target_run], mode)
                        all_data[target_run][mode] = Dict{String, Dict}()
                    end
                    unique_run_id = "$(target_run)_$(run_dir_name)_$(global_run_counter)"
                    global_run_counter += 1
                    empty_metrics = Dict{Symbol, Float64}()
                    all_data[target_run][mode][unique_run_id] = Dict(
                        :metrics => empty_metrics,
                        :uncertainty_evolution => Float64[],
                        :reward_log => reward_log,
                        :filepath => reward_log_files[1],
                        :source_run_dir => run_dir_name,
                        :source_timestamp => target_run
                    )
                    println("      Mode $(mode): no .txt metrics; stored reward-only entry as $(unique_run_id)")
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
Calculate averages across runs for each planning mode.
Accumulates all values from all TARGET_RUNS (timestamps), then computes mean per metric,
so bar plots use the same combined data as the boxplots.
"""
function calculate_run_averages(all_data::Dict{String, Dict})
    println("📊 Calculating averages across runs...")
    
    averages = Dict{Symbol, Dict{Symbol, Float64}}()
    
    for mode in PLANNING_MODES
        # Accumulate values from timestamps/runs for this mode
        metric_values = Dict{Symbol, Vector{Float64}}()
        
        for (timestamp, timestamp_data) in all_data
            if haskey(timestamp_data, mode)
                for (run_num, run_data) in timestamp_data[mode]
                    for metric in METRICS
                        # Optional exception: for planning time, only use last TARGET_RUNS folder
                        if metric == :average_planning_time && USE_LAST_RUN_FOR_PLANNING_TIME
                            if timestamp != TARGET_RUNS[end]
                                continue
                            end
                        end
                        if haskey(run_data[:metrics], metric)
                            if !haskey(metric_values, metric)
                                metric_values[metric] = Float64[]
                            end
                            push!(metric_values[metric], run_data[:metrics][metric])
                        end
                    end
                end
            end
        end
        
        # Single mean per metric over all accumulated values
        mode_averages = Dict{Symbol, Float64}()
        for metric in METRICS
            if haskey(metric_values, metric) && !isempty(metric_values[metric])
                mode_averages[metric] = mean(metric_values[metric])
            end
        end
        averages[mode] = mode_averages
    end
    
    return averages
end

"""
Compute mean and std trajectories for a list of variable-length vectors.
Pads shorter vectors with NaN and ignores NaN in stats.
"""
function aggregate_variable_length_series(series_list::Vector{Vector{Float64}})
    if isempty(series_list)
        return Float64[], Float64[]
    end

    max_len = maximum(length.(series_list))
    padded = [vcat(s, fill(NaN, max_len - length(s))) for s in series_list]
    mat = hcat(padded...)

    means = Float64[]
    stds = Float64[]
    for i in 1:max_len
        vals = mat[i, :]
        clean_vals = vals[.!isnan.(vals)]
        if isempty(clean_vals)
            push!(means, NaN)
            push!(stds, NaN)
        else
            push!(means, mean(clean_vals))
            push!(stds, length(clean_vals) > 1 ? std(clean_vals) : 0.0)
        end
    end

    return means, stds
end

"""
Create reward comparison plots from action_reward_log CSV files.
Saves in a separate output subdirectory to avoid affecting existing plots.
"""
function create_reward_comparison_plots(all_data::Dict{String, Dict}, output_dir::String)
    println("💰 Creating reward comparison plots...")

    reward_output_dir = joinpath(output_dir, "reward_plots")
    mkpath(reward_output_dir)

    mode_step_series = Dict{Symbol, Vector{Vector{Float64}}}()
    mode_cum_series = Dict{Symbol, Vector{Vector{Float64}}}()

    for mode in PLANNING_MODES
        mode_step_series[mode] = Vector{Vector{Float64}}()
        mode_cum_series[mode] = Vector{Vector{Float64}}()

        for (_, timestamp_data) in all_data
            if haskey(timestamp_data, mode)
                for (_, run_data) in timestamp_data[mode]
                    if haskey(run_data, :reward_log)
                        rlog = run_data[:reward_log]
                        if haskey(rlog, :team_step_reward) && !isempty(rlog[:team_step_reward])
                            push!(mode_step_series[mode], rlog[:team_step_reward])
                        end
                        if haskey(rlog, :cumulative_discounted_team) && !isempty(rlog[:cumulative_discounted_team])
                            push!(mode_cum_series[mode], rlog[:cumulative_discounted_team])
                        end
                    end
                end
            end
        end
    end

    has_any_reward_data = any(!isempty(v) for v in values(mode_step_series))
    if !has_any_reward_data
        println("    ⚠️ No reward logs found. Skipping reward plots.")
        return nothing
    end

    p_step = plot()
    p_cum = plot()

    for mode in PLANNING_MODES
        if !isempty(mode_step_series[mode])
            means, stds = aggregate_variable_length_series(mode_step_series[mode])
            t = 0:(length(means)-1)
            plot!(p_step, t, means,
                ribbon=stds,
                label=get_mode_display_name(mode),
                linewidth=2.5,
                fillalpha=0.2)
        end

        if !isempty(mode_cum_series[mode])
            means, stds = aggregate_variable_length_series(mode_cum_series[mode])
            t = 0:(length(means)-1)
            plot!(p_cum, t, means,
                ribbon=stds,
                label=get_mode_display_name(mode),
                linewidth=2.5,
                fillalpha=0.2)
        end
    end

    plot!(p_step,
        title="Team Step Reward Evolution",
        xlabel="Timestep",
        ylabel="Team Step Reward",
        legend=true,
        grid=true,
        gridwidth=0.5,
        gridalpha=0.3)

    plot!(p_cum,
        title="Cumulative Discounted Team Reward",
        xlabel="Timestep",
        ylabel="Discounted Return",
        legend=true,
        grid=true,
        gridwidth=0.5,
        gridalpha=0.3)

    combined = plot(p_step, p_cum,
        layout=(2, 1),
        size=(1300, 1200),
        margin=6Plots.mm,
        link=:none)

    combined_filename = joinpath(reward_output_dir, "reward_comparison_combined.pdf")
    savefig_both(combined, combined_filename)

    return combined
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
        
        # For planning time, optionally restrict to last TARGET_RUNS folder only
        use_last_run_only = (metric == :average_planning_time && USE_LAST_RUN_FOR_PLANNING_TIME)
        last_run_id = TARGET_RUNS[end]
        
        for mode in PLANNING_MODES
            mode_data[mode] = Float64[]
            
            # Navigate the new data structure: all_data[timestamp][mode][unique_run_id][:metrics][metric]
            for (timestamp, timestamp_data) in all_data
                if use_last_run_only && timestamp != last_run_id
                    continue
                end
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
                     linewidth=1.5,
                     linecolor=:black)
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
            xlabel="",
            ylabel=metric_title,
            xticks=(1:length(PLANNING_MODES), [get_mode_display_name(m) for m in PLANNING_MODES]),
            # Keep angled labels but give them extra padding so they
            # don’t collide with neighboring boxes.
            xrotation=45,
            legend=false,
            grid=true,
            gridwidth=0.5,
            gridalpha=0.3,
            size=(1000, 900),
            titlefontsize=36,
            xlabelfontsize=32,
            ylabelfontsize=32,
            xtickfontsize=26,
            ytickfontsize=30,
            bottom_margin=22Plots.mm,
            left_margin=18Plots.mm,
            top_margin=15Plots.mm)
        
        # Save plot (PDF and PNG)
        plot_filename = joinpath(output_dir, "boxplot_$(metric).pdf")
        savefig_both(p, plot_filename)
        
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
                        linewidth=1.5,
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
                gridwidth=0.5,
                gridalpha=0.3,
                size=(1000, 800),
                titlefontsize=36,
                xlabelfontsize=32,
                ylabelfontsize=32,
                xtickfontsize=30,
                ytickfontsize=30,
                legendfontsize=30)
            
            # Save plot (PDF and PNG)
            plot_filename = joinpath(output_dir, "uncertainty_evolution_$(timestamp).pdf")
            savefig_both(p, plot_filename)
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
        gridwidth=0.5,
        gridalpha=0.3,
        size=(1200, 800),
        titlefontsize=36,
        xlabelfontsize=32,
        ylabelfontsize=32,
        xtickfontsize=30,
        ytickfontsize=30,
        legendfontsize=32)
    
    # Save plot (PDF and PNG)
    plot_filename = joinpath(output_dir, "average_uncertainty_comparison.pdf")
    savefig_both(p, plot_filename)
    
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
            # Track maximum value for y-axis padding and annotations
            max_val = maximum(values)

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
            
            # Choose number formatting:
            # - 3 decimals for NDD and final uncertainty
            # - 2 decimals for planning time
            # - 1 decimal for everything else
            label_values =
                if metric in (:ndd_expected, :ndd_actual, :final_uncertainty)
                    round.(values, digits=3)
                elseif metric == :average_planning_time
                    round.(values, digits=2)
                else
                    round.(values, digits=1)
                end
            label_strings = string.(label_values)

            # Create bar plot with better formatting, clearer spacing, and extra headroom
            # on the y-axis so value labels don't overlap the title.
            p = bar(mode_labels, values,
                title=metric_title,
                ylabel=metric_title,
                color=:steelblue,
                alpha=0.7,
                # Narrower bars to create more separation
                bar_width=0.45,
                # Extra headroom above the tallest bar
                ylims=(0, max_val * 1.3),
                legend=false,
                grid=true,
                gridwidth=0.5,
                gridalpha=0.3,
                size=(700, 780),
                titlefontsize=30,
                xlabelfontsize=26,
                ylabelfontsize=26,
                xtickfontsize=24,
                ytickfontsize=24,
                xrotation=45,
                bottom_margin=15Plots.mm,
                left_margin=18Plots.mm,
                top_margin=15Plots.mm)

            # Add centered value labels on bars with a vertical offset to avoid overlap
            for (j, label) in enumerate(label_strings)
                # Slight left shift to visually center text over the bar
                x_pos = j - 0.20
                # Position text above the bar with some offset relative to max value
                y_pos = values[j] + 0.08 * max_val
                # Explicitly set horizontal and vertical alignment; color as last arg
                annotate!(p, x_pos, y_pos, text(label, 24, :center, :bottom, :black))
            end
            
            push!(plots, p)
        end
    end
    
    # Check if we have any plots to combine
    if isempty(plots)
        println("    ⚠️ No valid plots to create")
        return nothing
    end
    
    # Combine plots with better layout and even more space between subplots
    if length(plots) == 4
        combined_plot = plot(plots[1], plots[2], plots[3], plots[4],
            layout=(2,2),
            size=(2000, 1800),
            margin=5Plots.mm,
            link=:none,
            wspace=0.3,
            hspace=0.3)
    elseif length(plots) == 3
        combined_plot = plot(plots[1], plots[2], plots[3],
            layout=(1,3),
            size=(2000, 1000),
            margin=5Plots.mm,
            link=:none,
            wspace=0.18,
            hspace=0.22)
    elseif length(plots) == 2
        combined_plot = plot(plots[1], plots[2],
            layout=(1,2),
            size=(1600, 1000),
            margin=5Plots.mm,
            link=:none,
            wspace=0.18,
            hspace=0.22)
    else
        combined_plot = plot(plots[1],
            size=(800, 1000),
            margin=15Plots.mm)
    end
    
    # Save plot (PDF and PNG)
    plot_filename = joinpath(output_dir, "averages_bar_plot.pdf")
    savefig_both(combined_plot, plot_filename)
    
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
        
        # For planning time, optionally restrict to last TARGET_RUNS folder only
        use_last_run_only = (metric == :average_planning_time && USE_LAST_RUN_FOR_PLANNING_TIME)
        last_run_id = TARGET_RUNS[end]
        
        for mode in PLANNING_MODES
            values = Float64[]
            
            # Navigate the new data structure: all_data[timestamp][mode][unique_run_id][:metrics][metric]
            for (timestamp, timestamp_data) in all_data
                if use_last_run_only && timestamp != last_run_id
                    continue
                end
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
                     linewidth=1.5,
                     linecolor=:black)
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
            xlabel="",
            ylabel=metric_title,
            xticks=(1:length(PLANNING_MODES), [get_mode_display_name(m) for m in PLANNING_MODES]),
            # Angled labels, but with more space between subplots so
            # rows don’t overlap each other.
            xrotation=45,
            legend=false,
            grid=true,
            gridwidth=0.5,
            gridalpha=0.3,
            titlefontsize=32,
            xlabelfontsize=30,
            ylabelfontsize=30,
            xtickfontsize=24,
            ytickfontsize=28,
            bottom_margin=22Plots.mm,
            left_margin=18Plots.mm,
            top_margin=15Plots.mm)
    end
    
    # Combine plots with dynamic layout
    num_metrics = length(METRICS)
    
    # Determine optimal layout based on number of metrics
    if num_metrics == 1
        layout = (1, 1)
        plot_size = (900, 900)
    elseif num_metrics == 2
        layout = (1, 2)
        plot_size = (1800, 900)
    elseif num_metrics == 3
        layout = (1, 3)
        plot_size = (2400, 900)
    elseif num_metrics == 4
        layout = (2, 2)
        plot_size = (2000, 1800)
    elseif num_metrics <= 6
        layout = (2, 3)
        plot_size = (2400, 1200)
    else
        # For more than 6 metrics, use a grid layout
        cols = ceil(Int, sqrt(num_metrics))
        rows = ceil(Int, num_metrics / cols)
        layout = (rows, cols)
        plot_size = (500 * cols, 500 * rows)
    end
    
    combined_plot = plot(plots..., 
        layout=layout, 
        size=plot_size,
        # Add positive spacing and a small margin so the subplots have
        # clear separation and the x‑tick labels don’t overlap across rows.
        margin=5Plots.mm,
        link=:none,
        wspace=0.28,
        hspace=0.32)
    
    # Save plot (PDF and PNG)
    plot_filename = joinpath(output_dir, "combined_comparison.pdf")
    savefig_both(combined_plot, plot_filename)
    
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

    # Create reward comparison plots (saved separately)
    create_reward_comparison_plots(all_data, output_dir)
    
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
    println("  - Reward comparison: separate combined plot in reward_plots/")
    println("  - Summary statistics: CSV table")
    println("  - Run averages: Text file with averages")
    
    return all_data, df, averages
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 
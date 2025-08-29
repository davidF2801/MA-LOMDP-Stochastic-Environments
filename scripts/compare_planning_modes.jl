#!/usr/bin/env julia

"""
Compare two planning modes and find runs with the best/worst performance differences
Usage: julia compare_planning_modes.jl [mode1] [mode2] [metric] [direction]
Example: julia compare_planning_modes.jl pbvi_0_5_0_5 prior_based event_observation_percentage best
Example: julia compare_planning_modes.jl pbvi_0_5_0_5 prior_based ndd_expected worst
Example: julia compare_planning_modes.jl pbvi_0_5_0_5 prior_based all best

Parameters:
- mode1: First planning mode (e.g., pbvi_0_5_0_5)
- mode2: Second planning mode to compare against (e.g., prior_based)  
- metric: Performance metric to compare (or 'all' for all metrics)
- direction: 'best' finds runs where mode1 >> mode2, 'worst' finds runs where mode1 << mode2

Available metrics:
- event_observation_percentage (higher is better)
- ndd_expected (lower is better)  
- ndd_actual (lower is better)
- final_uncertainty (lower is better)
- average_planning_time (lower is better)
- all (analyze all metrics simultaneously)
"""

using Statistics
using Glob

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default settings (can be overridden by command line arguments)
DEFAULT_MODE1 = :pbvi_0_5_0_5
DEFAULT_MODE2 = :prior_based
DEFAULT_METRIC = :all
DEFAULT_DIRECTION = "best"  # "best" or "worst"

# Multiple results directories to analyze - add as many as needed
TARGET_RUNS = [
    #"run_2025-08-17T14-08-17-424",
    "run_2025-08-19T10-23-17-927-new",
    # Add more run directories here as needed
    # "run_2025-08-16T16-52-26-473",
    # "run_2025-08-16T16-52-42-231",
]

# Metric definitions: true means higher is better, false means lower is better
METRIC_DEFINITIONS = Dict(
    :event_observation_percentage => (name="Event Observation %", higher_is_better=true),
    :ndd_expected => (name="NDD (Expected Lifetime)", higher_is_better=false),
    :ndd_actual => (name="NDD (Actual Lifetime)", higher_is_better=false),
    :final_uncertainty => (name="Final Uncertainty", higher_is_better=false),
    :average_planning_time => (name="Average Planning Time (seconds)", higher_is_better=false)
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
Calculate performance difference between two modes for a specific metric
Returns positive value if mode1 is better, negative if mode2 is better
"""
function calculate_performance_difference(value1::Float64, value2::Float64, metric::Symbol)
    metric_def = METRIC_DEFINITIONS[metric]
    
    if metric_def.higher_is_better
        # Higher is better, so mode1 - mode2 (positive means mode1 is better)
        return value1 - value2
    else
        # Lower is better, so mode2 - mode1 (positive means mode1 is better)
        return value2 - value1
    end
end

"""
Compare two planning modes across all metrics and provide a comprehensive summary
"""
function compare_all_metrics(mode1::Symbol, mode2::Symbol, direction::String="best")
    direction_emoji = direction == "best" ? "🏆" : "📉"
    direction_word = direction == "best" ? "best" : "worst"
    
    println("$(direction_emoji) Comparing planning modes across ALL METRICS:")
    println("  Mode 1: $(mode1)")
    println("  Mode 2: $(mode2)")
    println("  Direction: Finding runs where $(mode1) performs $(direction_word) relative to $(mode2)")
    println("="^80)
    
    # Collect run data for both modes
    println("\n🔍 Collecting data for $(mode1)...")
    mode1_results = collect_run_data_for_mode(mode1)
    
    println("\n🔍 Collecting data for $(mode2)...")
    mode2_results = collect_run_data_for_mode(mode2)
    
    if isempty(mode1_results)
        println("❌ No runs found for planning mode: $(mode1)")
        return nothing
    end
    
    if isempty(mode2_results)
        println("❌ No runs found for planning mode: $(mode2)")
        return nothing
    end
    
    # Create lookup dictionaries by run directory and timestamp
    mode1_lookup = Dict()
    mode2_lookup = Dict()
    
    for run in mode1_results
        key = (run[:timestamp], run[:run_dir])
        mode1_lookup[key] = run
    end
    
    for run in mode2_results
        key = (run[:timestamp], run[:run_dir])
        mode2_lookup[key] = run
    end
    
    # Find matching runs (same timestamp and run directory)
    matching_runs = []
    for key in keys(mode1_lookup)
        if haskey(mode2_lookup, key)
            push!(matching_runs, key)
        end
    end
    
    if isempty(matching_runs)
        println("❌ No matching runs found between $(mode1) and $(mode2)")
        println("   Make sure both planning modes have been run on the same run directories")
        return nothing
    end
    
    println("\n📊 Found $(length(matching_runs)) matching runs for comparison")
    
    # Analyze each metric - only include event observation and NDD for overall comparison
    all_metrics = [:event_observation_percentage, :ndd_expected]  # Only these two metrics
    metric_summaries = Dict()
    overall_scores = Dict()  # Track overall performance for each run
    
    for metric in all_metrics
        println("\n" * "="^60)
        println("📊 ANALYZING METRIC: $(METRIC_DEFINITIONS[metric].name)")
        println("="^60)
        
        metric_def = METRIC_DEFINITIONS[metric]
        comparisons = []
        
        # Calculate comparisons for this metric
        for key in matching_runs
            mode1_run = mode1_lookup[key]
            mode2_run = mode2_lookup[key]
            
            if haskey(mode1_run[:metrics], metric) && haskey(mode2_run[:metrics], metric)
                value1 = mode1_run[:metrics][metric]
                value2 = mode2_run[:metrics][metric]
                
                # Calculate performance difference (positive means mode1 is better)
                difference = calculate_performance_difference(value1, value2, metric)
                
                push!(comparisons, Dict(
                    :timestamp => key[1],
                    :run_dir => key[2],
                    :mode1_value => value1,
                    :mode2_value => value2,
                    :difference => difference,
                    :mode1_run => mode1_run,
                    :mode2_run => mode2_run
                ))
                
                # Add to overall scores
                if !haskey(overall_scores, key)
                    overall_scores[key] = []
                end
                push!(overall_scores[key], difference)
            end
        end
        
        if !isempty(comparisons)
            # Find extreme case for this metric
            if direction == "best"
                sorted_comparisons = sort(comparisons, by=x->x[:difference], rev=true)
                extreme_comparison = sorted_comparisons[1]
            else
                sorted_comparisons = sort(comparisons, by=x->x[:difference])
                extreme_comparison = sorted_comparisons[1]
            end
            
            # Calculate statistics
            differences = [comp[:difference] for comp in comparisons]
            mode1_values = [comp[:mode1_value] for comp in comparisons]
            mode2_values = [comp[:mode2_value] for comp in comparisons]
            
            println("Direction: $(metric_def.higher_is_better ? "Higher is better" : "Lower is better")")
            println("Runs analyzed: $(length(comparisons))")
            println("$(mode1) average: $(round(mean(mode1_values), digits=3))")
            println("$(mode2) average: $(round(mean(mode2_values), digits=3))")
            println("Average difference: $(round(mean(differences), digits=3)) (positive = $(mode1) better)")
            println("Std deviation: $(round(std(differences), digits=3))")
            
            println("\n$(direction_emoji) $(uppercase(direction_word)) CASE:")
            println("  Run: $(extreme_comparison[:run_dir]) ($(extreme_comparison[:timestamp]))")
            println("  $(mode1): $(extreme_comparison[:mode1_value])")
            println("  $(mode2): $(extreme_comparison[:mode2_value])")
            println("  Difference: $(round(extreme_comparison[:difference], digits=3))")
            
            # Store summary for this metric
            metric_summaries[metric] = Dict(
                :extreme_case => extreme_comparison,
                :avg_difference => mean(differences),
                :std_difference => std(differences),
                :mode1_avg => mean(mode1_values),
                :mode2_avg => mean(mode2_values),
                :comparisons => comparisons
            )
        else
            println("⚠️ No matching data found for this metric")
            metric_summaries[metric] = nothing
        end
    end
    
    # Calculate overall performance scores for each run with normalization
    println("\n" * "="^80)
    println("🎯 OVERALL PERFORMANCE SUMMARY (Normalized)")
    println("="^80)
    
    # First, collect all differences for each metric to calculate normalization parameters
    metric_differences = Dict()
    for metric in all_metrics
        metric_differences[metric] = []
    end
    
    # Collect all differences by metric
    for (key, scores) in overall_scores
        if !isempty(scores) && length(scores) == length(all_metrics)
            for (i, metric) in enumerate(all_metrics)
                push!(metric_differences[metric], scores[i])
            end
        end
    end
    
    # Calculate normalization parameters (mean and std) for each metric
    metric_normalization = Dict()
    for metric in all_metrics
        if !isempty(metric_differences[metric])
            metric_mean = mean(metric_differences[metric])
            metric_std = std(metric_differences[metric])
            # Avoid division by zero
            if metric_std == 0
                metric_std = 1.0
            end
            metric_normalization[metric] = (mean=metric_mean, std=metric_std)
            println("📊 $(METRIC_DEFINITIONS[metric].name) normalization: mean=$(round(metric_mean, digits=3)), std=$(round(metric_std, digits=3))")
        end
    end
    
    # Calculate normalized overall scores
    run_overall_scores = []
    for (key, scores) in overall_scores
        if !isempty(scores) && length(scores) == length(all_metrics)
            # Normalize each metric difference using z-score
            normalized_scores = []
            for (i, metric) in enumerate(all_metrics)
                if haskey(metric_normalization, metric)
                    norm_params = metric_normalization[metric]
                    normalized_score = (scores[i] - norm_params.mean) / norm_params.std
                    push!(normalized_scores, normalized_score)
                end
            end
            
            # Calculate average of normalized scores
            avg_normalized_score = mean(normalized_scores)
            
            push!(run_overall_scores, Dict(
                :key => key,
                :timestamp => key[1],
                :run_dir => key[2],
                :avg_score => avg_normalized_score,
                :raw_scores => scores,
                :normalized_scores => normalized_scores
            ))
        end
    end
    
    # Sort by overall performance
    if direction == "best"
        sorted_overall = sort(run_overall_scores, by=x->x[:avg_score], rev=true)
        best_overall = sorted_overall[1]
        println("🏆 BEST OVERALL RUN ($(mode1) vs $(mode2)):")
    else
        sorted_overall = sort(run_overall_scores, by=x->x[:avg_score])
        best_overall = sorted_overall[1]
        println("📉 WORST OVERALL RUN ($(mode1) vs $(mode2)):")
    end
    
    println("  Run: $(best_overall[:run_dir]) ($(best_overall[:timestamp]))")
    println("  Average normalized score: $(round(best_overall[:avg_score], digits=3))")
    println("  Individual metric differences (raw | normalized):")
    
    # Show individual metrics for the best overall run
    best_key = best_overall[:key]
    mode1_run = mode1_lookup[best_key]
    mode2_run = mode2_lookup[best_key]
    
    for (i, metric) in enumerate(all_metrics)
        if haskey(mode1_run[:metrics], metric) && haskey(mode2_run[:metrics], metric)
            raw_diff = best_overall[:raw_scores][i]
            normalized_diff = best_overall[:normalized_scores][i]
            metric_def = METRIC_DEFINITIONS[metric]
            winner = raw_diff >= 0 ? mode1 : mode2
            
            raw_str = raw_diff >= 0 ? "+$(round(raw_diff, digits=3))" : "$(round(raw_diff, digits=3))"
            norm_str = normalized_diff >= 0 ? "+$(round(normalized_diff, digits=3))" : "$(round(normalized_diff, digits=3))"
            
            println("    $(metric_def.name): $(raw_str) | $(norm_str) ($(winner) wins)")
        end
    end
    
    # Show ranking of all runs by overall performance
    println("\n📋 ALL RUNS RANKED BY OVERALL PERFORMANCE:")
    for (i, run_score) in enumerate(sorted_overall)
        if direction == "best"
            status = i == 1 ? "← BEST OVERALL" : ""
        else
            status = i == 1 ? "← WORST OVERALL" : ""
        end
        
        diff_sign = run_score[:avg_score] >= 0 ? "+" : ""
        winner = run_score[:avg_score] >= 0 ? mode1 : mode2
        println("  $(i). $(run_score[:run_dir]) ($(run_score[:timestamp])): $(diff_sign)$(round(run_score[:avg_score], digits=3)) ($(winner) wins overall) $(status)")
    end
    
    # Summary table by metric
    println("\n📊 SUMMARY TABLE BY METRIC:")
    println("-"^80)
    println(rpad("Metric", 25) * "| " * rpad("Mode1 Avg", 10) * "| " * rpad("Mode2 Avg", 10) * "| " * rpad("Avg Diff", 10) * "| Winner")
    println("-"^80)
    
    for metric in all_metrics
        if metric_summaries[metric] !== nothing
            summary = metric_summaries[metric]
            metric_name = METRIC_DEFINITIONS[metric].name
            mode1_avg = round(summary[:mode1_avg], digits=2)
            mode2_avg = round(summary[:mode2_avg], digits=2)
            avg_diff = round(summary[:avg_difference], digits=3)
            winner = avg_diff >= 0 ? string(mode1) : string(mode2)
            
            # Truncate metric name if too long
            display_name = length(metric_name) > 24 ? metric_name[1:21] * "..." : metric_name
            diff_str = avg_diff >= 0 ? "+$(avg_diff)" : "$(avg_diff)"
            
            println(rpad(display_name, 25) * "| " * 
                   rpad(string(mode1_avg), 10) * "| " * 
                   rpad(string(mode2_avg), 10) * "| " * 
                   rpad(diff_str, 10) * "| $(winner)")
        end
    end
    println("-"^80)
    
    return Dict(
        :metric_summaries => metric_summaries,
        :overall_best => best_overall,
        :all_runs_ranked => sorted_overall
    )
end

"""
Compare two planning modes and find runs with best/worst differences
"""
function compare_planning_modes(mode1::Symbol, mode2::Symbol, target_metric::Symbol, direction::String="best")
    # Validate direction
    if !(direction in ["best", "worst"])
        println("❌ Invalid direction: $(direction). Must be 'best' or 'worst'")
        return nothing
    end
    
    direction_emoji = direction == "best" ? "🏆" : "📉"
    direction_word = direction == "best" ? "best" : "worst"
    
    println("$(direction_emoji) Comparing planning modes to find $(direction_word) differences:")
    println("  Mode 1: $(mode1)")
    println("  Mode 2: $(mode2)")
    println("  Metric: $(target_metric)")
    println("  Direction: Finding runs where $(mode1) performs $(direction_word) relative to $(mode2)")
    println("="^80)
    
    # Check if metric is valid
    if !haskey(METRIC_DEFINITIONS, target_metric)
        println("❌ Unknown metric: $(target_metric)")
        println("\n📋 Available metrics:")
        for (metric, def) in METRIC_DEFINITIONS
            direction_desc = def.higher_is_better ? "(higher is better)" : "(lower is better)"
            println("  - $(metric): $(def.name) $(direction_desc)")
        end
        println("  - all: Analyze all metrics simultaneously")
        return nothing
    end
    
    metric_def = METRIC_DEFINITIONS[target_metric]
    
    # Collect run data for both modes
    println("\n🔍 Collecting data for $(mode1)...")
    mode1_results = collect_run_data_for_mode(mode1)
    
    println("\n🔍 Collecting data for $(mode2)...")
    mode2_results = collect_run_data_for_mode(mode2)
    
    if isempty(mode1_results)
        println("❌ No runs found for planning mode: $(mode1)")
        return nothing
    end
    
    if isempty(mode2_results)
        println("❌ No runs found for planning mode: $(mode2)")
        return nothing
    end
    
    # Create lookup dictionaries by run directory and timestamp
    mode1_lookup = Dict()
    mode2_lookup = Dict()
    
    for run in mode1_results
        if haskey(run[:metrics], target_metric)
            key = (run[:timestamp], run[:run_dir])
            mode1_lookup[key] = run
        end
    end
    
    for run in mode2_results
        if haskey(run[:metrics], target_metric)
            key = (run[:timestamp], run[:run_dir])
            mode2_lookup[key] = run
        end
    end
    
    # Find matching runs (same timestamp and run directory)
    comparisons = []
    for key in keys(mode1_lookup)
        if haskey(mode2_lookup, key)
            mode1_run = mode1_lookup[key]
            mode2_run = mode2_lookup[key]
            
            value1 = mode1_run[:metrics][target_metric]
            value2 = mode2_run[:metrics][target_metric]
            
            # Calculate performance difference (positive means mode1 is better)
            difference = calculate_performance_difference(value1, value2, target_metric)
            
            push!(comparisons, Dict(
                :timestamp => key[1],
                :run_dir => key[2],
                :mode1_value => value1,
                :mode2_value => value2,
                :difference => difference,
                :mode1_run => mode1_run,
                :mode2_run => mode2_run
            ))
        end
    end
    
    if isempty(comparisons)
        println("❌ No matching runs found between $(mode1) and $(mode2)")
        println("   Make sure both planning modes have been run on the same run directories")
        return nothing
    end
    
    println("\n📊 Found $(length(comparisons)) matching runs for comparison")
    
    # Sort comparisons based on direction
    if direction == "best"
        # Best means largest positive difference (mode1 >> mode2)
        sorted_comparisons = sort(comparisons, by=x->x[:difference], rev=true)
        target_comparison = sorted_comparisons[1]
        comparison_desc = "$(mode1) performs BEST relative to $(mode2)"
    else
        # Worst means most negative difference (mode1 << mode2)
        sorted_comparisons = sort(comparisons, by=x->x[:difference])
        target_comparison = sorted_comparisons[1]
        comparison_desc = "$(mode1) performs WORST relative to $(mode2)"
    end
    
    # Calculate statistics
    differences = [comp[:difference] for comp in comparisons]
    avg_difference = mean(differences)
    std_difference = std(differences)
    
    mode1_values = [comp[:mode1_value] for comp in comparisons]
    mode2_values = [comp[:mode2_value] for comp in comparisons]
    
    println("\n🔍 COMPARISON RESULTS:")
    println("="^50)
    println("Planning Modes: $(mode1) vs $(mode2)")
    println("Metric: $(metric_def.name)")
    println("Direction: $(metric_def.higher_is_better ? "Higher is better" : "Lower is better")")
    println("Total Matching Runs: $(length(comparisons))")
    println()
    println("📊 STATISTICS:")
    println("  $(mode1) average: $(round(mean(mode1_values), digits=3))")
    println("  $(mode2) average: $(round(mean(mode2_values), digits=3))")
    println("  Average difference: $(round(avg_difference, digits=3)) (positive = $(mode1) better)")
    println("  Std deviation of differences: $(round(std_difference, digits=3))")
    println()
    println("$(direction_emoji) $(uppercase(direction_word)) CASE ($(comparison_desc)):")
    println("  Timestamp: $(target_comparison[:timestamp])")
    println("  Run Directory: $(target_comparison[:run_dir])")
    println("  $(mode1) $(metric_def.name): $(target_comparison[:mode1_value])")
    println("  $(mode2) $(metric_def.name): $(target_comparison[:mode2_value])")
    println("  Difference: $(round(target_comparison[:difference], digits=3)) ($(mode1) - $(mode2))")
    
    # Calculate percentage difference if applicable
    if target_comparison[:mode2_value] != 0
        pct_diff = (target_comparison[:difference] / abs(target_comparison[:mode2_value])) * 100
        println("  Percentage difference: $(round(pct_diff, digits=1))%")
    end
    
    println("  $(mode1) metrics file: $(target_comparison[:mode1_run][:metric_file])")
    println("  $(mode2) metrics file: $(target_comparison[:mode2_run][:metric_file])")
    
    # Show all metrics for both runs in the target case
    println("\n📊 ALL METRICS FOR $(uppercase(direction_word)) CASE:")
    println("$(mode1):")
    for (metric, value) in target_comparison[:mode1_run][:metrics]
        if haskey(METRIC_DEFINITIONS, metric)
            def = METRIC_DEFINITIONS[metric]
            marker = metric == target_metric ? "← TARGET METRIC" : ""
            println("  $(def.name): $(value) $(marker)")
        end
    end
    
    println("\n$(mode2):")
    for (metric, value) in target_comparison[:mode2_run][:metrics]
        if haskey(METRIC_DEFINITIONS, metric)
            def = METRIC_DEFINITIONS[metric]
            marker = metric == target_metric ? "← TARGET METRIC" : ""
            println("  $(def.name): $(value) $(marker)")
        end
    end
    
    # Show ranking of all comparisons
    println("\n📋 ALL COMPARISONS (sorted by difference, best $(mode1) performance first):")
    for (i, comp) in enumerate(sort(comparisons, by=x->x[:difference], rev=true))
        status = ""
        if direction == "best" && i == 1
            status = "← BEST CASE"
        elseif direction == "worst" && comp == target_comparison
            status = "← WORST CASE"
        end
        
        diff_sign = comp[:difference] >= 0 ? "+" : ""
        winner = comp[:difference] >= 0 ? mode1 : mode2
        println("  $(i). $(comp[:run_dir]) ($(comp[:timestamp])): $(diff_sign)$(round(comp[:difference], digits=3)) ($(winner) wins) $(status)")
    end
    
    return target_comparison
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

function main()
    # Parse command line arguments
    mode1 = DEFAULT_MODE1
    mode2 = DEFAULT_MODE2
    target_metric = DEFAULT_METRIC
    direction = DEFAULT_DIRECTION
    
    if length(ARGS) > 0
        mode1 = Symbol(ARGS[1])
    end
    
    if length(ARGS) > 1
        mode2 = Symbol(ARGS[2])
    end
    
    if length(ARGS) > 2
        target_metric = Symbol(ARGS[3])
    end
    
    if length(ARGS) > 3
        direction = ARGS[4]
    end
    
    println("🚀 Comparing Planning Modes")
    println("Target run directories: $(join(TARGET_RUNS, ", "))")
    println("Mode 1: $(mode1)")
    println("Mode 2: $(mode2)")
    println("Target metric: $(target_metric)")
    println("Direction: $(direction)")
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
    
    # Compare the planning modes
    if target_metric == :all
        result = compare_all_metrics(mode1, mode2, direction)
    else
        result = compare_planning_modes(mode1, mode2, target_metric, direction)
    end
    
    if result !== nothing
        println("\n✅ Comparison completed!")
        if target_metric != :all
            println("📄 Use the metrics file paths above to examine the detailed results.")
        end
    else
        println("\n❌ Comparison failed. Check the planning mode names and ensure matching runs exist.")
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

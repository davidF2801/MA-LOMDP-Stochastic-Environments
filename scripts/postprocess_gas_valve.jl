#!/usr/bin/env julia

"""
Postprocessing script for the Gas-Valve Coordination results.

Reads every `results/gas_valve_run_<timestamp>/Run N/<mode>/metrics/`
folder listed in `TARGET_RUNS`, aggregates the per-run metrics saved
by `main_gas_valve.jl`, and produces comparison plots + a summary CSV.

Expected per-run artifacts (written by `main_gas_valve.jl`):
  * performance_metrics_<mode>_run<N>.txt   - scalar summary stats
  * reward_log_<mode>_run<N>.csv            - per-step timeseries
  * action_history_<mode>_run<N>.csv        - joint action log

If `TARGET_RUNS` is empty, the script auto-picks the latest
`gas_valve_run_*` directory under `results/`.
"""

using Statistics
using Plots
using Dates
using DataFrames
using CSV
using StatsPlots
using Printf

gr()
ENV["GKSwstype"] = "nul"
ENV["GR_COLORSPACE"] = "sRGB"
default(palette = :default)

println("Gas-Valve postprocessing starting...")

# =============================================================================
# CONFIGURATION
# =============================================================================

# List the timestamped batches you want combined in a single analysis. If
# left empty, the latest `gas_valve_run_*` directory is auto-selected.
const TARGET_RUNS = String[
    "gas_valve_run_2026-04-18T15-40-47-663",
]

const RESULTS_BASE = joinpath(@__DIR__, "..", "results")

const PLANNING_MODES = [:public_belief, :do_sb_abba, :do_alpha, :pomcp_independent, :random]

function get_mode_display_name(mode::Symbol)
    if mode == :public_belief
        return "Public belief\n(joint B_clean)"
    elseif mode == :do_sb_abba
        return "DO-SB-ABBA-α"
    elseif mode == :do_alpha
        return "DO-α\n(1-step blend)"
    elseif mode == :pomcp_independent
        return "Indep. POMCP"
    elseif mode == :random
        return "Random"
    end
    return replace(string(mode), "_" => " ")
end

function mode_color(mode::Symbol)
    mode == :public_belief     && return :steelblue
    mode == :do_sb_abba        && return :seagreen
    mode == :do_alpha          && return :mediumpurple
    mode == :pomcp_independent && return :orange
    mode == :random            && return :gray
    return :purple
end

# =============================================================================
# FILE DISCOVERY
# =============================================================================
function pick_target_runs()::Vector{String}
    if !isempty(TARGET_RUNS)
        return TARGET_RUNS
    end
    isdir(RESULTS_BASE) || error("Results directory not found: $(RESULTS_BASE)")
    candidates = String[]
    for item in readdir(RESULTS_BASE)
        isdir(joinpath(RESULTS_BASE, item)) || continue
        startswith(item, "gas_valve_run_") || continue
        push!(candidates, item)
    end
    isempty(candidates) && error("No gas_valve_run_* directories in $(RESULTS_BASE)")
    sort!(candidates)
    println("Auto-selected latest batch: $(candidates[end])")
    return [candidates[end]]
end

function find_run_dirs(batch_dir::String)::Vector{String}
    out = String[]
    for item in readdir(batch_dir)
        path = joinpath(batch_dir, item)
        if isdir(path) && (startswith(item, "Run") || startswith(item, "run"))
            push!(out, item)
        end
    end
    return sort(out)
end

# =============================================================================
# PARSING
# =============================================================================

"""Extract scalar stats from `performance_metrics_*.txt`."""
function parse_perf_txt(filepath::String)::Dict{Symbol, Float64}
    out = Dict{Symbol, Float64}()
    isfile(filepath) || return out

    re_map = [
        (r"Total reward:\s*([-+\d\.eE]+)",                 :total_reward),
        (r"Mean per-step reward:\s*([-+\d\.eE]+)",         :mean_reward),
        (r"Total successful Seal\+Vent operations:\s*(\d+)", :n_success_vent),
        (r"Total explosions triggered:\s*(\d+)",           :n_explosions),
        (r"Final pockets exploded:\s*(\d+)",               :n_final_exploded),
        (r"Seal actions issued:\s*(\d+)",                  :n_seal),
        (r"Vent actions issued:\s*(\d+)",                  :n_vent),
        (r"Observe actions issued:\s*(\d+)",               :n_observe),
        (r"Mean rover-rover distance:\s*([-+\d\.eE]+)",    :mean_rover_distance),
        (r"Mean planning time per step:\s*([-+\d\.eE]+)\s*ms", :mean_plan_ms),
        (r"Std  planning time per step:\s*([-+\d\.eE]+)\s*ms", :std_plan_ms),
        (r"Max  planning time per step:\s*([-+\d\.eE]+)\s*ms", :max_plan_ms),
        (r"Total planning time:\s*([-+\d\.eE]+)\s*seconds", :total_plan_s),
        (r"Steps simulated:\s*(\d+)",                      :steps_simulated),
    ]

    try
        for line in readlines(filepath)
            line = strip(line)
            for (re, key) in re_map
                m = match(re, line)
                if m !== nothing
                    out[key] = parse(Float64, m.captures[1])
                end
            end
        end
    catch e
        @warn "Could not parse $(filepath): $(e)"
    end
    return out
end

"""Return `DataFrame` of the per-step reward log (or empty DF if missing)."""
function parse_reward_log_csv(filepath::String)::DataFrame
    isfile(filepath) || return DataFrame()
    try
        return DataFrame(CSV.File(filepath))
    catch e
        @warn "Could not parse reward log $(filepath): $(e)"
        return DataFrame()
    end
end

# =============================================================================
# DATA COLLECTION
# =============================================================================

"""
    RunRecord: one completed simulation (one mode, one run, one batch).
"""
struct RunRecord
    batch::String
    run_label::String
    mode::Symbol
    scalars::Dict{Symbol, Float64}
    timeseries::DataFrame
end

function collect_all_records(target_runs::Vector{String})::Vector{RunRecord}
    records = RunRecord[]
    for batch in target_runs
        batch_dir = joinpath(RESULTS_BASE, batch)
        if !isdir(batch_dir)
            @warn "Skipping missing batch: $(batch_dir)"
            continue
        end
        println("Scanning: $(batch)")
        for run_label in find_run_dirs(batch_dir)
            run_dir = joinpath(batch_dir, run_label)
            for mode in PLANNING_MODES
                metrics_dir = joinpath(run_dir, string(mode), "metrics")
                isdir(metrics_dir) || continue

                # scalar metrics
                perf_files = filter(f -> startswith(f, "performance_metrics_") &&
                                         endswith(f, ".txt"),
                                    readdir(metrics_dir))
                isempty(perf_files) && continue
                scalars = parse_perf_txt(joinpath(metrics_dir, first(perf_files)))

                # timeseries (reward_log_*.csv)
                log_files = filter(f -> startswith(f, "reward_log_") &&
                                        endswith(f, ".csv"),
                                   readdir(metrics_dir))
                ts = isempty(log_files) ? DataFrame() :
                    parse_reward_log_csv(joinpath(metrics_dir, first(log_files)))

                push!(records, RunRecord(batch, run_label, mode, scalars, ts))
            end
        end
    end
    println("Total records collected: $(length(records))")
    return records
end

# =============================================================================
# PLOT HELPERS
# =============================================================================

function _savefig_both(p, out_path_pdf::String)
    mkpath(dirname(out_path_pdf))
    png_path = replace(out_path_pdf, r"\.pdf$" => ".png")
    try
        savefig(p, out_path_pdf)
    catch e
        @warn "PDF save failed for $(out_path_pdf): $(e)"
    end
    try
        savefig(p, png_path)
    catch e
        @warn "PNG save failed for $(png_path): $(e)"
    end
    println("  Saved: $(basename(out_path_pdf)) & $(basename(png_path))")
end

function per_mode_values(records::Vector{RunRecord}, key::Symbol)
    out = Dict{Symbol, Vector{Float64}}()
    for mode in PLANNING_MODES
        out[mode] = Float64[]
        for r in records
            r.mode == mode || continue
            if haskey(r.scalars, key)
                push!(out[mode], r.scalars[key])
            end
        end
    end
    return out
end

function scalar_boxplot(records::Vector{RunRecord}, key::Symbol,
                        title_str::String, ylabel_str::String)
    data_by_mode = per_mode_values(records, key)
    labels = String[]
    data   = Vector{Vector{Float64}}()
    colors = Symbol[]
    for mode in PLANNING_MODES
        vals = get(data_by_mode, mode, Float64[])
        isempty(vals) && continue
        push!(labels, get_mode_display_name(mode))
        push!(data, vals)
        push!(colors, mode_color(mode))
    end
    isempty(data) && return nothing

    p = plot(title = title_str, ylabel = ylabel_str,
             xlabel = "", legend = false, grid = true, gridalpha = 0.3)
    for (i, vals) in enumerate(data)
        boxplot!(p, fill(i, length(vals)), vals, color = colors[i],
                 alpha = 0.75, markerstrokecolor = :black, label = "")
        scatter!(p, fill(i, length(vals)) .+ 0.15 .* randn(length(vals)), vals,
                 color = :black, markersize = 3, label = "", alpha = 0.6)
    end
    xticks!(p, (1:length(labels), labels))
    return p
end

function scalar_bar(records::Vector{RunRecord}, key::Symbol,
                    title_str::String, ylabel_str::String)
    data_by_mode = per_mode_values(records, key)
    labels = String[]
    means  = Float64[]
    stds   = Float64[]
    colors = Symbol[]
    for mode in PLANNING_MODES
        vals = get(data_by_mode, mode, Float64[])
        isempty(vals) && continue
        push!(labels, get_mode_display_name(mode))
        push!(means, mean(vals))
        push!(stds, length(vals) > 1 ? std(vals) : 0.0)
        push!(colors, mode_color(mode))
    end
    isempty(means) && return nothing

    p = bar(1:length(means), means, yerror = stds, color = colors,
            legend = false, grid = true, gridalpha = 0.3,
            title = title_str, ylabel = ylabel_str, xlabel = "")
    xticks!(p, (1:length(labels), labels))
    return p
end

"""
Aggregate variable-length per-step vectors into (mean, std) series.
Shorter vectors are padded with NaN and ignored in the stats.
"""
function aggregate_var_length(series::Vector{Vector{Float64}})
    isempty(series) && return Float64[], Float64[]
    maxlen = maximum(length, series)
    padded = [vcat(s, fill(NaN, maxlen - length(s))) for s in series]
    mat = hcat(padded...)
    means = Float64[]
    stds  = Float64[]
    for i in 1:maxlen
        row = mat[i, :]
        clean = row[.!isnan.(row)]
        if isempty(clean)
            push!(means, NaN); push!(stds, NaN)
        else
            push!(means, mean(clean))
            push!(stds, length(clean) > 1 ? std(clean) : 0.0)
        end
    end
    return means, stds
end

function collect_timeseries(records::Vector{RunRecord}, col::Symbol)
    out = Dict{Symbol, Vector{Vector{Float64}}}()
    for mode in PLANNING_MODES
        out[mode] = Vector{Vector{Float64}}()
        for r in records
            r.mode == mode || continue
            nrow(r.timeseries) == 0 && continue
            col in propertynames(r.timeseries) || continue
            push!(out[mode], Float64.(r.timeseries[!, col]))
        end
    end
    return out
end

function timeseries_plot(records::Vector{RunRecord}, col::Symbol,
                         title_str::String, ylabel_str::String)
    series_by_mode = collect_timeseries(records, col)
    p = plot(title = title_str, xlabel = "Timestep", ylabel = ylabel_str,
             legend = :outertopright, grid = true, gridalpha = 0.3)
    any_plotted = false
    for mode in PLANNING_MODES
        ss = get(series_by_mode, mode, Vector{Vector{Float64}}())
        isempty(ss) && continue
        means, stds = aggregate_var_length(ss)
        t = 0:(length(means) - 1)
        plot!(p, t, means, ribbon = stds, fillalpha = 0.2, linewidth = 2.2,
              color = mode_color(mode),
              label = get_mode_display_name(mode))
        any_plotted = true
    end
    return any_plotted ? p : nothing
end

# =============================================================================
# SUMMARY CSV
# =============================================================================

function write_summary_csv(records::Vector{RunRecord}, output_dir::String)
    mkpath(output_dir)
    rows = NamedTuple[]
    for r in records
        get_f(k) = get(r.scalars, k, NaN)
        push!(rows, (
            batch                 = r.batch,
            run                   = r.run_label,
            mode                  = string(r.mode),
            steps_simulated       = get_f(:steps_simulated),
            total_reward          = get_f(:total_reward),
            mean_reward           = get_f(:mean_reward),
            n_success_vent        = get_f(:n_success_vent),
            n_explosions          = get_f(:n_explosions),
            n_final_exploded      = get_f(:n_final_exploded),
            n_seal                = get_f(:n_seal),
            n_vent                = get_f(:n_vent),
            n_observe             = get_f(:n_observe),
            mean_rover_distance   = get_f(:mean_rover_distance),
            mean_plan_time_ms     = get_f(:mean_plan_ms),
            std_plan_time_ms      = get_f(:std_plan_ms),
            max_plan_time_ms      = get_f(:max_plan_ms),
            total_plan_time_s     = get_f(:total_plan_s),
        ))
    end
    filepath = joinpath(output_dir, "gas_valve_summary.csv")
    CSV.write(filepath, DataFrame(rows))
    println("Wrote summary: $(filepath)")
    return filepath
end

function print_mode_table(records::Vector{RunRecord})
    println("\n" * "="^80)
    println("MEAN ± STD BY MODE")
    println("="^80)
    header = @sprintf "%-22s %12s %12s %12s %12s %10s" "Mode" "TotalReward" "Success" "Explosions" "MeanDist" "PlanMs"
    println(header)
    println("-"^80)
    for mode in PLANNING_MODES
        subset = filter(r -> r.mode == mode, records)
        isempty(subset) && continue
        tr = [get(r.scalars, :total_reward, NaN)       for r in subset]
        sv = [get(r.scalars, :n_success_vent, NaN)     for r in subset]
        ex = [get(r.scalars, :n_explosions, NaN)       for r in subset]
        md = [get(r.scalars, :mean_rover_distance, NaN) for r in subset]
        pm = [get(r.scalars, :mean_plan_ms, NaN)       for r in subset]
        fmt(v) = isempty(v) || all(isnan, v) ? "n/a" :
            @sprintf "%6.2f±%5.2f" mean(filter(!isnan, v)) (length(filter(!isnan, v)) > 1 ? std(filter(!isnan, v)) : 0.0)
        println(@sprintf "%-22s %12s %12s %12s %12s %10s" get_mode_display_name(mode) fmt(tr) fmt(sv) fmt(ex) fmt(md) fmt(pm))
    end
    println("="^80)
end

# =============================================================================
# MAIN
# =============================================================================

target_runs = pick_target_runs()
records = collect_all_records(target_runs)

if isempty(records)
    error("No records collected. Check TARGET_RUNS and results layout.")
end

output_dir = joinpath(RESULTS_BASE, target_runs[end], "postprocess")
mkpath(output_dir)
println("Output: $(output_dir)")

println("\nGenerating plots...")

# Boxplots of scalar metrics (one run == one dot)
p1 = scalar_boxplot(records, :total_reward,
                    "Total reward per mode (per run)", "Total reward")
p1 !== nothing && _savefig_both(p1, joinpath(output_dir, "total_reward_boxplot.pdf"))

p2 = scalar_bar(records, :n_success_vent,
                "Successful Seal+Vent operations (mean ± std)",
                "# successful vents")
p2 !== nothing && _savefig_both(p2, joinpath(output_dir, "success_vent_bar.pdf"))

p3 = scalar_bar(records, :n_explosions,
                "Triggered explosions (mean ± std)", "# explosions")
p3 !== nothing && _savefig_both(p3, joinpath(output_dir, "explosions_bar.pdf"))

p4 = scalar_bar(records, :mean_plan_ms,
                "Mean per-step planning time (ms)", "Planning time [ms]")
p4 !== nothing && _savefig_both(p4, joinpath(output_dir, "planning_time_bar.pdf"))

p5 = scalar_bar(records, :mean_rover_distance,
                "Mean rover-rover Manhattan distance", "Distance [cells]")
p5 !== nothing && _savefig_both(p5, joinpath(output_dir, "rover_distance_bar.pdf"))

# Action mix bars
for (key, ttl, fname) in (
        (:n_seal,    "Seal actions issued",    "seal_actions_bar.pdf"),
        (:n_vent,    "Vent actions issued",    "vent_actions_bar.pdf"),
        (:n_observe, "Observe actions issued", "observe_actions_bar.pdf"),
    )
    pp = scalar_bar(records, key, "$(ttl) (mean ± std)", "# actions")
    pp !== nothing && _savefig_both(pp, joinpath(output_dir, fname))
end

# Timeseries plots
pt1 = timeseries_plot(records, :step_reward,
                      "Per-step team reward (mean ± std)", "r_t")
pt1 !== nothing && _savefig_both(pt1, joinpath(output_dir, "reward_vs_time.pdf"))

pt2 = timeseries_plot(records, :cumulative_reward,
                      "Cumulative team reward", "Σ r_t")
pt2 !== nothing && _savefig_both(pt2, joinpath(output_dir, "cum_reward_vs_time.pdf"))

pt3 = timeseries_plot(records, :cumulative_disc_reward,
                      "Cumulative discounted team reward", "Σ γᵗ r_t")
pt3 !== nothing && _savefig_both(pt3, joinpath(output_dir, "cum_disc_reward_vs_time.pdf"))

pt4 = timeseries_plot(records, :planning_time_ms,
                      "Per-step planning time [ms]", "ms")
pt4 !== nothing && _savefig_both(pt4, joinpath(output_dir, "planning_time_vs_time.pdf"))

pt5 = timeseries_plot(records, :cumulative_success_vent,
                      "Cumulative successful Seal+Vent", "count")
pt5 !== nothing && _savefig_both(pt5, joinpath(output_dir, "cum_success_vent_vs_time.pdf"))

pt6 = timeseries_plot(records, :cumulative_explosions,
                      "Cumulative explosions", "count")
pt6 !== nothing && _savefig_both(pt6, joinpath(output_dir, "cum_explosions_vs_time.pdf"))

pt7 = timeseries_plot(records, :n_critical_pockets,
                      "Number of CRITICAL pockets", "count")
pt7 !== nothing && _savefig_both(pt7, joinpath(output_dir, "critical_pockets_vs_time.pdf"))

pt8 = timeseries_plot(records, :rover_distance,
                      "Rover-rover distance over time", "Manhattan cells")
pt8 !== nothing && _savefig_both(pt8, joinpath(output_dir, "rover_distance_vs_time.pdf"))

# Summary CSV + console table
write_summary_csv(records, output_dir)
print_mode_table(records)

println("\nGas-Valve postprocessing done.")
println("Artifacts in: $(output_dir)")

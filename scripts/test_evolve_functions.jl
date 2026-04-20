# Simple test for evolve_no_obs vs evolve_no_obs_fast
# This test directly includes the necessary files without loading the full project

# Add the src directory to the path
push!(LOAD_PATH, "../src")

# Include the necessary modules directly
include("../src/types/Types.jl")
include("../src/agents/Agents.jl")
include("../src/environment/Environment.jl")

using .Types
using .Environment
using .Agents.BeliefManagement
using Plots
using Statistics

# Import specific functions from Types
import .Types: create_uniform_rsp_maps, RSPParameterMaps

# Import both evolve functions from BeliefManagement
import .Agents.BeliefManagement: evolve_no_obs, evolve_no_obs_fast, clear_belief_evolution_cache!,
    calculate_uncertainty_map_from_distributions

# Test environment: parameters tuned so belief evolves a lot (strong contagion, moderate ignition/persistence)
function create_test_env()
    width, height = 3, 3
    event_dynamics = EventDynamics(0.15, 0.25, 0.4, 0.15, 0.6)
    agents = Agent[]
    sensor_range = 2.0
    discount = 0.95
    initial_events = 1
    max_sensing_targets = 1
    ground_station_pos = (1, 1)
    rsp_params = create_uniform_rsp_maps(3, 3, lambda=0.15, beta0=0.22, alpha=0.85, delta=0.55)
    env = SpatialGrid(width, height, event_dynamics, agents, sensor_range, discount, initial_events, max_sensing_targets, ground_station_pos, rsp_params)
    return env
end

# Create a test belief: 2-state per cell (P(no event), P(event)), same for all cells.
function create_test_belief()
    # 2-state belief: state 1 = NO_EVENT, state 2 = EVENT_PRESENT
    event_distributions = zeros(2, 3, 3)
    # Start with 0.5, 0.5 (maximum uncertainty) for every cell
    event_distributions[1, :, :] .= 0.5  # P(NO_EVENT)
    event_distributions[2, :, :] .= 0.5  # P(EVENT_PRESENT)

    # Create uncertainty map
    uncertainty_map = calculate_uncertainty_map_from_distributions(event_distributions)

    return Belief(event_distributions, uncertainty_map, 0, [])
end

# Test function
function test_evolve_functions()
    println("🧪 Testing evolve_no_obs vs evolve_no_obs_fast (10 iterations)")
    
    # Create test environment and belief
    env = create_test_env()
    belief = create_test_belief()
    
    println("Initial belief shape: $(size(belief.event_distributions))")
    println("Initial event probabilities:")
    println(belief.event_distributions[2, :, :])
    
    # Clear cache to ensure fair comparison
    clear_belief_evolution_cache!()
    
    # Test original function - 100 iterations for better timing
    println("\n📊 Testing evolve_no_obs (100 iterations)...")
    start_time = time()
    result_original = deepcopy(belief)
    for i in 1:100
        result_original = evolve_no_obs(result_original, env, calculate_uncertainty=true)
    end
    original_time = time() - start_time
    
    # Clear cache again and create a fresh belief to avoid cache effects
    clear_belief_evolution_cache!()
    belief_fresh = deepcopy(belief)
    
    # Test fast function - 100 iterations (no cache)
    println("📊 Testing evolve_no_obs_fast (100 iterations)...")
    start_time = time()
    result_fast = deepcopy(belief_fresh)
    for i in 1:100
        if i % 10 == 0
            println("  Iteration $i...")
        end
        iter_start = time()
        result_fast = evolve_no_obs_fast(result_fast, env, calculate_uncertainty=true)
        iter_time = time() - iter_start
        if i % 10 == 0
            println("    Iteration $i took: $(round(iter_time * 1000, digits=6)) ms")
        end
    end
    fast_time = time() - start_time
    
    # Compare results
    println("\n🔍 Results comparison:")
    println("Original time: $(round(original_time * 1000, digits=6)) ms")
    println("Fast time: $(round(fast_time * 1000, digits=6)) ms")
    println("Speedup: $(round(original_time / fast_time, digits=2))x")
    
    # Also test with @elapsed for more precision
    println("\n📊 Precise timing with @elapsed:")
    clear_belief_evolution_cache!()
    belief_test = deepcopy(belief)
    original_elapsed = @elapsed for i in 1:100
        belief_test = evolve_no_obs(belief_test, env, calculate_uncertainty=true)
    end
    
    clear_belief_evolution_cache!()
    belief_test = deepcopy(belief)
    fast_elapsed = @elapsed for i in 1:100
        belief_test = evolve_no_obs_fast(belief_test, env, calculate_uncertainty=true)
    end
    
    println("Original @elapsed: $(round(original_elapsed * 1000, digits=6)) ms")
    println("Fast @elapsed: $(round(fast_elapsed * 1000, digits=6)) ms")
    println("Speedup @elapsed: $(round(original_elapsed / fast_elapsed, digits=2))x")
    
    # Compare event distributions
    dist_diff = abs.(result_original.event_distributions - result_fast.event_distributions)
    max_dist_diff = maximum(dist_diff)
    mean_dist_diff = mean(dist_diff)
    
    println("\n📈 Distribution differences:")
    println("Max difference: $(round(max_dist_diff, digits=6))")
    println("Mean difference: $(round(mean_dist_diff, digits=6))")
    
    # Compare uncertainty maps
    uncert_diff = abs.(result_original.uncertainty_map - result_fast.uncertainty_map)
    max_uncert_diff = maximum(uncert_diff)
    mean_uncert_diff = mean(uncert_diff)
    
    println("\n📊 Uncertainty map differences:")
    println("Max difference: $(round(max_uncert_diff, digits=6))")
    println("Mean difference: $(round(mean_uncert_diff, digits=6))")
    
    # Check if results are essentially identical
    tolerance = 1e-06
    identical = max_dist_diff < tolerance && max_uncert_diff < tolerance
    
    if identical
        println("\n✅ SUCCESS: Functions produce identical results!")
    else
        println("\n❌ WARNING: Functions produce different results!")
        println("Original event probs:")
        println(result_original.event_distributions[2, :, :])
        println("Fast event probs:")
        println(result_fast.event_distributions[2, :, :])
    end
    
    return identical, original_time, fast_time
end

"""
Collect data over evolution steps for plotting: error (exact vs fast), timing, and belief evolution.
"""
function run_evolve_comparison_for_plots(; n_steps::Int=1000, n_timing_repeats::Int=5)
    env = create_test_env()
    steps = 1:n_steps
    errors_max = Float64[]
    errors_L1 = Float64[]
    mean_no_event_exact = Float64[]   # mean P(state 1) over grid
    mean_no_event_fast = Float64[]
    mean_event_prob_exact = Float64[] # mean P(state 2) over grid
    mean_event_prob_fast = Float64[]
    mean_uncertainty_exact = Float64[]
    mean_uncertainty_fast = Float64[]

    for k in steps
        # Same initial belief for both
        b0 = create_test_belief()
        b_exact = deepcopy(b0)
        b_fast = deepcopy(b0)

        # Evolve exact k steps
        clear_belief_evolution_cache!()
        for _ in 1:k
            b_exact = evolve_no_obs(b_exact, env, calculate_uncertainty=true)
        end

        # Evolve fast k steps
        for _ in 1:k
            b_fast = evolve_no_obs_fast(b_fast, env, calculate_uncertainty=true)
        end

        # Error
        diff_dist = abs.(b_exact.event_distributions - b_fast.event_distributions)
        push!(errors_max, maximum(diff_dist))
        push!(errors_L1, sum(diff_dist) / length(diff_dist))

        # Belief evolution: both dimensions (P(no event), P(event)) and mean uncertainty
        push!(mean_no_event_exact, mean(b_exact.event_distributions[1, :, :]))
        push!(mean_no_event_fast, mean(b_fast.event_distributions[1, :, :]))
        push!(mean_event_prob_exact, mean(b_exact.event_distributions[2, :, :]))
        push!(mean_event_prob_fast, mean(b_fast.event_distributions[2, :, :]))
        push!(mean_uncertainty_exact, mean(b_exact.uncertainty_map))
        push!(mean_uncertainty_fast, mean(b_fast.uncertainty_map))
    end

    # Timing: average time per single evolution over n_timing_repeats runs of n_steps each
    clear_belief_evolution_cache!()
    times_exact = Float64[]
    for _ in 1:n_timing_repeats
        b = deepcopy(create_test_belief())
        t = @elapsed for _ in 1:n_steps
            b = evolve_no_obs(b, env, calculate_uncertainty=true)
        end
        push!(times_exact, t / n_steps)
    end
    times_fast = Float64[]
    for _ in 1:n_timing_repeats
        b = deepcopy(create_test_belief())
        t = @elapsed for _ in 1:n_steps
            b = evolve_no_obs_fast(b, env, calculate_uncertainty=true)
        end
        push!(times_fast, t / n_steps)
    end
    t_exact = mean(times_exact)
    t_fast = mean(times_fast)

    return (; steps, errors_max, errors_L1,
            time_exact_per_evolve=t_exact, time_fast_per_evolve=t_fast,
            mean_no_event_exact, mean_no_event_fast,
            mean_event_prob_exact, mean_event_prob_fast,
            mean_uncertainty_exact, mean_uncertainty_fast)
end

# Plot style: larger fonts and thick lines for readability
const _PLOT_OPTS = (;
    size=(900, 560),
    titlefontsize=14,
    guidefontsize=12,
    tickfontsize=11,
    legendfontsize=11,
    linewidth=2.5,
    legend=:right,
    grid=true,
    minorgrid=false,
)

"""
Save comparison plots: error vs step, timing bar, belief evolution.
Clear colors, larger fonts, and one combined summary figure.
"""
function save_comparison_plots(data; outdir::String="evolve_comparison_plots")
    mkpath(outdir)
    steps = collect(data.steps)
    t_exact_ms = data.time_exact_per_evolve * 1000
    t_fast_ms = data.time_fast_per_evolve * 1000
    speedup = t_exact_ms / t_fast_ms

    # ---- 1) Error between exact and fast ----
    p1 = plot(steps, data.errors_max,
              label="Max difference (any cell)",
              color=:coral2, linewidth=2.5;
              _PLOT_OPTS...)
    plot!(steps, data.errors_L1,
          label="Average difference per cell",
          color=:steelblue, linewidth=2.5)
    plot!(xlabel="Evolution step", ylabel="Error  (exact vs fast)")
    title!("Approximation error: evolve_no_obs vs evolve_no_obs_fast")
    savefig(p1, joinpath(outdir, "error_vs_step.png"))

    # ---- 2) Timing: bar chart with value labels ----
    labels = ["Exact\n(evolve_no_obs)", "Fast\n(evolve_no_obs_fast)"]
    times_ms = [t_exact_ms, t_fast_ms]
    bar_colors = [:coral2, :seagreen]
    p2 = bar(labels, times_ms, color=bar_colors, legend=false; _PLOT_OPTS...)
    plot!(ylabel="Time per evolution (ms)", xlabel="")
    title!("Runtime: exact vs fast  (fast is $(round(speedup, digits=1))× faster)")
    # Annotate bar values above each bar
    y_max = maximum(times_ms)
    for (i, v) in enumerate(times_ms)
        annotate!(i, v + 0.04 * y_max, text("$(round(v, digits=2)) ms", 10, :center))
    end
    savefig(p2, joinpath(outdir, "time_comparison.png"))

    # ---- 3) Belief evolution: both dimensions (P(no event), P(event)); initial 0.5, 0.5 ----
    p3 = plot(steps, data.mean_no_event_exact,
              label="Exact P(no event)",
              color=:coral2, linewidth=2.5; _PLOT_OPTS...)
    plot!(steps, data.mean_event_prob_exact,
          label="Exact P(event)",
          color=:brown2, linewidth=2.5)
    plot!(steps, data.mean_no_event_fast,
          label="Fast P(no event)",
          color=:seagreen, linewidth=2.5, linestyle=:dash)
    plot!(steps, data.mean_event_prob_fast,
          label="Fast P(event)",
          color=:darkgreen, linewidth=2.5, linestyle=:dash)
    plot!(xlabel="Evolution step", ylabel="Mean probability")
    title!("Belief evolution (2-state, initial 0.5/0.5)")
    savefig(p3, joinpath(outdir, "belief_evolution_event_prob.png"))

    # ---- 4) Belief evolution: mean uncertainty ----
    p4 = plot(steps, data.mean_uncertainty_exact,
              label="Exact (evolve_no_obs)",
              color=:coral2, linewidth=2.5; _PLOT_OPTS...)
    plot!(steps, data.mean_uncertainty_fast,
          label="Fast (evolve_no_obs_fast)",
          color=:seagreen, linewidth=2.5, linestyle=:dash)
    plot!(xlabel="Evolution step", ylabel="Mean uncertainty (entropy)")
    title!("Belief evolution: mean uncertainty")
    savefig(p4, joinpath(outdir, "belief_evolution_uncertainty.png"))

    # ---- 5) Error and time only (single figure, two panels) ----
    p_error_only = plot(steps, data.errors_max,
                        label="Max difference", color=:coral2, linewidth=2.5; _PLOT_OPTS...)
    plot!(steps, data.errors_L1, label="Mean L1", color=:steelblue, linewidth=2.5)
    plot!(xlabel="Evolution step", ylabel="Error")
    title!("Error (exact vs fast)")
    p_time_only = bar(["Exact", "Fast"], [t_exact_ms, t_fast_ms], color=[:coral2, :seagreen], legend=false; _PLOT_OPTS...)
    plot!(ylabel="Time per evolution (ms)")
    title!("Time  ($(round(speedup, digits=1))× speedup)")
    for (i, v) in enumerate([t_exact_ms, t_fast_ms])
        annotate!(i, v + 0.04 * maximum([t_exact_ms, t_fast_ms]), text("$(round(v, digits=2)) ms", 10, :center))
    end
    p_error_time = plot(p_error_only, p_time_only, layout=(1, 2), size=(900, 420))
    savefig(p_error_time, joinpath(outdir, "error_and_time_only.png"))

    # ---- 6) One-page summary: 2×2 layout ----
    p_summary = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 900))
    savefig(p_summary, joinpath(outdir, "summary_evolve_comparison.png"))

    println("📁 Plots saved to $(outdir)/")
    return outdir
end

# Run the test
println("🚀 Running evolve function comparison test...")
identical, orig_time, fast_time = test_evolve_functions()

# Generate and save comparison plots
println("\n📊 Generating comparison plots...")
data = run_evolve_comparison_for_plots(n_steps=100, n_timing_repeats=5)
save_comparison_plots(data)

println("\n" * "="^50)
println("FINAL SUMMARY")
println("="^50)
println("Functions identical: $(identical ? "✅ YES" : "❌ NO")")
println("Speedup: $(round(orig_time / fast_time, digits=2))x")
println("Original time: $(round(orig_time * 1000, digits=2)) ms")
println("Fast time: $(round(fast_time * 1000, digits=2)) ms")
println("Time per evolution (exact): $(round(data.time_exact_per_evolve * 1000, digits=4)) ms")
println("Time per evolution (fast): $(round(data.time_fast_per_evolve * 1000, digits=4)) ms") 
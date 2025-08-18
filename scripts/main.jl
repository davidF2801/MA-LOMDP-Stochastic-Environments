#!/usr/bin/env julia

"""
Test script for RSP (Random Spread Process) with 3x2 grid and row-only visibility
Demonstrates exact world enumeration for macro-script evaluation
"""

println("🚀 RSP 3x2 Row Visibility Test starting...")

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Plots
using Dates  # Add this for timestamping
Plots.plotlyjs()
using Infiltrator

# Set random seed for reproducibility
#Random.seed!(42)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# 🎯 MAIN SIMULATION PARAMETERS
const NUM_STEPS = 100            # Total simulation steps
const PLANNING_MODE = :policy         # Use policy tree planning (:script, :policy, :random, :sweep, :greedy, :future_actions, :prior_based, :pbvi)
#const modes = [:pbvi, :script, :macro_approx_095, :macro_approx_090, :macro_approx_085]
const modes = [:macro_approx_090,:greedy, :pbvi, :script, :prior_based, :sweep, :random]
const N_RUNS = 200
const MAX_BATTERY = 10000.0
const CHARGING_RATE = 3.0
const OBSERVATION_COST = 0.0
# Planning modes:
#   :script - Exact belief evolution with macro-script planning
#   :policy - Policy tree planning
#   :random - Random action selection (baseline for comparison)
#   :sweep - Systematic sweep over columns in each row
#   :greedy - Greedy selection maximizing entropy * event_probability
#   :future_actions - Exact planning considering other agents' possible future actions
#   :macro_approx - Approximate macro-script planning with branch pruning (uses MAX_PROB_MASS threshold)
#   :macro_approx_099 - Approximate macro-script planning with 0.99 belief mass threshold
#   :macro_approx_097 - Approximate macro-script planning with 0.97 belief mass threshold
#   :macro_approx_095 - Approximate macro-script planning with 0.95 belief mass threshold
#   :prior_based - Prior-based planning using static probability map (no belief updates)
#   :pbvi - Point-Based Value Iteration planning

# 🌍 ENVIRONMENT PARAMETERS
const GRID_WIDTH = 3                  # Grid width (columns)
const GRID_HEIGHT = 4                 # Grid height (rows)
const INITIAL_EVENTS = 1              # Number of initial events
const MAX_SENSING_TARGETS = 1         # Maximum cells an agent can sense per step
const SENSOR_RANGE = 0.0              # Sensor range for agents (0.0 = row-only visibility)
const DISCOUNT_FACTOR = 0.95        # POMDP discount factor
const MAX_PROB_MASS = 0.98            # Maximum probability mass to keep when pruning belief branches (for macro_approx)

# 🤖 AGENT PARAMETERS
const NUM_AGENTS = 2                  # Number of agents (one per row)
const PLANNING_HORIZON = 4            # Planning horizon for macro-scripts
const SENSOR_FOV = pi/2               # Field of view angle (radians)
const SENSOR_NOISE = 0.0              # Perfect observations

# 📡 COMMUNICATION PARAMETERS
const CONTACT_HORIZON = 4             # Steps until next sync opportunity
const GROUND_STATION_X = 2            # Ground station X position
const GROUND_STATION_Y = 1            # Ground station Y position

# 🎯 REWARD FUNCTION CONFIGURATION
const ENTROPY_WEIGHT = 1.0            # w_H: Weight for entropy reduction (coordination)
const VALUE_WEIGHT = 0.0             # w_F: Weight for state value (detection priority)
const INFORMATION_STATES = [1, 2]     # I_1: No event, I_2: Event
const STATE_VALUES = [0.1, 1.0]      # F_1: No event value, F_2: Event value

# =============================================================================
# HETEROGENEOUS RSP (Random Spread Process) MODEL PARAMETERS
# -----------------------------------------------------------------------------
# The environment now uses heterogeneous cell types with different parameters
# randomly distributed across the grid. Cell types are defined in Types.jl:
#
#   HETEROGENEOUS_CELL_TYPES:
#   - Immune cells: λ=0.0002, β₀=0.0002, α=0.03, δ=0.05 (events rarely start/die quickly)
#   - Fleeting events: λ=0.0050, β₀=0.0150, α=0.01, δ=0.99 (ignite occasionally, burn out fast)
#   - Long-lasting events: λ=0.0020, β₀=0.0020, α=0.01, δ=0.95 (rare ignition, ~10-step lifetime)
#   - Moderate cells: λ=0.0100, β₀=0.0100, α=0.01, δ=0.99 (balanced ignition and lifetime)
#   - High-contagion cells: λ=0.0200, β₀=0.0100, α=0.1, δ=0.9 (ignite easily and spread)
#
# Each cell gets randomly assigned one of these types, creating a realistic
# non-uniform environment with different event behaviors.
# =============================================================================

# =============================================================================

# Import the existing environment and planner modules
include("../src/MyProject.jl")
using .MyProject

# Import specific types from MyProject
using .MyProject: Agent, SensingAction, GridObservation, EventState, NO_EVENT, EVENT_PRESENT
using .MyProject: CircularTrajectory, LinearTrajectory, RangeLimitedSensor
using .MyProject: EventState2, NO_EVENT_2, EVENT_PRESENT_2, EventMap, DynamicsMode, toy_dbn, rsp
using .MyProject: EventDynamics, SpatialGrid

# Import functions
using .MyProject.Agents.TrajectoryPlanner: get_position_at_time, execute_plan
using .MyProject.Types: save_agent_actions_to_csv, calculate_and_save_ndd_metrics, EnhancedEventTracker, initialize_enhanced_event_tracker, update_enhanced_event_tracking!, mark_observed_events_with_time!, get_event_statistics, save_event_tracking_data, save_uncertainty_evolution_data, save_sync_event_data, create_observation_heatmap

# Sync reward configuration with PBVI planner
using .MyProject.Planners.GroundStation.MacroPlannerPBVI: set_reward_config_from_main
set_reward_config_from_main(ENTROPY_WEIGHT, VALUE_WEIGHT, INFORMATION_STATES, STATE_VALUES)

# Sync reward configuration with Policy Tree planner
using .MyProject.Planners.GroundStation.AsyncPBVIPolicyTree: set_reward_config_from_main as set_reward_config_policy_tree
set_reward_config_policy_tree(ENTROPY_WEIGHT, VALUE_WEIGHT, INFORMATION_STATES, STATE_VALUES)

# Import specific modules
using .Environment
using .Environment: GridState
using .Environment.EventDynamicsModule
using .Planners.GroundStation
using .Planners.MacroPlannerAsync
using .Planners.PolicyTreePlanner
using .Planners.MacroPlannerRandom # Added for random planner
using .Planners.MacroPlannerSweep # Added for sweep planner

# Import RSP functions
import .Environment.EventDynamicsModule: transition_rsp!
# Import functions from MacroPlannerAsync
import .MacroPlannerAsync: initialize_uniform_belief, get_known_observations_at_time, has_known_observation, get_known_observation, evolve_no_obs, collapse_belief_to

# Import planning time function from GroundStation
import .GroundStation: get_average_planning_time

# =============================================================================
# REPLAY SYSTEM STRUCTURES
# =============================================================================

"""
Replay environment that stores and replays the exact same environmental changes
"""
mutable struct ReplayEnvironment
    env::SpatialGrid
    event_evolution::Vector{Matrix{EventState}}
    rng_state::Vector{Int}  # Store RNG state for reproducibility
end

"""
Save planning time statistics and performance metrics to a file
"""
function save_performance_metrics(gs_state, avg_uncertainty, event_observation_percentage, ndd_expected_lifetime, ndd_actual_lifetime, env, agents, results_dir, run_number, planning_mode)
    # Create the metrics directory path
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "metrics")
    if !isdir(metrics_dir)
        mkpath(metrics_dir)
    end
    
    filename = "performance_metrics_$(planning_mode)_run$(run_number).txt"
    filepath = joinpath(metrics_dir, filename)
    
    open(filepath, "w") do file
        println(file, "="^60)
        println(file, "PERFORMANCE METRICS REPORT")
        println(file, "="^60)
        println(file, "Generated: $(now())")
        println(file, "Planning Mode: $(planning_mode)")
        println(file, "Run Number: $(run_number)")
        println(file, "")
        
        # Environment parameters
        println(file, "ENVIRONMENT PARAMETERS:")
        println(file, "  Grid size: $(env.width) x $(env.height)")
        println(file, "  RSP parameters: Heterogeneous (cell-specific)")
        println(file, "  Cell types: $(length(Types.HETEROGENEOUS_CELL_TYPES)) different types")
        
        # Print cell type distribution
        cell_counts, total_cells = Types.analyze_cell_type_distribution(env.rsp_params)
        println(file, "  Cell type distribution:")
        for (cell_name, count) in cell_counts
            percentage = round(100 * count / total_cells, digits=1)
            println(file, "    $(cell_name): $(count) ($(percentage)%)")
        end
        
        # Print average parameters
        avg_lambda = mean(env.rsp_params.lambda_map)
        avg_alpha = mean(env.rsp_params.alpha_map)
        avg_delta = mean(env.rsp_params.delta_map)
        avg_mu = mean(env.rsp_params.mu_map)
        println(file, "  Average parameters:")
        println(file, "    λ (lambda): $(round(avg_lambda, digits=4))")
        println(file, "    α (alpha): $(round(avg_alpha, digits=3))")
        println(file, "    δ (delta): $(round(avg_delta, digits=3))")
        println(file, "    μ (mu): $(round(avg_mu, digits=3))")
        println(file, "")
        
        # Agent information
        println(file, "AGENT INFORMATION:")
        println(file, "  Number of agents: $(length(agents))")
        for (i, agent) in enumerate(agents)
            println(file, "  Agent $(agent.id): trajectory type $(typeof(agent.trajectory)), phase offset $(agent.phase_offset)")
        end
        println(file, "")
        
        # Planning time statistics
        println(file, "PLANNING TIME STATISTICS:")
        if gs_state.num_plans_computed > 0
            avg_planning_time = get_average_planning_time(gs_state)
            println(file, "  Total plans computed: $(gs_state.num_plans_computed)")
            println(file, "  Total planning time: $(round(gs_state.total_planning_time, digits=3)) seconds")
            println(file, "  Average planning time per plan: $(round(avg_planning_time, digits=3)) seconds")
            println(file, "")
            
            # Per-agent planning times
            println(file, "  Per-agent planning times:")
            for (agent_id, times) in gs_state.planning_times
                if !isempty(times)
                    agent_avg = sum(times) / length(times)
                    agent_min = minimum(times)
                    agent_max = maximum(times)
                    println(file, "    Agent $(agent_id): $(length(times)) plans")
                    println(file, "      Average: $(round(agent_avg, digits=3)) seconds")
                    println(file, "      Min: $(round(agent_min, digits=3)) seconds")
                    println(file, "      Max: $(round(agent_max, digits=3)) seconds")
                end
            end
        else
            println(file, "  No planning time data available")
        end
        println(file, "")
        
        # Performance metrics
        println(file, "PERFORMANCE METRICS:")
        println(file, "  Final event observation percentage: $(round(event_observation_percentage, digits=1))%")
        println(file, "  Final average uncertainty: $(round(avg_uncertainty[end], digits=3))")
        println(file, "  Normalized Detection Delay (expected lifetime): $(round(ndd_expected_lifetime, digits=3))")
        println(file, "  Normalized Detection Delay (actual lifetime): $(round(ndd_actual_lifetime, digits=3))")
        println(file, "  Uncertainty evolution:")
        for (i, uncertainty) in enumerate(avg_uncertainty)
            if i % 10 == 1 || i == length(avg_uncertainty)  # Print every 10th step and the last step
                println(file, "    Step $(i): $(round(uncertainty, digits=3))")
            end
        end
        println(file, "")
        
        # Cache statistics (if available)
        try
            cache_stats = BeliefManagement.get_cache_stats()
            println(file, "CACHE STATISTICS:")
            println(file, "  Hits: $(cache_stats[:hits])")
            println(file, "  Misses: $(cache_stats[:misses])")
            println(file, "  Hit rate: $(round(cache_stats[:hit_rate] * 100, digits=1))%")
            println(file, "  Cache size: $(cache_stats[:cache_size])")
        catch
            println(file, "CACHE STATISTICS: Not available")
        end
        println(file, "")
        
        println(file, "="^60)
    end
    
    println("📁 Performance metrics saved to: $(filepath)")
    return filepath
end

println("✅ All modules imported successfully")

# =============================================================================
# ROW-ONLY VISIBILITY FUNCTIONS
# =============================================================================

"""
Get field of regard for an agent at a specific position (row-only visibility)
"""
function get_row_field_of_regard(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Row-only visibility: agent can only see cells in its own row
    for col in 1:env.width
        if col != x  # Don't include current position
            push!(fov_cells, (col, y))
        end
    end
    
    return fov_cells
end

"""
Debug function to show agent positions and trajectory information
"""
function debug_agent_positions(agents)
    println("\n🔍 DEBUG: Agent Positions and Trajectories")
    println("==========================================")
    
    for agent in agents
        println("Agent $(agent.id):")
        println("  Trajectory: $(agent.trajectory)")
        println("  Phase offset: $(agent.phase_offset)")
        println("  Sensor range: $(agent.sensor.range)")
        println("  Positions over time:")
        for t in 0:10
            pos = get_position_at_time(agent.trajectory, t, agent.phase_offset)
            println("    Time $(t): $(pos)")
        end
        println()
    end
end

"""
Create agents with row-only visibility (two agents)
"""
function create_row_agents()
    agents = Agent[]
    
    # Create two agents: one in row 1, one in row 3
    agent_rows = [1, 3]
    
    for (i, row) in enumerate(agent_rows)
        # Create trajectory that cycles through all four rows
        # For a 4-row grid, we need to cycle through positions (2,1), (2,2), (2,3), (2,4)
        trajectory = LinearTrajectory(2, 1, 2, 4, 4, 1.0)  # Period = 4, moves from row 1 to row 4
        phase_offset = (row - 1)  # Start at the correct row
        # Sensor with row-only visibility
        sensor = RangeLimitedSensor(SENSOR_RANGE, SENSOR_FOV, SENSOR_NOISE, :row_only)
        # Agent with phase offset based on starting row and custom battery parameters
        # Battery parameters: max_battery, charging_rate, observation_cost
        agent = Agent(i, trajectory, sensor, phase_offset, MAX_BATTERY, CHARGING_RATE, OBSERVATION_COST)
        push!(agents, agent)
    end
    
    println("🤖 Created $(NUM_AGENTS) agents with row-only visibility:")
    for agent in agents
        println("  Agent $(agent.id): starts in row $(agent.trajectory.start_y + agent.phase_offset), moves upward 1→2→3→1→2→3..., Phase offset $(agent.phase_offset)")
    end
    
    return agents
end

"""
Create test environment with heterogeneous RSP dynamics
"""
function create_rsp_environment(max_prob_mass::Float64=MAX_PROB_MASS)
    # Create event dynamics (not used for RSP, but required by constructor)
    event_dynamics = EventDynamics(0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Create agents
    agents = create_row_agents()
    
    # Create spatial grid environment with RSP dynamics
    env = SpatialGrid(GRID_WIDTH, GRID_HEIGHT, event_dynamics, agents, SENSOR_RANGE, DISCOUNT_FACTOR, INITIAL_EVENTS, MAX_SENSING_TARGETS, (GROUND_STATION_X, GROUND_STATION_Y), nothing, max_prob_mass)
    
    # Update to RSP dynamics
    env.dynamics = rsp  # Use RSP dynamics (enum value)
    
    # Create heterogeneous parameter maps
    param_maps = Types.create_heterogeneous_rsp_maps(GRID_HEIGHT, GRID_WIDTH)
    
    # Use lambda map as ignition probability map for backward compatibility
    env.ignition_prob = param_maps.lambda_map
    
    # Add RSP parameter maps to environment
    env.rsp_params = param_maps
    
    println("🌍 Created heterogeneous RSP test environment:")
    println("  Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
    println("  Initial events: $(INITIAL_EVENTS)")
    println("  Max sensing targets: $(MAX_SENSING_TARGETS)")
    println("  Dynamics: Heterogeneous RSP (Random Spread Process)")
    println("  Cell types: $(length(Types.HETEROGENEOUS_CELL_TYPES)) different types randomly distributed")
    println("  Max probability mass: $(max_prob_mass) (branch pruning threshold)")
    
    # Print cell type distribution
    cell_counts, total_cells = Types.analyze_cell_type_distribution(param_maps)
    println("  Cell type distribution:")
    for (cell_name, count) in cell_counts
        percentage = round(100 * count / total_cells, digits=1)
        println("    $(cell_name): $(count) ($(percentage)%)")
    end
    
    return env
end

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

"""
Create and save environment parameter distribution visualization
"""
function create_environment_distribution_plot(param_maps::Types.RSPParameterMaps, results_dir::String="", run_number::Int=1)
    println("🌍 Creating environment parameter distribution visualization...")
    
    # Create plots directory path
    plots_dir = joinpath(results_dir, "Run $(run_number)", "environment")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end
    
    # Create subplots for each parameter
    p1 = heatmap(param_maps.lambda_map, 
        title="λ (Ignition Intensity)", 
        colorbar_title="λ",
        colormap=:plasma,
        aspect_ratio=:equal,
        size=(400, 300))
    
    p2 = heatmap(param_maps.alpha_map, 
        title="α (Contagion Strength)", 
        colorbar_title="α",
        colormap=:plasma,
        aspect_ratio=:equal,
        size=(400, 300))
    
    p3 = heatmap(param_maps.delta_map, 
        title="δ (Persistence Probability)", 
        colorbar_title="δ",
        colormap=:plasma,
        aspect_ratio=:equal,
        size=(400, 300))
    
    p4 = heatmap(param_maps.mu_map, 
        title="μ (Death Probability)", 
        colorbar_title="μ",
        colormap=:plasma,
        aspect_ratio=:equal,
        size=(400, 300))
    
    # Combine all plots
    combined_plot = plot(p1, p2, p3, p4, 
        layout=(2,2), 
        size=(800, 600),
        title="Heterogeneous RSP Environment Parameters - Grid $(GRID_WIDTH)x$(GRID_HEIGHT), Run $(run_number)",
        titlefontsize=12)
    
    # Save the plot
    plot_filename = joinpath(plots_dir, "environment_parameters_run$(run_number).png")
    savefig(combined_plot, plot_filename)
    println("✓ Saved environment distribution plot: $(basename(plot_filename))")
    
    return combined_plot
end

"""
Visualize the current state of the environment and agents
"""
function visualize_rsp_state(
    time_step::Int,
    agents::Vector{Agent},
    environment_state::Matrix{EventState}=fill(NO_EVENT, GRID_HEIGHT, GRID_WIDTH),
    actions::Vector{SensingAction}=SensingAction[],
    ground_station_pos::Tuple{Int, Int}=(GROUND_STATION_X, GROUND_STATION_Y)
)
    height, width = size(environment_state)
    agent_colors = [:red, :blue, :green, :orange]  # Add more if needed

    # Create a blank plot with correct limits and aspect
    p = plot(; xlim=(0.5, width+0.5), ylim=(0.5, height+0.5),
        aspect_ratio=:equal, size=(600, 800), legend=false,
        xlabel="X Coordinate", ylabel="Y Coordinate",
        grid=false,
        title="RSP $(width)x$(height) Test - Time Step $(time_step) - γ=$(DISCOUNT_FACTOR), Events: $(count(==(EVENT_PRESENT), environment_state)) | Agents: $(length(agents))",
        titlefontsize=12,
        background_color=:white
    )

    # Draw grid cells as white squares with black borders
    for x in 1:width, y in 1:height
        xs = [x-0.5, x+0.5, x+0.5, x-0.5]
        ys = [y-0.5, y-0.5, y+0.5, y+0.5]
        plot!(p, xs, ys, seriestype=:shape, fillcolor=:white, linecolor=:black, linewidth=1, alpha=1, label=false)
    end

    # Overlay: FOR and action for each agent
    for (i, agent) in enumerate(agents)
        color = agent_colors[i]
        pos = get_position_at_time(agent.trajectory, time_step, agent.phase_offset)
        # Get FOR cells
        for_cells = get_row_field_of_regard(agent, pos, (width=width, height=height))
        # Highlight FOR (light color)
        for (x, y) in for_cells
            xs = [x-0.5, x+0.5, x+0.5, x-0.5]
            ys = [y-0.5, y-0.5, y+0.5, y+0.5]
            plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.18, label=false)
        end
        # Highlight agent's own position in FOR (light color, but slightly more opaque)
        x, y = pos
        xs = [x-0.5, x+0.5, x+0.5, x-0.5]
        ys = [y-0.5, y-0.5, y+0.5, y+0.5]
        plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.28, label=false)
        # Highlight action (dark color), matching by agent id
        action_idx = findfirst(a -> a.agent_id == agent.id, actions)
        if action_idx !== nothing && !isempty(actions[action_idx].target_cells)
            for (x, y) in actions[action_idx].target_cells
                xs = [x-0.5, x+0.5, x+0.5, x-0.5]
                ys = [y-0.5, y-0.5, y+0.5, y+0.5]
                plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.5, label=false)
            end
        end
    end

    # Overlay: Ground station as green star
    scatter!(p, [ground_station_pos[1]], [ground_station_pos[2]]; marker=:star, markersize=14, color=:green, alpha=0.9, label=false)

    # Overlay: Agents as small circles
    for (i, agent) in enumerate(agents)
        pos = get_position_at_time(agent.trajectory, time_step, agent.phase_offset)
        agent_color = agent_colors[i]
        scatter!(p, [pos[1]], [pos[2]]; marker=:circle, markersize=10, color=agent_color, alpha=0.9, label=false)
    end

    # Overlay: Fire emoji for events
    for y in 1:height, x in 1:width
        if environment_state[y, x] == EVENT_PRESENT
            annotate!(p, x, y, text("🔥", :center, 18))
        end
    end

    return p
end

"""
Create animation of the RSP simulation
"""
function create_rsp_animation(
    agents::Vector{Agent},
    num_steps::Int,
    environment_evolution::Vector{Matrix{EventState}}=Vector{Matrix{EventState}}(),
    action_history::Vector{Vector{SensingAction}}=Vector{Vector{SensingAction}}(),
    ground_station_pos::Tuple{Int, Int}=(GROUND_STATION_X, GROUND_STATION_Y),
    results_dir::String="",
    run_number::Int=1,
    planning_mode::Symbol=:script
)
    println("\n🎬 Creating RSP Simulation Animation")
    println("====================================")
    
    # Create animations directory path
    animations_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "animations")
    if !isdir(animations_dir)
        mkpath(animations_dir)
    end
    
    # Create frames for animation
    frames = []
    
    for step in 0:(num_steps-1)
        # Get environment state for this step
        env_state = if step < length(environment_evolution)
            environment_evolution[step + 1]
        else
            fill(NO_EVENT, GRID_HEIGHT, GRID_WIDTH)
        end
        
        # Get actions for this step
        actions = if step < length(action_history)
            action_history[step + 1]
        else
            SensingAction[]
        end
        
        # Create frame
        frame = visualize_rsp_state(step, agents, env_state, actions, ground_station_pos)
        push!(frames, frame)
    end
    
    # Create animation
    anim = @animate for frame in frames
        plot(frame, size=(600, 800))
    end
    
    # Save animation with new naming convention
    animation_filename = joinpath(animations_dir, "rsp_$(GRID_WIDTH)x$(GRID_HEIGHT)_$(planning_mode)_run$(run_number).gif")
    gif(anim, animation_filename, fps=0.5)  # Slower animation
    println("✓ Saved animation: $(basename(animation_filename))")
    
    return anim
end

"""
Create animation (GIF) of the belief P(Event Present) per cell over time
"""
function create_belief_event_animation(
    belief_event_present_evolution::Vector{Matrix{Float64}},
    results_dir::String="",
    run_number::Int=1,
    planning_mode::Symbol=:script
)
    println("\n🎬 Creating Belief Animation (P(Event Present))")
    println("=============================================")

    # Create animations directory path
    animations_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "animations")
    if !isdir(animations_dir)
        mkpath(animations_dir)
    end

    # Build frames as heatmaps with fixed color limits [0, 1]
    anim = @animate for (t, prob_map) in enumerate(belief_event_present_evolution)
        heatmap(
            prob_map;
            title = "Belief P(Event) — t=$(t - 1)",
            xlabel = "X Coordinate",
            ylabel = "Y Coordinate",
            aspect_ratio = :equal,
            colorbar_title = "P(Event)",
            c = :viridis,
            clims = (0.0, 1.0),
            size = (600, 600)
        )
    end

    gif_filename = joinpath(animations_dir, "belief_event_present_$(GRID_WIDTH)x$(GRID_HEIGHT)_$(planning_mode)_run$(run_number).gif")
    gif(anim, gif_filename, fps=1.5)
    println("✓ Saved belief animation: $(basename(gif_filename))")

    return anim
end

"""
Create uncertainty visualizations including uncertainty map animation and uncertainty over time plot
"""
function create_uncertainty_visualizations(
    uncertainty_evolution::Vector{Matrix{Float64}},
    average_uncertainty_per_timestep::Vector{Float64},
    environment_evolution::Vector{Matrix{EventState}},
    num_steps::Int,
    results_dir::String="",
    run_number::Int=1,
    planning_mode::Symbol=:script
)
    println("📊 Creating uncertainty visualizations...")
    
    # Create plots directory path
    plots_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "plots")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end
    
    # 1. Create uncertainty over time plot
    println("  📈 Creating uncertainty over time plot...")
    time_points = 1:length(average_uncertainty_per_timestep)
    
    uncertainty_plot = plot(
        time_points, 
        average_uncertainty_per_timestep, 
        seriestype=:line,
        color=:blue,
        linewidth=2,
        marker=:circle,
        markersize=4,
        title="Average Uncertainty Over Time - Grid $(GRID_WIDTH)x$(GRID_HEIGHT), $(planning_mode), Run $(run_number)",
        xlabel="Time Step",
        ylabel="Average Uncertainty (Entropy)",
        legend=false,
        grid=true,
        size=(800, 600)
    )
    
    # Add horizontal line for uniform distribution entropy (log2(2) ≈ 0.693)
    hline!([1], color=:red, linestyle=:dash, linewidth=1, label="Uniform Distribution")
    
    # Save the plot with new naming convention
    plot_filename = joinpath(plots_dir, "uncertainty_evolution_$(planning_mode)_run$(run_number).png")
    savefig(uncertainty_plot, plot_filename)
    println("  ✓ Saved uncertainty plot: $(basename(plot_filename))")
    
    # # 2. Create uncertainty map animation
    # println("  🎬 Creating uncertainty map animation...")
    
    # # Find the range of uncertainty values for consistent coloring
    # all_uncertainties = vcat([vec(u) for u in uncertainty_evolution]...)
    # min_uncertainty = minimum(all_uncertainties)
    # max_uncertainty = maximum(all_uncertainties)
    
    # # Create frames for uncertainty animation
    # uncertainty_frames = []
    
    # for (step, uncertainty_map) in enumerate(uncertainty_evolution)
    #     frame = heatmap(
    #         uncertainty_map,
    #         colormap=:plasma,
    #         colorrange=(min_uncertainty, max_uncertainty),
    #         title="Uncertainty Map (t=$(step)) - Grid $(GRID_WIDTH)x$(GRID_HEIGHT), $(planning_mode), Run $(run_number)",
    #         xlabel="X",
    #         ylabel="Y",
    #         aspect_ratio=:equal,
    #         size=(600, 400),
    #         colorbar_title="Uncertainty (Entropy)"
    #     )
    #     push!(uncertainty_frames, frame)
    # end
    
    # # Create animation
    # uncertainty_anim = @animate for frame in uncertainty_frames
    #     plot(frame)
    # end
    
    # # Save animation with new naming convention
    # uncertainty_animation_filename = joinpath(plots_dir, "uncertainty_animation_$(planning_mode)_run$(run_number).gif")
    # gif(uncertainty_anim, uncertainty_animation_filename, fps=1)
    # println("  ✓ Saved uncertainty animation: $(basename(uncertainty_animation_filename))")
    
    println("✅ All uncertainty visualizations created!")
end

# =============================================================================
# EVENT TRACKING STRUCTURES
# =============================================================================

"""
Event tracking structure to maintain unique event IDs and observation status
"""
mutable struct EventTracker
    cell_active_event_id::Dict{Tuple{Int,Int}, Union{Nothing,Int}}  # Maps cell -> current active event ID
    event_registry::Dict{Int, Dict}  # Maps event_id -> event info
    next_event_id::Int  # Next available event ID
end

"""
Initialize event tracker
"""
function initialize_event_tracker()
    return EventTracker(
        Dict{Tuple{Int,Int}, Union{Nothing,Int}}(),
        Dict{Int, Dict}(),
        1
    )
end

"""
Update event tracking for a timestep
"""
function update_event_tracking!(tracker::EventTracker, prev_environment::Matrix{EventState}, 
                               curr_environment::Matrix{EventState}, timestep::Int)
    height, width = size(curr_environment)
    
    for y in 1:height, x in 1:width
        cell = (x, y)
        prev_state = prev_environment[y, x]
        curr_state = curr_environment[y, x]
        
        if prev_state == NO_EVENT && curr_state == EVENT_PRESENT
            # New event started
            event_id = tracker.next_event_id
            tracker.cell_active_event_id[cell] = event_id
            tracker.event_registry[event_id] = Dict(
                :cell => cell,
                :start_time => timestep,
                :observed => false,
                :end_time => nothing
            )
            tracker.next_event_id += 1
            
        elseif prev_state == EVENT_PRESENT && curr_state == NO_EVENT
            # Event ended
            event_id = tracker.cell_active_event_id[cell]
            if event_id !== nothing
                tracker.event_registry[event_id][:end_time] = timestep
                tracker.cell_active_event_id[cell] = nothing
            end
        end
    end
end

"""
Mark events as observed based on agent observations
"""
function mark_observed_events!(tracker::EventTracker, agent_observations::Vector{Tuple{Int, Vector{Tuple{Tuple{Int,Int}, EventState}}}})
    for (agent_id, observations) in agent_observations
        for (cell, observed_state) in observations
            if observed_state == EVENT_PRESENT
                event_id = tracker.cell_active_event_id[cell]
                if event_id !== nothing
                    tracker.event_registry[event_id][:observed] = true
                end
            end
        end
    end
end





# =============================================================================
# RSP SIMULATION FUNCTIONS
# =============================================================================

"""
Simulate RSP environment evolution
"""
function simulate_rsp_environment(env, num_steps::Int,  planning_mode::Symbol)
    println("\n🔥 Simulating RSP Environment Evolution")
    println("======================================")
    
    # Initialize environment state
    current_state = Matrix{EventState}(undef, GRID_HEIGHT, GRID_WIDTH)
    current_state .= NO_EVENT
    
    # Add initial events
    for _ in 1:INITIAL_EVENTS
        x = rand(1:GRID_WIDTH)
        y = rand(1:GRID_HEIGHT)
        current_state[y, x] = EVENT_PRESENT
    end
    
    # Track evolution
    evolution = [copy(current_state)]
    
    println("Initial state:")
    display(current_state)
    println("Ignition probability map:")
    display(env.ignition_prob)
    
    # Simulate evolution
    for step in 1:num_steps
        new_state = similar(current_state)
        
        # Use heterogeneous RSP transition with parameter maps
        transition_rsp!(new_state, current_state, env.rsp_params, Random.GLOBAL_RNG)
        
        current_state = new_state
        push!(evolution, copy(current_state))
        
        println("\nStep $(step):")
        display(current_state)
        
        # Debug: show transition probabilities for a specific cell
        if step == 1
            println("\nDebug: Transition probabilities for cell (1,2) - neighbor of initial event:")
            x, y = 1, 2
            current_cell_state = current_state[y, x] == NO_EVENT ? 1 : 2
            
            # Get neighbor states
            neighbor_states = Int[]
            for dx in -1:1, dy in -1:1
                if dx == 0 && dy == 0
                    continue
                end
                nx, ny = x + dx, y + dy
                if 1 <= nx <= GRID_WIDTH && 1 <= ny <= GRID_HEIGHT
                    neighbor_state = current_state[ny, nx] == NO_EVENT ? 1 : 2
                    push!(neighbor_states, neighbor_state)
                end
            end
            
            # Get cell-specific parameters
            cell_params = Types.get_cell_rsp_params(env.rsp_params, y, x)
            
            prob_no_event = get_transition_probability_rsp(1, current_cell_state, neighbor_states;
                λ=cell_params.lambda,
                β0=cell_params.beta0,
                α=cell_params.alpha,
                δ=cell_params.delta)
            prob_event = get_transition_probability_rsp(2, current_cell_state, neighbor_states;
                λ=cell_params.lambda,
                β0=cell_params.beta0,
                α=cell_params.alpha,
                δ=cell_params.delta)
            println("  Current state: $(current_cell_state == 1 ? "NO_EVENT" : "EVENT_PRESENT")")
            println("  Active neighbors: $(count(x -> x == 2, neighbor_states))")
            println("  P(NO_EVENT): $(round(prob_no_event, digits=3))")
            println("  P(EVENT_PRESENT): $(round(prob_event, digits=3))")
            println("  Local ignition: $(env.ignition_prob[y, x])")
        end
    end
    
    return evolution
end

"""
Simulate asynchronous centralized planning with RSP using replay environment
"""
function simulate_rsp_async_planning_replay(replay_env::ReplayEnvironment, num_steps::Int=NUM_STEPS, run_number::Int=1, planning_mode::Symbol=:script)
    println("🚀 Starting RSP Asynchronous Centralized Planning Simulation (Replay)")
    println("====================================================================")
    println("Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
    println("Agents: $(NUM_AGENTS) (rows 1 and 3)")
    println("Planning mode: $(planning_mode)")
    println("Dynamics: RSP (Replay)")
    println("Run: $(run_number)")
    
    # Create environment and agents from replay
    env = replay_env.env
    agents = env.agents
    
    # Initialize ground station
    gs_state = GroundStation.initialize_ground_station(env, agents, num_states=2)
    
    # Initialize enhanced event tracker
    event_tracker = initialize_enhanced_event_tracker()
    
    # Track performance metrics
    sync_events = []
    
    # Track environment and actions for visualization
    environment_evolution = Matrix{EventState}[]
    action_history = Vector{Vector{SensingAction}}()
    
    # Track uncertainty evolution for visualization
    uncertainty_evolution = Matrix{Float64}[]
    average_uncertainty_per_timestep = Float64[]
    # Track belief of EVENT_PRESENT per cell over time
    belief_event_present_evolution = Matrix{Float64}[]

    # Get initial environment state from replay
    current_environment = get_replay_state(replay_env, 0)

    # Initialize previous environment state for event tracking
    prev_environment = copy(current_environment)
    
    # Update event tracking for initial state (t=0)
    update_enhanced_event_tracking!(event_tracker, fill(NO_EVENT, GRID_HEIGHT, GRID_WIDTH), current_environment, 0)

    println("\n📊 Starting simulation with replay...")
    
    for t in 0:(num_steps-1)
        println("\n⏰ Time step $(t)")        
        # Check for synchronization opportunities
        old_sync_times = copy(gs_state.agent_last_sync)
        GroundStation.maybe_sync!(env, gs_state, agents, t, planning_mode=planning_mode)
        
        # Record sync events
        for (agent_id, old_time) in old_sync_times
            if gs_state.agent_last_sync[agent_id] != old_time
                push!(sync_events, (t, agent_id))
                println("📡 Sync event: Agent $(agent_id) at time $(t)")
                println("🔍 Using exact evaluation for agent $(agent_id)...")
            end
        end
        
        # Execute agent actions
        joint_actions = SensingAction[]
        agent_observations = Vector{Tuple{Int, Vector{Tuple{Tuple{Int,Int}, EventState}}}}()
        
        for agent in agents
            # Get plan from ground station and execute it
            plan, plan_type = GroundStation.get_agent_plan(agent, gs_state)
            action = execute_plan(agent, plan, plan_type, agent.observation_history, t)
            push!(joint_actions, action)
            
            # Ensure every agent charges every timestep (this is the key fix!)
            # The charging rate is applied every timestep regardless of action execution
            agent.battery_level = min(agent.max_battery, agent.battery_level + agent.charging_rate)
            # Use POMDP interface to get observations
            if !isempty(action.target_cells)
                # Get observation using POMDP observation model
                # For replay, we need to get the actual state from replay
                # Create proper GridState with all required parameters
                agent_positions = [get_position_at_time(agent.trajectory, t, agent.phase_offset) for agent in agents]
                agent_trajectories = [agent.trajectory for agent in agents]
                grid_state = GridState(current_environment, agent_positions, agent_trajectories, t)
                observation_dist = POMDPs.observation(env, action, grid_state)
                observation = rand(observation_dist)
                push!(agent.observation_history, observation)
                
                # Collect observations for event tracking
                observations = Vector{Tuple{Tuple{Int,Int}, EventState}}()
                for (i, cell) in enumerate(observation.sensed_cells)
                    if i <= length(observation.event_states)
                        push!(observations, (cell, observation.event_states[i]))
                    end
                end
                push!(agent_observations, (agent.id, observations))
                
                # Debug: print what was observed
                events_found = count(==(EVENT_PRESENT), observation.event_states)
                println("  Agent $(agent.id): $(length(action.target_cells)) cells sensed, $(events_found) events found")
            else
                # Wait action - create empty observation
                empty_observation = GridObservation(agent.id, Tuple{Int,Int}[], EventState[], [])
                push!(agent.observation_history, empty_observation)
                println("  Agent $(agent.id): wait action")
            end
        end
        
        # Mark events as observed with detection time tracking
        mark_observed_events_with_time!(event_tracker, agent_observations, t)
        
        # Record environment state and actions for visualization
        push!(environment_evolution, copy(current_environment))
        push!(action_history, joint_actions)
        
        # Update global belief with new observations using t_clean log2ic
        if gs_state.global_belief !== nothing
            # Determine t_clean (last time where all observation outcomes are known)
            tau = gs_state.agent_last_sync
            t_clean = minimum([tau[j] for j in keys(tau)])
            
            # Roll forward deterministically from uniform belief to t_clean using known observations
            B = GroundStation.initialize_global_belief(env)
            for t_roll in 0:(t_clean-1)
                B = evolve_no_obs(B, env)  # Contagion-aware update
                # Apply known observations (perfect observations)
                for (agent_j, action_j) in get_known_observations_at_time(t_roll, gs_state)
                    for cell in action_j.target_cells
                        if has_known_observation(t_roll, cell, gs_state)
                            observed_value = get_known_observation(t_roll, cell, gs_state)
                            B = collapse_belief_to(B, cell, observed_value)
                        end
                    end
                end
            end
            
            # Update the global belief with the belief at t_clean
            gs_state.global_belief.event_distributions = B.event_distributions
            gs_state.global_belief.uncertainty_map = B.uncertainty_map
            gs_state.global_belief.last_update = t_clean
        end
        
        # Record uncertainty state for visualization
        if gs_state.global_belief !== nothing
            push!(uncertainty_evolution, copy(gs_state.global_belief.uncertainty_map))
            avg_uncertainty = mean(gs_state.global_belief.uncertainty_map)
            push!(average_uncertainty_per_timestep, avg_uncertainty)
            # Record P(EVENT_PRESENT) per cell
            prob_map = copy(gs_state.global_belief.event_distributions[Int(EVENT_PRESENT) + 1, :, :])
            push!(belief_event_present_evolution, prob_map)
        else
            # If no global belief yet, use uniform uncertainty
            uniform_uncertainty = fill(1, GRID_HEIGHT, GRID_WIDTH)  # log2(2) for uniform distribution
            push!(uncertainty_evolution, uniform_uncertainty)
            push!(average_uncertainty_per_timestep, 1)
            # Uniform belief over 2 states → P(event)=0.5
            uniform_prob = fill(0.5, GRID_HEIGHT, GRID_WIDTH)
            push!(belief_event_present_evolution, uniform_prob)
        end
        
        # Update environment using replay (not POMDP transition)
        if t < num_steps - 1
            # Get next state from replay
            current_environment = get_replay_state(replay_env, t + 1)
            
            # Update event tracking for the new timestep
            update_enhanced_event_tracking!(event_tracker, prev_environment, current_environment, t + 1)
            prev_environment .= current_environment
        end
        
        # Print status every few steps
        if t % 50 == 0
            total_events, observed_events = get_event_statistics(event_tracker)
            println("📊 Status: $(count(==(EVENT_PRESENT), current_environment)) events active, $(total_events) total events, $(observed_events) observed")
        end
    end
    
    # Calculate final statistics
    total_events, observed_events = get_event_statistics(event_tracker)
    event_observation_percentage = total_events > 0 ? (observed_events / total_events) * 100.0 : 0.0
    
    # Calculate both NDD metrics
    ndd_expected_lifetime = Types.calculate_ndd_expected_lifetime(event_tracker, env, NUM_STEPS)
    ndd_actual_lifetime = Types.calculate_ndd_actual_lifetime(event_tracker, NUM_STEPS)
    
    # Print final results
    println("\n📈 RSP Simulation Results (Replay)")
    println("===================================")
    println("Event Observation Performance:")
    println("  Total unique events that appeared: $(total_events)")
    println("  Total unique events observed: $(observed_events)")
    println("  Event observation percentage: $(round(event_observation_percentage, digits=1))%")
    println("  Normalized Detection Delay (expected lifetime): $(round(ndd_expected_lifetime, digits=3))")
    println("  Normalized Detection Delay (actual lifetime): $(round(ndd_actual_lifetime, digits=3))")
    println("")
    println("Event Details:")
    for (event_id, event_info) in event_tracker.event_registry
        status = event_info[:observed] ? "✅ OBSERVED" : "❌ MISSED"
        end_time_str = event_info[:end_time] !== nothing ? "$(event_info[:end_time])" : "ongoing"
        detection_str = event_info[:observed] ? " (detected at t=$(event_info[:detection_time]))" : ""
        println("  Event $(event_id): cell $(event_info[:cell]), time $(event_info[:start_time])-$(end_time_str), $(status)$(detection_str)")
    end
    println("")
    println("System Performance:")
    println("  Total sync events: $(length(sync_events))")
    println("  Grid size: $(GRID_WIDTH)x$(GRID_HEIGHT)")
    println("  Planning horizon: $(PLANNING_HORIZON)")
    println("  Dynamics: RSP (Replay)")
    
    return gs_state, agents, event_observation_percentage, sync_events, environment_evolution, action_history, event_tracker, uncertainty_evolution, average_uncertainty_per_timestep, ndd_expected_lifetime, ndd_actual_lifetime, belief_event_present_evolution
end

# =============================================================================
# REPLAY SYSTEM FOR FAIR COMPARISON
# =============================================================================

"""
Simulate environment evolution once and store it for replay
"""
function simulate_environment_once(num_steps::Int)
    println("\n🔥 Simulating Environment Evolution for Replay")
    println("=============================================")
    
    # Create environment
    env = create_rsp_environment()
    
    # Store initial RNG state - use a fixed seed for reproducibility
    initial_rng_state = [42]  # Use a simple fixed seed
    
    # Initialize environment state
    current_state = Matrix{EventState}(undef, GRID_HEIGHT, GRID_WIDTH)
    current_state .= NO_EVENT
    
    # Add initial events
    for _ in 1:INITIAL_EVENTS
        x = rand(1:GRID_WIDTH)
        y = rand(1:GRID_HEIGHT)
        current_state[y, x] = EVENT_PRESENT
    end
    
    # Track evolution
    event_evolution = [copy(current_state)]
    
    println("Initial state:")
    display(current_state)
    
    # Simulate evolution
    for step in 1:num_steps
        new_state = similar(current_state)
        
        # Use heterogeneous RSP transition with parameter maps
        transition_rsp!(new_state, current_state, env.rsp_params, Random.GLOBAL_RNG)
        
        current_state = new_state
        push!(event_evolution, copy(current_state))
        
        if step % 100 == 0
            println("Step $(step): $(count(==(EVENT_PRESENT), current_state)) events active")
        end
    end
    
    println("✅ Environment evolution simulated and stored for replay")
    println("  Total steps: $(length(event_evolution))")
    println("  Final events: $(count(==(EVENT_PRESENT), event_evolution[end]))")
    
    return ReplayEnvironment(env, event_evolution, initial_rng_state)
end

"""
Get environment state at a specific timestep from replay
"""
function get_replay_state(replay_env::ReplayEnvironment, timestep::Int)
    if timestep < length(replay_env.event_evolution)
        return replay_env.event_evolution[timestep + 1]  # +1 because evolution starts at t=0
    else
        # If beyond stored evolution, return last state
        return replay_env.event_evolution[end]
    end
end

"""
Simulate RSP environment evolution using replay
"""
function simulate_rsp_environment_replay(replay_env::ReplayEnvironment, num_steps::Int, planning_mode::Symbol)
    println("\n🔥 Replaying RSP Environment Evolution")
    println("======================================")
    
    # Restore RNG state for reproducibility
    Random.seed!(replay_env.rng_state[1])
    
    println("Initial state (from replay):")
    display(replay_env.event_evolution[1])
    
    # Return the stored evolution (truncated to requested steps)
    return replay_env.event_evolution[1:min(num_steps+1, length(replay_env.event_evolution))]
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Create main results directory with timestamp
timestamp = replace(string(now()), ":" => "-", "." => "-")
# Create results directory at project root (relative to this file's location)
results_base_dir = joinpath(@__DIR__, "..", "results", "run_$(timestamp)")
if !isdir(results_base_dir)
    mkpath(results_base_dir)
end

println("📁 Results will be saved in: $(results_base_dir)")


for n in 1:N_RUNS
    # Simulate environment once for fair comparison
    println("\n🔥 Simulating environment once for replay system...")
    replay_env = simulate_environment_once(NUM_STEPS)
    for mode in modes
        PLANNING_MODE = mode
        
        # Determine the actual planning mode and max_prob_mass based on the mode
        actual_planning_mode = mode
        max_prob_mass = MAX_PROB_MASS
        
        # Parse probability mass from symbol name if it's a macro_approx variant
        mode_str = string(mode)
        if startswith(mode_str, "macro_approx_")
            actual_planning_mode = :macro_approx
            # Extract everything after "macro_approx_" and convert to probability mass
            prob_mass_str = mode_str[length("macro_approx_")+1:end]
            # Handle decimal values by parsing as Float64 directly
            # If the string represents a decimal (e.g., "09999999" should be 0.9999999)
            # we need to insert a decimal point after the first digit
            if length(prob_mass_str) > 1
                # Insert decimal point after first digit: "09999999" -> "0.9999999"
                prob_mass_str = prob_mass_str[1] * "." * prob_mass_str[2:end]
            end
            max_prob_mass = parse(Float64, prob_mass_str)
        end
        @infiltrate
        
        println("🎯 RSP 3x2 Row Visibility Test")
        println("==============================")
        println("Configuration:")
        println("  Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
        println("  Agents: $(NUM_AGENTS) (rows 1 and 3)")
        println("  Planning horizon: $(PLANNING_HORIZON)")
        println("  Planning mode: $(actual_planning_mode) (:script, :policy, :random, :sweep, :greedy, :future_actions, :macro_approx, :pbvi)")
        if actual_planning_mode == :macro_approx
            println("  Max probability mass: $(max_prob_mass) (branch pruning threshold)")
        end
        println("  Dynamics: Heterogeneous RSP (Replay)")
        println("  Row-only visibility: true")
        println("  Run: $(n)/$(N_RUNS)")

        # Debug: show agent positions and trajectories
        debug_agent_positions(replay_env.env.agents)

        # Create environment distribution visualization
        create_environment_distribution_plot(replay_env.env.rsp_params, results_base_dir, n)

        # Create a new environment with the appropriate max_prob_mass for this mode
        # We need to update the replay environment's env with the correct max_prob_mass
        if actual_planning_mode == :macro_approx
            # Create new environment with correct max_prob_mass
            new_env = create_rsp_environment(max_prob_mass)
            # Copy the RSP parameters from the original environment
            new_env.rsp_params = replay_env.env.rsp_params
            new_env.ignition_prob = replay_env.env.ignition_prob
            # Update the replay environment
            replay_env.env = new_env
        end

        # Run the simulation with replay
        gs_state, agents, percentage, sync_events, env_evolution, action_history, event_tracker, uncertainty_evolution, avg_uncertainty, ndd_expected_lifetime, ndd_actual_lifetime, belief_event_present_evolution = simulate_rsp_async_planning_replay(replay_env, NUM_STEPS, n, actual_planning_mode)

        println("\n✅ RSP test completed!")
        println("📊 Final event observation percentage: $(round(percentage, digits=1))%")
        println("📊 Final average uncertainty: $(round(avg_uncertainty[end], digits=3))") 
        println("📊 Final Normalized Detection Delay (expected lifetime): $(round(ndd_expected_lifetime, digits=3))")
        println("📊 Final Normalized Detection Delay (actual lifetime): $(round(ndd_actual_lifetime, digits=3))")

        # Print final status
        GroundStation.print_status(gs_state)

        # Print final planning time statistics
        println("\n" * "="^60)
        println("📊 FINAL PLANNING TIME STATISTICS")
        println("="^60)
        avg_planning_time = get_average_planning_time(gs_state)
        println("Average planning time per plan: $(round(avg_planning_time, digits=3)) seconds")
        println("Total planning time: $(round(gs_state.total_planning_time, digits=3)) seconds")
        println("Total plans computed: $(gs_state.num_plans_computed)")

        # Per-agent breakdown
        for (agent_id, times) in gs_state.planning_times
            if !isempty(times)
                agent_avg = sum(times) / length(times)
                println("Agent $(agent_id): $(length(times)) plans, avg=$(round(agent_avg, digits=3))s")
            end
        end
        println("="^60) 

        # Create visualizations
        println("\n🎨 Creating Visualizations...")
        println("============================")
        
        # Create simulation animation
        anim = create_rsp_animation(agents, NUM_STEPS, env_evolution, action_history, (GROUND_STATION_X, GROUND_STATION_Y), results_base_dir, n, mode)
        
        # Create belief animation for P(Event Present)
        create_belief_event_animation(belief_event_present_evolution, results_base_dir, n, mode)
        
        # Create uncertainty visualization
        println("\n📊 Creating Uncertainty Visualizations...")
        create_uncertainty_visualizations(uncertainty_evolution, avg_uncertainty, env_evolution, NUM_STEPS, results_base_dir, n, mode)
        
        # Save performance metrics
        save_performance_metrics(gs_state, avg_uncertainty, percentage, ndd_expected_lifetime, ndd_actual_lifetime, replay_env.env, agents, results_base_dir, n, mode)
        
        # Save agent actions to CSV
        Types.save_agent_actions_to_csv(action_history, results_base_dir, n, mode, NUM_STEPS)
        
        # Calculate and save NDD metrics with both expected and actual lifetimes
        ndd_expected, ndd_actual, ndd_filepath = Types.calculate_and_save_ndd_metrics(event_tracker, replay_env.env, NUM_STEPS, results_base_dir, n, mode)
        
        # Save detailed event tracking data
        Types.save_event_tracking_data(event_tracker, results_base_dir, n, mode)
        
        # Save uncertainty evolution data
        Types.save_uncertainty_evolution_data(uncertainty_evolution, avg_uncertainty, results_base_dir, n, mode)
        
        # Save sync event data
        Types.save_sync_event_data(sync_events, results_base_dir, n, mode)
        
        # Create and save observation heatmap
        Types.create_observation_heatmap(action_history, GRID_WIDTH, GRID_HEIGHT, results_base_dir, n, mode)
        
        println("\n✅ RSP simulation completed!")
        println("📁 Check the results folder for:")
        println("  - Run $(n)/$(mode)/animations/ (main simulation animation)")
        println("  - Run $(n)/$(mode)/plots/ (uncertainty visualizations)")
        println("  - Run $(n)/$(mode)/metrics/ (performance metrics, agent actions, NDD metrics)")
    end
end

println("\n🎉 All simulations completed!")
println("📁 Final results saved in: $(results_base_dir)")
#!/usr/bin/env julia

"""
Test script for RSP (Random Spread Process) with 5x5 grid and cross-shaped sensor
Demonstrates complex trajectory with cross-shaped sensor range
"""

println("🚀 RSP 5x5 Complex Trajectory Test starting...")

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Plots
using Dates  # Add this for timestamping
using DataFrames  # For saving metrics to CSV
using CSV  # For saving metrics to CSV
Plots.plotlyjs()
using Infiltrator

# Set random seed for reproducibility
#Random.seed!(42)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# 🎯 MAIN SIMULATION PARAMETERS
const NUM_STEPS = 20            # Total simulation steps
const PLANNING_MODE = :sweep         # Use policy tree planning (:script, :policy, :random, :sweep, :greedy, :future_actions, :prior_based)
#const modes = [:pbvi, :prior_based, :random]
const modes = [:pbvi, :prior_based, :random]
const N_RUNS = 200
const MAX_BATTERY = 10000.0
const CHARGING_RATE = 3.0
const OBSERVATION_COST = 0.0

# 🌍 ENVIRONMENT PARAMETERS
const GRID_WIDTH = 5                  # Grid width (columns)
const GRID_HEIGHT = 5                 # Grid height (rows)
const INITIAL_EVENTS = 1              # Number of initial events
const MAX_SENSING_TARGETS = 1         # Maximum cells an agent can sense per step
const DISCOUNT_FACTOR = 0.95        # POMDP discount factor
const MAX_PROB_MASS = 0.6            # Maximum probability mass to keep when pruning belief branches (for macro_approx)

# 🤖 AGENT PARAMETERS
const NUM_AGENTS = 2                  # Number of agents (two agents with complex trajectories)
const PLANNING_HORIZON = 10           # Planning horizon for macro-scripts (matches trajectory period)
const SENSOR_NOISE = 0.0              # Perfect observations

# 📡 COMMUNICATION PARAMETERS
const CONTACT_HORIZON = 10             # Steps until next sync opportunity
const GROUND_STATION_X = 2            # Ground station X position (center)
const GROUND_STATION_Y = 1            # Ground station Y position (center)

# 🎯 REWARD FUNCTION CONFIGURATION
const ENTROPY_WEIGHT = 1.0            # w_H: Weight for entropy reduction (coordination)
const VALUE_WEIGHT = 0.0              # w_F: Weight for state value (detection priority)
const INFORMATION_STATES = [1, 2]     # I_1: No event, I_2: Event
const STATE_VALUES = [0.1, 0.9]      # F_1: No event value, F_2: Event value

# =============================================================================
# HETEROGENEOUS RSP (Random Spread Process) MODEL PARAMETERS
# =============================================================================

# Import the existing environment and planner modules
include("../src/MyProject.jl")
using .MyProject

# Import specific types from MyProject
using .MyProject: Agent, SensingAction, GridObservation, EventState, NO_EVENT, EVENT_PRESENT
using .MyProject: ComplexTrajectory, RangeLimitedSensor
using .MyProject: EventState2, NO_EVENT_2, EVENT_PRESENT_2, EventMap, DynamicsMode, toy_dbn, rsp
using .MyProject: EventDynamics, SpatialGrid

# Import functions
using .MyProject.Agents.TrajectoryPlanner: get_position_at_time, execute_plan, create_complex_trajectory
using .MyProject.Types: save_agent_actions_to_csv, calculate_and_save_ndd_metrics, EnhancedEventTracker, initialize_enhanced_event_tracker, update_enhanced_event_tracking!, mark_observed_events_with_time!, get_event_statistics, save_event_tracking_data, save_uncertainty_evolution_data, save_sync_event_data, create_observation_heatmap

# Sync reward configuration with PBVI planner
using .MyProject.Planners.GroundStation.MacroPlannerPBVI: set_reward_config_from_main
set_reward_config_from_main(ENTROPY_WEIGHT, VALUE_WEIGHT, INFORMATION_STATES, STATE_VALUES)

# Import specific modules
using .Environment
using .Environment: GridState
using .Environment.EventDynamicsModule
using .Planners.GroundStation
using .Planners.MacroPlannerAsync
using .Planners.PolicyTreePlanner
using .Planners.MacroPlannerRandom
# using .Agents.BeliefManagement: initialize_global_belief

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

println("✅ All modules imported successfully")

# =============================================================================
# CROSS-SHAPED SENSOR FUNCTIONS
# =============================================================================

"""
Get field of regard for an agent at a specific position (cross-shaped visibility)
"""
function get_cross_field_of_regard(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Cross-shaped sensor: agent's position and adjacent cells in cardinal directions
    for dx in -1:1, dy in -1:1
        nx, ny = x + dx, y + dy
        if 1 <= nx <= env.width && 1 <= ny <= env.height
            # Only include cross pattern (not diagonal)
            if (dx == 0 && dy == 0) || (dx == 0 && dy != 0) || (dx != 0 && dy == 0)
                push!(fov_cells, (nx, ny))
            end
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
        println("  Sensor type: $(typeof(agent.sensor))")
        println("  Positions over time:")
        for t in 0:15
            pos = get_position_at_time(agent.trajectory, t, agent.phase_offset)
            println("    Time $(t): $(pos)")
        end
        println()
    end
end

"""
Create agents with complex trajectories and cross-shaped sensors
"""
function create_complex_agents()
    agents = Agent[]
    
    # Agent 1: Complex trajectory starting from second column
    # Waypoints: (2,1) -> (2,2) -> (2,3) -> (2,4) -> (2,5) -> (4,1) -> (4,2) -> (4,3) -> (4,4) -> (4,5) -> repeat
    waypoints1 = [
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),  # Second column, going up
        (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)   # Fourth column, going up
    ]
    trajectory1 = create_complex_trajectory(waypoints1, 10)  # Period = 10
    
    # Agent 2: Complex trajectory starting from first column, different pattern
    # Waypoints: (1,1) -> (1,2) -> (1,3) -> (1,4) -> (1,5) -> (3,1) -> (3,2) -> (3,3) -> (3,4) -> (3,5) -> repeat
    waypoints2 = [
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),  # Second column, going up
        (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)   # Fourth column, going up
    ]
    trajectory2 = create_complex_trajectory(waypoints2, 10)  # Period = 10
    
    # Cross-shaped sensors for both agents
    sensor1 = RangeLimitedSensor(0.0, pi/2, SENSOR_NOISE, :cross)
    sensor2 = RangeLimitedSensor(0.0, pi/2, SENSOR_NOISE, :cross)
    
    # Create agents with different phase offsets
    agent1 = Agent(1, trajectory1, sensor1, 0, MAX_BATTERY, CHARGING_RATE, OBSERVATION_COST)
    agent2 = Agent(2, trajectory2, sensor2, 3, MAX_BATTERY, CHARGING_RATE, OBSERVATION_COST)  # Phase offset of 5
    
    push!(agents, agent1)
    push!(agents, agent2)
    
    println("🤖 Created $(NUM_AGENTS) agents with complex trajectories and cross-shaped sensors:")
    for agent in agents
        println("  Agent $(agent.id): complex trajectory with period $(agent.trajectory.period), phase offset $(agent.phase_offset)")
        println("    Sensor: RangeLimitedSensor with :cross pattern (cross-shaped visibility)")
        println("    Starting position: $(get_position_at_time(agent.trajectory, 0, agent.phase_offset))")
    end
    
    return agents
end

"""
Create test environment with heterogeneous RSP dynamics
"""
function create_rsp_environment()
    # Create event dynamics (not used for RSP, but required by constructor)
    event_dynamics = EventDynamics(0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Create agents
    agents = create_complex_agents()
    
    # Create spatial grid environment with RSP dynamics
    env = SpatialGrid(GRID_WIDTH, GRID_HEIGHT, event_dynamics, agents, 0.0, DISCOUNT_FACTOR, INITIAL_EVENTS, MAX_SENSING_TARGETS, (GROUND_STATION_X, GROUND_STATION_Y), nothing, MAX_PROB_MASS)
    
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
        aspect_ratio=:equal, size=(600, 600), legend=false,
        xlabel="X Coordinate", ylabel="Y Coordinate",
        grid=false,
        title="RSP $(width)x$(height) Complex Trajectories - Time Step $(time_step) - γ=$(DISCOUNT_FACTOR), Events: $(count(==(EVENT_PRESENT), environment_state)) | Agents: $(length(agents))",
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
        for_cells = get_cross_field_of_regard(agent, pos, (width=width, height=height))
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
        plot(frame, size=(600, 600))
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

# =============================================================================
# EVENT TRACKING STRUCTURES
# =============================================================================



# =============================================================================
# METRIC SAVING FUNCTIONS
# =============================================================================

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
        println(file, "  Total plans computed: $(gs_state.num_plans_computed)")
        println(file, "  Total planning time: $(round(gs_state.total_planning_time, digits=3)) seconds")
        if gs_state.num_plans_computed > 0
            avg_planning_time = gs_state.total_planning_time / gs_state.num_plans_computed
            println(file, "  Average planning time per plan: $(round(avg_planning_time, digits=3)) seconds")
        end
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
        println(file, "")
        
        # Performance metrics
        println(file, "PERFORMANCE METRICS:")
        println(file, "  Final event observation percentage: $(round(event_observation_percentage, digits=1))%")
        println(file, "  Final average uncertainty: $(round(avg_uncertainty[end], digits=3))")
        println(file, "  Normalized Detection Delay (expected lifetime): $(round(ndd_expected_lifetime, digits=3))")
        println(file, "  Normalized Detection Delay (actual lifetime): $(round(ndd_actual_lifetime, digits=3))")
        println(file, "")
        
        # Uncertainty evolution
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

# =============================================================================
# RSP SIMULATION FUNCTIONS
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
Simulate asynchronous centralized planning with RSP using replay environment
"""
function simulate_rsp_async_planning_replay(replay_env::ReplayEnvironment, num_steps::Int=NUM_STEPS, run_number::Int=1, planning_mode::Symbol=:script)
    println("🚀 Starting RSP Asynchronous Centralized Planning Simulation (Replay)")
    println("====================================================================")
    println("Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
    println("Agents: $(NUM_AGENTS) (complex trajectories)")
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
        
        # Update global belief with new observations using t_clean logic
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
            # If no global belief yet, use uniform values
            uniform_uncertainty = fill(1, GRID_HEIGHT, GRID_WIDTH)
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
    
    # Calculate Normalized Detection Delay (lifetime-normalized)
    ndd_life = Types.calculate_ndd_expected_lifetime(event_tracker, env, NUM_STEPS)
    
    # Print final results
    println("\n📈 RSP Simulation Results (Replay)")
    println("===================================")
    println("Event Observation Performance:")
    println("  Total unique events that appeared: $(total_events)")
    println("  Total unique events observed: $(observed_events)")
    println("  Event observation percentage: $(round(event_observation_percentage, digits=1))%")
    println("  Normalized Detection Delay (lifetime): $(round(ndd_life, digits=3))")
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
    
    return gs_state, agents, event_observation_percentage, sync_events, environment_evolution, action_history, event_tracker, uncertainty_evolution, average_uncertainty_per_timestep, ndd_life, belief_event_present_evolution
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
        println("🎯 RSP 5x5 Complex Trajectory Test")
        println("==================================")
        println("Configuration:")
        println("  Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
        println("  Agents: $(NUM_AGENTS) (complex trajectories)")
        println("  Planning horizon: $(PLANNING_HORIZON)")
        println("  Planning mode: $(PLANNING_MODE)")
        println("  Dynamics: Heterogeneous RSP (Replay)")
        println("  Cross-shaped sensor: true")
        println("  Run: $(n)/$(N_RUNS)")

        # Debug: show agent positions and trajectories
        debug_agent_positions(replay_env.env.agents)

        # Create environment distribution visualization
        create_environment_distribution_plot(replay_env.env.rsp_params, results_base_dir, n)

        # Run the simulation with replay
        gs_state, agents, percentage, sync_events, env_evolution, action_history, event_tracker, uncertainty_evolution, uncertainty_avg, ndd_life, belief_event_present_evolution = simulate_rsp_async_planning_replay(replay_env, NUM_STEPS, n, mode)

        println("\n✅ RSP test completed!")
        println("📊 Final event observation percentage: $(round(percentage, digits=1))%")
        println("📊 Final average uncertainty: $(round(uncertainty_avg[end], digits=3))") 
        println("📊 Final Normalized Detection Delay (lifetime): $(round(ndd_life, digits=3))")

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

        # Save performance metrics
        println("\n📊 Saving Performance Metrics...")
        println("===============================")
        
        # Calculate both NDD metrics for performance report
        ndd_expected_lifetime = Types.calculate_ndd_expected_lifetime(event_tracker, replay_env.env, NUM_STEPS)
        ndd_actual_lifetime = Types.calculate_ndd_actual_lifetime(event_tracker, NUM_STEPS)
        
        # Save main performance metrics
        save_performance_metrics(gs_state, uncertainty_avg, percentage, ndd_expected_lifetime, ndd_actual_lifetime, replay_env.env, agents, results_base_dir, n, PLANNING_MODE)
        
        # Save agent actions to CSV
        Types.save_agent_actions_to_csv(action_history, results_base_dir, n, PLANNING_MODE, NUM_STEPS)
        
        # Calculate and save NDD metrics with both expected and actual lifetimes
        ndd_expected, ndd_actual, ndd_filepath = Types.calculate_and_save_ndd_metrics(event_tracker, replay_env.env, NUM_STEPS, results_base_dir, n, PLANNING_MODE)
        
        # Save detailed event tracking data
        Types.save_event_tracking_data(event_tracker, results_base_dir, n, PLANNING_MODE)
        
        # Save uncertainty evolution data
        Types.save_uncertainty_evolution_data(uncertainty_evolution, uncertainty_avg, results_base_dir, n, PLANNING_MODE)
        
        # Save sync event data
        Types.save_sync_event_data(sync_events, results_base_dir, n, PLANNING_MODE)
        
        # Create and save observation heatmap
        Types.create_observation_heatmap(action_history, GRID_WIDTH, GRID_HEIGHT, results_base_dir, n, PLANNING_MODE)
        
        # Create visualizations
        println("\n🎨 Creating Visualizations...")
        println("============================")
        
        # Create simulation animation
        anim = create_rsp_animation(agents, NUM_STEPS, env_evolution, action_history, (GROUND_STATION_X, GROUND_STATION_Y), results_base_dir, n, PLANNING_MODE)
        
        # Create belief animation for P(Event Present)
        create_belief_event_animation(belief_event_present_evolution, results_base_dir, n, PLANNING_MODE)
        
        println("\n✅ RSP simulation completed!")
        println("📁 Check the results folder for:")
        println("  - Run $(n)/$(PLANNING_MODE)/animations/ (main simulation animation and belief animation)")
        println("  - Run $(n)/$(PLANNING_MODE)/metrics/ (performance metrics, agent actions, NDD metrics)")
        println("  - Run $(n)/environment/ (environment parameter distribution)")
    end
end

println("\n🎉 All simulations completed!")
println("📁 Final results saved in: $(results_base_dir)") 
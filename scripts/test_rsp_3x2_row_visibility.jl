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
const NUM_STEPS = 10             # Total simulation steps
const PLANNING_MODE = :future_actions         # Use macro-script planning (:script, :policy, :random, :sweep, :greedy, :future_actions)
const modes = [:future_actions]
const N_RUNS = 1
# Planning modes:
#   :script - Exact belief evolution with macro-script planning
#   :policy - Policy tree planning
#   :random - Random action selection (baseline for comparison)
#   :sweep - Systematic sweep over columns in each row
#   :greedy - Greedy selection maximizing entropy * event_probability
#   :future_actions - Exact planning considering other agents' possible future actions

# 🌍 ENVIRONMENT PARAMETERS
const GRID_WIDTH = 3                  # Grid width (columns)
const GRID_HEIGHT = 4                 # Grid height (rows)
const INITIAL_EVENTS = 1              # Number of initial events
const MAX_SENSING_TARGETS = 1         # Maximum cells an agent can sense per step
const SENSOR_RANGE = 0.0              # Sensor range for agents (0.0 = row-only visibility)
const DISCOUNT_FACTOR = 0.95        # POMDP discount factor

# 🤖 AGENT PARAMETERS
const NUM_AGENTS = 2                  # Number of agents (one per row)
const PLANNING_HORIZON = 5            # Planning horizon for macro-scripts
const SENSOR_FOV = pi/2               # Field of view angle (radians)
const SENSOR_NOISE = 0.0              # Perfect observations

# 📡 COMMUNICATION PARAMETERS
const CONTACT_HORIZON = 5             # Steps until next sync opportunity
const GROUND_STATION_X = 2            # Ground station X position
const GROUND_STATION_Y = 1            # Ground station Y position

# =============================================================================
# RSP (Random Spread Process) MODEL PARAMETERS
# -----------------------------------------------------------------------------
# These constants control the stochastic event dynamics in the RSP model.
# Each parameter has a specific meaning in the context of event spread and decay:
#
#   RSP_LAMBDA:  Local ignition intensity (λ) - could be λmap[y,x]; 0–1
#                - Additional ignition probability from local conditions
#   RSP_BETA0:   Spontaneous (background) ignition probability when no neighbors burn
#                - Base probability of spontaneous event birth
#   RSP_ALPHA:   Contagion contribution of each active neighbor
#                - How much each neighboring event increases birth probability
#   RSP_DELTA:   Probability the fire persists (EVENT→EVENT)
#                - Probability of event survival/continuation
#   RSP_MU:      Probability the fire dies (EVENT→NO_EVENT)
#                - Probability of event death/extinction
#
# Tune these parameters to explore different behaviors.
# -----------------------------------------------------------------------------
const RSP_LAMBDA = 0.001   # Local ignition intensity (λ)
const RSP_BETA0  = 0.001   # Spontaneous ignition probability (β₀)
const RSP_ALPHA  = 0.5   # Contagion contribution per neighbor (α)
const RSP_DELTA  = 0.8    # Event persistence probability (δ)
const RSP_MU     = 1.0 - RSP_DELTA  # Event death probability (μ = 1 - δ)
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

# Import specific modules
using .Environment
using .Environment: GridState
using .Environment.EventDynamicsModule
using .Planners.GroundStation
using .Planners.MacroPlannerAsync
using .Planners.PolicyTreePlanner
using .Planners.MacroPlannerRandom # Added for random planner
using .Planners.MacroPlannerAsyncFutureActions # Added for future actions planner

# Import RSP functions
import .Environment.EventDynamicsModule: transition_rsp!, get_transition_probability_rsp
import .Agents.BeliefManagement: predict_belief_rsp

# Import functions from MacroPlannerAsync
import .MacroPlannerAsync: initialize_uniform_belief, get_known_observations_at_time, has_known_observation, get_known_observation, evolve_no_obs, collapse_belief_to
import .MacroPlannerAsyncFutureActions: initialize_uniform_belief, get_known_observations_at_time, has_known_observation, get_known_observation, evolve_no_obs, collapse_belief_to

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
Enhanced event tracking that also tracks detection times
"""
mutable struct EnhancedEventTracker
    cell_active_event_id::Dict{Tuple{Int,Int}, Union{Nothing,Int}}
    event_registry::Dict{Int, Dict}
    next_event_id::Int
end

"""
Save planning time statistics and performance metrics to a file
"""
function save_performance_metrics(gs_state, avg_uncertainty, event_observation_percentage, ndd_life, env, agents, results_dir, run_number, planning_mode)
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
        println(file, "  RSP parameters:")
        println(file, "    λ (lambda): $(RSP_LAMBDA)")
        println(file, "    β₀ (beta0): $(RSP_BETA0)")
        println(file, "    α (alpha): $(RSP_ALPHA)")
        println(file, "    δ (delta): $(RSP_DELTA)")
        println(file, "    μ (mu): $(RSP_MU)")
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
        println(file, "  Normalized Detection Delay (lifetime): $(round(ndd_life, digits=3))")
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
        sensor = RangeLimitedSensor(SENSOR_RANGE, SENSOR_FOV, SENSOR_NOISE)
        # Agent with phase offset based on starting row
        agent = Agent(i, trajectory, sensor, phase_offset)
        push!(agents, agent)
    end
    
    println("🤖 Created $(NUM_AGENTS) agents with row-only visibility:")
    for agent in agents
        println("  Agent $(agent.id): starts in row $(agent.trajectory.start_y + agent.phase_offset), moves upward 1→2→3→1→2→3..., Phase offset $(agent.phase_offset)")
    end
    
    return agents
end

"""
Create test environment with RSP dynamics
"""
function create_rsp_environment()
    # Create event dynamics (not used for RSP, but required by constructor)
    event_dynamics = EventDynamics(0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Create agents
    agents = create_row_agents()
    
    # Create spatial grid environment with RSP dynamics
    env = SpatialGrid(GRID_WIDTH, GRID_HEIGHT, event_dynamics, agents, SENSOR_RANGE, DISCOUNT_FACTOR, INITIAL_EVENTS, MAX_SENSING_TARGETS, (GROUND_STATION_X, GROUND_STATION_Y))
    
    # Update to RSP dynamics
    env.dynamics = rsp  # Use RSP dynamics (enum value)
    
    # Create ignition probability map for RSP (using RSP_LAMBDA as base)
    env.ignition_prob = fill(RSP_LAMBDA, GRID_HEIGHT, GRID_WIDTH)
    
    # Add RSP parameters to environment
    env.rsp_params = (
        lambda=RSP_LAMBDA,
        beta0=RSP_BETA0,
        alpha=RSP_ALPHA,
        delta=RSP_DELTA,
        mu=RSP_MU
    )
    
    println("🌍 Created RSP test environment:")
    println("  Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
    println("  Initial events: $(INITIAL_EVENTS)")
    println("  Max sensing targets: $(MAX_SENSING_TARGETS)")
    println("  Dynamics: RSP (Random Spread Process)")
    println("  RSP params: $(env.rsp_params)")
    
    return env
end

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

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
    hline!([0.693], color=:red, linestyle=:dash, linewidth=1, label="Uniform Distribution")
    
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

"""
Get event statistics from tracker
"""
function get_event_statistics(tracker::EventTracker)
    total_events = length(tracker.event_registry)
    observed_events = count(e -> e[:observed], collect(values(tracker.event_registry)))
    return total_events, observed_events
end

"""
Get event statistics from enhanced tracker
"""
function get_event_statistics(tracker::EnhancedEventTracker)
    total_events = length(tracker.event_registry)
    observed_events = count(e -> e[:observed], collect(values(tracker.event_registry)))
    return total_events, observed_events
end

"""
Calculate Normalized Detection Delay (lifetime-normalized)
NDD_life = (1/|E_det|) * sum_{e in E_det} (t_detect(e) - t_start(e)) / E[L_e]
where E[L_e] is the expected duration of event e
"""
function calculate_normalized_detection_delay_lifetime(event_tracker::EventTracker)
    detected_events = filter(e -> e[:observed], collect(values(event_tracker.event_registry)))
    
    if isempty(detected_events)
        return 0.0  # No detected events
    end
    
    total_ndd = 0.0
    
    for event in detected_events
        # Calculate detection delay
        t_start = event[:start_time]
        t_detect = event[:detection_time]  # We need to track this
        
        # Calculate expected lifetime E[L_e] for RSP events
        # For RSP, E[L] = 1/μ where μ is the death probability
        # From the RSP parameters: μ = 1 - δ
        μ = 1.0 - RSP_DELTA
        expected_lifetime = 1.0 / μ
        
        # Calculate normalized delay for this event
        detection_delay = t_detect - t_start
        normalized_delay = detection_delay / expected_lifetime
        
        total_ndd += normalized_delay
    end
    
    # Average over all detected events
    ndd_life = total_ndd / length(detected_events)
    
    return ndd_life
end

"""
Calculate Normalized Detection Delay (lifetime-normalized) for enhanced tracker
"""
function calculate_normalized_detection_delay_lifetime(event_tracker::EnhancedEventTracker)
    detected_events = filter(e -> e[:observed], collect(values(event_tracker.event_registry)))
    
    if isempty(detected_events)
        return 0.0  # No detected events
    end
    
    total_ndd = 0.0
    
    for event in detected_events
        # Calculate detection delay
        t_start = event[:start_time]
        t_detect = event[:detection_time]  # We need to track this
        
        # Calculate expected lifetime E[L_e] for RSP events
        # For RSP, E[L] = 1/μ where μ is the death probability
        # From the RSP parameters: μ = 1 - δ
        μ = 1.0 - RSP_DELTA
        expected_lifetime = 1.0 / μ
        
        # Calculate normalized delay for this event
        detection_delay = t_detect - t_start
        normalized_delay = detection_delay / expected_lifetime
        
        total_ndd += normalized_delay
    end
    
    # Average over all detected events
    ndd_life = total_ndd / length(detected_events)
    
    return ndd_life
end

"""
Enhanced event tracking that also tracks detection times
"""
mutable struct EnhancedEventTracker
    cell_active_event_id::Dict{Tuple{Int,Int}, Union{Nothing,Int}}
    event_registry::Dict{Int, Dict}
    next_event_id::Int
end

"""
Initialize enhanced event tracker
"""
function initialize_enhanced_event_tracker()
    return EnhancedEventTracker(
        Dict{Tuple{Int,Int}, Union{Nothing,Int}}(),
        Dict{Int, Dict}(),
        1
    )
end

"""
Update enhanced event tracking for a timestep
"""
function update_enhanced_event_tracking!(tracker::EnhancedEventTracker, prev_environment::Matrix{EventState}, 
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
                :detection_time => nothing,  # Track when event was first detected
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
Mark events as observed with detection time tracking
"""
function mark_observed_events_with_time!(tracker::EnhancedEventTracker, agent_observations::Vector{Tuple{Int, Vector{Tuple{Tuple{Int,Int}, EventState}}}}, current_timestep::Int)
    for (agent_id, observations) in agent_observations
        for (cell, observed_state) in observations
            if observed_state == EVENT_PRESENT
                event_id = tracker.cell_active_event_id[cell]
                if event_id !== nothing
                    event_info = tracker.event_registry[event_id]
                    if !event_info[:observed]
                        # First time this event is observed
                        event_info[:observed] = true
                        event_info[:detection_time] = current_timestep
                    end
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
        
        # Use RSP transition, passing RSP parameters from env
        transition_rsp!(new_state, current_state, env.ignition_prob, Random.GLOBAL_RNG; rsp_params=env.rsp_params)
        
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
            
            prob_no_event = get_transition_probability_rsp(1, current_cell_state, neighbor_states;
                λ=env.rsp_params.lambda,
                β0=env.rsp_params.beta0,
                α=env.rsp_params.alpha,
                δ=env.rsp_params.delta,
                μ=env.rsp_params.mu)
            prob_event = get_transition_probability_rsp(2, current_cell_state, neighbor_states;
                λ=env.rsp_params.lambda,
                β0=env.rsp_params.beta0,
                α=env.rsp_params.alpha,
                δ=env.rsp_params.delta,
                μ=env.rsp_params.mu)
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
            action = execute_plan(agent, plan, plan_type, agent.observation_history)
            push!(joint_actions, action)
            
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
            B = initialize_uniform_belief(env)
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
        else
            # If no global belief yet, use uniform uncertainty
            uniform_uncertainty = fill(0.693, GRID_HEIGHT, GRID_WIDTH)  # log2(2) for uniform distribution
            push!(uncertainty_evolution, uniform_uncertainty)
            push!(average_uncertainty_per_timestep, 0.693)
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
    ndd_life = calculate_normalized_detection_delay_lifetime(event_tracker)
    
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
    
    return gs_state, agents, event_observation_percentage, sync_events, environment_evolution, action_history, event_tracker, uncertainty_evolution, average_uncertainty_per_timestep, ndd_life
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
        
        # Use RSP transition
        transition_rsp!(new_state, current_state, env.ignition_prob, Random.GLOBAL_RNG; rsp_params=env.rsp_params)
        
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
results_base_dir = joinpath("results", "run_$(timestamp)")
if !isdir(results_base_dir)
    mkpath(results_base_dir)
end

println("📁 Results will be saved in: $(results_base_dir)")

# Simulate environment once for fair comparison
println("\n🔥 Simulating environment once for replay system...")
replay_env = simulate_environment_once(NUM_STEPS)

for n in 1:N_RUNS
    for mode in modes
        PLANNING_MODE = mode
        println("🎯 RSP 3x2 Row Visibility Test")
        println("==============================")
        println("Configuration:")
        println("  Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
        println("  Agents: $(NUM_AGENTS) (rows 1 and 3)")
        println("  Planning horizon: $(PLANNING_HORIZON)")
        println("  Planning mode: $(PLANNING_MODE) (:script, :policy, :random, :sweep, :greedy, :future_actions)")
        println("  Dynamics: RSP (Replay)")
        println("  Row-only visibility: true")
        println("  Run: $(n)/5")

        # Debug: show agent positions and trajectories
        debug_agent_positions(replay_env.env.agents)

        # Run the simulation with replay
        gs_state, agents, percentage, sync_events, env_evolution, action_history, event_tracker, uncertainty_evolution, avg_uncertainty, ndd_life = simulate_rsp_async_planning_replay(replay_env, NUM_STEPS, n, mode)

        println("\n✅ RSP test completed!")
        println("📊 Final event observation percentage: $(round(percentage, digits=1))%")
        println("📊 Final average uncertainty: $(round(avg_uncertainty[end], digits=3))") 
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

        # Create visualizations
        println("\n🎨 Creating Visualizations...")
        println("============================")
        
        # Create simulation animation
        anim = create_rsp_animation(agents, NUM_STEPS, env_evolution, action_history, (GROUND_STATION_X, GROUND_STATION_Y), results_base_dir, n, PLANNING_MODE)
        
        # Create uncertainty visualization
        println("\n📊 Creating Uncertainty Visualizations...")
        create_uncertainty_visualizations(uncertainty_evolution, avg_uncertainty, env_evolution, NUM_STEPS, results_base_dir, n, PLANNING_MODE)
        
        # Save performance metrics
        save_performance_metrics(gs_state, avg_uncertainty, percentage, ndd_life, replay_env.env, agents, results_base_dir, n, PLANNING_MODE)
        
        println("\n✅ RSP simulation completed!")
        println("📁 Check the results folder for:")
        println("  - Run $(n)/$(PLANNING_MODE)/animations/ (main simulation animation)")
        println("  - Run $(n)/$(PLANNING_MODE)/plots/ (uncertainty visualizations)")
        println("  - Run $(n)/$(PLANNING_MODE)/metrics/ (performance metrics)")
    end
end

println("\n🎉 All simulations completed!")
println("📁 Final results saved in: $(results_base_dir)")
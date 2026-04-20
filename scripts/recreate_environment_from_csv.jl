"""
Script to recreate environment evolution from event tracking CSV and run additional planning modes
This allows you to run additional planning modes (like prior_based, random) on the same environment
that was used for existing runs, without re-running the entire simulation.
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
using Infiltrator

println("🔄 Starting environment recreation from CSV...")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Configuration - modify these as needed
const GRID_WIDTH = 9
const GRID_HEIGHT = 9
const NUM_AGENTS = 4
const NUM_STEPS = 120
const MAX_BATTERY = 10000.0
const CHARGING_RATE = 3.0
const OBSERVATION_COST = 0.0
const PLANNING_HORIZON = 3
const SENSOR_NOISE = 0.0
const CONTACT_HORIZON = 10
const GROUND_STATION_X = 5
const GROUND_STATION_Y = 5
const DISCOUNT_FACTOR = 0.95
const MAX_SENSING_TARGETS = 1
const MAX_PROB_MASS = 0.6

# Reward function configuration
const ENTROPY_WEIGHT = 0.5
const VALUE_WEIGHT = 0.5
const INFORMATION_STATES = [1, 2]
const STATE_VALUES = [0.1, 0.9]

# Import the existing environment and planner modules
include("../src/MyProject.jl")
using .MyProject

# Import specific types and functions
using .MyProject: Agent, SensingAction, GridObservation, EventState, NO_EVENT, EVENT_PRESENT
using .MyProject: CircularTrajectory, RangeLimitedSensor
using .MyProject: EventState2, NO_EVENT_2, EVENT_PRESENT_2, EventMap, DynamicsMode, toy_dbn, rsp
using .MyProject: EventDynamics, SpatialGrid

# Import functions
using .MyProject.Agents.TrajectoryPlanner: get_position_at_time, execute_plan, create_circular_trajectory
using .MyProject.Types: save_agent_actions_to_csv, calculate_and_save_ndd_metrics, EnhancedEventTracker, initialize_enhanced_event_tracker, update_enhanced_event_tracking!, mark_observed_events_with_time!, get_event_statistics, save_event_tracking_data, save_uncertainty_evolution_data, save_sync_event_data, create_observation_heatmap
using .MyProject.Planners.GroundStation.MacroPlannerPBVI: set_reward_config_from_main

# Import specific modules
using .Environment
using .Environment: GridState
using .Environment.EventDynamicsModule
using .Planners.GroundStation
using .Planners.MacroPlannerAsync
using .Planners.PolicyTreePlanner
using .Planners.MacroPlannerRandom
using .Planners.MacroPlannerGreedy

# Import RSP functions
import .Environment.EventDynamicsModule: transition_rsp!
import .MacroPlannerAsync: initialize_uniform_belief, get_known_observations_at_time, has_known_observation, get_known_observation, evolve_no_obs, collapse_belief_to
import .GroundStation: get_average_planning_time

println("✅ All modules imported successfully")

# =============================================================================
# ENVIRONMENT RECREATION FUNCTIONS
# =============================================================================

"""
Parse event tracking CSV to extract event timeline
Returns a dictionary mapping timestep -> list of events that are active
"""
function parse_event_tracking_csv(csv_path::String)
    println("📊 Parsing event tracking CSV: $(basename(csv_path))")
    
    # Read the CSV file
    df = CSV.read(csv_path, DataFrame)
    
    # Create timeline of events
    event_timeline = Dict{Int, Vector{Tuple{Int, Int, Int}}}()  # timestep -> [(event_id, x, y)]
    
    for row in eachrow(df)
        event_id = row.event_id
        start_time = row.start_time
        end_time = row.end_time == -1 ? NUM_STEPS : row.end_time
        cell_x = row.cell_x
        cell_y = row.cell_y
        
        # Add this event to all timesteps it's active
        for t in start_time:end_time
            if !haskey(event_timeline, t)
                event_timeline[t] = Tuple{Int, Int, Int}[]
            end
            push!(event_timeline[t], (event_id, cell_x, cell_y))
        end
    end
    
    println("  Found $(nrow(df)) unique events")
    println("  Timeline spans $(minimum(keys(event_timeline))) to $(maximum(keys(event_timeline))) timesteps")
    
    return event_timeline
end

"""
Recreate environment state at a specific timestep from event timeline
"""
function recreate_environment_state(event_timeline::Dict{Int, Vector{Tuple{Int, Int, Int}}}, timestep::Int)
    state = fill(NO_EVENT, GRID_HEIGHT, GRID_WIDTH)
    
    if haskey(event_timeline, timestep)
        for (event_id, x, y) in event_timeline[timestep]
            if 1 <= x <= GRID_WIDTH && 1 <= y <= GRID_HEIGHT
                state[y, x] = EVENT_PRESENT
            end
        end
    end
    
    return state
end

"""
Create agents with the same circular trajectories as the original simulation
"""
function create_circular_agents()
    agents = Agent[]
    
    # Same trajectory configuration as in main_9x9_circular.jl
    trajectory1 = create_circular_trajectory(3.0, 4.0, 2.0, 12)
    trajectory2 = create_circular_trajectory(7.0, 4.0, 2.0, 12)
    trajectory3 = create_circular_trajectory(3.0, 6.0, 2.0, 12)
    trajectory4 = create_circular_trajectory(7.0, 6.0, 2.0, 12)
    
    # 9-cell sensors for all agents
    sensor1 = RangeLimitedSensor(1.5, pi/2, SENSOR_NOISE, :circular)
    sensor2 = RangeLimitedSensor(1.5, pi/2, SENSOR_NOISE, :circular)
    sensor3 = RangeLimitedSensor(1.5, pi/2, SENSOR_NOISE, :circular)
    sensor4 = RangeLimitedSensor(1.5, pi/2, SENSOR_NOISE, :circular)
    
    # Create agents with same phase offsets
    agent1 = Agent(1, trajectory1, sensor1, 0, MAX_BATTERY, CHARGING_RATE, OBSERVATION_COST)
    agent2 = Agent(2, trajectory2, sensor2, 3, MAX_BATTERY, CHARGING_RATE, OBSERVATION_COST)
    agent3 = Agent(3, trajectory3, sensor3, 6, MAX_BATTERY, CHARGING_RATE, OBSERVATION_COST)
    agent4 = Agent(4, trajectory4, sensor4, 9, MAX_BATTERY, CHARGING_RATE, OBSERVATION_COST)
    
    push!(agents, agent1, agent2, agent3, agent4)
    
    println("🤖 Created $(NUM_AGENTS) agents with circular trajectories")
    return agents
end

"""
Create environment with RSP dynamics (we'll use the recreated states instead of RSP transitions)
"""
function create_rsp_environment()
    # Create event dynamics (not used for recreated environment, but required by constructor)
    event_dynamics = EventDynamics(0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Create agents
    agents = create_circular_agents()
    
    # Create spatial grid environment
    env = SpatialGrid(GRID_WIDTH, GRID_HEIGHT, event_dynamics, agents, 0.0, DISCOUNT_FACTOR, 0, MAX_SENSING_TARGETS, (GROUND_STATION_X, GROUND_STATION_Y), nothing, MAX_PROB_MASS)
    
    # Set to RSP dynamics
    env.dynamics = rsp
    
    # Create dummy parameter maps (we won't use RSP transitions)
    param_maps = Types.create_heterogeneous_rsp_maps(GRID_HEIGHT, GRID_WIDTH)
    env.ignition_prob = param_maps.lambda_map
    env.rsp_params = param_maps
    
    println("🌍 Created environment for recreated simulation")
    return env
end

"""
Get field of regard for an agent at a specific position (9-cell: agent + 8 neighbors)
"""
function get_nine_cell_field_of_regard(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # 9-cell sensor: agent's position and all 8 adjacent cells (including diagonals)
    for dx in -1:1, dy in -1:1
        nx, ny = x + dx, y + dy
        if 1 <= nx <= env.width && 1 <= ny <= env.height
            push!(fov_cells, (nx, ny))
        end
    end
    
    return fov_cells
end

"""
Simulate with recreated environment using a specific planning mode
"""
function simulate_recreated_environment(
    event_timeline::Dict{Int, Vector{Tuple{Int, Int, Int}}},
    planning_mode::Symbol,
    run_number::Int,
    results_dir::String
)
    println("🚀 Starting recreated simulation with planning mode: $(planning_mode)")
    
    # Create environment and agents
    env = create_rsp_environment()
    agents = env.agents
    
    # Initialize ground station
    gs_state = GroundStation.initialize_ground_station(env, agents, num_states=2)
    
    # Initialize enhanced event tracker
    event_tracker = initialize_enhanced_event_tracker()
    
    # Track performance metrics
    sync_events = []
    environment_evolution = Matrix{EventState}[]
    action_history = Vector{Vector{SensingAction}}()
    uncertainty_evolution = Matrix{Float64}[]
    average_uncertainty_per_timestep = Float64[]
    belief_event_present_evolution = Matrix{Float64}[]
    
    # Get initial environment state from timeline
    current_environment = recreate_environment_state(event_timeline, 0)
    prev_environment = copy(current_environment)
    
    # Update event tracking for initial state
    update_enhanced_event_tracking!(event_tracker, fill(NO_EVENT, GRID_HEIGHT, GRID_WIDTH), current_environment, 0)
    
    println("📊 Starting recreated simulation...")
    
    for t in 0:(NUM_STEPS-1)
        if t % 20 == 0
            println("⏰ Time step $(t)")
        end
        
        # Check for synchronization opportunities
        old_sync_times = copy(gs_state.agent_last_sync)
        GroundStation.maybe_sync!(env, gs_state, agents, t, planning_mode=planning_mode)
        
        # Record sync events
        for (agent_id, old_time) in old_sync_times
            if gs_state.agent_last_sync[agent_id] != old_time
                push!(sync_events, (t, agent_id))
                if t % 20 == 0
                    println("📡 Sync event: Agent $(agent_id) at time $(t)")
                end
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
            
            # Charge battery every timestep
            agent.battery_level = min(agent.max_battery, agent.battery_level + agent.charging_rate)
            
            if !isempty(action.target_cells)
                # Get observation using POMDP observation model
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
            else
                # Wait action - create empty observation
                empty_observation = GridObservation(agent.id, Tuple{Int,Int}[], EventState[], [])
                push!(agent.observation_history, empty_observation)
            end
        end
        
        # Mark events as observed with detection time tracking
        mark_observed_events_with_time!(event_tracker, agent_observations, t)
        
        # Record environment state and actions for visualization
        push!(environment_evolution, copy(current_environment))
        push!(action_history, joint_actions)
        
        # Update global belief with new observations
        if gs_state.global_belief !== nothing
            # Determine t_clean (last time where all observation outcomes are known)
            tau = gs_state.agent_last_sync
            t_clean = minimum([tau[j] for j in keys(tau)])
            
            # Roll forward deterministically from uniform belief to t_clean using known observations
            B = MacroPlannerAsync.initialize_uniform_belief(env)
            for t_roll in 0:(t_clean-1)
                B = evolve_no_obs(B, env)
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
            uniform_prob = fill(0.5, GRID_HEIGHT, GRID_WIDTH)
            push!(belief_event_present_evolution, uniform_prob)
        end
        
        # Update environment using recreated timeline (not RSP transition)
        if t < NUM_STEPS - 1
            # Get next state from recreated timeline
            current_environment = recreate_environment_state(event_timeline, t + 1)
            
            # Update event tracking for the new timestep
            update_enhanced_event_tracking!(event_tracker, prev_environment, current_environment, t + 1)
            prev_environment .= current_environment
        end
    end
    
    # Calculate final statistics
    total_events, observed_events = get_event_statistics(event_tracker)
    event_observation_percentage = total_events > 0 ? (observed_events / total_events) * 100.0 : 0.0
    
    # Calculate Normalized Detection Delay
    ndd_life = Types.calculate_ndd_expected_lifetime(event_tracker, env, NUM_STEPS)
    
    # Print final results
    println("\n📈 Recreated Simulation Results")
    println("==============================")
    println("Planning mode: $(planning_mode)")
    println("Event Observation Performance:")
    println("  Total unique events that appeared: $(total_events)")
    println("  Total unique events observed: $(observed_events)")
    println("  Event observation percentage: $(round(event_observation_percentage, digits=1))%")
    println("  Normalized Detection Delay (lifetime): $(round(ndd_life, digits=3))")
    
    return gs_state, agents, event_observation_percentage, sync_events, environment_evolution, action_history, event_tracker, uncertainty_evolution, average_uncertainty_per_timestep, ndd_life, belief_event_present_evolution
end

"""
Save performance metrics for the recreated simulation
"""
function save_recreated_performance_metrics(gs_state, avg_uncertainty, event_observation_percentage, ndd_expected_lifetime, ndd_actual_lifetime, env, agents, results_dir, run_number, planning_mode)
    # Create the metrics directory path
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(planning_mode), "metrics")
    if !isdir(metrics_dir)
        mkpath(metrics_dir)
    end
    
    filename = "performance_metrics_$(planning_mode)_run$(run_number).txt"
    filepath = joinpath(metrics_dir, filename)
    
    open(filepath, "w") do file
        println(file, "="^60)
        println(file, "PERFORMANCE METRICS REPORT (RECREATED FROM CSV)")
        println(file, "="^60)
        println(file, "Generated: $(now())")
        println(file, "Planning Mode: $(planning_mode)")
        println(file, "Run Number: $(run_number)")
        println(file, "Source: Recreated from existing event tracking CSV")
        println(file, "")
        
        # Environment parameters
        println(file, "ENVIRONMENT PARAMETERS:")
        println(file, "  Grid size: $(env.width) x $(env.height)")
        println(file, "  Environment: Recreated from event tracking CSV")
        println(file, "  Dynamics: RSP (Recreated)")
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
            if i % 10 == 1 || i == length(avg_uncertainty)
                println(file, "    Step $(i): $(round(uncertainty, digits=3))")
            end
        end
        println(file, "")
        
        println(file, "="^60)
    end
    
    println("📁 Performance metrics saved to: $(filepath)")
    return filepath
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

function main()
    println("🔄 Environment Recreation from CSV")
    println("==================================")
    # Configuration - modify these paths as needed
    source_csv_path = "../results/run_2025-09-02T18-43-26-307/Run 3/pbvi_0_5_0_5/metrics/event_tracking_pbvi_0_5_0_5_run3.csv"
    results_base_dir = "../results/run_2025-09-02T18-43-26-307"
    run_number = 3
    
    # Planning modes to run on the recreated environment
    additional_modes = [:prior_based, :random]  # Add more modes as needed
    
    # Check if source CSV exists
    if !isfile(source_csv_path)
        println("❌ Source CSV file not found: $(source_csv_path)")
        println("Please update the source_csv_path variable with the correct path.")
        return
    end
    
    println("📊 Source CSV: $(source_csv_path)")
    println("📁 Results directory: $(results_base_dir)")
    println("🎯 Additional modes to run: $(join(additional_modes, ", "))")
    
    # Parse event tracking CSV
    event_timeline = parse_event_tracking_csv(source_csv_path)
    
    # Run each additional planning mode
    for planning_mode in additional_modes
        println("\n" * "="^60)
        println("🚀 Running planning mode: $(planning_mode)")
        println("="^60)
        
        # Set reward configuration if needed
        if planning_mode == :pbvi
            set_reward_config_from_main(ENTROPY_WEIGHT, VALUE_WEIGHT, INFORMATION_STATES, STATE_VALUES)
        end
        
        # Run simulation with recreated environment
        gs_state, agents, percentage, sync_events, env_evolution, action_history, event_tracker, uncertainty_evolution, uncertainty_avg, ndd_life, belief_event_present_evolution = simulate_recreated_environment(
            event_timeline, planning_mode, run_number, results_base_dir
        )
        
        println("\n✅ Recreated simulation completed for $(planning_mode)!")
        println("📊 Final event observation percentage: $(round(percentage, digits=1))%")
        println("📊 Final average uncertainty: $(round(uncertainty_avg[end], digits=3))")
        println("📊 Final Normalized Detection Delay: $(round(ndd_life, digits=3))")
        
        # Save performance metrics
        println("\n📊 Saving Performance Metrics...")
        ndd_expected_lifetime = Types.calculate_ndd_expected_lifetime(event_tracker, create_rsp_environment(), NUM_STEPS)
        ndd_actual_lifetime = Types.calculate_ndd_actual_lifetime(event_tracker, NUM_STEPS)
        
        save_recreated_performance_metrics(gs_state, uncertainty_avg, percentage, ndd_expected_lifetime, ndd_actual_lifetime, create_rsp_environment(), agents, results_base_dir, run_number, planning_mode)
        
        # Save other data files
        Types.save_agent_actions_to_csv(action_history, results_base_dir, run_number, planning_mode, NUM_STEPS)
        Types.calculate_and_save_ndd_metrics(event_tracker, create_rsp_environment(), NUM_STEPS, results_base_dir, run_number, planning_mode)
        Types.save_event_tracking_data(event_tracker, results_base_dir, run_number, planning_mode)
        Types.save_uncertainty_evolution_data(uncertainty_evolution, uncertainty_avg, results_base_dir, run_number, planning_mode)
        Types.save_sync_event_data(sync_events, results_base_dir, run_number, planning_mode)
        Types.create_observation_heatmap(action_history, GRID_WIDTH, GRID_HEIGHT, results_base_dir, run_number, planning_mode)
        
        println("✅ All data saved for $(planning_mode)")
    end
    
    println("\n🎉 All recreated simulations completed!")
    println("📁 Results saved in: $(results_base_dir)")
    println("📊 You can now run postprocess_results.jl to analyze all modes together!")
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

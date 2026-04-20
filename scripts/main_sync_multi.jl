#!/usr/bin/env julia

"""
Synchronized Multi-Agent Test Script
Tests synchronized multi-agent planning where all agents have phase 0 and sync with ground station simultaneously.
Uses different trajectories and compares with SB-ABBA (PBVI) approach.
"""

println("🚀 Synchronized Multi-Agent Test starting...")

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Plots
using Dates
using DataFrames
using CSV
using Statistics
Plots.plotlyjs()
using Infiltrator

# Load the project early
include("../src/MyProject.jl")
using .MyProject

# Import specific types from MyProject
using .MyProject: Agent, SensingAction, GridObservation, EventState, NO_EVENT, EVENT_PRESENT
using .MyProject: CircularTrajectory, LinearTrajectory, RangeLimitedSensor
using .MyProject: EventState2, NO_EVENT_2, EVENT_PRESENT_2, EventMap, DynamicsMode, toy_dbn, rsp
using .MyProject: EventDynamics, SpatialGrid

# Import functions
using .MyProject.Agents.TrajectoryPlanner: get_position_at_time, create_linear_trajectory, create_circular_trajectory, execute_plan
using .MyProject.Planners.GroundStation: initialize_ground_station, initialize_global_belief, maybe_sync!, get_agent_plan
using .MyProject.Agents.BeliefManagement: calculate_cell_entropy, get_event_probability, collapse_belief_to, predict_belief_evolution_dbn
using .MyProject.Environment.EventDynamicsModule

# Include the new synchronized multi-agent planner
include("../src/planners/macro_planner_sync_multi.jl")
using .MacroPlannerSyncMulti

# Set random seed for reproducibility
Random.seed!(42)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# 🎯 MAIN SIMULATION PARAMETERS
const NUM_STEPS = 50             # Total simulation steps
const PLANNING_MODE = :sync_multi # Synchronized multi-agent planning
const modes = [:sync_multi, :pbvi]  # Compare sync multi-agent with SB-ABBA (PBVI)
const N_RUNS = 10                # Number of runs for comparison
const MAX_BATTERY = 10000.0
const CHARGING_RATE = 3.0
const OBSERVATION_COST = 0.0

# 🌍 ENVIRONMENT PARAMETERS
const GRID_WIDTH = 7                  # Grid width (columns)
const GRID_HEIGHT = 7                 # Grid height (rows)
const INITIAL_EVENTS = 2              # Number of initial events
const MAX_SENSING_TARGETS = 1         # Maximum cells an agent can sense per step
const SENSOR_RANGE = 0.0              # Sensor range for agents (0.0 = row-only visibility)
const DISCOUNT_FACTOR = 0.95          # POMDP discount factor
const MAX_PROB_MASS = 0.6             # Maximum probability mass to keep when pruning belief branches

# 📡 COMMUNICATION PARAMETERS
const GROUND_STATION_X = 4            # Ground station X position (center of 7x7)
const GROUND_STATION_Y = 4            # Ground station Y position (center of 7x7)

# 🤖 AGENT PARAMETERS
const NUM_AGENTS = 3                  # Number of agents
const PLANNING_HORIZON = 5            # Planning horizon
const SENSOR_NOISE = 0.0              # Perfect observations
const PHASE_OFFSET = 0                # All agents synchronized (phase 0)

# 🎯 REWARD CONFIGURATION
const ENTROPY_WEIGHT = 1.0            # Weight for entropy reduction (coordination)
const VALUE_WEIGHT = 0.5              # Weight for state value (detection priority)
const INFORMATION_STATES = [1, 2]     # I_1: No event, I_2: Event
const STATE_VALUES = [0.1, 1.0]       # F_1: No event value, F_2: Event value

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
get_field_of_regard_at_position(agent, position, env)
Gets field of regard for an agent at a specific position.
"""
function get_field_of_regard_at_position(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Check sensor pattern
    if agent.sensor.pattern == :cross
        # Cross-shaped sensor: agent's position and adjacent cells
        ax, ay = position
        for dx in -1:1, dy in -1:1
            nx, ny = ax + dx, ay + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                # Only include cross pattern (not diagonal)
                if (dx == 0 && dy == 0) || (dx == 0 && dy != 0) || (dx != 0 && dy == 0)
                    push!(fov_cells, (nx, ny))
                end
            end
        end
    elseif agent.sensor.pattern == :circular
        # Circular sensor: agent's position and all 8 adjacent cells (9-cell pattern)
        for dx in -1:1, dy in -1:1
            nx, ny = x + dx, y + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                push!(fov_cells, (nx, ny))
            end
        end
    elseif agent.sensor.pattern == :row_only || agent.sensor.range == 0.0
        # Row-only visibility: agent can only see cells in its current row
        for nx in 1:env.width
            push!(fov_cells, (nx, y))
        end
    else
        # Standard sensor range visibility
        sensor_range = round(Int, agent.sensor.range)
        for dx in -sensor_range:sensor_range
            for dy in -sensor_range:sensor_range
                nx, ny = x + dx, y + dy
                if 1 <= nx <= env.width && 1 <= ny <= env.height
                    # Check if within sensor range
                    distance = sqrt(dx^2 + dy^2)
                    if distance <= agent.sensor.range
                        push!(fov_cells, (nx, ny))
                    end
                end
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
        for t in 0:11  # Show one complete period
            pos = get_position_at_time(agent.trajectory, t, agent.phase_offset)
            println("    Time $(t): $(pos)")
        end
        println()
    end
end

"""
Create agents with different synchronized trajectories
"""
function create_synchronized_agents()
    agents = Agent[]
    
    # Design 3 different trajectories for a 7x7 grid
    # All agents have phase 0 (synchronized)
    # Ground station at (4,4) - center of 7x7 grid
    
    # Agent 1: Linear trajectory - horizontal sweep
    trajectory1 = create_linear_trajectory(1, 2, 7, 2, 8)  # From (1,2) to (7,2), period 8
    
    # Agent 2: Linear trajectory - vertical sweep  
    trajectory2 = create_linear_trajectory(2, 1, 2, 7, 8)  # From (2,1) to (2,7), period 8
    
    # Agent 3: Circular trajectory - around center
    trajectory3 = create_circular_trajectory(4.0, 4.0, 2.0, 8)  # Center (4,4), radius 2, period 8
    
    # Cross-shaped sensors for all agents (5-cell pattern)
    sensor1 = RangeLimitedSensor(1.0, pi/2, SENSOR_NOISE, :cross)
    sensor2 = RangeLimitedSensor(1.0, pi/2, SENSOR_NOISE, :cross)
    sensor3 = RangeLimitedSensor(1.0, pi/2, SENSOR_NOISE, :cross)
    
    # Create agents with synchronized phase (all phase 0)
    agent1 = Agent(1, trajectory1, sensor1, PHASE_OFFSET, MAX_BATTERY, CHARGING_RATE, OBSERVATION_COST)
    agent2 = Agent(2, trajectory2, sensor2, PHASE_OFFSET, MAX_BATTERY, CHARGING_RATE, OBSERVATION_COST)
    agent3 = Agent(3, trajectory3, sensor3, PHASE_OFFSET, MAX_BATTERY, CHARGING_RATE, OBSERVATION_COST)
    
    push!(agents, agent1, agent2, agent3)
    
    println("🤖 Created $(NUM_AGENTS) synchronized agents with different trajectories")
    return agents
end

"""
Create environment with RSP dynamics
"""
function create_rsp_environment()
    # Create event dynamics (not used for RSP, but required by constructor)
    event_dynamics = EventDynamics(0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Create agents
    agents = create_synchronized_agents()
    
    # Create spatial grid environment with RSP dynamics
    env = SpatialGrid(GRID_WIDTH, GRID_HEIGHT, event_dynamics, agents, SENSOR_RANGE, DISCOUNT_FACTOR, INITIAL_EVENTS, MAX_SENSING_TARGETS, (GROUND_STATION_X, GROUND_STATION_Y), nothing, MAX_PROB_MASS)
    
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

"""
Run simulation for a specific planning mode
"""
function run_simulation_mode(env, agents, mode::Symbol, run_id::Int)
    println("\n🔄 Running simulation $(run_id) with mode: $(mode)")
    
    # Initialize ground station
    gs_state = initialize_ground_station(env, agents, num_states=2)
    
    # Initialize global belief
    global_belief = initialize_global_belief(env, num_states=2)
    gs_state.global_belief = global_belief
    
    # Initialize agent plans
    agent_plans = Dict{Int, Any}()
    for agent in agents
        agent_plans[agent.id] = nothing
    end
    
    # Initialize metrics
    total_reward = 0.0
    total_planning_time = 0.0
    num_plans = 0
    belief_entropy_history = Float64[]
    detection_count = 0
    
    # Initialize environment state (ground truth)
    current_environment = Matrix{EventState}(undef, env.height, env.width)
    current_environment .= NO_EVENT
    
    # Add initial events
    for _ in 1:INITIAL_EVENTS
        x = rand(1:env.width)
        y = rand(1:env.height)
        current_environment[y, x] = EVENT_PRESENT
    end
    
    println("  Initialized environment with $(INITIAL_EVENTS) initial events")
    
    # Track evolution for animation
    environment_evolution = [copy(current_environment)]
    action_history = Vector{Vector{SensingAction}}()
    events_detected_per_timestep = Int[]
    uncertainty_per_timestep = Float64[]
    
    # For sync_multi mode: store the full period joint policy
    joint_policy_full_period = nothing
    
    # Run simulation
    for t in 1:NUM_STEPS
        println("  Step $(t)/$(NUM_STEPS)")
        
        # Update ground station timestep
        gs_state.time_step = t
        
        # Synchronized planning - all agents plan together
        if mode == :sync_multi
            # Get trajectory period
            period = agents[1].trajectory.period
            phase_in_period = mod(t - 1, period)  # Which phase in the period (0 to period-1)
            
            # Compute joint policy once per period (when phase resets to 0)
            if phase_in_period == 0 || joint_policy_full_period === nothing
                # Use synchronized multi-agent planner for full period
                joint_policy_full_period, planning_time = MacroPlannerSyncMulti.best_joint_policy(
                    env, global_belief, agents, PLANNING_HORIZON, gs_state, solver_type=:greedy_lookahead
                )
                
                total_planning_time += planning_time
                num_plans += 1
                println("  ✅ Computed joint policy for full period ($(length(joint_policy_full_period)) phases) in $(round(planning_time, digits=3))s")
            end
            
            # Execute action from policy based on current phase
            action_index = phase_in_period + 1  # Julia is 1-indexed
            
            if action_index <= length(joint_policy_full_period)
                joint_action = joint_policy_full_period[action_index]
                println("  📍 Executing phase $(phase_in_period) action (index $(action_index)) from joint policy")
                
                # Track actions for animation
                push!(action_history, joint_action.actions)
                
                # Execute actions for all agents
                joint_observation = GridObservation[]
                for (i, agent) in enumerate(agents)
                    action = joint_action.actions[i]
                    
                    # Execute action and get observation
                    obs = execute_agent_action(agent, action, env, t, current_environment)
                    push!(joint_observation, obs)
                end
                
                # Update global belief with joint observations
                global_belief = update_global_belief_sync(global_belief, joint_action.actions, joint_observation, env)
                gs_state.global_belief = global_belief
                
                # Calculate reward
                step_reward = calculate_joint_reward(global_belief, joint_action.actions, agents, env)
                total_reward += step_reward
                
                # Count detections
                for obs in joint_observation
                    for state in obs.event_states
                        if state == EVENT_PRESENT
                            detection_count += 1
                        end
                    end
                end
            end
            
        elseif mode == :pbvi
            # Use individual PBVI planning (SB-ABBA)
            maybe_sync!(env, gs_state, agents, t, planning_mode=:pbvi)
            
            # Execute individual agent actions
            step_reward = 0.0
            step_actions = SensingAction[]
            for agent in agents
                plan, plan_type = get_agent_plan(agent, gs_state)
                action = execute_plan(agent, plan, plan_type, agent.observation_history, t)
                push!(step_actions, action)
                
                # Execute action and get observation
                obs = execute_agent_action(agent, action, env, t, current_environment)
                
                # Update individual agent belief
                if !isempty(obs.sensed_cells)
                    for (i, cell) in enumerate(obs.sensed_cells)
                        if i <= length(obs.event_states)
                            observed_state = obs.event_states[i]
                            global_belief = collapse_belief_to(global_belief, cell, observed_state)
                        end
                    end
                end
                
                # Calculate individual reward
                if !isempty(action.target_cells)
                    for cell in action.target_cells
                        cell_reward = calculate_sophisticated_reward(global_belief, cell)
                        step_reward += cell_reward
                    end
                end
                
                # Count detections
                for state in obs.event_states
                    if state == EVENT_PRESENT
                        detection_count += 1
                    end
                end
            end
            
            # Track actions for animation
            push!(action_history, step_actions)
            
            total_reward += step_reward
        end
        
        # Calculate belief entropy
        belief_entropy = calculate_belief_entropy(global_belief)
        push!(belief_entropy_history, belief_entropy)
        
        # Track uncertainty
        total_uncertainty = sum([calculate_cell_entropy(global_belief, (x, y)) for x in 1:env.width, y in 1:env.height])
        avg_uncertainty = total_uncertainty / (env.width * env.height)
        push!(uncertainty_per_timestep, avg_uncertainty)
        
        # Track step detections for animation (already counted in detection_count above)
        # We don't need to recompute, just track zero for now
        # Detections are already being counted properly in the execution loops above
        push!(events_detected_per_timestep, 0)
        
        # Evolve environment state (ground truth)
        new_environment = similar(current_environment)
        EventDynamicsModule.transition_rsp!(new_environment, current_environment, env.rsp_params, Random.GLOBAL_RNG)
        current_environment = new_environment
        push!(environment_evolution, copy(current_environment))
    end
    
    # Calculate final metrics
    avg_planning_time = num_plans > 0 ? total_planning_time / num_plans : 0.0
    avg_belief_entropy = length(belief_entropy_history) > 0 ? mean(belief_entropy_history) : 0.0
    
    return Dict(
        :mode => mode,
        :run_id => run_id,
        :total_reward => total_reward,
        :avg_planning_time => avg_planning_time,
        :total_planning_time => total_planning_time,
        :num_plans => num_plans,
        :detection_count => detection_count,
        :avg_belief_entropy => avg_belief_entropy,
        :final_belief_entropy => belief_entropy_history[end],
        :environment_evolution => environment_evolution,
        :action_history => action_history,
        :events_detected_per_timestep => events_detected_per_timestep,
        :uncertainty_per_timestep => uncertainty_per_timestep
    )
end

"""
Execute agent action and return observation
"""
function execute_agent_action(agent::Agent, action::SensingAction, env, timestep::Int, current_environment::Matrix{EventState})
    # Get agent position
    pos = get_position_at_time(agent.trajectory, timestep, agent.phase_offset)
    
    # Get field of regard
    for_cells = get_field_of_regard_at_position(agent, pos, env)
    
    # Simulate observation
    sensed_cells = Tuple{Int, Int}[]
    event_states = EventState[]
    
    for cell in action.target_cells
        if cell in for_cells
            x, y = cell
            push!(sensed_cells, cell)
            # Use actual environment state
            if 1 <= x <= env.width && 1 <= y <= env.height
                if current_environment[y, x] == EVENT_PRESENT
                    push!(event_states, EVENT_PRESENT)
                else
                    push!(event_states, NO_EVENT)
                end
            end
        end
    end
    
    # Add to agent's observation history
    obs = GridObservation(agent.id, sensed_cells, event_states, [])
    push!(agent.observation_history, obs)
    
    return obs
end

"""
Update global belief with synchronized observations
"""
function update_global_belief_sync(global_belief::Belief, joint_action::Vector{SensingAction}, joint_observation::Vector{GridObservation}, env)
    # For now, use a simplified belief evolution since predict_belief_evolution_dbn is not available
    # In a full implementation, this would use the DBN transition model
    evolved_belief = global_belief  # Simplified: no evolution for now
    
    # Then update with all observations
    updated_belief = copy(evolved_belief)
    
    # Process observations in reverse order to give priority to most recent
    for obs_idx in length(joint_observation):-1:1
        observation = joint_observation[obs_idx]
        action = joint_action[obs_idx]
        
        # Update belief with this observation
        for (i, cell) in enumerate(observation.sensed_cells)
            x, y = cell
            if 1 <= x <= env.width && 1 <= y <= env.height
                # Get observed state
                observed_state = observation.event_states[i]
                
                # Update belief for this cell with perfect observation
                if observed_state == EVENT_PRESENT
                    # Set to certain event present
                    updated_belief.event_distributions[2, y, x] = 1.0  # EVENT_PRESENT
                    updated_belief.event_distributions[1, y, x] = 0.0  # NO_EVENT
                else
                    # Set to certain no event
                    updated_belief.event_distributions[1, y, x] = 1.0  # NO_EVENT
                    updated_belief.event_distributions[2, y, x] = 0.0  # EVENT_PRESENT
                end
            end
        end
    end
    
    return updated_belief
end

"""
Calculate joint reward for synchronized actions
"""
function calculate_joint_reward(belief::Belief, joint_action::Vector{SensingAction}, agents::Vector{Agent}, env)
    total_reward = 0.0
    
    for action in joint_action
        if !isempty(action.target_cells)
            for cell in action.target_cells
                # Use sophisticated reward function
                cell_reward = calculate_sophisticated_reward(belief, cell)
                total_reward += cell_reward
            end
        end
    end
    
    return total_reward
end

"""
Calculate sophisticated reward for sensing actions
"""
function calculate_sophisticated_reward(belief::Belief, cell::Tuple{Int, Int})
    # 1. Entropy-based reward: w_H * (H_prior - H_post)
    H_before = calculate_cell_entropy(belief, cell)
    H_after = 0.0  # Simplified: assume perfect observation
    entropy_reward = ENTROPY_WEIGHT * (H_before - H_after)
    
    # 2. State value reward: w_F * E[F_I_j]
    event_prob = get_event_probability(belief, cell)
    no_event_prob = 1.0 - event_prob
    
    # E[F_I_j] = Σ_k p(I_k) * F_k
    expected_value = no_event_prob * STATE_VALUES[1] + event_prob * STATE_VALUES[2]
    value_reward = VALUE_WEIGHT * expected_value
    
    # Total reward for this cell
    return entropy_reward + value_reward
end

"""
Calculate belief entropy
"""
function calculate_belief_entropy(belief::Belief)
    total_entropy = 0.0
    num_states, height, width = size(belief.event_distributions)
    
    for x in 1:width, y in 1:height
        cell_dist = belief.event_distributions[:, y, x]
        # Normalize to ensure it's a proper probability distribution
        cell_dist = cell_dist ./ sum(cell_dist)
        
        # Calculate entropy: -Σ p * log(p)
        for p in cell_dist
            if p > 0
                total_entropy -= p * log(p)
            end
        end
    end
    
    return total_entropy
end

"""
Get field of regard for agent at position (helper for visualization)
"""
function get_field_of_regard_at_position(agent::Agent, position::Tuple{Int, Int}, grid_dims)
    env_like = (width=grid_dims.width, height=grid_dims.height)
    return MacroPlannerSyncMulti.get_field_of_regard_at_position(agent, position, env_like)
end

"""
Visualize the current state of the environment and agents
"""
function visualize_rsp_state(
    time_step::Int,
    agents::Vector{Agent},
    environment_state::Matrix{EventState},
    actions::Vector{SensingAction}=SensingAction[],
    ground_station_pos::Tuple{Int, Int}=(GROUND_STATION_X, GROUND_STATION_Y);
    events_detected::Int=0,
    avg_uncertainty::Float64=0.0
)
    height, width = size(environment_state)
    agent_colors = [:red, :blue, :green, :orange]

    p = plot(; xlim=(0.5, width+0.5), ylim=(0.5, height+0.5),
        aspect_ratio=:equal, size=(600, 800), legend=false,
        xlabel="X", ylabel="Y", grid=false,
        title="Time $(time_step) | Events: $(count(==(EVENT_PRESENT), environment_state)) | Det: $(events_detected) | Unc: $(round(avg_uncertainty, digits=2))",
        titlefontsize=10, background_color=:white
    )

    # Draw grid
    for x in 1:width, y in 1:height
        xs = [x-0.5, x+0.5, x+0.5, x-0.5]
        ys = [y-0.5, y-0.5, y+0.5, y+0.5]
        plot!(p, xs, ys, seriestype=:shape, fillcolor=:white, linecolor=:black, linewidth=1, alpha=1, label=false)
    end

    # Draw field of regard and actions
    for (i, agent) in enumerate(agents)
        color = agent_colors[i]
        pos = get_position_at_time(agent.trajectory, time_step, agent.phase_offset)
        for_cells = get_field_of_regard_at_position(agent, pos, (width=width, height=height))
        
        for (x, y) in for_cells
            xs = [x-0.5, x+0.5, x+0.5, x-0.5]
            ys = [y-0.5, y-0.5, y+0.5, y+0.5]
            plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.15, label=false)
        end
        
        action_idx = findfirst(a -> a.agent_id == agent.id, actions)
        if action_idx !== nothing && !isempty(actions[action_idx].target_cells)
            for (x, y) in actions[action_idx].target_cells
                xs = [x-0.5, x+0.5, x+0.5, x-0.5]
                ys = [y-0.5, y-0.5, y+0.5, y+0.5]
                plot!(p, xs, ys, seriestype=:shape, fillcolor=color, linecolor=:black, alpha=0.5, label=false)
            end
        end
    end

    scatter!(p, [ground_station_pos[1]], [ground_station_pos[2]]; marker=:star, markersize=14, color=:green, alpha=0.9, label=false)

    for (i, agent) in enumerate(agents)
        pos = get_position_at_time(agent.trajectory, time_step, agent.phase_offset)
        scatter!(p, [pos[1]], [pos[2]]; marker=:circle, markersize=10, color=agent_colors[i], alpha=0.9, label=false)
    end

    for y in 1:height, x in 1:width
        if environment_state[y, x] == EVENT_PRESENT
            annotate!(p, x, y, text("🔥", :center, 18))
        end
    end

    return p
end

"""
Create animation of the simulation
"""
function create_rsp_animation(
    agents::Vector{Agent},
    num_steps::Int,
    environment_evolution::Vector{Matrix{EventState}},
    action_history::Vector{Vector{SensingAction}},
    results_dir::String,
    run_number::Int,
    planning_mode::Symbol;
    events_detected_per_timestep::Vector{Int}=Int[],
    uncertainty_per_timestep::Vector{Float64}=Float64[]
)
    println("  🎬 Creating animation...")
    
    animations_dir = joinpath(results_dir, "Run_$(run_number)", string(planning_mode))
    mkpath(animations_dir)
    
    frames = []
    for step in 0:(num_steps-1)
        env_state = step < length(environment_evolution) ? environment_evolution[step + 1] : fill(NO_EVENT, GRID_HEIGHT, GRID_WIDTH)
        actions = step < length(action_history) ? action_history[step + 1] : SensingAction[]
        events_detected = step < length(events_detected_per_timestep) ? events_detected_per_timestep[step + 1] : 0
        avg_uncertainty = step < length(uncertainty_per_timestep) ? uncertainty_per_timestep[step + 1] : 0.0
        
        frame = visualize_rsp_state(step, agents, env_state, actions, (GROUND_STATION_X, GROUND_STATION_Y); 
                                  events_detected=events_detected, avg_uncertainty=avg_uncertainty)
        push!(frames, frame)
    end
    
    anim = @animate for frame in frames
        plot(frame, size=(600, 800))
    end
    
    animation_filename = joinpath(animations_dir, "sync_multi_$(GRID_WIDTH)x$(GRID_HEIGHT)_$(planning_mode)_run$(run_number).gif")
    gif(anim, animation_filename, fps=1.0)
    println("  ✓ Animation saved: Run_$(run_number)/$(planning_mode)/$(basename(animation_filename))")
    
    return anim
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

println("🚀 Starting Synchronized Multi-Agent Test...")
println("Grid size: $(GRID_WIDTH)x$(GRID_HEIGHT)")
println("Number of agents: $(NUM_AGENTS)")
println("Planning modes: $(modes)")
println("Number of runs: $(N_RUNS)")

# Project and planner modules already loaded at the top

# Set reward configuration
set_reward_config_from_main(ENTROPY_WEIGHT, VALUE_WEIGHT, INFORMATION_STATES, STATE_VALUES)

# Create environment and agents
env = create_rsp_environment()
agents = create_synchronized_agents()

# Debug agent positions
debug_agent_positions(agents)

# Create results directory
timestamp = Dates.format(now(), "yyyy-mm-ddTHH-MM-SS")
results_dir = "results/run_$(timestamp)"
mkpath(results_dir)
println("📁 Results will be saved to: $(results_dir)")

# Run simulations for all modes
all_results = DataFrame()

for mode in modes
    println("\n" * "="^60)
    println("🔄 Testing mode: $(mode)")
    println("="^60)
    
    mode_results = []
    
    for run in 1:N_RUNS
        println("\n--- Run $(run)/$(N_RUNS) ---")
        
        # Reset environment and agents for each run
        env = create_rsp_environment()
        agents = create_synchronized_agents()
        
        # Run simulation
        result = run_simulation_mode(env, agents, mode, run)
        
        # Create animation for this run
        create_rsp_animation(
            agents, NUM_STEPS,
            result[:environment_evolution],
            result[:action_history],
            results_dir, run, mode;
            events_detected_per_timestep=result[:events_detected_per_timestep],
            uncertainty_per_timestep=result[:uncertainty_per_timestep]
        )
        
        # Add to results
        push!(mode_results, result)
    end
    
    # Calculate statistics for this mode
    avg_reward = mean([r[:total_reward] for r in mode_results])
    std_reward = std([r[:total_reward] for r in mode_results])
    avg_planning_time = mean([r[:avg_planning_time] for r in mode_results])
    avg_detections = mean([r[:detection_count] for r in mode_results])
    avg_entropy = mean([r[:avg_belief_entropy] for r in mode_results])
    
    println("\n📊 Results for $(mode):")
    println("  Average total reward: $(round(avg_reward, digits=3)) ± $(round(std_reward, digits=3))")
    println("  Average planning time: $(round(avg_planning_time * 1000, digits=2)) ms")
    println("  Average detections: $(round(avg_detections, digits=1))")
    println("  Average belief entropy: $(round(avg_entropy, digits=3))")
    
    # Convert results to DataFrame
    mode_df = DataFrame(mode_results)
    append!(all_results, mode_df)
end

# Save detailed results
CSV.write("$(results_dir)/detailed_results.csv", all_results)

# Create summary
summary_list = []
for mode in modes
    mode_data = all_results[all_results.mode .== mode, :]
    summary = Dict(
        :mode => mode,
        :avg_reward => mean(mode_data.total_reward),
        :std_reward => std(mode_data.total_reward),
        :avg_planning_time => mean(mode_data.avg_planning_time),
        :avg_detections => mean(mode_data.detection_count),
        :avg_entropy => mean(mode_data.avg_belief_entropy)
    )
    push!(summary_list, summary)
end
summary_results = DataFrame(summary_list)

CSV.write("$(results_dir)/summary_results.csv", summary_results)

println("\n✅ Synchronized Multi-Agent Test completed!")
println("Results saved to: $(results_dir)")

# Print final comparison
println("\n📊 Final Comparison:")
println("="^50)
for row in eachrow(summary_results)
    println("$(row.mode):")
    println("  Reward: $(round(row.avg_reward, digits=3)) ± $(round(row.std_reward, digits=3))")
    println("  Planning time: $(round(row.avg_planning_time * 1000, digits=2)) ms")
    println("  Detections: $(round(row.avg_detections, digits=1))")
    println("  Entropy: $(round(row.avg_entropy, digits=3))")
    println()
end

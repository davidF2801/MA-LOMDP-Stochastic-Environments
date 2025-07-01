#!/usr/bin/env julia

"""
Test script for Asynchronous Centralized Planning with Decentralized Execution
Demonstrates the ground station coordinating multiple agents with periodic synchronization
"""

println("🚀 Script starting...")

using POMDPs
println("✅ POMDPs loaded")

using POMDPTools
println("✅ POMDPTools loaded")

using Random
println("✅ Random loaded")

using Plots
Plots.plotlyjs()  # Use PlotlyJS backend for emoji support
println("✅ Using PlotlyJS backend for emoji support.")

# Set random seed for reproducibility
Random.seed!(42)
println("✅ Random seed set")

# =============================================================================
# CONFIGURATION PARAMETERS - CHANGE THESE TO MODIFY THE SIMULATION
# =============================================================================
# 
# 🎯 QUICK START: Change these main parameters to experiment:
#   - NUM_STEPS: Total simulation duration
#   - GRID_WIDTH/GRID_HEIGHT: Environment size
#   - NUM_AGENTS: Number of agents
#   - PLANNING_MODE: :script (open-loop) or :policy (closed-loop)
#   - INITIAL_EVENTS: How many events start in the environment
#   - BIRTH_RATE/DEATH_RATE: How quickly events appear/disappear
#
# 📊 For advanced tuning, modify the other parameters below.
# =============================================================================

# 🎯 MAIN SIMULATION PARAMETERS
const NUM_STEPS = 15                  # Total simulation steps
const COMPARISON_STEPS = 3000         # Steps for planning mode comparison
const CONTACT_FREQUENCY_STEPS = 25    # Steps for contact frequency analysis
const PLANNING_MODE = :script         # :script or :policy

# 🌍 ENVIRONMENT PARAMETERS
const GRID_WIDTH = 5                  # Grid width (columns)
const GRID_HEIGHT = 5                 # Grid height (rows)
const INITIAL_EVENTS = 0             # Number of initial events
const MAX_SENSING_TARGETS = 1         # Maximum cells an agent can sense per step
const SENSOR_RANGE = 1.5              # Sensor range for agents
const DISCOUNT_FACTOR = 0.95          # POMDP discount factor

# 📊 EVENT DYNAMICS PARAMETERS
const BIRTH_RATE = 0.01            # Rate of new events appearing
const DEATH_RATE = 0.03               # Rate of events disappearing
const SPREAD_RATE = 0.01              # Rate of events spreading to neighbors
const DECAY_RATE = 0.03               # Rate of events decaying
const NEIGHBOR_INFLUENCE = 0.02       # Influence of neighboring cells

# 🤖 AGENT PARAMETERS
const NUM_AGENTS = 2                  # Number of agents
const AGENT1_PHASE_OFFSET = 0         # Phase offset for agent 1
const AGENT2_PHASE_OFFSET = 3         # Phase offset for agent 2
const SENSOR_FOV = pi/2               # Field of view angle (radians)
const SENSOR_NOISE = 0.1              # Observation noise level

# 📡 COMMUNICATION PARAMETERS
const CONTACT_HORIZON = 5             # Steps until next sync opportunity
const GROUND_STATION_X = 3            # Center column for 5x5 grid
const GROUND_STATION_Y = 1            # Ground station Y position

# 🎨 VISUALIZATION PARAMETERS
const ANIMATION_FPS = 2               # Frames per second for animation
const STATUS_UPDATE_INTERVAL = 10     # Print status every N steps

# =============================================================================
# PARAMETER DESCRIPTIONS
# =============================================================================
# 
# 🎯 MAIN SIMULATION PARAMETERS:
#   - NUM_STEPS: Total simulation duration (higher = longer simulation)
#   - COMPARISON_STEPS: Steps for planning mode comparison (higher = more accurate comparison)
#   - CONTACT_FREQUENCY_STEPS: Steps for contact frequency analysis
#   - PLANNING_MODE: :script (open-loop planning) or :policy (closed-loop planning)
#
# 🌍 ENVIRONMENT PARAMETERS:
#   - GRID_WIDTH/GRID_HEIGHT: Size of the environment grid
#   - INITIAL_EVENTS: Number of events that start in the environment
#   - MAX_SENSING_TARGETS: Maximum cells an agent can sense per step
#   - SENSOR_RANGE: How far agents can sense (in grid cells)
#   - DISCOUNT_FACTOR: POMDP discount factor (0.95 = standard)
#
# 📊 EVENT DYNAMICS PARAMETERS:
#   - BIRTH_RATE: Probability of new events appearing per cell per step
#   - DEATH_RATE: Probability of events disappearing per step
#   - SPREAD_RATE: Probability of events spreading to neighbors
#   - DECAY_RATE: Probability of events decaying over time
#   - NEIGHBOR_INFLUENCE: How much neighboring cells influence each other
#
# 🤖 AGENT PARAMETERS:
#   - NUM_AGENTS: Number of agents in the simulation
#   - AGENT1_PHASE_OFFSET/AGENT2_PHASE_OFFSET: Phase offsets for agent trajectories
#   - SENSOR_FOV: Field of view angle in radians (pi/2 = 90 degrees)
#   - SENSOR_NOISE: Observation noise level (0 = perfect, 1 = completely random)
#
# 📡 COMMUNICATION PARAMETERS:
#   - CONTACT_HORIZON: Steps until next sync opportunity (calculated automatically)
#   - GROUND_STATION_X/Y: Position of the ground station for synchronization
#
# 🎨 VISUALIZATION PARAMETERS:
#   - ANIMATION_FPS: Frames per second for the simulation animation
#   - STATUS_UPDATE_INTERVAL: How often to print status updates
#
# =============================================================================

# =============================================================================

# Import the existing environment and planner modules
include("../src/MyProject.jl")
using .MyProject

# Import specific types from MyProject
using .MyProject: Agent, SensingAction, GridObservation, EventState, NO_EVENT, EVENT_PRESENT
using .MyProject: CircularTrajectory, LinearTrajectory, RangeLimitedSensor
using .MyProject: EventState2, NO_EVENT_2, EVENT_PRESENT_2

# Import functions
using .MyProject.Agents.TrajectoryPlanner: get_position_at_time

# Import specific modules
using .Environment
using .Environment.EventDynamicsModule
using .Planners.GroundStation
using .Planners.MacroPlanner
using .Planners.PolicyTreePlanner

println("✅ All modules imported successfully")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

"""
Visualize the current state of the environment and agents
"""
function visualize_simulation_state(
    time_step::Int,
    agents::Vector{Agent},
    environment_state::Matrix{EventState}=fill(NO_EVENT, GRID_HEIGHT, GRID_WIDTH),
    actions::Vector{SensingAction}=SensingAction[],
    ground_station_pos::Tuple{Int, Int}=(GROUND_STATION_X, GROUND_STATION_Y)
)
    height, width = size(environment_state)
    
    # 1. Base: white background
    p = plot(; xlim=(0.5, width+0.5), ylim=(0.5, height+0.5),
        aspect_ratio=:equal, size=(900, 700), legend=false,
        xlabel="X Coordinate", ylabel="Y Coordinate",
        grid=true, gridalpha=0.3,
        title="Time Step $(time_step)\nEvents: $(count(==(EVENT_PRESENT), environment_state)) | Agents: $(length(agents)) | GS: ($(ground_station_pos[1]),$(ground_station_pos[2]))",
        titlefontsize=12)

    # 2. Overlay: FOR (Field of Regard) as semi-transparent blue
    for_cells_all = Tuple{Int,Int}[]
    for (i, agent) in enumerate(agents)
        pos = get_position_at_time(agent.trajectory, time_step, agent.phase_offset)
        for dx in -round(Int, agent.sensor.range):round(Int, agent.sensor.range)
            for dy in -round(Int, agent.sensor.range):round(Int, agent.sensor.range)
                cell = (pos[1] + dx, pos[2] + dy)
                if 1 <= cell[1] <= width && 1 <= cell[2] <= height
                    distance = sqrt(dx^2 + dy^2)
                    if distance <= agent.sensor.range
                        push!(for_cells_all, cell)
                    end
                end
            end
        end
    end
    for_cells_all = unique(for_cells_all)
    if !isempty(for_cells_all)
        for_x = [cell[1] for cell in for_cells_all]
        for_y = [cell[2] for cell in for_cells_all]
        scatter!(p, for_x, for_y; marker=:rect, markersize=32, color=:lightblue, alpha=0.18, label="Field of Regard (FOR)")
    else
        scatter!(p, [], []; marker=:rect, markersize=32, color=:lightblue, alpha=0.18, label="Field of Regard (FOR)")
    end

    # 3. Overlay: FOV (Field of View) as semi-transparent, agent-specific color
    fov_colors = [:red, :green, :orange, :purple, :cyan]
    for (i, agent) in enumerate(agents)
        if i <= length(actions) && !isempty(actions[i].target_cells)
            fov_cells = actions[i].target_cells
            fov_x = [cell[1] for cell in fov_cells]
            fov_y = [cell[2] for cell in fov_cells]
            scatter!(p, fov_x, fov_y; marker=:rect, markersize=32, color=fov_colors[i], alpha=0.35, label="FOV Agent $(agent.id)")
            # Annotate with agent number
            for (x, y) in zip(fov_x, fov_y)
                annotate!(p, x, y, text("$(agent.id)", :center, 12, fov_colors[i]))
            end
        end
    end

    # 4. Overlay: Ground station as green star
    if 1 <= ground_station_pos[1] <= width && 1 <= ground_station_pos[2] <= height
        scatter!(p, [ground_station_pos[1]], [ground_station_pos[2]]; marker=:star, markersize=14, color=:green, alpha=0.9, label="Ground Station")
    else
        scatter!(p, [], []; marker=:star, markersize=14, color=:green, alpha=0.9, label="Ground Station")
    end

    # 5. Overlay: Agents as small circles/diamonds (always on top)
    agent_colors = [:blue, :purple, :orange, :brown, :pink]
    agent_positions = [get_position_at_time(agent.trajectory, time_step, agent.phase_offset) for agent in agents]
    # Check for overlap
    if length(unique(agent_positions)) < length(agent_positions)
        println("⚠️  WARNING: Agents overlap at time step $time_step at position(s): $(agent_positions)")
    end
    plotted_positions = Dict{Tuple{Int,Int}, Int}()
    for (i, agent) in enumerate(agents)
        pos = agent_positions[i]
        agent_color = agent_colors[mod(i-1, length(agent_colors))+1]
        # If another agent is already at this position, offset or use a different marker
        if get(plotted_positions, pos, 0) == 0
            scatter!(p, [pos[1]], [pos[2]]; marker=:circle, markersize=10, color=agent_color, alpha=0.9, label="Agent $(agent.id)")
            plotted_positions[pos] = 1
        else
            # Offset slightly or use diamond
            scatter!(p, [pos[1]+0.15], [pos[2]-0.15]; marker=:diamond, markersize=11, color=agent_color, alpha=0.9, label="Agent $(agent.id) (overlap)")
        end
    end

    # 6. Overlay: Fire emoji for events
    for y in 1:height, x in 1:width
        if environment_state[y, x] == EVENT_PRESENT
            annotate!(p, x, y, text("🔥", :center, 18))
        end
    end

    # 7. Legend (only one entry per type)
    plot!(p, legend=:topright, legendfontsize=10)
    return p
end

"""
Create animation of the simulation
"""
function create_simulation_animation(
    agents::Vector{Agent},
    num_steps::Int,
    environment_evolution::Vector{Matrix{EventState}}=Vector{Matrix{EventState}}(),
    action_history::Vector{Vector{SensingAction}}=Vector{Vector{SensingAction}}(),
    ground_station_pos::Tuple{Int, Int}=(GROUND_STATION_X, GROUND_STATION_Y)
)
    println("\n🎬 Creating Simulation Animation")
    println("================================")
    
    # Create output directory
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    # Create frames for animation
    frames = []
    
    for step in 0:(num_steps-1)
        # Get environment state for this step
        env_state = if step < length(environment_evolution)
            environment_evolution[step + 1]
        else
            fill(NO_EVENT, GRID_HEIGHT, GRID_WIDTH)  # Default empty environment
        end
        
        # Get actions for this step
        actions = if step < length(action_history)
            action_history[step + 1]
        else
            SensingAction[]
        end
        
        # Create frame
        frame = visualize_simulation_state(step, agents, env_state, actions, ground_station_pos)
        push!(frames, frame)
    end
    
    # Create animation
    anim = @animate for frame in frames
        plot(frame, size=(900, 700))
    end
    
    # Save animation
    animation_filename = joinpath(output_dir, "async_centralized_simulation.gif")
    gif(anim, animation_filename, fps=ANIMATION_FPS)
    println("✓ Saved animation: $(basename(animation_filename))")
    
    return anim
end

"""
Visualize agent trajectories over time
"""
function visualize_agent_trajectories(agents::Vector{Agent}, num_steps::Int, ground_station_pos::Tuple{Int, Int}=(GROUND_STATION_X, GROUND_STATION_Y))
    println("\n🎯 Creating Agent Trajectory Visualization")
    println("=========================================")
    

    
    # Create output directory
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    # Collect trajectory points
    trajectory_data = Dict{Int, Vector{Tuple{Int, Int}}}()
    for agent in agents
        trajectory_data[agent.id] = []
        for step in 0:(num_steps-1)
            pos = get_position_at_time(agent.trajectory, step, agent.phase_offset)
            push!(trajectory_data[agent.id], pos)
        end
    end
    
    # Create plot
    p = plot(size=(800, 600))
    
    colors = [:blue, :red, :green, :purple, :orange]
    for (i, agent) in enumerate(agents)
        trajectory = trajectory_data[agent.id]
        x_coords = [pos[1] for pos in trajectory]
        y_coords = [pos[2] for pos in trajectory]
        
        # Plot trajectory line
        plot!(p, x_coords, y_coords, 
              label="Agent $(agent.id)", 
              color=colors[mod(i-1, length(colors))+1],
              linewidth=2,
              alpha=0.7)
        
        # Mark start and end points
        scatter!(p, [x_coords[1]], [y_coords[1]], 
                marker=:circle, 
                markersize=8, 
                color=colors[mod(i-1, length(colors))+1],
                label="")
        scatter!(p, [x_coords[end]], [y_coords[end]], 
                marker=:square, 
                markersize=8, 
                color=colors[mod(i-1, length(colors))+1],
                label="")
    end
    
    # Mark ground station position
    scatter!(p, [ground_station_pos[1]], [ground_station_pos[2]], 
            marker=:star, 
            markersize=12, 
            color=:green,
            label="Ground Station")
    
    # Add grid and labels
    plot!(p, xlabel="X", ylabel="Y", 
          title="Agent Trajectories Over $(num_steps) Steps",
          grid=true,
          legend=:topright)
    
    # Save plot
    plot_filename = joinpath(output_dir, "agent_trajectories.png")
    savefig(p, plot_filename)
    println("✓ Saved trajectory plot: $(basename(plot_filename))")
    
    return p
end

"""
Visualize action statistics over time
"""
function visualize_action_statistics(action_history::Vector{Vector{SensingAction}}, num_steps::Int)
    println("\n📊 Creating Action Statistics Visualization")
    println("=========================================")
    
    # Create output directory
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    # Collect statistics
    cells_sensed_per_step = []
    wait_actions_per_step = []
    communication_events = []
    
    for step in 1:min(length(action_history), num_steps)
        actions = action_history[step]
        
        # Count cells sensed
        total_cells = sum(length(action.target_cells) for action in actions)
        push!(cells_sensed_per_step, total_cells)
        
        # Count wait actions
        wait_count = count(isempty(action.target_cells) for action in actions)
        push!(wait_actions_per_step, wait_count)
        
        # Count communication events
        comm_count = count(action.communicate for action in actions)
        push!(communication_events, comm_count)
    end
    
    # Create plots
    p1 = plot(1:length(cells_sensed_per_step), cells_sensed_per_step,
              label="Cells Sensed",
              xlabel="Time Step",
              ylabel="Number of Cells",
              title="Sensing Activity Over Time",
              marker=:circle,
              linewidth=2,
              grid=true)
    
    p2 = plot(1:length(wait_actions_per_step), wait_actions_per_step,
              label="Wait Actions",
              xlabel="Time Step", 
              ylabel="Number of Agents",
              title="Wait Actions Over Time",
              marker=:square,
              linewidth=2,
              grid=true)
    
    p3 = plot(1:length(communication_events), communication_events,
              label="Communication Events",
              xlabel="Time Step",
              ylabel="Number of Events", 
              title="Communication Activity Over Time",
              marker=:diamond,
              linewidth=2,
              grid=true)
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
    
    # Save plot
    plot_filename = joinpath(output_dir, "action_statistics.png")
    savefig(combined_plot, plot_filename)
    println("✓ Saved action statistics: $(basename(plot_filename))")
    
    return combined_plot
end

# =============================================================================

"""
Create test agents with linear trajectories moving upward in a 5x5 grid
"""
function create_test_agents()
    agents = Agent[]
    
    # Grid dimensions
    grid_height = GRID_HEIGHT
    grid_width = GRID_WIDTH
    period = grid_height  # Period equals number of rows
    
    # Agent 1: Starts at row 0, moves upward
    trajectory1 = LinearTrajectory(3, 1, 3, grid_height, period)  # Start at (1,1), end at (1,5), period 5
    
    # Agent 2: Starts at row 4, moves upward (phase offset of 3)
    trajectory2 = LinearTrajectory(3, 1, 3, grid_height, period)  # Same trajectory, different phase
    
    # Both agents use same sensor
    sensor = RangeLimitedSensor(SENSOR_RANGE, SENSOR_FOV, SENSOR_NOISE)
    
    # Agent 1: No phase offset (starts at row 1)
    agent1 = Agent(1, trajectory1, sensor, AGENT1_PHASE_OFFSET)
    
    # Agent 2: Phase offset of 3 (starts at row 4)
    agent2 = Agent(2, trajectory2, sensor, AGENT2_PHASE_OFFSET)
    
    println("🤖 Created $(NUM_AGENTS) agents with linear trajectories:")
    println("  Agent 1: Phase offset $(AGENT1_PHASE_OFFSET) (starts at row 1)")
    println("  Agent 2: Phase offset $(AGENT2_PHASE_OFFSET) (starts at row 4)")
    println("  Trajectory: Linear, column 1, rows 1-$(grid_height), period $(period)")
    println("  Sensor: Range $(SENSOR_RANGE), FOV $(SENSOR_FOV)")
    
    return [agent1, agent2]
end

"""
Create test environment using existing environment simulation
"""
function create_test_environment()
    # Create event dynamics (more events for denser environment)
    event_dynamics = EventDynamics(BIRTH_RATE, DEATH_RATE, SPREAD_RATE, DECAY_RATE, NEIGHBOR_INFLUENCE)
    
    # Create agents
    agents = create_test_agents()
    
    # Create spatial grid environment, now with ground_station_pos
    env = SpatialGrid(GRID_WIDTH, GRID_HEIGHT, event_dynamics, agents, SENSOR_RANGE, DISCOUNT_FACTOR, INITIAL_EVENTS, MAX_SENSING_TARGETS, (GROUND_STATION_X, GROUND_STATION_Y))
    
    println("🌍 Created test environment:")
    println("  Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
    println("  Initial events: $(INITIAL_EVENTS)")
    println("  Max sensing targets: $(MAX_SENSING_TARGETS)")
    println("  Event dynamics: birth=$(BIRTH_RATE), death=$(DEATH_RATE), spread=$(SPREAD_RATE), decay=$(DECAY_RATE), influence=$(NEIGHBOR_INFLUENCE)")
    
    return env
end

"""
Simulate asynchronous centralized planning with proper environment simulation
"""
function simulate_async_centralized_planning(num_steps::Int=NUM_STEPS; planning_mode::Symbol=PLANNING_MODE)
    println("🚀 Starting Asynchronous Centralized Planning Simulation")
    println("=======================================================")
    println("Planning mode: $(planning_mode)")
    println("Simulation steps: $(num_steps)")
    
    # Create environment and agents
    env = create_test_environment()
    agents = env.agents
    
    # Initialize ground station
    gs_state = GroundStation.initialize_ground_station(env, agents)
    
    # Ground station position
    ground_station_pos = (GROUND_STATION_X, GROUND_STATION_Y)  # Position where agents sync
    
    # Track performance metrics
    total_reward = 0.0
    sync_events = []
    agent_rewards = Dict{Int, Float64}()
    for agent in agents
        agent_rewards[agent.id] = 0.0
    end
    
    # Track environment and actions for visualization
    environment_evolution = Matrix{EventState}[]
    action_history = Vector{Vector{SensingAction}}()
    
    # Get initial environment state
    initial_state = POMDPs.initialstate(env)
    if hasproperty(initial_state, :value)
        current_environment = initial_state.value.event_map
    else
        current_environment = initial_state.event_map
    end
    
    println("\n📊 Starting simulation...")
    
    for t in 0:(num_steps-1)
        println("\n⏰ Time step $(t)")
        
        # Check for synchronization opportunities
        old_sync_times = copy(gs_state.agent_last_sync)
        GroundStation.maybe_sync!(env, gs_state, agents, t, planning_mode=planning_mode)
        
        # Record sync events
        for (agent_id, old_time) in old_sync_times
            if gs_state.agent_last_sync[agent_id] != old_time
                push!(sync_events, (t, agent_id))
                println("📡 Sync event recorded: Agent $(agent_id) at time $(t)")
            end
        end
        
        # Execute agent actions
        joint_actions = SensingAction[]
        step_reward = 0.0
        
        for agent in agents
            # Execute agent's current plan
            action = GroundStation.execute_plan(agent, gs_state, agent.observation_history)
            push!(joint_actions, action)
            
            # Calculate reward for this agent
            if !isempty(action.target_cells)
                # Simple reward: number of cells sensed
                agent_reward = length(action.target_cells) * 0.1
                agent_rewards[agent.id] += agent_reward
                step_reward += agent_reward
            end
            
            # Simulate observation (simplified)
            observation = GridObservation(agent.id, action.target_cells, EventState[], [])
            push!(agent.observation_history, observation)
            
            println("  Agent $(agent.id): $(length(action.target_cells)) cells sensed")
        end
        
        # Record environment state and actions for visualization
        push!(environment_evolution, copy(current_environment))
        push!(action_history, joint_actions)
        
        total_reward += step_reward
        
        # Update environment using existing event dynamics
        if t < num_steps - 1  # Don't update on last step
            # Convert to DBN model for environment update
            dbn_model = DBNTransitionModel2(env.event_dynamics)
            
            # Convert EventState to EventState2 for DBN update
            event_map_2 = Matrix{EventState2}(undef, env.height, env.width)
            for y in 1:env.height
                for x in 1:env.width
                    event_map_2[y, x] = current_environment[y, x] == EVENT_PRESENT ? EVENT_PRESENT_2 : NO_EVENT_2
                end
            end
            
            # Update environment
            EventDynamicsModule.update_events!(dbn_model, event_map_2, Random.GLOBAL_RNG)
            
            # Convert back to EventState
            for y in 1:env.height
                for x in 1:env.width
                    current_environment[y, x] = event_map_2[y, x] == EVENT_PRESENT_2 ? EVENT_PRESENT : NO_EVENT
                end
            end
        end
        
        # Print ground station status every STATUS_UPDATE_INTERVAL steps
        if t % STATUS_UPDATE_INTERVAL == 0
            println("📊 Ground Station Status:")
            println("  Planning mode: $(planning_mode)")
            println("  Last sync times: $(gs_state.agent_last_sync)")
        end
    end
    
    # Print final results
    println("\n📈 Simulation Results")
    println("====================")
    println("Total reward: $(round(total_reward, digits=3))")
    println("Total sync events: $(length(sync_events))")
    println("Agent rewards:")
    for (agent_id, reward) in agent_rewards
        println("  Agent $(agent_id): $(round(reward, digits=3))")
    end
    
    println("\n📡 Sync Events:")
    for (time, agent_id) in sync_events
        println("  Time $(time): Agent $(agent_id)")
    end
    
    # Create visualizations
    println("\n🎨 Creating Visualizations...")
    println("============================")
    
    # Create simulation animation
    anim = create_simulation_animation(agents, num_steps, environment_evolution, action_history, ground_station_pos)
    
    # Create trajectory visualization
    trajectory_plot = visualize_agent_trajectories(agents, num_steps, ground_station_pos)
    
    # Create action statistics
    action_stats = visualize_action_statistics(action_history, num_steps)
    
    println("\n✅ Visualizations completed!")
    println("📁 Check the 'visualizations' folder for:")
    println("  - async_centralized_simulation.gif (main simulation animation)")
    println("  - agent_trajectories.png (agent paths with ground station)")
    println("  - action_statistics.png (sensing and communication stats)")
    
    return gs_state, agents, total_reward, sync_events, agent_rewards, environment_evolution, action_history
end

"""
Compare different planning modes
"""
function compare_planning_modes(num_steps::Int=COMPARISON_STEPS)
    println("🔬 Comparing Planning Modes")
    println("===========================")
    
    # Test macro-script planning
    println("\n1️⃣ Testing Macro-Script Planning")
    gs_state_script, agents_script, reward_script, sync_script, agent_rewards_script, env_script, action_script = 
        simulate_async_centralized_planning(num_steps, planning_mode=:script)
    
    # Test policy tree planning
    println("\n2️⃣ Testing Policy Tree Planning")
    gs_state_policy, agents_policy, reward_policy, sync_policy, agent_rewards_policy, env_policy, action_policy = 
        simulate_async_centralized_planning(num_steps, planning_mode=:policy)
    
    # Compare results
    println("\n📊 Comparison Results")
    println("====================")
    println("Macro-Script Planning:")
    println("  Total reward: $(round(reward_script, digits=3))")
    println("  Sync events: $(length(sync_script))")
    
    println("\nPolicy Tree Planning:")
    println("  Total reward: $(round(reward_policy, digits=3))")
    println("  Sync events: $(length(sync_policy))")
    
    # Create comparison plot
    p = plot([reward_script, reward_policy], 
             label=["Macro-Script" "Policy Tree"],
             xlabel="Planning Mode",
             ylabel="Total Reward",
             title="Planning Mode Comparison",
             marker=:circle,
             markersize=8,
             grid=true)
    
    # Save plot
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    plot_filename = joinpath(output_dir, "planning_mode_comparison.png")
    savefig(p, plot_filename)
    println("\n📁 Comparison plot saved as '$(plot_filename)'")
    
    return p, gs_state_script, gs_state_policy
end

"""
Simple test with configurable parameters
"""
function simple_test(num_steps::Int=NUM_STEPS, planning_mode::Symbol=PLANNING_MODE)
    println("🧪 Simple Test Configuration")
    println("============================")
    println("Number of steps: $(num_steps)")
    println("Planning mode: $(planning_mode)")
    println("Number of agents: 2")
    
    gs_state, agents, reward, sync_events, agent_rewards, env_evolution, action_history = 
        simulate_async_centralized_planning(num_steps, planning_mode=planning_mode)
    
    return gs_state, agents, reward, sync_events, agent_rewards, env_evolution, action_history
end

"""
Quick test with custom number of steps
"""
function quick_test(steps::Int)
    println("⚡ Quick Test with $(steps) steps")
    println("=================================")
    return simple_test(steps, :script)
end

# Main execution
println("🎯 Asynchronous Centralized Planning Test")
println("========================================")
println("Configuration:")
println("  NUM_STEPS: $(NUM_STEPS)")
println("  PLANNING_MODE: $(PLANNING_MODE)")
println("  Number of agents: 2 (same circular trajectory, different phases)")
    println("  Ground Station: Position ($(GROUND_STATION_X),$(GROUND_STATION_Y))")

# Test basic functionality
println("\n🧪 Basic Functionality Test")
gs_state, agents, reward, sync_events, agent_rewards, env_evolution, action_history = simple_test()

# # Compare planning modes (only if NUM_STEPS is reasonable)
# if NUM_STEPS <= 100
#     println("\n🔬 Planning Mode Comparison")
#     comparison_plot, gs_script, gs_policy = compare_planning_modes()
# end

# println("\n✅ All tests completed!")
# println("\n📁 Generated files in 'visualizations' folder:")
# println("- async_centralized_simulation.gif")
# println("- agent_trajectories.png")
# println("- action_statistics.png")
# if NUM_STEPS <= 100
#     println("- planning_mode_comparison.png")
# end

# println("\n" * "="^60)
# println("🚨 HOW TO CHANGE NUMBER OF STEPS:")
# println("="^60)
# println("Option 1: Change line 18 in this file:")
# println("           const NUM_STEPS = 30  ← Change this number")
# println()
# println("Option 2: Call quick_test() function:")
# println("           quick_test(15)        ← This will run 15 steps")
# println()
# println("Option 3: Call simple_test() function:")
# println("           simple_test(20, :policy)  ← 20 steps with policy tree")
# println("="^60) 
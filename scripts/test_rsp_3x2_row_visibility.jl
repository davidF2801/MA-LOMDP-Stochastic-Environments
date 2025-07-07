#!/usr/bin/env julia

"""
Test script for RSP (Random Spread Process) with 3x2 grid and row-only visibility
Demonstrates exact world enumeration for macro-script evaluation
"""

println("🚀 RSP 3x2 Row Visibility Test starting...")

using POMDPs
using POMDPTools
using Random
using Plots
Plots.plotlyjs()
using Infiltrator

# Set random seed for reproducibility
Random.seed!(42)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# 🎯 MAIN SIMULATION PARAMETERS
const NUM_STEPS = 50                # Total simulation steps
const PLANNING_MODE = :script         # Use macro-script planning

# 🌍 ENVIRONMENT PARAMETERS
const GRID_WIDTH = 2                  # Grid width (columns)
const GRID_HEIGHT = 3                 # Grid height (rows)
const INITIAL_EVENTS = 1              # Number of initial events
const MAX_SENSING_TARGETS = 1         # Maximum cells an agent can sense per step
const SENSOR_RANGE = 0.0              # Sensor range for agents (0.0 = row-only visibility)
const DISCOUNT_FACTOR = 0.95          # POMDP discount factor

# 📊 RSP PARAMETERS
const IGNITION_PROBABILITY = 0.2       # Base ignition probability for λmap
const DEATH_RATE = 0.3                 # Death rate for events (30% chance of death per timestep)
const BIRTH_RATE = 0.01                # Spontaneous birth rate

# 🤖 AGENT PARAMETERS
const NUM_AGENTS = 2                  # Number of agents (one per row)
const PLANNING_HORIZON = 3            # Planning horizon for macro-scripts
const SENSOR_FOV = pi/2               # Field of view angle (radians)
const SENSOR_NOISE = 0.0              # Perfect observations

# 📡 COMMUNICATION PARAMETERS
const CONTACT_HORIZON = 3             # Steps until next sync opportunity
const GROUND_STATION_X = 1            # Ground station X position
const GROUND_STATION_Y = 1            # Ground station Y position

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

# Import RSP functions
import .Environment.EventDynamicsModule: transition_rsp!, get_transition_probability_rsp
import .Agents.BeliefManagement: predict_belief_rsp

println("✅ All modules imported successfully")

# =============================================================================
# ROW-ONLY VISIBILITY FUNCTIONS
# =============================================================================

"""
Get agent position at time t for row-cycling agents
Agents cycle through rows: Agent 1: 1→2→3→1→2→3..., Agent 2: 3→1→2→3→1→2...
"""
function get_agent_position_row_cycle(agent_id::Int, time::Int, phase_offset::Int)
    # Calculate which row the agent should be in at this time
    # Each agent cycles through rows 1, 2, 3 with period 3
    cycle_position = (time + phase_offset) % 3
    
    # Map cycle position to actual row
    if cycle_position == 0
        row = 1
    elseif cycle_position == 1
        row = 2
    else  # cycle_position == 2
        row = 3
    end
    
    # Always in column 1
    return (1, row)
end

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
Create agents with row-only visibility (two agents)
"""
function create_row_agents()
    agents = Agent[]
    
    # Create two agents: one in row 1, one in row 3
    agent_rows = [1, 3]
    
    for (i, row) in enumerate(agent_rows)
        # Create trajectory that cycles through all three rows
        # For a 3-row grid, we need to cycle through positions (1,1), (1,2), (1,3)
        if i == 1
            # Agent 1: starts in row 1, cycles 1→2→3→1→2→3...
            trajectory = LinearTrajectory(1, 1, 1, 3, 3)  # Period = 3, moves from row 1 to row 3
            phase_offset = 0  # Start at row 1
        else
            # Agent 2: starts in row 3, cycles 3→1→2→3→1→2...
            trajectory = LinearTrajectory(1, 1, 1, 3, 3)  # Period = 3, moves from row 1 to row 3
            phase_offset = 2  # Start at row 3 (phase offset of 2)
        end
        
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
    # Create event dynamics (simplified for RSP)
    event_dynamics = EventDynamics(BIRTH_RATE, DEATH_RATE, 0.0, 0.0, 0.0)
    
    # Create agents
    agents = create_row_agents()
    
    # Create spatial grid environment with RSP dynamics
    env = SpatialGrid(GRID_WIDTH, GRID_HEIGHT, event_dynamics, agents, SENSOR_RANGE, DISCOUNT_FACTOR, INITIAL_EVENTS, MAX_SENSING_TARGETS, (GROUND_STATION_X, GROUND_STATION_Y))
    
    # Update to RSP dynamics
    env.dynamics = rsp  # Use RSP dynamics (enum value)
    env.ignition_prob = fill(IGNITION_PROBABILITY, GRID_HEIGHT, GRID_WIDTH)
    
    println("🌍 Created RSP test environment:")
    println("  Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
    println("  Initial events: $(INITIAL_EVENTS)")
    println("  Max sensing targets: $(MAX_SENSING_TARGETS)")
    println("  Dynamics: RSP (Random Spread Process)")
    println("  Ignition probability: $(IGNITION_PROBABILITY)")
    
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
    
    # Create plot
    p = plot(; xlim=(0.5, width+0.5), ylim=(0.5, height+0.5),
        aspect_ratio=:equal, size=(600, 800), legend=false,
        xlabel="X Coordinate", ylabel="Y Coordinate",
        grid=true, gridalpha=0.3,
        title="RSP 3x2 Test - Time Step $(time_step)\nEvents: $(count(==(EVENT_PRESENT), environment_state)) | Agents: $(length(agents))",
        titlefontsize=12)

    # Overlay: Row visibility as semi-transparent colored bands
    agent_colors = [:red, :green]
    for (i, agent) in enumerate(agents)
        row = agent.trajectory.start_y
        # Draw row band
        for col in 1:width
            scatter!(p, [col], [row]; marker=:rect, markersize=40, color=agent_colors[i], alpha=0.15, label="")
        end
    end

    # Overlay: Ground station as green star
    scatter!(p, [ground_station_pos[1]], [ground_station_pos[2]]; marker=:star, markersize=14, color=:green, alpha=0.9, label="Ground Station")

    # Overlay: Agents as small circles
    for (i, agent) in enumerate(agents)
        pos = get_agent_position_row_cycle(agent.id, time_step, agent.phase_offset)
        agent_color = agent_colors[mod(i-1, length(agent_colors))+1]
        scatter!(p, [pos[1]], [pos[2]]; marker=:circle, markersize=10, color=agent_color, alpha=0.9, label="Agent $(agent.id)")
    end

    # Overlay: Fire emoji for events
    for y in 1:height, x in 1:width
        if environment_state[y, x] == EVENT_PRESENT
            annotate!(p, x, y, text("🔥", :center, 18))
        end
    end

    # Overlay: FOV (Field of View) as agent-specific colored cells
    for (i, agent) in enumerate(agents)
        if i <= length(actions) && !isempty(actions[i].target_cells)
            fov_cells = actions[i].target_cells
            for cell in fov_cells
                x, y = cell
                scatter!(p, [x], [y]; marker=:rect, markersize=32, color=agent_colors[i], alpha=0.4, label="")
                annotate!(p, x, y, text("$(agent.id)", :center, 10, agent_colors[i]))
            end
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
    ground_station_pos::Tuple{Int, Int}=(GROUND_STATION_X, GROUND_STATION_Y)
)
    println("\n🎬 Creating RSP Simulation Animation")
    println("====================================")
    
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
    
    # Save animation
    animation_filename = joinpath(output_dir, "rsp_3x2_row_visibility.gif")
    gif(anim, animation_filename, fps=2)
    println("✓ Saved animation: $(basename(animation_filename))")
    
    return anim
end

# =============================================================================
# RSP SIMULATION FUNCTIONS
# =============================================================================

"""
Simulate RSP environment evolution
"""
function simulate_rsp_environment(env, num_steps::Int)
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
        
        # Use RSP transition
        transition_rsp!(new_state, current_state, env.ignition_prob, Random.GLOBAL_RNG)
        
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
            
            prob_no_event = get_transition_probability_rsp(1, current_cell_state, neighbor_states, env.ignition_prob, x, y)
            prob_event = get_transition_probability_rsp(2, current_cell_state, neighbor_states, env.ignition_prob, x, y)
            
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
Simulate asynchronous centralized planning with RSP
"""
function simulate_rsp_async_planning(num_steps::Int=NUM_STEPS)
    println("🚀 Starting RSP Asynchronous Centralized Planning Simulation")
    println("============================================================")
    println("Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
    println("Agents: $(NUM_AGENTS) (rows 1 and 3)")
    println("Planning horizon: $(PLANNING_HORIZON)")
    println("Dynamics: RSP")
    
    # Create environment and agents
    env = create_rsp_environment()
    agents = env.agents
    
    # Initialize ground station
    gs_state = GroundStation.initialize_ground_station(env, agents, num_states=2)
    
    # Track performance metrics
    sync_events = []
    observed_events = Set{Tuple{Int, Tuple{Int, Int}}}()
    all_events_ever_appeared = Set{Tuple{Int, Tuple{Int, Int}}}()
    
    # Track environment and actions for visualization
    environment_evolution = Matrix{EventState}[]
    action_history = Vector{Vector{SensingAction}}()
    
    # Get initial environment state
    initial_state_dist = POMDPs.initialstate(env)
    current_state = rand(initial_state_dist)
    current_environment = current_state.event_map
    
    println("\n📊 Starting simulation...")
    
    for t in 0:(num_steps-1)
        println("\n⏰ Time step $(t)")        
        # Check for synchronization opportunities
        old_sync_times = copy(gs_state.agent_last_sync)
        GroundStation.maybe_sync!(env, gs_state, agents, t, planning_mode=PLANNING_MODE)
        
        # Record sync events
        for (agent_id, old_time) in old_sync_times
            if gs_state.agent_last_sync[agent_id] != old_time
                push!(sync_events, (t, agent_id))
                println("📡 Sync event: Agent $(agent_id) at time $(t)")
                
                # Note: We're using the new best_script function that does exact evaluation
                # without pre-computing worlds. The enumeration logic is no longer used.
                println("🔍 Using exact evaluation for agent $(agent_id)...")
            end
        end
        
        # Execute agent actions
        joint_actions = SensingAction[]
        
        for agent in agents
            # Get plan from ground station and execute it
            plan, plan_type = GroundStation.get_agent_plan(agent, gs_state)
            action = execute_plan(agent, plan, plan_type, agent.observation_history)
            push!(joint_actions, action)
            # Use POMDP interface to get observations
            if !isempty(action.target_cells)
                # Get observation using POMDP observation model
                observation_dist = POMDPs.observation(env, action, current_state)
                observation = rand(observation_dist)
                push!(agent.observation_history, observation)
                
                # Track observed events for reward calculation
                for (i, cell) in enumerate(observation.sensed_cells)
                    if i <= length(observation.event_states) && observation.event_states[i] == EVENT_PRESENT
                        push!(observed_events, (t, cell))  # Track (timestep, cell) to count multiple observations
                    end
                end
                
                # Debug: print what was observed
                events_found = count(==(EVENT_PRESENT), observation.event_states)
                println("  Agent $(agent.id): $(length(action.target_cells)) cells sensed, $(events_found) events found")
            else
                println("  Agent $(agent.id): wait action")
            end
        end
        
        # Track events that appeared in this step
        for y in 1:GRID_HEIGHT, x in 1:GRID_WIDTH
            if current_environment[y, x] == EVENT_PRESENT
                push!(all_events_ever_appeared, (t, (x, y)))  # Track (timestep, cell) to count multiple events
            end
        end
        
        # Record environment state and actions for visualization
        push!(environment_evolution, copy(current_environment))
        push!(action_history, joint_actions)
        # Update environment using POMDP transition (handles both environment and agent movement)
        if t < num_steps - 1
            # Use POMDP transition to update both environment and agent positions
            transition_dist = POMDPs.transition(env, current_state, SensingAction(1, [], false))  # Dummy action for transition
            current_state = rand(transition_dist)
            current_environment = current_state.event_map
        end
        
        # Print status every few steps
        if t % 5 == 0
            println("📊 Status: $(count(==(EVENT_PRESENT), current_environment)) events active")
        end
        
        # Debug: print agent positions
        println("  Agent positions at time $(t):")
        for (i, agent) in enumerate(agents)
            pos = get_agent_position_row_cycle(agent.id, t, agent.phase_offset)
            println("    Agent $(agent.id): $(pos)")
        end
    end
    
    # Calculate final reward metric
    total_events_appeared = length(all_events_ever_appeared)  # Count of (timestep, cell) pairs where events appeared
    total_events_observed = length(observed_events)  # Count of (timestep, cell) pairs where events were observed
    event_observation_percentage = total_events_appeared > 0 ? (total_events_observed / total_events_appeared) * 100.0 : 0.0
    
    # Print final results
    println("\n📈 RSP Simulation Results")
    println("=========================")
    println("Event Observation Performance:")
    println("  Total events that appeared: $(total_events_appeared)")
    println("  Total events observed: $(total_events_observed)")
    println("  Event observation percentage: $(round(event_observation_percentage, digits=1))%")
    println("  Unique cells with events: $(length(Set([cell for (_, cell) in all_events_ever_appeared])))")
    println("  Unique cells observed: $(length(Set([cell for (_, cell) in observed_events])))")
    println("")
    println("System Performance:")
    println("  Total sync events: $(length(sync_events))")
    println("  Grid size: $(GRID_WIDTH)x$(GRID_HEIGHT)")
    println("  Planning horizon: $(PLANNING_HORIZON)")
    println("  Dynamics: RSP")
    
    # Create visualizations
    println("\n🎨 Creating Visualizations...")
    println("============================")
    
    # Create simulation animation
    anim = create_rsp_animation(agents, num_steps, environment_evolution, action_history, (GROUND_STATION_X, GROUND_STATION_Y))
    
    println("\n✅ RSP simulation completed!")
    println("📁 Check the 'visualizations' folder for:")
    println("  - rsp_3x2_row_visibility.gif (main simulation animation)")
    
    return gs_state, agents, event_observation_percentage, sync_events, environment_evolution, action_history
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

println("🎯 RSP 3x2 Row Visibility Test")
println("==============================")
println("Configuration:")
println("  Grid: $(GRID_WIDTH)x$(GRID_HEIGHT)")
println("  Agents: $(NUM_AGENTS) (rows 1 and 3)")
println("  Planning horizon: $(PLANNING_HORIZON)")
println("  Dynamics: RSP")
println("  Row-only visibility: true")

# Test RSP environment evolution first
println("\n🔥 Testing RSP Environment Evolution...")
env = create_rsp_environment()
test_evolution = simulate_rsp_environment(env, 5)
println("✅ RSP environment evolution test completed!")

# Test trajectory calculations
println("\n🧭 Testing Trajectory Calculations...")
agents = env.agents
for agent in agents
    println("Agent $(agent.id): trajectory=$(agent.trajectory), phase_offset=$(agent.phase_offset)")
    println("Expected cycle: Agent $(agent.id) should cycle through rows $(agent.id == 1 ? "1→2→3→1→2→3..." : "3→1→2→3→1→2...")")
    for t in 0:9
        pos = get_agent_position_row_cycle(agent.id, t, agent.phase_offset)
        println("  Time $(t): $(pos)")
    end
end
println("✅ Trajectory test completed!")

# Run the simulation
gs_state, agents, percentage, sync_events, env_evolution, action_history = simulate_rsp_async_planning()

println("\n✅ RSP test completed!")
println("📊 Final event observation percentage: $(round(percentage, digits=1))%") 
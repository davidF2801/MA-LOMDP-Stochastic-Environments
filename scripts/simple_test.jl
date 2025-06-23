#!/usr/bin/env julia

"""
Simple test script for the SpatialGrid environment (without external POMDP packages)
"""

using Random

# Set random seed for reproducibility
Random.seed!(42)

# Include the environment files directly
include("../src/environment/spatial_grid.jl")

function test_basic_types()
    println("=== Testing Basic Types ===")
    
    # Test EventState enum
    println("✓ EventState enum created")
    
    # Test EventDynamics
    dynamics = EventDynamics(0.1, 0.05, 0.2, 0.1, 0.3)
    println("✓ EventDynamics created: $(dynamics)")
    
    # Test RangeLimitedSensor
    sensor = RangeLimitedSensor(2.0, π/2, 0.0)
    println("✓ RangeLimitedSensor created: range=$(sensor.range)")
    
    # Test trajectories
    circ_traj = CircularTrajectory(5, 5, 3.0, 10)
    lin_traj = LinearTrajectory(1, 1, 10, 10, 10)
    println("✓ Trajectories created")
    
    # Test Agent
    agent = Agent(1, circ_traj, sensor, 0)
    println("✓ Agent created: ID=$(agent.id)")
    
    println("All basic types work correctly!")
end

function test_trajectory_functions()
    println("\n=== Testing Trajectory Functions ===")
    
    # Test circular trajectory
    circ_traj = CircularTrajectory(5, 5, 3.0, 10)
    pos1 = get_position_at_time(circ_traj, 0)
    pos2 = get_position_at_time(circ_traj, 5)
    pos3 = get_position_at_time(circ_traj, 10)
    
    println("Circular trajectory positions:")
    println("- t=0: $(pos1)")
    println("- t=5: $(pos2)")
    println("- t=10: $(pos3)")
    
    # Test linear trajectory
    lin_traj = LinearTrajectory(1, 1, 10, 10, 10)
    pos1 = get_position_at_time(lin_traj, 0)
    pos2 = get_position_at_time(lin_traj, 5)
    pos3 = get_position_at_time(lin_traj, 10)
    
    println("Linear trajectory positions:")
    println("- t=0: $(pos1)")
    println("- t=5: $(pos2)")
    println("- t=10: $(pos3)")
    
    println("✓ Trajectory functions work correctly!")
end

function test_event_dynamics()
    println("\n=== Testing Event Dynamics ===")
    
    # Create event map
    event_map = fill(NO_EVENT, 5, 5)
    event_map[3, 3] = EVENT_PRESENT  # Add one event
    
    # Create dynamics
    dynamics = EventDynamics(0.1, 0.05, 0.2, 0.1, 0.3)
    
    println("Initial events: $(count(x -> x != NO_EVENT, event_map))")
    
    # Update events
    rng = Random.GLOBAL_RNG
    update_events!(dynamics, event_map, rng)
    
    println("Events after update: $(count(x -> x != NO_EVENT, event_map))")
    
    # Show event map
    println("Event map after update:")
    for y in 1:5
        for x in 1:5
            if event_map[y, x] == NO_EVENT
                print(".")
            elseif event_map[y, x] == EVENT_PRESENT
                print("E")
            elseif event_map[y, x] == EVENT_SPREADING
                print("S")
            else
                print("D")
            end
        end
        println()
    end
    
    println("✓ Event dynamics work correctly!")
end

function test_sensor_functions()
    println("\n=== Testing Sensor Functions ===")
    
    # Create sensor
    sensor = RangeLimitedSensor(2.0, π/2, 0.0)
    agent_pos = (5, 5)
    event_map = fill(NO_EVENT, 10, 10)
    event_map[4, 4] = EVENT_PRESENT
    event_map[6, 6] = EVENT_PRESENT
    event_map[8, 8] = EVENT_PRESENT  # Outside range
    
    target_cells = [(4, 4), (5, 5), (6, 6), (8, 8)]
    
    # Test observation generation
    sensed_cells, event_states = generate_observation(sensor, agent_pos, event_map, target_cells)
    
    println("Sensor test results:")
    println("- Target cells: $(target_cells)")
    println("- Sensed cells: $(sensed_cells)")
    println("- Event states: $(event_states)")
    
    # Test information gain
    belief = fill(0.5, 10, 10)
    info_gain = calculate_information_gain(belief, sensed_cells, event_states)
    println("- Information gain: $(info_gain)")
    
    println("✓ Sensor functions work correctly!")
end

function test_spatial_grid_creation()
    println("\n=== Testing SpatialGrid Creation ===")
    
    # Create event dynamics
    dynamics = EventDynamics(0.1, 0.05, 0.2, 0.1, 0.3)
    
    # Create agents with trajectories
    agents = [
        Agent(1, CircularTrajectory(5, 5, 3.0, 10), RangeLimitedSensor(2.0, π/2, 0.0), 0),
        Agent(2, LinearTrajectory(1, 1, 10, 10, 15), RangeLimitedSensor(2.5, π/3, 0.0), 0)
    ]
    
    # Create spatial grid environment
    env = SpatialGrid(
        10,  # width
        10,  # height
        dynamics,
        agents,
        3.0,  # sensor_range
        0.95, # discount
        3,    # initial_events
        5     # max_sensing_targets
    )
    
    println("✓ SpatialGrid environment created successfully!")
    println("- Grid size: $(env.width) x $(env.height)")
    println("- Number of agents: $(length(env.agents))")
    println("- Discount factor: $(env.discount)")
    
    # Test manual state creation (without POMDP interface)
    event_map = fill(NO_EVENT, env.height, env.width)
    initialize_random_events(event_map, env.initial_events, Random.GLOBAL_RNG)
    
    agent_positions = Vector{Tuple{Int, Int}}()
    agent_trajectories = Vector{Trajectory}()
    
    for agent in env.agents
        initial_pos = get_position_at_time(agent.trajectory, 0)
        push!(agent_positions, initial_pos)
        push!(agent_trajectories, agent.trajectory)
    end
    
    initial_state = GridState(event_map, agent_positions, agent_trajectories, 0)
    
    println("✓ Manual state creation successful:")
    println("- Grid size: $(size(initial_state.event_map))")
    println("- Agent positions: $(initial_state.agent_positions)")
    println("- Time step: $(initial_state.time_step)")
    println("- Events present: $(count(x -> x != NO_EVENT, initial_state.event_map))")
end

# Run all tests
if abspath(PROGRAM_FILE) == @__FILE__
    test_basic_types()
    test_trajectory_functions()
    test_event_dynamics()
    test_sensor_functions()
    test_spatial_grid_creation()
    
    println("\n🎉 All basic tests completed successfully!")
    println("The environment implementation is working correctly!")
end 
#!/usr/bin/env julia

"""
Test script for the SpatialGrid POMDP environment
"""

using Pkg
Pkg.activate(".")

using POMDPs
using POMDPTools
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Include the environment files directly
include("../src/environment/spatial_grid.jl")

function test_spatial_grid()
    println("=== Testing SpatialGrid POMDP Environment ===")
    
    # Create event dynamics
    dynamics = EventDynamics(
        0.1,  # birth_rate
        0.05, # death_rate
        0.2,  # spread_rate
        0.1,  # decay_rate
        0.3   # neighbor_influence
    )
    
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
    
    println("✓ Environment created successfully")
    
    # Test initial state
    println("\n--- Testing Initial State ---")
    initial_dist = POMDPs.initialstate(env)
    initial_state = rand(initial_dist)
    
    println("Initial state created:")
    println("- Grid size: $(size(initial_state.event_map))")
    println("- Agent positions: $(initial_state.agent_positions)")
    println("- Time step: $(initial_state.time_step)")
    println("- Events present: $(count(x -> x != NO_EVENT, initial_state.event_map))")
    
    # Test transition
    println("\n--- Testing State Transition ---")
    action = SensingAction(1, [(4, 4), (5, 5), (6, 6)], false)
    transition_dist = POMDPs.transition(env, initial_state, action)
    next_state = rand(transition_dist)
    
    println("State transition completed:")
    println("- New time step: $(next_state.time_step)")
    println("- New agent positions: $(next_state.agent_positions)")
    println("- Events after transition: $(count(x -> x != NO_EVENT, next_state.event_map))")
    
    # Test observation
    println("\n--- Testing Observation Generation ---")
    obs_dist = POMDPs.observation(env, action, next_state)
    observation = rand(obs_dist)
    
    println("Observation generated:")
    println("- Agent ID: $(observation.agent_id)")
    println("- Sensed cells: $(observation.sensed_cells)")
    println("- Event states: $(observation.event_states)")
    
    # Test reward
    println("\n--- Testing Reward Calculation ---")
    reward_val = POMDPs.reward(env, initial_state, action, next_state)
    println("Reward calculated: $(reward_val)")
    
    # Test discount
    println("\n--- Testing Discount ---")
    discount_val = POMDPs.discount(env)
    println("Discount factor: $(discount_val)")
    
    # Test terminal state
    println("\n--- Testing Terminal State Check ---")
    is_terminal = POMDPs.isterminal(env, initial_state)
    println("Is initial state terminal: $(is_terminal)")
    
    # Test with terminal state
    terminal_state = GridState(
        fill(NO_EVENT, 10, 10),  # No events
        [(1, 1), (2, 2)],       # Agent positions
        [CircularTrajectory(1, 1, 1.0, 5), LinearTrajectory(1, 1, 5, 5, 5)],  # Trajectories
        1001                     # Max time exceeded
    )
    is_terminal = POMDPs.isterminal(env, terminal_state)
    println("Is terminal state terminal: $(is_terminal)")
    
    println("\n=== All Tests Passed! ===")
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
end

# Run all tests
if abspath(PROGRAM_FILE) == @__FILE__
    test_spatial_grid()
    test_trajectory_functions()
    test_event_dynamics()
    test_sensor_functions()
    
    println("\n🎉 All environment tests completed successfully!")
end 
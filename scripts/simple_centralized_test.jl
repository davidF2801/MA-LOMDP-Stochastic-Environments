#!/usr/bin/env julia

"""
Simple test script for centralized planner to isolate issues
"""

using Pkg
Pkg.activate(".")

using Random

# Set random seed for reproducibility
Random.seed!(42)

# Include only the essential files
include("../src/types.jl")
include("../src/environment/spatial_grid.jl")

println("✓ Types and spatial grid loaded successfully")

# Test basic functionality
println("\n🔍 Testing Basic Functionality")
println("=============================")

# Create a simple environment
width = 6
height = 4

# Create event dynamics
event_dynamics = EventDynamics(0.01, 0.05, 0.01, 0.1, 0.2)

# Create a single agent
sensor = RangeLimitedSensor(1.5, π, 0.1)
trajectory = CircularTrajectory(3, 2, 1.0, 4)
agent = create_agent(1, trajectory, sensor, width, height)

# Create environment
env = SpatialGrid(width, height, event_dynamics, [agent], 1.5, 0.95, 1, 1)

println("✓ Environment created successfully")
println("Grid size: $(width)x$(height)")
println("Number of agents: $(length(env.agents))")

# Test POMDP interface
println("\n🎯 Testing POMDP Interface")
println("==========================")

# Test initial state
initial_state_dist = POMDPs.initialstate(env)
initial_state = rand(initial_state_dist)
println("✓ Initial state created")

# Test action space
actions = POMDPs.actions(env, initial_state)
println("✓ Action space generated: $(length(actions)) actions")

# Test a simple action
if !isempty(actions)
    test_action = actions[1]
    println("✓ Test action: $(test_action)")
    
    # Test transition
    next_state_dist = POMDPs.transition(env, initial_state, test_action)
    next_state = rand(next_state_dist)
    println("✓ State transition successful")
    
    # Test observation
    obs_dist = POMDPs.observation(env, test_action, next_state)
    obs = rand(obs_dist)
    println("✓ Observation generated")
    
    # Test reward
    reward = POMDPs.reward(env, initial_state, test_action, next_state)
    println("✓ Reward calculated: $(reward)")
end

println("\n✅ Basic functionality test completed successfully!") 
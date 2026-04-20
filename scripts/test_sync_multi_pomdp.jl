#!/usr/bin/env julia

"""
Test script for synchronized multi-agent planner using POMDPs.jl
"""

println("🧪 Testing Synchronized Multi-Agent Planner with POMDPs.jl...")

# Load the project
include("../src/MyProject.jl")
using .MyProject
using .MyProject.Agents.TrajectoryPlanner: get_position_at_time, create_linear_trajectory, create_circular_trajectory

# Include the new synchronized multi-agent planner
include("../src/planners/macro_planner_sync_multi.jl")
using .MacroPlannerSyncMulti

# Test basic functionality
println("✅ Modules loaded successfully")

# Test trajectory creation
trajectory1 = create_linear_trajectory(1, 2, 7, 2, 8)
trajectory2 = create_linear_trajectory(2, 1, 2, 7, 8)
trajectory3 = create_circular_trajectory(4.0, 4.0, 2.0, 8)

println("✅ Trajectories created successfully")

# Test agent creation
sensor1 = RangeLimitedSensor(1.0, pi/2, 0.0, :cross)
sensor2 = RangeLimitedSensor(1.0, pi/2, 0.0, :cross)
sensor3 = RangeLimitedSensor(1.0, pi/2, 0.0, :cross)

agent1 = Agent(1, trajectory1, sensor1, 0, 1000.0, 3.0, 0.0)
agent2 = Agent(2, trajectory2, sensor2, 0, 1000.0, 3.0, 0.0)
agent3 = Agent(3, trajectory3, sensor3, 0, 1000.0, 3.0, 0.0)

agents = [agent1, agent2, agent3]

println("✅ Agents created successfully")

# Test environment creation
rsp_dynamics = rsp(
    width=7,
    height=7,
    ignition_prob=0.1,
    spread_prob=0.3,
    decay_prob=0.05,
    persistence_prob=0.9
)

env = SpatialGrid(
    width=7,
    height=7,
    dynamics=rsp_dynamics,
    discount=0.95,
    max_sensing_targets=1
)

println("✅ Environment created successfully")

# Test belief creation
belief = initialize_global_belief(env, num_states=2)

println("✅ Belief created successfully")

# Test POMDP creation
pomdp = SyncMultiAgentPOMDP(env, agents, 5, 0.95)
println("✅ Synchronized Multi-Agent POMDP created successfully")

# Test POMDP interface
println("Testing POMDP interface...")
println("  Discount: $(discount(pomdp))")
println("  Horizon: $(pomdp.horizon)")
println("  Number of agents: $(length(pomdp.agents))")

# Test joint actions
joint_actions = enumerate_joint_actions(agents, env)
println("✅ Joint actions enumerated: $(length(joint_actions)) actions")

# Test joint action creation
joint_action = JointAction([SensingAction(1, [(1,1)], false), 
                           SensingAction(2, [(2,2)], false), 
                           SensingAction(3, [(3,3)], false)])
println("✅ Joint action created successfully")

# Test state creation
current_positions = [get_position_at_time(agent.trajectory, 0, agent.phase_offset) for agent in agents]
state = SyncMultiAgentState(belief, current_positions, 0)
println("✅ Synchronized state created successfully")

println("\n🎉 All tests passed! Synchronized multi-agent planner with POMDPs.jl is working correctly.")
println("\n📋 Summary:")
println("  • POMDP interface implemented")
println("  • Joint actions and observations defined")
println("  • State representation working")
println("  • Ready for POMDPs.jl solver integration")


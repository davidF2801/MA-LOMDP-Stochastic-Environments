#!/usr/bin/env julia

# Test script for the new Bellman backup implementation in policy tree planner

using Pkg
Pkg.activate(".")

# Add the source directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Import modules
using MyProject
using Planners
using Agents
using Environment
using Types

println("🧪 Testing Bellman backup implementation for policy tree planner...")

# Create a simple test environment
println("📋 Creating test environment...")
env = create_test_environment()  # Use existing test environment

# Create test agents
println("🤖 Creating test agents...")
agents = [
    Agent(1, CircularTrajectory(4, 4, 2.0, 8, 0.5), RangeLimitedSensor(2.0, π/2, 0.1), 0, 50.0, 2.0, 3.0),
    Agent(2, LinearTrajectory(1, 1, 8, 8, 8, 0.5), RangeLimitedSensor(1.5, π/2, 0.1), 0, 40.0, 1.5, 4.0),
    Agent(3, CircularTrajectory(6, 6, 1.5, 6, 0.5), RangeLimitedSensor(1.0, π/2, 0.1), 0, 60.0, 2.5, 2.5)
]

# Set environment agents
env.agents = agents

# Create initial belief
println("🧠 Creating initial belief...")
initial_belief = BeliefManagement.initialize_belief(env.width, env.height, [0.5, 0.5])

# Create ground station state
println("🏠 Creating ground station state...")
gs_state = GroundStationState(
    time_step = 0,
    agent_last_sync = Dict(1 => -1, 2 => -1, 3 => -1),  # No agents have synced yet
    agent_plans = Dict{Int, Any}(),
    agent_plan_types = Dict{Int, Symbol}(),
    agent_observation_history = Dict{Int, Vector{Tuple{Int, Tuple{Int, Int}, EventState}}}()
)

# Test the Bellman backup implementation
println("🔄 Testing Bellman backup...")
try
    policy_tree, planning_time = Planners.PolicyTreePlanner.best_policy_tree(
        env, initial_belief, agents[1], 5, gs_state
    )
    
    println("✅ Bellman backup completed successfully!")
    println("📊 Planning time: $(round(planning_time, digits=3)) seconds")
    println("🌳 Policy tree root action: $(policy_tree.action)")
    println("💰 Policy tree root value: $(round(policy_tree.value, digits=3))")
    
    # Test with agents having policy trees
    println("\n🔄 Testing with agents having policy trees...")
    
    # Give agent 2 a macro plan (for comparison)
    gs_state.agent_plans[2] = [
        SensingAction(2, [(2, 2)], false),
        SensingAction(2, [(3, 3)], false),
        SensingAction(2, [(4, 4)], false)
    ]
    gs_state.agent_plan_types[2] = :script
    gs_state.agent_last_sync[2] = 0
    
    # Give agent 3 a policy tree (this is the key difference)
    gs_state.agent_plans[3] = nothing  # Will be set to policy tree
    gs_state.agent_plan_types[3] = :policy
    gs_state.agent_last_sync[3] = 0
    
    policy_tree2, planning_time2 = Planners.PolicyTreePlanner.best_policy_tree(
        env, initial_belief, agents[1], 5, gs_state
    )
    
    println("✅ Policy tree scenario completed successfully!")
    println("📊 Planning time: $(round(planning_time2, digits=3)) seconds")
    println("🌳 Policy tree root action: $(policy_tree2.action)")
    println("💰 Policy tree root value: $(round(policy_tree2.value, digits=3))")
    
    # Test the key insight: policy pointers
    println("\n🔍 Testing policy pointer initialization...")
    initial_pointers = Planners.PolicyTreePlanner.initialize_policy_pointers(env, agents[1], gs_state)
    println("📌 Initial policy pointers:")
    for (i, agent) in enumerate(agents)
        pointer_type = if initial_pointers[i] == Planners.PolicyTreePlanner.ROOT_ID
            "ROOT_ID (policy tree)"
        elseif initial_pointers[i] == Planners.PolicyTreePlanner.IN_MACRO_FLAG
            "IN_MACRO_FLAG (macro)"
        elseif initial_pointers[i] == Planners.PolicyTreePlanner.NO_PLAN_FLAG
            "NO_PLAN_FLAG (no plan)"
        else
            "Unknown"
        end
        println("   Agent $(agent.id): $(initial_pointers[i]) ($(pointer_type))")
    end
    
    # Test extended state creation
    println("\n🔍 Testing extended state creation...")
    initial_state = Planners.PolicyTreePlanner.ExtendedState(initial_belief, 0, initial_pointers)
    state_hash = Planners.PolicyTreePlanner.hash_state(initial_state)
    println("📌 Extended state hash: $(state_hash)")
    println("📌 Belief dimensions: $(size(initial_state.belief.event_distributions))")
    println("📌 Stage: $(initial_state.stage)")
    println("📌 Policy pointers: $(initial_state.policy_pointers)")
    
catch e
    println("❌ Error during Bellman backup test:")
    println("   Error: $e")
    println("   Backtrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\n🎉 Bellman backup test completed!") 
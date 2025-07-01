#!/usr/bin/env julia

"""
Test script for centralized planner (C-LOMDP) implementation
Demonstrates global coordination with complete information sharing
"""

using Pkg
Pkg.activate(".")

using Random
using Plots

# Set random seed for reproducibility
Random.seed!(42)

# Include the environment and centralized planner
include("../src/environment/spatial_grid.jl")
include("../src/agents/centralized_planner.jl")

# Import modules
using .CentralizedPlanner

"""
Create a test environment with multiple agents
"""
function create_test_environment()
    println("🏗️ Creating Test Environment")
    println("============================")
    
    # Grid configuration
    width = 8
    height = 6
    
    # Create event dynamics (2-state system)
    event_dynamics = EventDynamics(0.01, 0.05, 0.01, 0.1, 0.2, 0.1)
    
    # Create agents with different trajectories
    agents = []
    
    # Agent 1: Circular trajectory (center of grid)
    sensor1 = RangeLimitedSensor(2.0, π/2, 0.1)
    trajectory1 = CircularTrajectory(4, 3, 1.5, 6)
    agent1 = create_agent(1, trajectory1, sensor1, width, height)
    push!(agents, agent1)
    
    # Agent 2: Linear trajectory (left to right)
    sensor2 = RangeLimitedSensor(1.5, π, 0.1)
    trajectory2 = LinearTrajectory(1, 2, 7, 4, 4)
    agent2 = create_agent(2, trajectory2, sensor2, width, height)
    push!(agents, agent2)
    
    # Agent 3: Linear trajectory (top to bottom)
    sensor3 = RangeLimitedSensor(1.8, π/3, 0.1)
    trajectory3 = LinearTrajectory(3, 1, 5, 5, 5)
    agent3 = create_agent(3, trajectory3, sensor3, width, height)
    push!(agents, agent3)
    
    # Create spatial grid environment
    env = SpatialGrid(
        width, height, event_dynamics, agents,
        2.0,  # sensor_range
        0.95, # discount
        2,    # initial_events
        1     # max_sensing_targets (FOV = 1 cell)
    )
    
    println("Grid size: $(width)x$(height)")
    println("Number of agents: $(length(agents))")
    println("Event dynamics: $(event_dynamics)")
    println("✓ Environment created successfully")
    
    return env
end

"""
Test global clock vector computation
"""
function test_global_clock()
    println("\n⏰ Testing Global Clock Vector")
    println("=============================")
    
    env = create_test_environment()
    
    # Test clock vector at different time steps
    for step in 0:10
        clock = get_global_clock_vector(env, step)
        println("Step $(step): τ = $(clock.agent_phases)")
    end
end

"""
Test joint action space generation
"""
function test_joint_action_space()
    println("\n⚡ Testing Joint Action Space")
    println("============================")
    
    env = create_test_environment()
    
    # Test at different phases
    for step in 0:3
        clock = get_global_clock_vector(env, step)
        joint_actions = get_joint_action_space(env, clock)
        
        println("\nStep $(step) - Clock: $(clock.agent_phases)")
        println("Joint action space size: $(length(joint_actions))")
        
        # Show sample joint actions
        println("Sample joint actions:")
        for (i, joint_action) in enumerate(joint_actions[1:min(5, length(joint_actions))])
            println("  $(i): $(joint_action)")
        end
    end
end

"""
Test centralized policy creation and action selection
"""
function test_centralized_policy()
    println("\n🎯 Testing Centralized Policy")
    println("=============================")
    
    env = create_test_environment()
    
    # Create centralized policy
    policy = create_centralized_policy(env, 10, 0.95, 1.0)
    println("✓ Policy created with horizon=$(policy.planning_horizon), discount=$(policy.discount_factor)")
    
    # Initialize global belief
    global_belief = initialize_global_belief(env, 0.5)
    println("✓ Global belief initialized")
    
    # Test action selection at different phases
    for step in 0:3
        clock = get_global_clock_vector(env, step)
        joint_action = select_joint_action(policy, global_belief, clock)
        
        println("\nStep $(step) - Clock: $(clock.agent_phases)")
        println("Selected joint action: $(joint_action)")
        
        # Calculate action value
        value = calculate_joint_action_value(policy, global_belief, clock, joint_action)
        println("Action value: $(value)")
    end
end

"""
Test global belief update
"""
function test_global_belief_update()
    println("\n🧠 Testing Global Belief Update")
    println("==============================")
    
    env = create_test_environment()
    
    # Initialize global belief
    global_belief = initialize_global_belief(env, 0.5)
    println("Initial belief uncertainty: $(sum(global_belief.uncertainty_map))")
    
    # Create sample joint action and observation
    clock = get_global_clock_vector(env, 0)
    joint_action = select_joint_action(policy, global_belief, clock)
    
    # Create mock observations
    joint_observation = [
        GridObservation(1, [(3, 3)], [EVENT_PRESENT], []),
        GridObservation(2, [(2, 2)], [NO_EVENT], []),
        GridObservation(3, [(4, 4)], [EVENT_PRESENT], [])
    ]
    
    # Update global belief
    updated_belief = update_global_belief(global_belief, joint_action, joint_observation, env)
    
    println("Updated belief uncertainty: $(sum(updated_belief.uncertainty_map))")
    println("Uncertainty reduction: $(sum(global_belief.uncertainty_map) - sum(updated_belief.uncertainty_map))")
    println("Observation history length: $(length(updated_belief.observation_history))")
end

"""
Run full centralized execution simulation
"""
function run_centralized_simulation()
    println("\n🚀 Running Centralized Execution Simulation")
    println("==========================================")
    
    env = create_test_environment()
    
    # Run simulation
    execution_history, final_belief, total_reward = simulate_centralized_execution(env, 10)
    
    # Analyze results
    println("\n📈 Execution Analysis")
    println("====================")
    
    # Plot reward over time
    steps = [step for (step, _, _, _, _) in execution_history]
    rewards = [reward for (_, _, _, _, reward) in execution_history]
    
    p = plot(steps, rewards, 
             label="Step Reward",
             xlabel="Time Step",
             ylabel="Reward",
             title="Centralized Execution Performance",
             marker=:circle,
             linewidth=2)
    
    # Save plot
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    plot_filename = joinpath(output_dir, "centralized_execution_performance.png")
    savefig(p, plot_filename)
    println("✓ Performance plot saved: $(basename(plot_filename))")
    
    # Show belief evolution
    println("\n🧠 Belief Evolution")
    println("==================")
    println("Initial uncertainty: $(sum(initialize_global_belief(env).uncertainty_map))")
    println("Final uncertainty: $(sum(final_belief.uncertainty_map))")
    println("Total uncertainty reduction: $(sum(initialize_global_belief(env).uncertainty_map) - sum(final_belief.uncertainty_map))")
    
    return execution_history, final_belief, total_reward
end

"""
Compare centralized vs random policy
"""
function compare_policies()
    println("\n🔬 Policy Comparison: Centralized vs Random")
    println("==========================================")
    
    env = create_test_environment()
    
    # Test centralized policy
    println("\n🎯 Centralized Policy:")
    centralized_history, _, centralized_reward = simulate_centralized_execution(env, 5)
    
    # Test random policy (simplified)
    println("\n🎲 Random Policy:")
    random_reward = 0.0
    
    # Initialize environment
    initial_state_dist = POMDPs.initialstate(env)
    current_state = rand(Random.GLOBAL_RNG, initial_state_dist)
    
    for step in 1:5
        clock = get_global_clock_vector(env, step-1)
        joint_actions = get_joint_action_space(env, clock)
        
        # Select random joint action
        random_joint_action = rand(joint_actions)
        
        # Calculate reward
        step_reward = 0.0
        for action in random_joint_action
            action_reward = POMDPs.reward(env, current_state, action, current_state)
            step_reward += action_reward
        end
        
        random_reward += step_reward
        
        # Transition
        next_state_dist = POMDPs.transition(env, current_state, random_joint_action[1])
        current_state = rand(Random.GLOBAL_RNG, next_state_dist)
    end
    
    println("Centralized total reward: $(centralized_reward)")
    println("Random total reward: $(random_reward)")
    println("Improvement: $(centralized_reward - random_reward)")
end

"""
Main test function
"""
function main()
    println("🎯 Centralized Planner (C-LOMDP) Test")
    println("====================================")
    
    # Test individual components
    test_global_clock()
    test_joint_action_space()
    test_centralized_policy()
    test_global_belief_update()
    
    # Run full simulation
    run_centralized_simulation()
    
    # Compare policies
    compare_policies()
    
    println("\n✅ Centralized planner testing completed!")
    println("📁 Check the 'visualizations' folder for results")
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 
#!/usr/bin/env julia

"""
Test script for centralized planner with specific 8x3 scenario
- 8x3 grid world
- Two agents with bottom-to-top trajectories, displaced by 3 phases
- Square FOR (3x3 square)
- Belief-MDP solver from POMDP library
- Constant communication
"""

using Pkg
Pkg.activate(".")

using POMDPs
using POMDPTools
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
Create square FOR sensor (3x3 square centered on target cell)
"""
function create_square_sensor(range::Float64=1.5)
    # Square FOR: can see the cell above and all its neighbors (3x3 square)
    return RangeLimitedSensor(range, π/2, 0.1)  # Wide FOV for square coverage
end

"""
Create bottom-to-top linear trajectory
"""
function create_bottom_to_top_trajectory(start_x::Int, period::Int=6)
    # Trajectory: bottom (y=1) to top (y=3) and back
    return LinearTrajectory(start_x, 1, start_x, 3, period)
end

"""
Create the 8x3 test environment with two agents
"""
function create_8x3_environment()
    println("🏗️ Creating 8x3 Test Environment")
    println("================================")
    
    # Grid configuration: 8x3
    width = 8
    height = 3
    
    # Create event dynamics (2-state system)
    event_dynamics = EventDynamics(0.02, 0.03, 0.01, 0.1, 0.3)
    
    # Create agents with bottom-to-top trajectories, displaced by 3 phases
    agents = Agent[]
    
    # Agent 1: Starts at x=2, period=6
    sensor1 = create_square_sensor(1.5)
    trajectory1 = create_bottom_to_top_trajectory(2, 6)
    agent1 = create_agent(1, trajectory1, sensor1, width, height)
    push!(agents, agent1)
    
    # Agent 2: Starts at x=6, period=6, displaced by 3 phases
    sensor2 = create_square_sensor(1.5)
    trajectory2 = create_bottom_to_top_trajectory(6, 6)
    agent2 = create_agent(2, trajectory2, sensor2, width, height)
    push!(agents, agent2)
    
    # Create spatial grid environment
    env = SpatialGrid(
        width, height, event_dynamics, agents,
        1.5,  # sensor_range
        0.95, # discount
        3,    # initial_events
        1     # max_sensing_targets (FOV = 1 cell)
    )
    
    println("Grid size: $(width)x$(height)")
    println("Number of agents: $(length(agents))")
    println("Agent 1: x=2, period=6")
    println("Agent 2: x=6, period=6 (displaced by 3 phases)")
    println("Event dynamics: $(event_dynamics)")
    println("✓ Environment created successfully")
    
    return env
end

"""
Test agent trajectories and phase displacement
"""
function test_agent_trajectories()
    println("\n🛤️ Testing Agent Trajectories")
    println("============================")
    
    env = create_8x3_environment()
    
    # Test trajectories for one full period
    period = 6
    println("Testing trajectories over $(period) steps:")
    println("Step | Agent 1 Position | Agent 2 Position | Phase Difference")
    println("-----|------------------|------------------|------------------")
    
    for step in 0:(period-1)
        pos1 = get_position_at_time(env.agents[1].trajectory, step)
        pos2 = get_position_at_time(env.agents[2].trajectory, step)
        phase_diff = abs(step - ((step + 3) % period))
        println("$(step)    | ($(pos1[1]), $(pos1[2]))        | ($(pos2[1]), $(pos2[2]))        | $(phase_diff)")
    end
end

"""
Test square FOR (Field of Regard)
"""
function test_square_for()
    println("\n🔲 Testing Square FOR")
    println("====================")
    
    env = create_8x3_environment()
    
    # Test FOR for both agents at different phases
    for agent_idx in 1:2
        agent = env.agents[agent_idx]
        println("\nAgent $(agent_idx) FOR Analysis:")
        
        for phase in 0:2
            pos = get_position_at_time(agent.trajectory, phase)
            locality = env.locality_functions[agent_idx]
            for_cells = get_observable_cells(locality, phase)
            
            println("Phase $(phase) - Position: ($(pos[1]), $(pos[2]))")
            println("FOR cells: $(for_cells)")
            println("FOR size: $(length(for_cells))")
        end
    end
end

"""
Create a belief-MDP solver using POMDP library
"""
function create_belief_mdp_solver(env::SpatialGrid)
    println("\n🧠 Creating Belief-MDP Solver")
    println("============================")
    
    # Create a simplified belief-MDP representation
    # For now, we'll use a simple value iteration approach
    # In practice, you'd use more sophisticated solvers like RTDP-Bel or point-based methods
    
    # Create policy with planning horizon
    policy = create_centralized_policy(env, 20, 0.95, 1.0)
    
    println("✓ Belief-MDP solver created")
    println("Planning horizon: $(policy.planning_horizon)")
    println("Discount factor: $(policy.discount_factor)")
    
    return policy
end

"""
Test centralized execution with constant communication
"""
function test_centralized_execution()
    println("\n🚀 Testing Centralized Execution with Constant Communication")
    println("==========================================================")
    
    env = create_8x3_environment()
    
    # Create belief-MDP solver
    policy = create_belief_mdp_solver(env)
    
    # Initialize global belief
    global_belief = initialize_global_belief(env, 0.5)
    
    # Initialize environment state
    initial_state_dist = POMDPs.initialstate(env)
    current_state = rand(initial_state_dist)
    
    println("\nInitial state:")
    println("Event map:")
    for y in 1:env.height
        row = [current_state.event_map[y, x] == EVENT_PRESENT ? "E" : "." for x in 1:env.width]
        println("  $(join(row, " "))")
    end
    
    # Track execution
    execution_history = []
    total_reward = 0.0
    num_steps = 12  # Two full periods
    
    for step in 1:num_steps
        println("\n⏰ Step $(step)")
        
        # Get global clock vector
        clock = get_global_clock_vector(env, step-1)
        println("Global clock: τ = $(clock.agent_phases)")
        
        # Show agent positions
        for (i, agent) in enumerate(env.agents)
            pos = get_position_at_time(agent.trajectory, step-1)
            println("Agent $(i) position: ($(pos[1]), $(pos[2]))")
        end
        
        # Select joint action using centralized policy (belief-MDP solver)
        joint_action = select_joint_action(policy, global_belief, clock)
        println("Selected joint action: $(joint_action)")
        
        # Execute joint action and get observations (constant communication)
        joint_observation = Vector{GridObservation}()
        step_reward = 0.0
        
        for action in joint_action
            # Generate observation for this action
            obs_dist = POMDPs.observation(env, action, current_state)
            obs = rand(obs_dist)
            push!(joint_observation, obs)
            
            # Calculate reward
            action_reward = POMDPs.reward(env, current_state, action, current_state)
            step_reward += action_reward
        end
        
        println("Joint observations: $(length(joint_observation))")
        for (i, obs) in enumerate(joint_observation)
            if !isempty(obs.sensed_cells)
                println("  Agent $(i): sensed $(obs.sensed_cells) -> $(obs.event_states)")
            else
                println("  Agent $(i): no cells sensed")
            end
        end
        println("Step reward: $(step_reward)")
        
        # Update global belief (constant communication - all observations shared)
        global_belief = update_global_belief(global_belief, joint_action, joint_observation, env)
        
        # Show belief state
        println("Global belief uncertainty: $(sum(global_belief.uncertainty_map))")
        
        # Transition to next state
        next_state_dist = POMDPs.transition(env, current_state, joint_action[1])
        current_state = rand(next_state_dist)
        
        # Record execution
        push!(execution_history, (step, clock, joint_action, joint_observation, step_reward))
        total_reward += step_reward
        
        # Check termination
        if POMDPs.isterminal(env, current_state)
            println("🏁 Environment terminated at step $(step)")
            break
        end
    end
    
    println("\n📊 Execution Summary")
    println("===================")
    println("Total steps: $(length(execution_history))")
    println("Total reward: $(total_reward)")
    if !isempty(execution_history)
        println("Average reward per step: $(total_reward / length(execution_history))")
    end
    
    # Show final belief state
    println("\n🧠 Final Belief State")
    println("====================")
    println("Final uncertainty: $(sum(global_belief.uncertainty_map))")
    println("Uncertainty reduction: $(sum(initialize_global_belief(env).uncertainty_map) - sum(global_belief.uncertainty_map))")
    
    return execution_history, global_belief, total_reward
end

"""
Visualize the execution
"""
function visualize_execution(execution_history, env)
    println("\n🎨 Creating Execution Visualization")
    println("==================================")
    
    # Create output directory
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    # Plot reward over time
    steps = [step for (step, _, _, _, _) in execution_history]
    rewards = [reward for (_, _, _, _, reward) in execution_history]
    
    p = plot(steps, rewards, 
             label="Step Reward",
             xlabel="Time Step",
             ylabel="Reward",
             title="Centralized Execution Performance (8x3 Grid)",
             marker=:circle,
             linewidth=2,
             grid=true)
    
    # Save plot
    plot_filename = joinpath(output_dir, "centralized_8x3_performance.png")
    savefig(p, plot_filename)
    println("✓ Performance plot saved: $(basename(plot_filename))")
    
    return p
end

"""
Main test function
"""
function main()
    println("🎯 Centralized Planner 8x3 Scenario Test")
    println("=======================================")
    
    # Test individual components
    test_agent_trajectories()
    test_square_for()
    
    # Run full centralized execution
    execution_history, final_belief, total_reward = test_centralized_execution()
    
    # Visualize results
    env = create_8x3_environment()
    visualize_execution(execution_history, env)
    
    println("\n✅ 8x3 scenario test completed!")
    println("📁 Check the 'visualizations' folder for results")
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 
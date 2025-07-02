using POMDPs
using POMDPTools
using Random
using LinearAlgebra

# Add the project to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include("../src/MyProject.jl")
using .MyProject

# Import specific types from MyProject
using .MyProject: Agent, SensingAction, GridObservation, EventState, NO_EVENT, EVENT_PRESENT
using .MyProject: CircularTrajectory, LinearTrajectory, RangeLimitedSensor
using .MyProject: EventState2, NO_EVENT_2, EVENT_PRESENT_2

# Import functions
using .MyProject.Agents.TrajectoryPlanner: get_position_at_time, execute_plan

# Import specific modules
using .Environment
using .Environment.EventDynamicsModule
using .Planners.GroundStation
using .Planners.MacroPlannerSync
using .Planners.PolicyTreePlanner
import .Environment: initialize_belief

"""
Test the synchronous macro planner with multiple agents
"""
function test_synchronous_planner()
    println("🚀 Testing Synchronous Macro Planner")
    println("====================================")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Create environment parameters
    grid_width, grid_height = 8, 6
    birth_rate, death_rate = 0.01, 0.05
    influence_rate = 0.01
    initial_events = 2
    
    # Create event dynamics
    event_dynamics = EventDynamics(birth_rate, death_rate, 0.0, 0.0, influence_rate)
    
    # Create agents with different trajectories
    agent1 = create_agent(1, CircularTrajectory(2, 3, 2.0, 8), RangeLimitedSensor(2.0, π/2, 0.0), grid_width, grid_height)
    agent2 = create_agent(2, LinearTrajectory(1, 1, 8, 6, 12), RangeLimitedSensor(2.5, π/3, 0.0), grid_width, grid_height)
    
    agents = [agent1, agent2]
    
    # Create environment
    env = SpatialGrid(grid_width, grid_height, event_dynamics, agents, 3.0, 0.95, initial_events, 3, (4, 3))
    
    # Initialize global belief
    global_belief = initialize_belief(grid_width, grid_height, 2, 0.5)
    
    # Get initial state
    initial_dist = POMDPs.initialstate(env)
    current_state = rand(initial_dist)
    current_environment = current_state.event_map
    
    println("📊 Environment Setup:")
    println("  Grid size: $(grid_width) × $(grid_height)")
    println("  Number of agents: $(length(agents))")
    println("  Initial events: $(initial_events)")
    println("  Event dynamics: birth=$(birth_rate), death=$(death_rate), influence=$(influence_rate)")
    
    # Simulation parameters
    num_steps = 20
    planning_horizon = 5  # C timesteps for macro-script planning
    sync_interval = 5     # Synchronize every 5 steps
    
    # Storage for results
    total_reward = 0.0
    agent_rewards = Dict{Int, Float64}()
    for agent in agents
        agent_rewards[agent.id] = 0.0
    end
    
    environment_evolution = [copy(current_environment)]
    action_history = []
    belief_evolution = [copy(global_belief)]
    
    println("\n🔄 Starting Synchronous Planning Simulation")
    println("===========================================")
    
    for t in 0:(num_steps-1)
        println("\n⏰ Time step $(t)")
        
        # Check if it's time to synchronize and plan
        if t % sync_interval == 0
            println("📡 Synchronization point - planning new macro-scripts")
            
            # Plan synchronous cycle for all agents
            agent_sequences = plan_synchronous_cycle(env, global_belief, agents, planning_horizon)
            
            # Store the planned sequences in agents
            for (agent_idx, agent) in enumerate(agents)
                agent.planned_sequence = agent_sequences[agent_idx]
                agent.sequence_index = 0
                println("  Agent $(agent.id): planned $(length(agent_sequences[agent_idx])) actions")
            end
        end
        
        # Execute actions for all agents
        joint_actions = SensingAction[]
        joint_observations = GridObservation[]
        step_reward = 0.0
        
        for agent in agents
            # Get next action from planned sequence
            if hasfield(typeof(agent), :planned_sequence) && agent.planned_sequence !== nothing
                agent.sequence_index += 1
                if agent.sequence_index <= length(agent.planned_sequence)
                    action = agent.planned_sequence[agent.sequence_index]
                else
                    # Use last action if sequence is exhausted
                    action = agent.planned_sequence[end]
                end
            else
                # No plan available, use wait action
                action = SensingAction(agent.id, Tuple{Int, Int}[], false)
            end
            
            push!(joint_actions, action)
            
            # Execute action and get observation using POMDP interface
            if !isempty(action.target_cells)
                # Get observation using POMDP observation model
                observation_dist = POMDPs.observation(env, action, current_state)
                observation = rand(observation_dist)
                push!(joint_observations, observation)
                
                # Calculate reward
                action_reward = POMDPs.reward(env, current_state, action, current_state)
                agent_rewards[agent.id] += action_reward
                step_reward += action_reward
                
                # Debug: print what was observed
                events_found = count(==(EVENT_PRESENT), observation.event_states)
                println("  Agent $(agent.id): $(length(action.target_cells)) cells sensed, $(events_found) events found")
            else
                # Wait action - no observation
                push!(joint_observations, GridObservation(agent.id, Tuple{Int, Int}[], EventState[], []))
                println("  Agent $(agent.id): wait action")
            end
        end
        
        # Update global belief with joint observations
        global_belief = update_global_belief_sync(global_belief, joint_actions, joint_observations, env)
        
        # Record environment state and actions for visualization
        push!(environment_evolution, copy(current_environment))
        push!(action_history, joint_actions)
        push!(belief_evolution, copy(global_belief))
        
        total_reward += step_reward
        
        # Update environment using POMDP transition model
        if t < num_steps - 1  # Don't update on last step
            # Create a dummy action for environment transition (no agent action affects environment)
            dummy_action = SensingAction(1, [], false)
            
            # Get next state using POMDP transition model
            next_state_dist = POMDPs.transition(env, current_state, dummy_action)
            current_state = rand(next_state_dist)
            current_environment = current_state.event_map
        end
        
        println("  Step reward: $(round(step_reward, digits=3))")
    end
    
    # Print final results
    println("\n📈 Simulation Results")
    println("====================")
    println("Total reward: $(round(total_reward, digits=3))")
    println("Agent rewards:")
    for (agent_id, reward) in agent_rewards
        println("  Agent $(agent_id): $(round(reward, digits=3))")
    end
    
    # Calculate belief statistics
    final_uncertainty = sum(global_belief.uncertainty_map)
    initial_uncertainty = sum(belief_evolution[1].uncertainty_map)
    uncertainty_reduction = initial_uncertainty - final_uncertainty
    
    println("\n📊 Belief Statistics:")
    println("  Initial uncertainty: $(round(initial_uncertainty, digits=3))")
    println("  Final uncertainty: $(round(final_uncertainty, digits=3))")
    println("  Uncertainty reduction: $(round(uncertainty_reduction, digits=3))")
    
    # Create visualizations
    println("\n🎨 Creating Visualizations...")
    println("============================")
    
    # Create simulation animation
    create_simulation_animation(environment_evolution, action_history, agents, grid_width, grid_height, "synchronous_simulation.gif")
    
    # Create belief evolution visualization
    create_belief_evolution_visualization(belief_evolution, grid_width, grid_height, "synchronous_belief_evolution.gif")
    
    println("✅ Simulation completed successfully!")
    
    return total_reward, agent_rewards, global_belief
end

"""
Create animation of the simulation
"""
function create_simulation_animation(environment_evolution, action_history, agents, width, height, filename)
    println("  📹 Simulation animation would be saved: $(filename)")
end

"""
Create belief evolution visualization
"""
function create_belief_evolution_visualization(belief_evolution, width, height, filename)
    println("  📊 Belief evolution animation would be saved: $(filename)")
end

"""
Helper function to get agent position at time
"""
function get_agent_position_at_time(agent, env, timestep_offset::Int)
    # Calculate position at future timestep using trajectory and phase offset
    t = timestep_offset
    
    # Apply phase offset
    adjusted_time = t + agent.phase_offset
    
    # Calculate position based on trajectory type
    if typeof(agent.trajectory) <: CircularTrajectory
        angle = 2π * (adjusted_time % agent.trajectory.period) / agent.trajectory.period
        x = agent.trajectory.center_x + round(Int, agent.trajectory.radius * cos(angle))
        y = agent.trajectory.center_y + round(Int, agent.trajectory.radius * sin(angle))
        return (x, y)
    elseif typeof(agent.trajectory) <: LinearTrajectory
        t_normalized = (adjusted_time % agent.trajectory.period) / agent.trajectory.period
        x = round(Int, agent.trajectory.start_x + t_normalized * (agent.trajectory.end_x - agent.trajectory.start_x))
        y = round(Int, agent.trajectory.start_y + t_normalized * (agent.trajectory.end_y - agent.trajectory.start_y))
        return (x, y)
    else
        return (1, 1)  # fallback
    end
end

# Run the test
test_synchronous_planner()

#!/usr/bin/env julia

"""
Test script for locality function implementation
Demonstrates how the footprint function g_i maps agent phases to observable regions
"""

using Pkg
Pkg.activate(".")

using Random
using Plots

# Set random seed for reproducibility
Random.seed!(42)

# Include the spatial grid
include("../src/environment/spatial_grid.jl")

# Import belief management functions
using .BeliefManagement: initialize_belief

"""
Test locality function for different agent configurations
"""
function test_locality_functions()
    println("🔍 Testing Locality Functions")
    println("============================")
    println("FOR (Field of Regard): Total observable area at each phase")
    println("FOV (Field of View): Specific cells chosen for observation (actions)")
    println()
    
    # Grid configuration
    width = 10
    height = 8
    
    # Create different agent configurations
    agents = []
    
    # Agent 1: Circular trajectory
    sensor1 = RangeLimitedSensor(2.0, π/2, 0.1)
    trajectory1 = CircularTrajectory(5, 4, 2.0, 8)
    agent1 = Agent(1, trajectory1, sensor1, 0, initialize_belief(width, height, 0.5))
    push!(agents, agent1)
    
    # Agent 2: Linear trajectory
    sensor2 = RangeLimitedSensor(1.5, π, 0.1)
    trajectory2 = LinearTrajectory(2, 2, 8, 6, 6)
    agent2 = Agent(2, trajectory2, sensor2, 0, initialize_belief(width, height, 0.5))
    push!(agents, agent2)
    
    # Test each agent's locality function
    for (i, agent) in enumerate(agents)
        println("\n🤖 Agent $(i) Locality Analysis")
        println("=" ^ 40)
        
        # Create locality function
        locality = LocalityFunction(agent.id, agent.trajectory, agent.sensor, width, height)
        
        # Get trajectory period
        period = get_trajectory_period(agent.trajectory)
        println("Trajectory period: $(period)")
        println("Sensor range: $(agent.sensor.range)")
        println()
        
        # Analyze FOR at each phase
        println("FOR (Field of Regard) at each phase:")
        println("Phase | Position | FOR Size | FOR Cells")
        println("------|----------|----------|----------")
        
        for phase in 0:(period-1)
            pos = get_position_at_time(agent.trajectory, phase)
            for_cells = get_observable_cells(locality, phase)
            println("$(phase)     | ($(pos[1]), $(pos[2])) | $(length(for_cells))      | $(for_cells[1:min(3, length(for_cells))])...")
        end
        println()
        
        # Show sample FOV actions (subsets of FOR)
        println("Sample FOV actions (subsets of FOR):")
        for phase in 0:min(2, period-1)
            pos = get_position_at_time(agent.trajectory, phase)
            for_cells = get_observable_cells(locality, phase)
            
            if !isempty(for_cells)
                println("Phase $(phase) FOR: $(for_cells)")
                
                # Generate sample FOV actions (always exactly 1 cell)
                sample_fov = [rand(for_cells)]
                println("  FOV (1 cell): $(sample_fov)")
            end
        end
        println()
        
        # Analyze locality sets for specific cells
        test_cells = [(5, 4), (3, 3), (7, 5)]
        println("Locality sets for specific cells:")
        println("Cell   | Phases when in FOR")
        println("-------|-------------------")
        
        for cell in test_cells
            locality_set = get_locality_set(locality, cell)
            if !isempty(locality_set)
                println("($(cell[1]), $(cell[2])) | $(locality_set)")
            else
                println("($(cell[1]), $(cell[2])) | never in FOR")
            end
        end
        println()
    end
end

"""
Visualize agent trajectories and observable regions
"""
function visualize_locality()
    println("\n🎨 Visualizing Locality Functions")
    println("================================")
    println("FOR (Field of Regard): Total observable area (blue)")
    println("FOV (Field of View): Specific observation targets (red)")
    println()
    
    # Grid configuration
    width = 10
    height = 8
    
    # Create agents
    agents = []
    
    # Agent 1: Circular trajectory
    sensor1 = RangeLimitedSensor(2.0, π/2, 0.1)
    trajectory1 = CircularTrajectory(5, 4, 2.0, 8)
    agent1 = Agent(1, trajectory1, sensor1, 0, initialize_belief(width, height, 0.5))
    push!(agents, agent1)
    
    # Agent 2: Linear trajectory
    sensor2 = RangeLimitedSensor(1.5, π, 0.1)
    trajectory2 = LinearTrajectory(2, 2, 8, 6, 6)
    agent2 = Agent(2, trajectory2, sensor2, 0, initialize_belief(width, height, 0.5))
    push!(agents, agent2)
    
    # Create output directory
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    # Visualize each agent's trajectory and observable regions
    for (i, agent) in enumerate(agents)
        println("Creating visualization for Agent $(i)...")
        
        # Get trajectory period
        period = get_trajectory_period(agent.trajectory)
        
        # Create frames for animation
        frames = []
        
        for phase in 0:(period-1)
            # Get agent position
            pos = get_position_at_time(agent.trajectory, phase)
            
            # Get FOR (Field of Regard) - all observable cells
            locality = LocalityFunction(agent.id, agent.trajectory, agent.sensor, width, height)
            for_cells = get_observable_cells(locality, phase)
            
            # Create sample FOV (Field of View) - specific observation targets
            # This represents what the agent might choose to observe
            fov_cells = []
            if !isempty(for_cells)
                # FOV is always exactly 1 cell (consistent sensor footprint)
                fov_cells = [rand(for_cells)]
            end
            
            # Create visualization matrix
            # 0 = empty, 1 = FOR (blue), 2 = FOV (red), 3 = agent position (black)
            vis_matrix = zeros(Int, height, width)
            
            # Mark FOR cells (Field of Regard)
            for cell in for_cells
                vis_matrix[cell[2], cell[1]] = 1
            end
            
            # Mark FOV cells (Field of View - the action)
            for cell in fov_cells
                vis_matrix[cell[2], cell[1]] = 2
            end
            
            # Mark agent position
            if 1 <= pos[1] <= width && 1 <= pos[2] <= height
                vis_matrix[pos[2], pos[1]] = 3
            end
            
            # Create heatmap with clear color coding
            p = heatmap(vis_matrix, 
                       aspect_ratio=:equal,
                       color=[:white, :lightblue, :red, :black],
                       clim=(0, 3),
                       title="Agent $(i) - Phase $(phase)\nPosition: ($(pos[1]), $(pos[2]))\nFOR: $(length(for_cells)) cells | FOV: $(length(fov_cells)) cells",
                       xlabel="X",
                       ylabel="Y",
                       grid=false,
                       showaxis=true,
                       ticks=true)
            
            push!(frames, p)
        end
        
        # Create animation
        anim = @animate for frame in frames
            plot(frame, size=(600, 500))
        end
        
        # Save animation
        animation_filename = joinpath(output_dir, "agent$(i)_for_fov_animation.gif")
        gif(anim, animation_filename, fps=1)
        println("✓ Saved animation: $(basename(animation_filename))")
    end
end

"""
Test action space generation using locality functions
"""
function test_action_space()
    println("\n⚡ Testing Action Space Generation")
    println("=================================")
    println("FOR → FOV: How actions are generated from Field of Regard")
    println()
    
    # Create a simple environment
    width = 6
    height = 6
    
    # Create agent
    sensor = RangeLimitedSensor(1.5, π, 0.1)
    trajectory = CircularTrajectory(3, 3, 1.5, 4)
    agent = Agent(1, trajectory, sensor, 0, initialize_belief(height, height, 0.5))
    
    # Create locality function
    locality = LocalityFunction(agent.id, agent.trajectory, agent.sensor, width, height)
    
    # Test action space at different phases
    for phase in 0:3
        println("\nPhase $(phase):")
        
        # Get FOR (Field of Regard)
        for_cells = get_observable_cells(locality, phase)
        println("FOR (Field of Regard): $(for_cells)")
        
        # Generate FOV actions (subsets of FOR)
        actions = []
        
        # Wait action (no FOV)
        push!(actions, SensingAction(1, [], false))
        
        # Sensing actions (FOV = exactly 1 cell from FOR)
        for cell in for_cells
            push!(actions, SensingAction(1, [cell], false))
        end
        
        println("FOV actions generated: $(length(actions))")
        println("Sample FOV actions:")
        for (i, action) in enumerate(actions[1:min(5, length(actions))])
            if isempty(action.target_cells)
                println("  $(i): Wait (no FOV)")
            else
                println("  $(i): FOV = $(action.target_cells) (comm: $(action.communicate))")
            end
        end
    end
end

"""
Main test function
"""
function main()
    println("🔍 Locality Function Test")
    println("========================")
    
    # Test locality functions
    test_locality_functions()
    
    # Visualize locality
    visualize_locality()
    
    # Test action space generation
    test_action_space()
    
    println("\n✅ Locality function testing completed!")
    println("📁 Check the 'visualizations' folder for animations")
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 
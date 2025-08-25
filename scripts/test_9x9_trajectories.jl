#!/usr/bin/env julia

"""
Test script to visualize the 9x9 circular trajectories and verify they match the image
"""

println("🔍 Testing 9x9 Circular Trajectories...")

# Import the main script functions
include("../src/MyProject.jl")
using .MyProject
using .MyProject.Agents.TrajectoryPlanner: get_position_at_time, create_circular_trajectory

# Create the exact same trajectories as in main_9x9_circular.jl
function test_trajectories()
    println("\n🎯 Creating 4 Circular Trajectories (Period 12 each)")
    println(repeat("=", 50))
    
    # Agent 1: Upper-left circular trajectory
    trajectory1 = create_circular_trajectory(3, 3, 2.8, 12)
    
    # Agent 2: Upper-right circular trajectory  
    trajectory2 = create_circular_trajectory(7, 3, 2.8, 12)
    
    # Agent 3: Lower-left circular trajectory
    trajectory3 = create_circular_trajectory(3, 7, 2.8, 12)
    
    # Agent 4: Lower-right circular trajectory
    trajectory4 = create_circular_trajectory(7, 7, 2.8, 12)
    
    trajectories = [trajectory1, trajectory2, trajectory3, trajectory4]
    phase_offsets = [0, 3, 6, 9]  # Same as in main script
    
    # Print trajectory information
    for (i, (traj, offset)) in enumerate(zip(trajectories, phase_offsets))
        println("\n🤖 Agent $(i):")
        println("  Center: ($(traj.center_x), $(traj.center_y))")
        println("  Radius: $(traj.radius)")
        println("  Period: $(traj.period)")
        println("  Phase offset: $(offset)")
        println("  Trajectory positions (one complete cycle):")
        
        for t in 0:11
            pos = get_position_at_time(traj, t, offset)
            println("    t=$(t): $(pos)")
        end
    end
    
    # Create a simple grid visualization
    println("\n📊 Grid Visualization (9x9)")
    println(repeat("=", 50))
    
    # Show positions at t=0
    grid = fill('.', 9, 9)
    
    # Mark ground station
    grid[5, 5] = 'G'
    
    # Mark agent positions at t=0
    agent_symbols = ['1', '2', '3', '4']
    for (i, (traj, offset)) in enumerate(zip(trajectories, phase_offsets))
        pos = get_position_at_time(traj, 0, offset)
        if 1 <= pos[1] <= 9 && 1 <= pos[2] <= 9
            grid[pos[2], pos[1]] = agent_symbols[i]  # Note: y,x for matrix indexing
        end
    end
    
    # Print grid (top to bottom)
    println("Grid at t=0 (G=Ground Station, 1-4=Agents):")
    for y in 1:9
        print("  ")
        for x in 1:9
            print(grid[y, x], " ")
        end
        println()
    end
    
    # Test: Check if each trajectory passes through or near the center (5,5)
    println("\n🔍 Testing Central Overlap with Ground Station (5,5)")
    println(repeat("=", 50))
    
    for (i, (traj, offset)) in enumerate(zip(trajectories, phase_offsets))
        closest_distance = Inf
        closest_time = -1
        closest_pos = (0, 0)
        
        # Check all positions in one complete cycle
        for t in 0:11
            pos = get_position_at_time(traj, t, offset)
            # Calculate distance to ground station (5,5)
            distance = sqrt((pos[1] - 5)^2 + (pos[2] - 5)^2)
            if distance < closest_distance
                closest_distance = distance
                closest_time = t
                closest_pos = pos
            end
        end
        
        println("Agent $(i): closest to GS at t=$(closest_time), pos=$(closest_pos), distance=$(round(closest_distance, digits=2))")
        
        if closest_distance <= 1.5  # Within reasonable range
            println("  ✅ GOOD: Passes close to ground station")
        else
            println("  ❌ BAD: Too far from ground station!")
        end
    end
    
    println("\n✅ Trajectory test completed!")
    println("🎯 Verify that the agent positions match the image:")
    println("   - All agents should pass close to Ground Station at (5,5)")
    println("   - Agent 1 (upper-left circle): center (3,3), radius 2.8")
    println("   - Agent 2 (upper-right circle): center (7,3), radius 2.8")  
    println("   - Agent 3 (lower-left circle): center (3,7), radius 2.8")
    println("   - Agent 4 (lower-right circle): center (7,7), radius 2.8")
end

# Run the test
test_trajectories()

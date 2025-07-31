#!/usr/bin/env julia

"""
Test script for heterogeneous RSP environment with different cell types
Demonstrates immune cells, fleeting events, and long-lasting events
"""

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Plots
Plots.plotlyjs()

# Import the project
include("../src/MyProject.jl")
using .MyProject
using .Types: EventState, NO_EVENT, EVENT_PRESENT
using .Environment.EventDynamicsModule: transition_rsp!

"""
Run heterogeneous RSP simulation with different cell types
"""
function run_heterogeneous_rsp_simulation(;
    grid_width::Int = 9,
    grid_height::Int = 4,
    num_steps::Int = 200,
    initial_events::Int = 1
)
    println("🔥 Testing Heterogeneous RSP Environment")
    println("=========================================")
    println("Grid: $(grid_width)x$(grid_height)")
    println("Steps: $(num_steps)")
    println("Initial events: $(initial_events)")

    # Create heterogeneous parameter maps
    println("\n🌍 Creating heterogeneous environment...")
    param_maps = Types.create_heterogeneous_rsp_maps(grid_height, grid_width)

    # Visualize parameter maps
    println("\n📊 Parameter Maps:")
    println("Randomly distributed cell types:")
    for cell_type in Types.HETEROGENEOUS_CELL_TYPES
        println("  • $(cell_type.name): λ=$(cell_type.lambda), β₀=$(cell_type.beta0), α=$(cell_type.alpha), δ=$(cell_type.delta)")
    end

    # Create and save parameter map visualization
    param_viz = visualize_parameter_maps(param_maps)
    savefig(param_viz, "heterogeneous_parameters.png")
    println("✅ Parameter maps saved to: heterogeneous_parameters.png")

    # Initialize environment state
    println("\n🔥 Initializing environment...")
    current_state = Matrix{EventState}(undef, grid_height, grid_width)
    current_state .= NO_EVENT

    # Add initial events in the middle section (fleeting events area)
    for _ in 1:initial_events
        x = rand(grid_width÷3 + 1:2*grid_width÷3)  # Middle third
        y = rand(1:grid_height)
        current_state[y, x] = EVENT_PRESENT
    end

    println("Initial state:")
    for y in 1:grid_height
        for x in 1:grid_width
            print(current_state[y, x] == EVENT_PRESENT ? "🔥 " : "⬜ ")
        end
        println()
    end

    # Track evolution
    evolution = [copy(current_state)]

    # Simulate evolution
    println("\n🔄 Simulating evolution...")
    event_map = copy(current_state)
    for step in 1:num_steps
        next_map = similar(event_map)
        
        # Use new heterogeneous RSP transition
        transition_rsp!(next_map, event_map, param_maps, Random.GLOBAL_RNG)
        
        event_map = next_map
        push!(evolution, copy(event_map))
        
        events_count = count(==(EVENT_PRESENT), event_map)
        println("Step $(step): $(events_count) events active")
        
        # Show state every 5 steps
        if step % 5 == 0
            println("  State at step $(step):")
            for y in 1:grid_height
                for x in 1:grid_width
                    print(event_map[y, x] == EVENT_PRESENT ? "🔥 " : "⬜ ")
                end
                println()
            end
        end
    end

    # Create animation
    println("\n🎬 Creating animation...")
    create_animation(evolution, param_maps, grid_width, grid_height)
    println("✅ Animation saved to: heterogeneous_rsp_simulation.gif")

    # Analyze results
    println("\n📈 Analysis:")
    println("=============")
    analyze_results(evolution, param_maps, grid_width, grid_height)

    println("\n✅ Heterogeneous RSP test completed!")
    
    return evolution, param_maps
end

"""
Create visualization of parameter maps
"""
function visualize_parameter_maps(param_maps::Types.RSPParameterMaps)
    fig = plot(layout=(2,2), size=(1200, 800))
    
    # Lambda map (ignition intensity)
    heatmap!(fig[1], param_maps.lambda_map, 
        title="λ (Ignition Intensity)", 
        colorbar_title="λ", 
        subplot=1)
    
    # Alpha map (contagion strength)
    heatmap!(fig[2], param_maps.alpha_map, 
        title="α (Contagion Strength)", 
        colorbar_title="α", 
        subplot=2)
    
    # Delta map (persistence probability)
    heatmap!(fig[3], param_maps.delta_map, 
        title="δ (Persistence Probability)", 
        colorbar_title="δ", 
        subplot=3)
    
    # Mu map (death probability)
    heatmap!(fig[4], param_maps.mu_map, 
        title="μ (Death Probability)", 
        colorbar_title="μ", 
        subplot=4)
    
    return fig
end

"""
Create animation from evolution data
"""
function create_animation(evolution::Vector{Matrix{EventState}}, param_maps::Types.RSPParameterMaps, grid_width::Int, grid_height::Int)
    frames = []
    for (step, state) in enumerate(evolution)
        # Create visualization for this frame
        height, width = size(state)
        
        # Create base plot
        p = plot(; xlim=(0.5, width+0.5), ylim=(0.5, height+0.5),
            aspect_ratio=:equal, size=(800, 400), legend=false,
            xlabel="X Coordinate", ylabel="Y Coordinate",
            grid=false,
            title="Heterogeneous RSP - Step $(step-1) - Events: $(count(==(EVENT_PRESENT), state))",
            titlefontsize=12,
            background_color=:white
        )
        
        # Draw grid cells with parameter-based coloring
        for x in 1:width, y in 1:height
            xs = [x-0.5, x+0.5, x+0.5, x-0.5]
            ys = [y-0.5, y-0.5, y+0.5, y+0.5]
            
            # Color based on cell type (using alpha as indicator)
            alpha_val = param_maps.alpha_map[y, x]
            if alpha_val < 0.1
                cell_color = :lightblue  # Immune cells
            elseif alpha_val < 0.3
                cell_color = :lightyellow  # Fleeting events
            else
                cell_color = :lightgreen  # Long-lasting events
            end
            
            plot!(p, xs, ys, seriestype=:shape, fillcolor=cell_color, linecolor=:black, linewidth=1, alpha=0.7)
            
            # Add event if present
            if state[y, x] == EVENT_PRESENT
                annotate!(p, x, y, text("🔥", :center, 18))
            end
        end
        
        push!(frames, p)
    end

    # Create animation
    anim = @animate for frame in frames
        plot(frame)
    end

    gif(anim, "heterogeneous_rsp_simulation.gif", fps=1)
end

"""
Analyze simulation results
"""
function analyze_results(evolution::Vector{Matrix{EventState}}, param_maps::Types.RSPParameterMaps, grid_width::Int, grid_height::Int)
    # Count total events
    total_events = count(==(EVENT_PRESENT), evolution[end])
    total_cells = grid_width * grid_height
    
    println("Final event distribution:")
    println("  Total events: $(total_events) out of $(total_cells) cells ($(round(100*total_events/total_cells, digits=1))%)")

    # Analyze cell type distribution using actual definitions
    cell_counts, total_cells = Types.analyze_cell_type_distribution(param_maps)
    
    println("\nCell type distribution:")
    for (cell_name, count) in cell_counts
        percentage = round(100 * count / total_cells, digits=1)
        println("  $(cell_name): $(count) ($(percentage)%)")
    end

    # Calculate average parameters
    avg_lambda = mean(param_maps.lambda_map)
    avg_alpha = mean(param_maps.alpha_map)
    avg_delta = mean(param_maps.delta_map)
    avg_mu = mean(param_maps.mu_map)
    
    println("\nAverage parameters:")
    println("  λ (ignition): $(round(avg_lambda, digits=4))")
    println("  α (contagion): $(round(avg_alpha, digits=3))")
    println("  δ (persistence): $(round(avg_delta, digits=3))")
    println("  μ (death): $(round(avg_mu, digits=3))")
    
    # Expected lifetime based on average death rate
    avg_lifetime = 1.0 / avg_mu
    println("  Expected average lifetime: $(round(avg_lifetime, digits=1)) steps")
end

# Run the simulation
if abspath(PROGRAM_FILE) == @__FILE__
    run_heterogeneous_rsp_simulation()
end 
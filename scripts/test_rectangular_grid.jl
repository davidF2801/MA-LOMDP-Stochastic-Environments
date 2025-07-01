#!/usr/bin/env julia

"""
Test script for rectangular grid functionality
"""

using Pkg
Pkg.activate(".")

using Random
using Plots

# Set random seed for reproducibility
Random.seed!(42)

# Include the event dynamics
include("../src/environment/event_dynamics.jl")
include("../src/environment/dbn_proper.jl")

"""
Test different rectangular grid configurations
"""
function test_rectangular_grids()
    println("🔲 Testing Rectangular Grid Configurations")
    println("=========================================")
    
    # Create output directory
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
        println("✓ Created output directory: $(output_dir)")
    end
    
    # Test different grid configurations
    grid_configs = [
        ("Square", 10, 10),
        ("Wide", 15, 8),
        ("Tall", 8, 15),
        ("Large_Wide", 20, 12),
        ("Large_Tall", 12, 20)
    ]
    
    # DBN model parameters
    birth_rate = 0.005
    death_rate = 0.05
    neighbor_influence = 0.01
    
    # Simulation parameters
    num_steps = 30
    initial_events = 3
    
    # Create DBN model
    dbn_model = DBNTransitionModel2(birth_rate, death_rate, neighbor_influence)
    
    println("DBN Model: $(dbn_model)")
    println("Simulation steps: $(num_steps)")
    println("Initial events: $(initial_events)")
    println()
    
    # Test each configuration
    for (name, width, height) in grid_configs
        println("🔲 Testing $(name) grid: $(width)x$(height)")
        println("Grid area: $(width * height) cells")
        println("Aspect ratio: $(round(width/height, digits=2))")
        
        # Simulate environment evolution
        evolution, event_counts = simulate_proper_dbn(dbn_model, width, height, num_steps, initial_events, Random.MersenneTwister(42))
        
        # Calculate statistics
        avg_events = mean(event_counts)
        max_events = maximum(event_counts)
        min_events = minimum(event_counts)
        final_events = event_counts[end]
        
        println("  Final events: $(final_events)")
        println("  Average events: $(round(avg_events, digits=2))")
        println("  Max events: $(max_events)")
        println("  Min events: $(min_events)")
        println("  Event density: $(round(final_events/(width*height), digits=4))")
        println()
        
        # Create visualization
        create_grid_visualization(evolution, event_counts, name, width, height, dbn_model, output_dir)
    end
    
    println("✅ All rectangular grid tests completed!")
    println("📁 Check the '$(output_dir)' folder for visualizations")
end

"""
Create visualization for a grid configuration
"""
function create_grid_visualization(evolution, event_counts, name, width, height, dbn_model, output_dir)
    # Create event count plot
    p = plot(event_counts, 
             label="Event Count",
             xlabel="Time Step",
             ylabel="Number of Events",
             title="$(name) Grid ($(width)x$(height)) Event Evolution\nDBN: birth=$(dbn_model.birth_rate), death=$(dbn_model.death_rate), influence=$(dbn_model.neighbor_influence)",
             linewidth=2,
             marker=:circle,
             markersize=3,
             grid=true)
    
    # Save plot
    plot_filename = joinpath(output_dir, "rectangular_grid_$(name)_$(width)x$(height)_event_evolution.png")
    savefig(p, plot_filename)
    println("  ✓ Saved event evolution plot: $(basename(plot_filename))")
    
    # Create final state heatmap
    final_state = evolution[end]
    final_plot = heatmap(Int.(final_state), 
                         aspect_ratio=:equal,
                         color=:Reds,
                         clim=(0, 1),
                         title="$(name) Grid Final State\n$(event_counts[end]) events",
                         xlabel="X",
                         ylabel="Y",
                         grid=false,
                         showaxis=true,
                         ticks=true)
    
    # Save final state plot
    final_filename = joinpath(output_dir, "rectangular_grid_$(name)_$(width)x$(height)_final_state.png")
    savefig(final_plot, final_filename)
    println("  ✓ Saved final state plot: $(basename(final_filename))")
end

"""
Compare square vs rectangular grids
"""
function compare_grid_shapes()
    println("\n📊 Comparing Grid Shapes")
    println("=======================")
    
    # Test configurations
    square_grid = (10, 10)      # 100 cells
    wide_grid = (15, 8)         # 120 cells (similar area)
    tall_grid = (8, 15)         # 120 cells (similar area)
    
    # DBN model
    dbn_model = DBNTransitionModel2(0.005, 0.05, 0.01)
    
    # Simulation parameters
    num_steps = 20
    initial_events = 2
    
    println("Comparing grid shapes with similar areas:")
    println("Square: $(square_grid[1])x$(square_grid[2]) = $(square_grid[1] * square_grid[2]) cells")
    println("Wide: $(wide_grid[1])x$(wide_grid[2]) = $(wide_grid[1] * wide_grid[2]) cells")
    println("Tall: $(tall_grid[1])x$(tall_grid[2]) = $(tall_grid[1] * tall_grid[2]) cells")
    println()
    
    # Simulate each configuration
    results = Dict()
    
    for (name, width, height) in [("Square", square_grid...), ("Wide", wide_grid...), ("Tall", tall_grid...)]
        println("Simulating $(name) grid...")
        evolution, event_counts = simulate_proper_dbn(dbn_model, width, height, num_steps, initial_events, Random.MersenneTwister(42))
        
        results[name] = Dict(
            "width" => width,
            "height" => height,
            "area" => width * height,
            "event_counts" => event_counts,
            "final_events" => event_counts[end],
            "avg_events" => mean(event_counts),
            "max_events" => maximum(event_counts),
            "min_events" => minimum(event_counts)
        )
    end
    
    # Display comparison
    println("\n📈 Grid Shape Comparison Results:")
    println("=================================")
    println("Grid Type | Width | Height | Area | Final | Avg   | Max | Min")
    println("----------|-------|--------|------|-------|-------|-----|-----")
    
    for name in ["Square", "Wide", "Tall"]
        r = results[name]
        println("$(name)     | $(r["width"])     | $(r["height"])      | $(r["area"])  | $(r["final_events"])     | $(round(r["avg_events"], digits=1))   | $(r["max_events"]) | $(r["min_events"])")
    end
    
    # Create comparison plot
    p = plot(title="Grid Shape Comparison\nEvent Evolution Over Time")
    
    for name in ["Square", "Wide", "Tall"]
        r = results[name]
        plot!(r["event_counts"], 
              label="$(name) ($(r["width"])x$(r["height"]))",
              linewidth=2,
              marker=:circle,
              markersize=3)
    end
    
    plot!(xlabel="Time Step", ylabel="Number of Events", grid=true)
    
    # Save comparison plot
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    comparison_filename = joinpath(output_dir, "grid_shape_comparison.png")
    savefig(p, comparison_filename)
    println("\n✓ Saved grid shape comparison: $(basename(comparison_filename))")
    
    return results
end

"""
Main test function
"""
function main()
    println("🔲 Rectangular Grid Test")
    println("========================")
    
    # Test different rectangular configurations
    test_rectangular_grids()
    
    # Compare grid shapes
    compare_grid_shapes()
    
    println("\n✅ Rectangular grid testing completed!")
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 
#!/usr/bin/env julia

"""
Test script comparing current DBN implementation with proper DBN
"""

using Pkg
Pkg.activate(".")

using Random
using Plots

# Set random seed for reproducibility
Random.seed!(42)

# Include both implementations
include("../src/environment/event_dynamics.jl")
include("../src/environment/dbn_proper.jl")

"""
Compare current vs proper DBN implementations
"""
function compare_dbn_implementations()
    println("🔬 DBN Implementation Comparison")
    println("================================")
    
    # Create models with same parameters
    birth_rate = 0.01
    death_rate = 0.05
    neighbor_influence = 0.1
    
    current_model = DBNTransitionModel2(birth_rate, death_rate, neighbor_influence)
    proper_model = ProperDBNModel(birth_rate, death_rate, neighbor_influence)
    
    println("Parameters: birth=$(birth_rate), death=$(death_rate), influence=$(neighbor_influence)")
    println()
    
    # Analyze proper DBN properties
    analyze_dbn_properties(proper_model)
    
    # Compare transition probabilities
    println("🔄 Transition Probability Comparison")
    println("====================================")
    
    test_scenarios = [
        ("No event, 0 neighbors", NO_EVENT_2, EventState2[]),
        ("No event, 1 neighbor", NO_EVENT_2, [EVENT_PRESENT_2]),
        ("No event, 2 neighbors", NO_EVENT_2, [EVENT_PRESENT_2, EVENT_PRESENT_2]),
        ("Event, 0 neighbors", EVENT_PRESENT_2, EventState2[]),
        ("Event, 1 neighbor", EVENT_PRESENT_2, [EVENT_PRESENT_2])
    ]
    
    for (name, current, neighbors) in test_scenarios
        p_current = transition_probability_dbn(current, neighbors, current_model)
        p_proper = transition_probability_proper_dbn(current, neighbors, proper_model)
        
        println("$(name):")
        println("  Current: $(round(p_current, digits=4))")
        println("  Proper:  $(round(p_proper, digits=4))")
        println("  Diff:    $(round(p_proper - p_current, digits=4))")
        println()
    end
    
    # Compare belief propagation
    println("🧠 Belief Propagation Comparison")
    println("================================")
    
    belief_scenarios = [
        ("Low belief, no neighbors", 0.1, [0.0, 0.0, 0.0]),
        ("High belief, no neighbors", 0.9, [0.0, 0.0, 0.0]),
        ("Low belief, high neighbors", 0.1, [0.8, 0.9, 0.7]),
        ("High belief, high neighbors", 0.9, [0.8, 0.9, 0.7])
    ]
    
    for (name, current_belief, neighbor_beliefs) in belief_scenarios
        # Current implementation (simplified)
        E_neighbors = sum(neighbor_beliefs)
        p1_current = 1.0 - death_rate
        p0_current = birth_rate + neighbor_influence * E_neighbors
        new_belief_current = current_belief * p1_current + (1.0 - current_belief) * p0_current
        
        # Proper DBN implementation
        new_belief_proper = belief_propagation_dbn(current_belief, neighbor_beliefs, proper_model)
        
        println("$(name):")
        println("  Current: $(round(current_belief, digits=3)) → $(round(new_belief_current, digits=3))")
        println("  Proper:  $(round(current_belief, digits=3)) → $(round(new_belief_proper, digits=3))")
        println("  Diff:    $(round(new_belief_proper - new_belief_current, digits=3))")
        println()
    end
end

"""
Simulate and compare both implementations
"""
function simulate_comparison()
    println("🎮 Simulation Comparison")
    println("=======================")
    
    # Parameters
    width = 8
    height = 6
    num_steps = 20
    initial_events = 2
    
    birth_rate = 0.005
    death_rate = 0.05
    neighbor_influence = 0.01
    
    # Create models
    current_model = DBNTransitionModel2(birth_rate, death_rate, neighbor_influence)
    proper_model = ProperDBNModel(birth_rate, death_rate, neighbor_influence)
    
    # Simulate with current implementation
    println("Simulating with current DBN...")
    event_map_current = fill(NO_EVENT_2, height, width)
    rng = Random.MersenneTwister(42)
    
    # Add initial events
    for _ in 1:initial_events
        x = rand(rng, 1:width)
        y = rand(rng, 1:height)
        event_map_current[y, x] = EVENT_PRESENT_2
    end
    
    evolution_current = [copy(event_map_current)]
    counts_current = [count(==(EVENT_PRESENT_2), event_map_current)]
    
    for step in 1:num_steps
        update_events!(current_model, event_map_current, rng)
        push!(evolution_current, copy(event_map_current))
        push!(counts_current, count(==(EVENT_PRESENT_2), event_map_current))
    end
    
    # Simulate with proper DBN
    println("Simulating with proper DBN...")
    rng = Random.MersenneTwister(42)  # Same seed for fair comparison
    evolution_proper, counts_proper = simulate_proper_dbn(proper_model, width, height, num_steps, initial_events, rng)
    
    # Compare results
    println("\n📊 Simulation Results Comparison")
    println("================================")
    println("Grid size: $(width)x$(height)")
    println("Steps: $(num_steps)")
    println("Initial events: $(initial_events)")
    println()
    
    println("Event Counts Over Time:")
    println("Step | Current | Proper | Diff")
    println("-----|---------|--------|------")
    for step in 1:min(length(counts_current), length(counts_proper))
        diff = counts_proper[step] - counts_current[step]
        println("$(step-1)    | $(counts_current[step])      | $(counts_proper[step])     | $(diff)")
    end
    
    # Calculate statistics
    avg_current = mean(counts_current)
    avg_proper = mean(counts_proper)
    max_current = maximum(counts_current)
    max_proper = maximum(counts_proper)
    
    println("\nStatistics:")
    println("Current DBN - Avg: $(round(avg_current, digits=2)), Max: $(max_current)")
    println("Proper DBN  - Avg: $(round(avg_proper, digits=2)), Max: $(max_proper)")
    println("Difference  - Avg: $(round(avg_proper - avg_current, digits=2)), Max: $(max_proper - max_current)")
    
    return evolution_current, counts_current, evolution_proper, counts_proper
end

"""
Visualize the differences
"""
function visualize_comparison(evolution_current, counts_current, evolution_proper, counts_proper)
    println("\n🎨 Creating Comparison Visualizations")
    println("====================================")
    
    # Create output directory
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    # Plot event counts comparison
    p = plot(counts_current, 
             label="Current DBN",
             linewidth=2,
             marker=:circle,
             markersize=3,
             title="Event Count Comparison: Current vs Proper DBN")
    
    plot!(counts_proper,
          label="Proper DBN",
          linewidth=2,
          marker=:square,
          markersize=3,
          xlabel="Time Step",
          ylabel="Number of Events",
          grid=true)
    
    # Save plot
    plot_filename = joinpath(output_dir, "dbn_comparison_event_counts.png")
    savefig(p, plot_filename)
    println("✓ Event count comparison saved as '$(basename(plot_filename))'")
    
    # Create side-by-side animation of final states
    if length(evolution_current) > 0 && length(evolution_proper) > 0
        final_current = evolution_current[end]
        final_proper = evolution_proper[end]
        
        # Create heatmaps
        p1 = heatmap(Int.(final_current), 
                     aspect_ratio=:equal,
                     color=:Reds,
                     clim=(0, 1),
                     title="Current DBN Final State\n$(counts_current[end]) events")
        
        p2 = heatmap(Int.(final_proper), 
                     aspect_ratio=:equal,
                     color=:Reds,
                     clim=(0, 1),
                     title="Proper DBN Final State\n$(counts_proper[end]) events")
        
        # Combine plots
        combined = plot(p1, p2, layout=(1, 2), size=(800, 400))
        
        # Save combined plot
        combined_filename = joinpath(output_dir, "dbn_comparison_final_states.png")
        savefig(combined, combined_filename)
        println("✓ Final states comparison saved as '$(basename(combined_filename))'")
    end
    
    return p
end

"""
Main test function
"""
function main()
    println("🧪 DBN Implementation Test")
    println("==========================")
    
    # Compare implementations
    compare_dbn_implementations()
    
    # Simulate and compare
    evolution_current, counts_current, evolution_proper, counts_proper = simulate_comparison()
    
    # Visualize results
    visualize_comparison(evolution_current, counts_current, evolution_proper, counts_proper)
    
    println("\n✅ DBN comparison completed!")
    println("\n📁 Check the 'visualizations' folder for comparison plots")
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 
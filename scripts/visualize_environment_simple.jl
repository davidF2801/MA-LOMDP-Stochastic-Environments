#!/usr/bin/env julia

"""
Simple text-based visualization script for the 2-state MA-LOMDP environment
Shows temporal evolution of events using ASCII art
"""

using Pkg
Pkg.activate(".")

using POMDPs
using POMDPTools
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Include the event dynamics
include("../src/environment/event_dynamics.jl")
include("../src/environment/dbn_proper.jl")

"""
Visualize a single grid state using ASCII art
"""
function visualize_grid_text(event_map::Matrix{EventState2}, title::String="Event Grid")
    height, width = size(event_map)
    
    println("\n" * "="^50)
    println(title)
    println("="^50)
    
    # Print grid with ASCII representation
    for y in 1:height
        print("| ")
        for x in 1:width
            if event_map[y, x] == EVENT_PRESENT_2
                print("🔴 ")  # Red circle for event present
            else
                print("⚪ ")  # White circle for no event
            end
        end
        println("|")
    end
    
    # Print legend
    println("Legend: 🔴 = Event Present, ⚪ = No Event")
    println("="^50)
end

"""
Simulate environment evolution (simple version)
"""
function simulate_environment_evolution_simple(width::Int, height::Int, num_steps::Int, initial_events::Int)
    println("🧠 Simulating 2-State Environment Evolution (Simple)")
    println("===================================================")
    println("Grid size: $(width)x$(height)")
    println("Simulation steps: $(num_steps)")
    println("Initial events: $(initial_events)")
    
    # Create 2-state DBN model
    #dbn_model = DBNTransitionModel2(0.001, 0.05, 0.01)
    dbn_model = ProperDBNModel(0.001, 0.05, 0.008)
    println("DBN model: $(dbn_model)")
    
    # Initialize event map
    event_map = fill(NO_EVENT_2, height, width)
    
    # Add initial random events
    rng = Random.GLOBAL_RNG
    for _ in 1:initial_events
        x = rand(rng, 1:width)
        y = rand(rng, 1:height)
        event_map[y, x] = EVENT_PRESENT_2
    end
    
    println("Initial events: $(count(==(EVENT_PRESENT_2), event_map))")
    
    # Store evolution
    evolution = [copy(event_map)]
    event_counts = [count(==(EVENT_PRESENT_2), event_map)]
    
    # Simulate evolution
    for step in 1:num_steps
        # Update events using DBN
        update_events!(dbn_model, event_map, rng)
        
        # Store current state
        push!(evolution, copy(event_map))
        push!(event_counts, count(==(EVENT_PRESENT_2), event_map))
        
        println("Step $(step): $(event_counts[end]) events")
    end
    
    return evolution, event_counts, dbn_model
end

"""
Analyze environment evolution
"""
function analyze_environment_evolution_text(evolution::Vector{Matrix{EventState2}}, event_counts::Vector{Int})
    println("\n📈 Environment Evolution Analysis")
    println("================================")
    
    # Calculate statistics
    total_steps = length(event_counts)
    max_events = maximum(event_counts)
    min_events = minimum(event_counts)
    avg_events = mean(event_counts)
    
    println("Total simulation steps: $(total_steps)")
    println("Maximum events: $(max_events)")
    println("Minimum events: $(min_events)")
    println("Average events: $(round(avg_events, digits=2))")
    
    # Find peak and trough
    peak_step = argmax(event_counts)
    trough_step = argmin(event_counts)
    
    println("Peak events at step $(peak_step-1): $(event_counts[peak_step])")
    println("Minimum events at step $(trough_step-1): $(event_counts[trough_step])")
    
    # Show event count progression
    println("\nEvent Count Progression:")
    println("Step | Events")
    println("-----|-------")
    for (step, count) in enumerate(event_counts)
        println("$(step-1)    | $(count)")
    end
    
    # Analyze spatial patterns
    if length(evolution) > 1
        println("\nSpatial Pattern Analysis:")
        
        # Count events in different regions
        height, width = size(evolution[1])
        mid_x = div(width, 2)
        mid_y = div(height, 2)
        
        for step in [1, div(total_steps, 2), total_steps]
            event_map = evolution[step]
            
            # Count events in different quadrants
            q1 = count(x -> x == EVENT_PRESENT_2, event_map[1:mid_y, 1:mid_x])
            q2 = count(x -> x == EVENT_PRESENT_2, event_map[1:mid_y, mid_x+1:end])
            q3 = count(x -> x == EVENT_PRESENT_2, event_map[mid_y+1:end, 1:mid_x])
            q4 = count(x -> x == EVENT_PRESENT_2, event_map[mid_y+1:end, mid_x+1:end])
            
            println("Step $(step-1): Q1=$(q1), Q2=$(q2), Q3=$(q3), Q4=$(q4)")
        end
    end
end

"""
Test different parameter combinations
"""
function test_parameter_combinations()
    println("\n🔬 Testing Different Parameter Combinations")
    println("==========================================")
    
    # Create output directory for parameter tests
    output_dir = "visualizations_text"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    # Test different parameter combinations
    parameter_sets = [
        ("High_Birth_Rate", DBNTransitionModel2(0.01, 0.05, 0.1)),
        ("High_Death_Rate", DBNTransitionModel2(0.005, 0.1, 0.01)),
        ("High_Neighbor_Influence", DBNTransitionModel2(0.01, 0.05, 0.3)),
        ("Balanced", DBNTransitionModel2(0.01, 0.01, 0.01))
    ]
    
    # Create summary file
    summary_filename = joinpath(output_dir, "parameter_comparison_summary.txt")
    open(summary_filename, "w") do io
        println(io, "Parameter Comparison Summary")
        println(io, "==========================")
        println(io, "")
        
        for (name, dbn_model) in parameter_sets
            println("\n🔬 Testing: $(name)")
            println("Parameters: birth=$(dbn_model.birth_rate), death=$(dbn_model.death_rate), influence=$(dbn_model.neighbor_influence)")
            
            # Write to summary file
            println(io, "Configuration: $(name)")
            println(io, "Parameters: birth=$(dbn_model.birth_rate), death=$(dbn_model.death_rate), influence=$(dbn_model.neighbor_influence)")
            
            # Simulate with these parameters
            width = 10
            height = 50
            num_steps = 100
            initial_events = 1
            
            # Initialize and simulate
            event_map = fill(NO_EVENT_2, height, width)
            event_map[3, 3] = EVENT_PRESENT_2  # Start with one event in center
            
            evolution = [copy(event_map)]
            event_counts = [1]
            
            rng = Random.GLOBAL_RNG
            for step in 1:num_steps
                update_events!(dbn_model, event_map, rng)
                push!(evolution, copy(event_map))
                push!(event_counts, count(==(EVENT_PRESENT_2), event_map))
            end
            
            # Show final state
            visualize_grid_text(evolution[end], "Final State for $(name): $(event_counts[end]) events")
            
            final_count = event_counts[end]
            avg_count = round(mean(event_counts), digits=2)
            
            println("Final event count: $(final_count)")
            println("Average event count: $(avg_count)")
            
            # Write results to summary file
            println(io, "Final event count: $(final_count)")
            println(io, "Average event count: $(avg_count)")
            println(io, "Event progression: $(event_counts)")
            println(io, "")
            
            # Save individual parameter test results
            config_str = "grid$(width)x$(height)_steps$(num_steps)_init$(initial_events)_birth$(dbn_model.birth_rate)_death$(dbn_model.death_rate)_influence$(dbn_model.neighbor_influence)"
            param_filename = joinpath(output_dir, "parameter_test_$(name)_$(config_str).txt")
            
            open(param_filename, "w") do param_io
                println(param_io, "Parameter Test: $(name)")
                println(param_io, "Configuration: $(config_str)")
                println(param_io, "DBN Model: birth=$(dbn_model.birth_rate), death=$(dbn_model.death_rate), influence=$(dbn_model.neighbor_influence)")
                println(param_io, "")
                println(param_io, "Event Count Progression:")
                for (step, count) in enumerate(event_counts)
                    println(param_io, "Step $(step-1): $(count) events")
                end
                println(param_io, "")
                println(param_io, "Statistics:")
                println(param_io, "Final count: $(final_count)")
                println(param_io, "Average count: $(avg_count)")
                println(param_io, "Max count: $(maximum(event_counts))")
                println(param_io, "Min count: $(minimum(event_counts))")
            end
            
            println("✓ Saved parameter test: $(basename(param_filename))")
        end
        
        println(io, "\nSummary:")
        println(io, "All parameter tests completed. Individual results saved in separate files.")
    end
    
    println("✓ Parameter comparison summary saved: $(basename(summary_filename))")
end

"""
Save evolution results to a text file
"""
function save_evolution_results(evolution::Vector{Matrix{EventState2}}, event_counts::Vector{Int}, filename::String)
    open(filename, "w") do io
        println(io, "2-State Environment Evolution Results")
        println(io, "====================================")
        println(io, "")
        
        # Grid information
        height, width = size(evolution[1])
        println(io, "Grid Configuration:")
        println(io, "- Width: $(width)")
        println(io, "- Height: $(height)")
        println(io, "- Total cells: $(width * height)")
        println(io, "")
        
        # Event count progression
        println(io, "Event Count Progression:")
        println(io, "Step | Events")
        println(io, "-----|-------")
        for (step, count) in enumerate(event_counts)
            println(io, "$(step-1)    | $(count)")
        end
        println(io, "")
        
        # Statistics
        total_steps = length(event_counts)
        max_events = maximum(event_counts)
        min_events = minimum(event_counts)
        avg_events = mean(event_counts)
        
        println(io, "Statistics:")
        println(io, "- Total simulation steps: $(total_steps)")
        println(io, "- Maximum events: $(max_events)")
        println(io, "- Minimum events: $(min_events)")
        println(io, "- Average events: $(round(avg_events, digits=2))")
        println(io, "")
        
        # Find peak and trough
        peak_step = argmax(event_counts)
        trough_step = argmin(event_counts)
        println(io, "Peak events at step $(peak_step-1): $(event_counts[peak_step])")
        println(io, "Minimum events at step $(trough_step-1): $(event_counts[trough_step])")
        println(io, "")
        
        # Spatial pattern analysis
        if length(evolution) > 1
            println(io, "Spatial Pattern Analysis:")
            mid_x = div(width, 2)
            mid_y = div(height, 2)
            
            for step in [1, div(total_steps, 2), total_steps]
                event_map = evolution[step]
                
                # Count events in different quadrants
                q1 = count(x -> x == EVENT_PRESENT_2, event_map[1:mid_y, 1:mid_x])
                q2 = count(x -> x == EVENT_PRESENT_2, event_map[1:mid_y, mid_x+1:end])
                q3 = count(x -> x == EVENT_PRESENT_2, event_map[mid_y+1:end, 1:mid_x])
                q4 = count(x -> x == EVENT_PRESENT_2, event_map[mid_y+1:end, mid_x+1:end])
                
                println(io, "Step $(step-1): Q1=$(q1), Q2=$(q2), Q3=$(q3), Q4=$(q4)")
            end
            println(io, "")
        end
        
        # Final state visualization
        println(io, "Final State Grid:")
        final_state = evolution[end]
        visualize_grid_text_to_file(io, final_state, "Final State: $(event_counts[end]) events")
    end
end

"""
Visualize grid as text and write to file
"""
function visualize_grid_text_to_file(io::IO, event_map::Matrix{EventState2}, title::String="Event Grid")
    height, width = size(event_map)
    
    println(io, title)
    println(io, "=" ^ length(title))
    println(io, "")
    
    # Print column headers
    print(io, "   ")
    for x in 1:width
        print(io, lpad(string(x), 2))
    end
    println(io, "")
    
    # Print grid with row headers
    for y in 1:height
        print(io, lpad(string(y), 2) * " ")
        for x in 1:width
            if event_map[y, x] == EVENT_PRESENT_2
                print(io, " ■")
            else
                print(io, " □")
            end
        end
        println(io, "")
    end
    println(io, "")
end

"""
Main visualization function
"""
function main_visualization_simple()
    println("🎨 Simple 2-State Environment Visualization")
    println("==========================================")
    
    # Create output directory
    output_dir = "visualizations"
    if !isdir(output_dir)
        mkdir(output_dir)
        println("✓ Created output directory: $(output_dir)")
    end
    
    # Simulation parameters
    width = 10
    height = 50
    num_steps = 1000
    initial_events = 1
    
    # Simulate environment evolution
    evolution, event_counts, dbn_model = simulate_environment_evolution_simple(width, height, num_steps, initial_events)
    
    # Analyze evolution
    analyze_environment_evolution_text(evolution, event_counts)
    
    # Save results
    println("\n💾 Saving Results...")
    
    # Create configuration string for filename
    config_str = "grid$(width)x$(height)_steps$(num_steps)_init$(initial_events)_birth$(dbn_model.birth_rate)_death$(dbn_model.death_rate)_influence$(dbn_model.neighbor_influence)"
    
    # Save evolution data
    results_filename = joinpath(output_dir, "evolution_results_$(config_str).txt")
    save_evolution_results(evolution, event_counts, results_filename)
    println("✓ Evolution results saved as '$(basename(results_filename))'")
    
    # Display final state
    println("\n🎯 Final Environment State:")
    visualize_grid_text(evolution[end], "Final State: $(event_counts[end]) events")
    
    println("\n✅ Simple visualization completed!")
    println("\n📁 Generated files in '$(output_dir)' folder:")
    println("- $(basename(results_filename)): Evolution results")
    
    return evolution, event_counts
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main_visualization_simple()
end 
#!/usr/bin/env julia

"""
Simple test script to verify the animation regeneration works
"""

using DataFrames
using CSV
using Plots
using Statistics

println("🎬 Testing Animation Script")
println("==========================")

# Test with a simple data load
function test_data_loading()
    results_folder = "E:\\MA-LOMDP-Stochastic-Environments\\results\\run_2025-09-22T10-36-50-358\\Run 1"
    
    println("📁 Testing data loading from: $(results_folder)")
    
    # Find planning mode folders
    planning_mode_folders = filter(x -> isdir(joinpath(results_folder, x)) && x != "environment", readdir(results_folder))
    
    if isempty(planning_mode_folders)
        println("❌ No planning mode folders found")
        return false
    end
    
    planning_mode = planning_mode_folders[1]
    planning_mode_path = joinpath(results_folder, planning_mode)
    
    println("  Using planning mode: $(planning_mode)")
    
    # Extract run number from folder path
    run_number = parse(Int, split(basename(results_folder), " ")[2])
    
    # Test loading actions CSV
    actions_csv_path = joinpath(planning_mode_path, "metrics", "agent_actions_$(planning_mode)_run$(run_number).csv")
    if !isfile(actions_csv_path)
        println("❌ Agent actions CSV not found: $(actions_csv_path)")
        return false
    end
    
    actions_df = CSV.read(actions_csv_path, DataFrame)
    println("✓ Loaded $(nrow(actions_df)) action records")
    
    # Test loading uncertainty CSV
    uncertainty_csv_path = joinpath(planning_mode_path, "metrics", "uncertainty_evolution_$(planning_mode)_run$(run_number).csv")
    if !isfile(uncertainty_csv_path)
        println("❌ Uncertainty evolution CSV not found: $(uncertainty_csv_path)")
        return false
    end
    
    uncertainty_df = CSV.read(uncertainty_csv_path, DataFrame)
    println("✓ Loaded $(nrow(uncertainty_df)) uncertainty records")
    
    # Test loading event tracking CSV
    event_tracking_csv_path = joinpath(planning_mode_path, "metrics", "event_tracking_$(planning_mode)_run$(run_number).csv")
    if !isfile(event_tracking_csv_path)
        println("❌ Event tracking CSV not found: $(event_tracking_csv_path)")
        return false
    end
    
    event_tracking_df = CSV.read(event_tracking_csv_path, DataFrame)
    println("✓ Loaded $(nrow(event_tracking_df)) event tracking records")
    
    # Test basic data analysis
    println("\n📊 Basic Data Analysis:")
    println("  Grid size from events: $(maximum(event_tracking_df.cell_x)) x $(maximum(event_tracking_df.cell_y))")
    println("  Number of timesteps: $(length(unique(actions_df.timestep)))")
    println("  Number of agents: $(length(unique(actions_df.agent_id)))")
    println("  Total events: $(nrow(event_tracking_df))")
    println("  Observed events: $(count(event_tracking_df.observed))")
    
    return true
end

# Test plotting functionality
function test_plotting()
    println("\n🎨 Testing plotting functionality...")
    
    # Create a simple test plot
    x = 1:10
    y = rand(10)
    
    p = plot(x, y, title="Test Plot", xlabel="X", ylabel="Y")
    
    # Try to save as PNG
    test_filename = "test_plot.png"
    try
        savefig(p, test_filename)
        println("✓ Plot saved successfully: $(test_filename)")
        
        # Clean up
        if isfile(test_filename)
            rm(test_filename)
            println("✓ Test file cleaned up")
        end
        
        return true
    catch e
        println("❌ Plot saving failed: $(e)")
        return false
    end
end

# Run tests
println("Running tests...")

data_test = test_data_loading()
plot_test = test_plotting()

if data_test && plot_test
    println("\n✅ All tests passed! The animation script should work.")
else
    println("\n❌ Some tests failed. Please check the errors above.")
end

println("\n🎬 Test completed!")

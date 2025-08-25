#!/usr/bin/env julia

"""
Script to offset run folder numbers in a results directory
Usage: julia offset_run_numbers.jl <folder_name> <offset>
Example: julia offset_run_numbers.jl run_2025-08-19T10-23-17-927 28
This will rename Run 1 -> Run 29, Run 2 -> Run 30, etc.
"""

using Base.Filesystem

function offset_run_numbers(results_folder::String, offset::Int)
    println("🔄 Offsetting run numbers in folder: $(results_folder)")
    println("📊 Offset value: $(offset)")
    
    # Check if results folder exists
    results_path = joinpath("results", results_folder)
    if !isdir(results_path)
        println("❌ Error: Directory $(results_path) not found!")
        return false
    end
    
    # Find all directories that start with "Run "
    run_dirs = []
    for item in readdir(results_path)
        item_path = joinpath(results_path, item)
        if isdir(item_path)
            # Match patterns like "Run 1", "Run 2", "Run 10", etc.
            run_match = match(r"^Run (\d+)(.*)$", item)
            if run_match !== nothing
                run_number = parse(Int, run_match[1])
                suffix = run_match[2]  # Capture any suffix like " copy"
                push!(run_dirs, (item, run_number, suffix))
            end
        end
    end
    
    if isempty(run_dirs)
        println("⚠️ No run directories found matching pattern 'Run X'")
        return false
    end
    
    # Sort by run number (descending) to avoid conflicts during renaming
    sort!(run_dirs, by=x->x[2], rev=true)
    
    println("\n📁 Found $(length(run_dirs)) run directories:")
    for (dir_name, run_num, suffix) in run_dirs
        new_num = run_num + offset
        new_name = "Run $(new_num)$(suffix)"
        println("  $(dir_name) -> $(new_name)")
    end
    
    # Ask for confirmation
    print("\n❓ Proceed with renaming? (y/N): ")
    response = strip(readline())
    if lowercase(response) != "y" && lowercase(response) != "yes"
        println("❌ Operation cancelled.")
        return false
    end
    
    # Perform the renaming
    println("\n🚀 Starting renaming process...")
    renamed_count = 0
    
    for (old_name, run_num, suffix) in run_dirs
        old_path = joinpath(results_path, old_name)
        new_num = run_num + offset
        new_name = "Run $(new_num)$(suffix)"
        new_path = joinpath(results_path, new_name)
        
        try
            mv(old_path, new_path)
            println("  ✅ $(old_name) -> $(new_name)")
            renamed_count += 1
        catch e
            println("  ❌ Failed to rename $(old_name): $(e)")
        end
    end
    
    println("\n✅ Renaming completed!")
    println("📊 Successfully renamed $(renamed_count) out of $(length(run_dirs)) directories")
    
    return renamed_count == length(run_dirs)
end

function main()
    # Parse command line arguments
    if length(ARGS) < 2
        println("❌ Usage: julia offset_run_numbers.jl <folder_name> <offset>")
        println("📝 Example: julia offset_run_numbers.jl run_2025-08-19T10-23-17-927 28")
        println("   This will rename Run 1 -> Run 29, Run 2 -> Run 30, etc.")
        return
    end
    
    folder_name = ARGS[1]
    
    # Parse offset
    local offset
    try
        offset = parse(Int, ARGS[2])
        if offset < 0
            println("⚠️ Warning: Negative offset will decrease run numbers")
        end
    catch
        println("❌ Error: Offset must be a valid integer")
        return
    end
    
    # Perform the operation
    success = offset_run_numbers(folder_name, offset)
    
    if success
        println("\n🎉 All operations completed successfully!")
    else
        println("\n⚠️ Some operations failed. Please check the output above.")
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

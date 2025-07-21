#!/usr/bin/env julia

"""
Script to remove empty folders from the results directory
This helps clean up any empty directories that might have been created during simulation runs
"""

using Base.Filesystem

function remove_empty_folders(root_dir::String)
    """
    Recursively remove empty folders starting from root_dir
    
    Args:
        root_dir: The root directory to start cleaning from
    """
    println("🧹 Cleaning empty folders in: $(root_dir)")
    
    if !isdir(root_dir)
        println("❌ Directory does not exist: $(root_dir)")
        return
    end
    
    # Get all subdirectories recursively
    all_dirs = String[]
    for (root, dirs, files) in walkdir(root_dir)
        for dir in dirs
            push!(all_dirs, joinpath(root, dir))
        end
    end
    
    # Sort by depth (deepest first) so we remove nested empty folders first
    sort!(all_dirs, by=dir -> count(c -> c == '/', dir), rev=true)
    
    removed_count = 0
    
    for dir in all_dirs
        if isdir(dir)
            # Check if directory is empty
            contents = readdir(dir)
            if isempty(contents)
                try
                    rm(dir)
                    println("🗑️  Removed empty folder: $(dir)")
                    removed_count += 1
                catch e
                    println("⚠️  Could not remove folder $(dir): $(e)")
                end
            end
        end
    end
    
    println("✅ Cleanup complete! Removed $(removed_count) empty folders.")
    return removed_count
end

function main()
    # Define the results directories to clean
    results_dirs = [
        "results",
        "results_new", 
        "visualizations",
        "visualizations_new",
        "Comparisons"
    ]
    
    total_removed = 0
    
    for dir in results_dirs
        if isdir(dir)
            println("\n" * "="^50)
            removed = remove_empty_folders(dir)
            total_removed += removed
        else
            println("📁 Directory does not exist: $(dir)")
        end
    end
    
    println("\n" * "="^50)
    println("🎉 Total cleanup summary:")
    println("   Removed $(total_removed) empty folders across all directories")
    println("="^50)
end

# Run the cleanup if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 
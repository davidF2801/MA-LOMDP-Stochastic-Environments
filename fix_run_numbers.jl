using Base.Filesystem

# Function to update run number in a single file
function update_run_number_in_file(file_path::String, correct_run_number::Int)
    try
        # Read the file content
        content = read(file_path, String)
        
        # Check if the file contains a run number line
        if occursin("Run Number:", content)
            # Replace the old run number with the correct one
            new_content = replace(content, 
                r"Run Number: \d+" => "Run Number: $(correct_run_number)")
            
            # Write the updated content back to the file
            write(file_path, new_content)
            println("✅ Updated: $file_path -> Run Number: $correct_run_number")
            return true
        else
            println("⚠️  No 'Run Number:' found in: $file_path")
            return false
        end
    catch e
        println("❌ Error updating $file_path: $e")
        return false
    end
end

# Function to process all files in a run directory
function process_run_directory(run_path::String, run_number::Int)
    println("🔄 Processing Run $run_number...")
    
    # Planning modes to check
    planning_modes = ["sweep", "script", "prior_based", "macro_approx_099", "macro_approx_095", "macro_approx_090"]
    
    total_files = 0
    updated_files = 0
    
    for mode in planning_modes
        metrics_dir = joinpath(run_path, mode, "metrics")
        
        if isdir(metrics_dir)
            # Find all performance metrics files in this directory
            for file in readdir(metrics_dir)
                if startswith(file, "performance_metrics_") && endswith(file, ".txt")
                    file_path = joinpath(metrics_dir, file)
                    total_files += 1
                    
                    if update_run_number_in_file(file_path, run_number)
                        updated_files += 1
                    end
                end
            end
        end
    end
    
    println("📊 Run $run_number: $updated_files/$total_files files updated")
    return updated_files, total_files
end

# Main execution
function main()
    base_dir = "results/run_2025-08-01T17-57-32-107"
    
    if !isdir(base_dir)
        println("❌ Base directory not found: $base_dir")
        return
    end
    
    println("🚀 Starting run number fixes for runs 31-45...")
    println("📍 Base directory: $base_dir")
    println()
    
    total_updated = 0
    total_files = 0
    
    # Process runs 31-45
    for run_num in 31:45
        run_dir = joinpath(base_dir, "Run $run_num")
        
        if isdir(run_dir)
            updated, files = process_run_directory(run_dir, run_num)
            total_updated += updated
            total_files += files
            println()
        else
            println("⚠️  Run directory not found: $run_dir")
        end
    end
    
    println("🎉 Run number fixes complete!")
    println("📈 Total files processed: $total_files")
    println("✅ Total files updated: $total_updated")
    println("⏭️  Total files skipped: $(total_files - total_updated)")
end

# Run the script
main() 
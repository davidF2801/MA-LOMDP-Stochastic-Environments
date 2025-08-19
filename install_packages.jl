#!/usr/bin/env julia

"""
Script to install ALL required packages for the MA-LOMDP project
This includes packages for POMDP modeling, data analysis, visualization, and postprocessing
"""

using Pkg

println("🚀 Installing all required packages for MA-LOMDP project...")
println("="^60)

# Core POMDP packages
println("\n📦 Installing core POMDP packages...")
packages_core = [
    "POMDPs",
    "POMDPTools"
]

for pkg in packages_core
    println("  Adding $(pkg)...")
    Pkg.add(pkg)
end

# Mathematical and statistical packages
println("\n🔢 Installing mathematical and statistical packages...")
packages_math = [
    "Random",
    "LinearAlgebra", 
    "Statistics",
    "Distributions",
    "Combinatorics"
]

for pkg in packages_math
    println("  Adding $(pkg)...")
    Pkg.add(pkg)
end

# Data handling packages
println("\n📊 Installing data handling packages...")
packages_data = [
    "DataFrames",
    "CSV",
    "JSON",
    "Dates"
]

for pkg in packages_data
    println("  Adding $(pkg)...")
    Pkg.add(pkg)
end

# Visualization packages
println("\n📈 Installing visualization packages...")
packages_viz = [
    "Plots",
    "StatsPlots"  # For boxplots and statistical visualizations
]

for pkg in packages_viz
    println("  Adding $(pkg)...")
    Pkg.add(pkg)
end

# File system and utility packages
println("\n🔧 Installing utility packages...")
packages_utils = [
    "Glob",           # For file pattern matching
    "Infiltrator",    # For debugging
    "Logging",        # For logging functionality
    "ImageFiltering", # Used in belief management
    "Base.Threads",   # For multithreading (built-in but ensuring compatibility)
    "Base.Filesystem" # For file operations (built-in but ensuring compatibility)
]

for pkg in packages_utils
    if pkg != "Base.Threads" && pkg != "Base.Filesystem"  # Skip built-in modules
        println("  Adding $(pkg)...")
        Pkg.add(pkg)
    else
        println("  $(pkg) (built-in module, skipping)")
    end
end

# Install all dependencies and precompile
println("\n⚙️ Installing all dependencies and precompiling...")
Pkg.instantiate()
Pkg.precompile()

println("\n✅ Package installation completed successfully!")
println("="^60)
println("📋 Summary of installed packages:")
println("  Core POMDP: POMDPs, POMDPTools")
println("  Mathematics: Random, LinearAlgebra, Statistics, Distributions, Combinatorics")
println("  Data: DataFrames, CSV, JSON, Dates")
println("  Visualization: Plots, StatsPlots")
println("  Utilities: Glob, Infiltrator, Logging, ImageFiltering")
println("\n🎯 You can now run any script in the project:")
println("  - julia scripts/main.jl")
println("  - julia scripts/postprocess_results.jl")
println("  - julia scripts/test_environment.jl")
println("  - And many more!")
println("\n💡 Tip: Use 'julia --project=.' to ensure you're using the project environment") 
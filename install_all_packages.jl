#!/usr/bin/env julia

"""
Install all required packages for the MA-LOMDP project
This script installs all packages listed in Project.toml and any additional ones needed
"""

println("=" ^ 80)
println("Installing all packages for MA-LOMDP project")
println("=" ^ 80)
println()

using Pkg

# Activate the project
Pkg.activate(".")
println("✓ Activated project directory")
println()

# List of all packages needed (from Project.toml + additional ones used in code)
# Note: POMDPModelTools, BeliefUpdaters, POMDPPolicies, POMDPSimulators, QMDP
# are excluded due to version compatibility issues with Julia 1.12
packages = [
    # Core POMDP packages
    "POMDPs",
    "POMDPTools",
    
    # Data and I/O
    "CSV",
    "DataFrames",
    "JSON",
    
    # Plotting and visualization
    "Plots",
    
    # Development and debugging
    "Infiltrator",
    
    # Standard library packages (usually already available, but ensuring they're in project)
    "Dates",
    "Random",
    "LinearAlgebra",
    "Statistics",
]

println("Installing packages:")
println("-" ^ 80)

# Install packages one by one with error handling
failed_packages = String[]
successful_packages = String[]

for pkg in packages
    try
        print("Installing $pkg... ")
        Pkg.add(pkg)
        println("✓")
        push!(successful_packages, pkg)
    catch e
        println("✗ Error: $e")
        push!(failed_packages, pkg)
    end
end

println()
println("=" ^ 80)
println("Installation Summary")
println("=" ^ 80)
println("✓ Successfully installed: $(length(successful_packages)) packages")
if !isempty(failed_packages)
    println("✗ Failed to install: $(length(failed_packages)) packages")
    for pkg in failed_packages
        println("  - $pkg")
    end
end
println()

# Now resolve and instantiate to ensure everything is set up correctly
println("Resolving dependencies...")
try
    Pkg.resolve()
    println("✓ Dependencies resolved")
catch e
    println("✗ Error resolving dependencies: $e")
end

println()
println("Instantiating project...")
try
    Pkg.instantiate()
    println("✓ Project instantiated")
catch e
    println("✗ Error instantiating project: $e")
end

println()
println("=" ^ 80)
println("Testing imports...")
println("=" ^ 80)

# Test critical imports
test_packages = ["POMDPs", "POMDPTools", "Plots", "DataFrames", "CSV", "Infiltrator"]
for pkg in test_packages
    try
        eval(Meta.parse("using $pkg"))
        println("✓ $pkg imports successfully")
    catch e
        println("✗ $pkg failed to import: $e")
    end
end

println()
println("=" ^ 80)
println("Installation complete!")
println("=" ^ 80)
println()
println("You can now run your scripts with:")
println("  julia --project=. scripts/main.jl")
println()


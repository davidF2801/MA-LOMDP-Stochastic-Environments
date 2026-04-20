#!/usr/bin/env julia

"""
Fix package version compatibility issues
This script removes problematic packages and updates version constraints
"""

println("=" ^ 80)
println("Fixing Package Version Compatibility")
println("=" ^ 80)
println()

using Pkg

# Activate the project
Pkg.activate(".")
println("✓ Activated project directory")
println()

# Remove problematic packages that are causing version conflicts
# These are transitive dependencies that aren't directly used
problematic_packages = [
    "POMDPModelTools",  # Uses old POMDPs API
    "BeliefUpdaters",   # Depends on POMDPModelTools
]

println("Removing problematic packages:")
println("-" ^ 80)
for pkg in problematic_packages
    try
        print("Removing $pkg... ")
        Pkg.rm(pkg)
        println("✓")
    catch e
        println("✗ (may not be installed): $e")
    end
end
println()

# Update POMDPs and related packages to latest compatible versions
println("Updating POMDP packages to compatible versions...")
println("-" ^ 80)

# Add packages with specific version constraints that work together
try
    # Update POMDPs first
    print("Updating POMDPs... ")
    Pkg.add("POMDPs")
    println("✓")
    
    # Update POMDPTools
    print("Updating POMDPTools... ")
    Pkg.add("POMDPTools")
    println("✓")
    
    # Try to add POMDPPolicies, POMDPSimulators, QMDP with version resolution
    # These may work if we don't have the problematic dependencies
    print("Updating POMDPPolicies... ")
    try
        Pkg.add("POMDPPolicies")
        println("✓")
    catch e
        println("✗ May have dependency issues: $e")
        println("  This package may not be critical for your main script")
    end
    
    print("Updating POMDPSimulators... ")
    try
        Pkg.add("POMDPSimulators")
        println("✓")
    catch e
        println("✗ May have dependency issues: $e")
    end
    
    print("Updating QMDP... ")
    try
        Pkg.add("QMDP")
        println("✓")
    catch e
        println("✗ May have dependency issues: $e")
    end
    
catch e
    println("✗ Error updating packages: $e")
end

println()
println("Resolving dependencies...")
try
    Pkg.resolve()
    println("✓ Dependencies resolved")
catch e
    println("⚠ Warning resolving dependencies: $e")
end

println()
println("=" ^ 80)
println("Testing critical imports...")
println("=" ^ 80)

# Test imports that are actually used
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
println("Fix complete!")
println("=" ^ 80)
println()
println("Note: If POMDPPolicies/POMDPSimulators/QMDP failed, they may not be")
println("      critical for your main script. The macro_planner_sync_multi.jl")
println("      file that uses them can be temporarily disabled if needed.")
println()



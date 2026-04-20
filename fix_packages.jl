#!/usr/bin/env julia

"""
Quick fix script to remove problematic packages and reinstall
"""

using Pkg

# Activate the project
Pkg.activate(".")

println("Removing problematic packages...")
problematic = ["POMDPModelTools", "BeliefUpdaters", "POMDPPolicies", "POMDPSimulators", "QMDP"]
for pkg in problematic
    try
        Pkg.rm(pkg)
        println("  ✓ Removed $pkg")
    catch e
        println("  - $pkg not found or already removed")
    end
end

println("\nResolving dependencies...")
Pkg.resolve()

println("Instantiating project...")
Pkg.instantiate()

println("\n✓ Done! You can now run: julia --project=. scripts/main.jl")

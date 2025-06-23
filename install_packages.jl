#!/usr/bin/env julia

"""
Script to install required packages for the MA-LOMDP project
"""

using Pkg

println("Installing POMDP packages...")

# Add POMDPs package
println("Adding POMDPs...")
Pkg.add("POMDPs")

# Add POMDPTools package
println("Adding POMDPTools...")
Pkg.add("POMDPTools")

# Add other dependencies
println("Adding Distributions...")
Pkg.add("Distributions")

println("Installing all dependencies...")
Pkg.instantiate()

println("Package installation completed!")
println("You can now run: julia scripts/test_environment.jl") 
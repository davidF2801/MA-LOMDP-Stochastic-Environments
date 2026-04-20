#!/usr/bin/env julia

println("🔧 Simple fix: Just add the packages we need...")

using Pkg
Pkg.activate(".")

println("Adding CSV...")
try
    Pkg.add("CSV")
catch e
    println("CSV error: $e")
end

println("Adding DataFrames...")
try
    Pkg.add("DataFrames")
catch e
    println("DataFrames error: $e")
end

println("Adding Plots...")
try
    Pkg.add("Plots")
catch e
    println("Plots error: $e")
end

println("Testing imports...")
try
    using CSV
    println("✅ CSV works!")
catch e
    println("❌ CSV failed: $e")
end

try
    using DataFrames
    println("✅ DataFrames works!")
catch e
    println("❌ DataFrames failed: $e")
end

try
    using Plots
    println("✅ Plots works!")
catch e
    println("❌ Plots failed: $e")
end

println("Done!")



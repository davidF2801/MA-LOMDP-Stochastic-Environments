#!/usr/bin/env julia

println("🧪 Testing imports...")

try
    # Test basic imports
    using POMDPs
    println("✅ POMDPs loaded")
    
    using POMDPTools
    println("✅ POMDPTools loaded")
    
    using Random
    println("✅ Random loaded")
    
    # Test project imports
    include("src/MyProject.jl")
    using .MyProject
    println("✅ MyProject loaded")
    
    # Test specific modules
    using .Environment
    println("✅ Environment loaded")
    
    using .Environment.EventDynamicsModule
    println("✅ EventDynamicsModule loaded")
    
    using .Planners.GroundStation
    println("✅ GroundStation loaded")
    
    using .Planners.MacroPlanner
    println("✅ MacroPlanner loaded")
    
    using .Planners.PolicyTreePlanner
    println("✅ PolicyTreePlanner loaded")
    
    using .Agents
    println("✅ Agents loaded")
    
    println("\n🎉 All imports successful!")
    
catch e
    println("❌ Import error: ", e)
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end 
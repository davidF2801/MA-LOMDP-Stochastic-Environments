module MyProject

# Main module for Multi-Agent Locally Observable MDP (MA-LOMDP)

# Export submodules
export Environment, Agents

# Include submodules
include("environment/Environment.jl")
include("agents/Agents.jl")

# Re-export commonly used functions
export simulate_environment, update_belief, plan_sensing, evaluate_policy

end # module 
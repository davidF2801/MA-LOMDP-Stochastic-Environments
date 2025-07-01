module MyProject

# Main module for Multi-Agent Locally Observable MDP (MA-LOMDP)

# Include and load Types module first
include("types/Types.jl")
using .Types

# Export submodules
export Agents, Environment, Planners, Types

# Include submodules in dependency order
include("agents/Agents.jl")
include("environment/Environment.jl")
include("planners/Planners.jl")

# Now that all modules are loaded, we can use them
using .Agents
using .Environment
using .Planners

# Re-export commonly used functions
export simulate_environment

# Re-export commonly used types from Types module
export Agent, SensingAction, EventState, NO_EVENT, EVENT_PRESENT, CircularTrajectory, LinearTrajectory, RangeLimitedSensor, Belief

end # module 
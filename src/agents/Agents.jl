module Agents

using POMDPs
using POMDPTools
using Distributions
using Random
using ..Types

# Export submodules
export TrajectoryPlanner, SensingPolicy, Communication, BeliefManagement

include("trajectory_planner.jl")
using .TrajectoryPlanner
export Agent  # ✅ Now this works because it's defined

include("sensing_policy.jl")
include("communication.jl")
include("belief_management.jl")

end # module

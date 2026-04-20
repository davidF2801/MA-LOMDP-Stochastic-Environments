module Planners

using POMDPs
using POMDPTools
using Random
using ..Types
using ..Environment
using ..Agents

# Export submodules
export GroundStation, MacroPlannerAsync, MacroPlannerSync, PolicyTreePlanner, MacroPlannerRandom, MacroPlannerSweep, MacroPlannerGreedy, MacroPlannerPriorBased, MacroPlannerPBVI, MacroPlannerPBVIRollout, MacroPlannerOracle, MacroPlannerMPOMDPOpenLoop, MacroPlannerPOMCP
# MacroPlannerSyncMulti temporarily disabled due to package version conflicts

# Include submodules
include("ground_station.jl")
# PBVI rollout lives in ground_station.jl's include tree; re-export for scripts (`using .Planners: MacroPlannerPBVIRollout`).
const MacroPlannerPBVIRollout = GroundStation.MacroPlannerPBVIRollout
include("macro_planner_async.jl")
include("macro_planner_sync.jl")
include("policy_tree_planner.jl")
include("macro_planner_random.jl")
include("macro_planner_sweep.jl")
include("macro_planner_greedy.jl")
include("macro_planner_prior_based.jl")
include("macro_planner_pbvi.jl")
# Temporarily disabled due to package version conflicts
# include("macro_planner_sync_multi.jl")
include("macro_planner_oracle.jl")
include("macro_planner_mpomdp_openloop.jl")
include("macro_planner_pomcp.jl")

end # module 
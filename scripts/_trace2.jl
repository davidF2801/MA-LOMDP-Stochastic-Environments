include(joinpath(@__DIR__, "..", "src", "gas_valve", "GasValve.jl"))
using .GasValve
using .GasValveTypes
using .GasValveBelief
using .GasValveEnv
using Random, Printf

dynamics = default_pocket_dynamics()
println("p_arrive=$(dynamics.p_arrive), p_acc=$(dynamics.p_acc)")

# Test: does obs at dist>=2 do anything?
b = [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]
println("initial: $b")
b[1] = GasValveBelief.bayes_update_pocket_obs(b[1], 2, 3)
println("after obs z=2 dist=3: $b  (should be unchanged)")

# Test: 3 steps of transition from uniform
b = [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]
for t in 1:3
    global b = GasValveBelief.transition_step_belief(b, dynamics)
    q = b[1][2] + b[1][3]
    println("after $t transitions: b[1]=$(round.(b[1], digits=3))  q=$(round(q, digits=3))")
end

# Full test: with obs at dist >=2
println("\n--- full rollforward with dist>=2 obs ---")
pockets = [PocketDef((1,2), (1,4), (1,3), 1), PocketDef((5,2), (5,4), (5,3), 2)]
obs = [
    ObsRecord(0, 1, (3,3), 0),  # dist=2 to P1
    ObsRecord(1, 1, (3,2), 1),  # dist=3 to P1
    ObsRecord(2, 1, (2,2), 2),  # dist=2 to P1
]
b0 = [[1/3,1/3,1/3], [1/3,1/3,1/3]]
b_after = GasValveBelief.PrivateBeliefRollforward(b0, 0, 3, obs, pockets, dynamics)
println("b_after=$(round.(b_after[1], digits=3))  q=$(round(b_after[1][2]+b_after[1][3], digits=3))")

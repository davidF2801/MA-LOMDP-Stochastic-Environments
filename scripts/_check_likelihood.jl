include(joinpath(@__DIR__, "..", "src", "gas_valve", "GasValve.jl"))
using .GasValve.GasValveEnv
for dist in 0:3
    println("dist=$dist:")
    for s in 0:2
        likelihoods = [observation_likelihood(z, s, dist) for z in 0:2]
        println("  s=$s: $(round.(likelihoods, digits=3))")
    end
end

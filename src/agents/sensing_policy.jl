using POMDPs
using POMDPTools
using POMDPPolicies
using Distributions

"""
SensingPolicy - Implements different strategies for sensing decisions
"""
module SensingPolicy

using POMDPs
using POMDPTools
using POMDPPolicies
using Distributions

export InformationGainPolicy, RandomSensingPolicy, GreedySensingPolicy, select_sensing_targets

"""
InformationGainPolicy - Policy that maximizes information gain
"""
struct InformationGainPolicy <: Policy
    exploration_weight::Float64
end

"""
RandomSensingPolicy - Random selection of sensing targets
"""
struct RandomSensingPolicy <: Policy
    rng::AbstractRNG
end

"""
GreedySensingPolicy - Greedy selection based on current belief
"""
struct GreedySensingPolicy <: Policy
    uncertainty_threshold::Float64
end

"""
select_sensing_targets(policy::InformationGainPolicy, belief::Matrix{Float64}, footprint::Vector{Tuple{Int, Int}}, max_targets::Int)
Selects sensing targets to maximize information gain
"""
function select_sensing_targets(policy::InformationGainPolicy, belief::Matrix{Float64}, footprint::Vector{Tuple{Int, Int}}, max_targets::Int)
    # TODO: Implement information gain-based target selection
    # - Calculate entropy for each cell in footprint
    # - Select cells with highest uncertainty
    # - Consider exploration vs exploitation
    
    # Calculate entropy for each cell
    entropies = Float64[]
    for cell in footprint
        x, y = cell
        entropy = calculate_entropy(belief[y, x])
        push!(entropies, entropy)
    end
    
    # Select cells with highest entropy
    sorted_indices = sortperm(entropies, rev=true)
    selected_cells = footprint[sorted_indices[1:min(max_targets, length(footprint))]]
    
    return selected_cells
end

"""
select_sensing_targets(policy::RandomSensingPolicy, belief::Matrix{Float64}, footprint::Vector{Tuple{Int, Int}}, max_targets::Int)
Randomly selects sensing targets
"""
function select_sensing_targets(policy::RandomSensingPolicy, belief::Matrix{Float64}, footprint::Vector{Tuple{Int, Int}}, max_targets::Int)
    # TODO: Implement random target selection
    num_targets = min(max_targets, length(footprint))
    selected_indices = randperm(policy.rng, length(footprint))[1:num_targets]
    return footprint[selected_indices]
end

"""
select_sensing_targets(policy::GreedySensingPolicy, belief::Matrix{Float64}, footprint::Vector{Tuple{Int, Int}}, max_targets::Int)
Greedily selects sensing targets based on uncertainty threshold
"""
function select_sensing_targets(policy::GreedySensingPolicy, belief::Matrix{Float64}, footprint::Vector{Tuple{Int, Int}}, max_targets::Int)
    # TODO: Implement greedy target selection
    # - Select cells with uncertainty above threshold
    # - Prioritize high-uncertainty regions
    
    selected_cells = Tuple{Int, Int}[]
    
    for cell in footprint
        x, y = cell
        uncertainty = 1.0 - abs(belief[y, x] - 0.5) * 2  # Uncertainty measure
        
        if uncertainty > policy.uncertainty_threshold && length(selected_cells) < max_targets
            push!(selected_cells, cell)
        end
    end
    
    return selected_cells
end

"""
calculate_entropy(probability::Float64)
Calculates entropy for a binary event
"""
function calculate_entropy(probability::Float64)
    if probability <= 0.0 || probability >= 1.0
        return 0.0
    end
    return -(probability * log(probability) + (1 - probability) * log(1 - probability))
end

"""
create_information_gain_policy(exploration_weight::Float64=0.1)
Creates an information gain policy
"""
function create_information_gain_policy(exploration_weight::Float64=0.1)
    return InformationGainPolicy(exploration_weight)
end

"""
create_random_sensing_policy(rng::AbstractRNG=Random.GLOBAL_RNG)
Creates a random sensing policy
"""
function create_random_sensing_policy(rng::AbstractRNG=Random.GLOBAL_RNG)
    return RandomSensingPolicy(rng)
end

"""
create_greedy_sensing_policy(uncertainty_threshold::Float64=0.3)
Creates a greedy sensing policy
"""
function create_greedy_sensing_policy(uncertainty_threshold::Float64=0.3)
    return GreedySensingPolicy(uncertainty_threshold)
end

end # module 
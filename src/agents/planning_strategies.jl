using POMDPs
using POMDPTools
using Distributions
using Random

"""
Abstract type for different planning strategies
"""
abstract type PlanningStrategy end

"""
Information Gain Planning - Plans actions to maximize information gain
"""
struct InformationGainPlanning <: PlanningStrategy
    horizon::Int
    exploration_weight::Float64
end

"""
Uncertainty Reduction Planning - Plans actions to minimize uncertainty
"""
struct UncertaintyReductionPlanning <: PlanningStrategy
    horizon::Int
    uncertainty_threshold::Float64
end

"""
Coverage Planning - Plans actions to maximize area coverage
"""
struct CoveragePlanning <: PlanningStrategy
    horizon::Int
    coverage_weight::Float64
end

"""
Multi-Objective Planning - Combines multiple objectives
"""
struct MultiObjectivePlanning <: PlanningStrategy
    horizon::Int
    info_weight::Float64
    uncertainty_weight::Float64
    coverage_weight::Float64
end

"""
Abstract type for action selection policies
"""
abstract type ActionPolicy end

"""
Greedy policy - selects action with highest immediate reward
"""
struct GreedyPolicy <: ActionPolicy end

"""
Lookahead policy - considers future rewards
"""
struct LookaheadPolicy <: ActionPolicy
    horizon::Int
end

"""
PlanningModule - Handles different planning strategies
"""
module PlanningModule

using POMDPs
using POMDPTools
using Distributions
using Random

export PlanningStrategy, InformationGainPlanning, UncertaintyReductionPlanning, 
       CoveragePlanning, MultiObjectivePlanning, ActionPolicy, GreedyPolicy, 
       LookaheadPolicy, plan_action, evaluate_action

"""
plan_action(strategy::InformationGainPlanning, agent::Agent, env::SpatialGrid, current_state::GridState)
Plans action to maximize information gain
"""
function plan_action(strategy::InformationGainPlanning, agent::Agent, env::SpatialGrid, current_state::GridState)
    # Get possible sensing targets within sensor range
    possible_targets = get_possible_targets(agent, current_state, env)
    
    best_action = nothing
    best_value = -Inf
    
    for targets in possible_targets
        action = SensingAction(agent.id, targets, false)
        
        # Calculate expected information gain
        expected_gain = calculate_expected_information_gain(action, agent, env, current_state)
        
        # Add exploration bonus
        exploration_bonus = strategy.exploration_weight * calculate_exploration_bonus(targets, agent.belief)
        
        total_value = expected_gain + exploration_bonus
        
        if total_value > best_value
            best_value = total_value
            best_action = action
        end
    end
    
    return best_action
end

"""
plan_action(strategy::UncertaintyReductionPlanning, agent::Agent, env::SpatialGrid, current_state::GridState)
Plans action to minimize uncertainty
"""
function plan_action(strategy::UncertaintyReductionPlanning, agent::Agent, env::SpatialGrid, current_state::GridState)
    possible_targets = get_possible_targets(agent, current_state, env)
    
    best_action = nothing
    best_uncertainty_reduction = -Inf
    
    for targets in possible_targets
        action = SensingAction(agent.id, targets, false)
        
        # Calculate expected uncertainty reduction
        uncertainty_reduction = calculate_expected_uncertainty_reduction(action, agent, env, current_state)
        
        if uncertainty_reduction > best_uncertainty_reduction
            best_uncertainty_reduction = uncertainty_reduction
            best_action = action
        end
    end
    
    return best_action
end

"""
plan_action(strategy::CoveragePlanning, agent::Agent, env::SpatialGrid, current_state::GridState)
Plans action to maximize area coverage
"""
function plan_action(strategy::CoveragePlanning, agent::Agent, env::SpatialGrid, current_state::GridState)
    possible_targets = get_possible_targets(agent, current_state, env)
    
    best_action = nothing
    best_coverage = -Inf
    
    for targets in possible_targets
        action = SensingAction(agent.id, targets, false)
        
        # Calculate coverage value
        coverage_value = calculate_coverage_value(targets, agent.belief, strategy.coverage_weight)
        
        if coverage_value > best_coverage
            best_coverage = coverage_value
            best_action = action
        end
    end
    
    return best_action
end

"""
plan_action(strategy::MultiObjectivePlanning, agent::Agent, env::SpatialGrid, current_state::GridState)
Plans action using multiple objectives
"""
function plan_action(strategy::MultiObjectivePlanning, agent::Agent, env::SpatialGrid, current_state::GridState)
    possible_targets = get_possible_targets(agent, current_state, env)
    
    best_action = nothing
    best_value = -Inf
    
    for targets in possible_targets
        action = SensingAction(agent.id, targets, false)
        
        # Calculate multi-objective value
        info_gain = strategy.info_weight * calculate_expected_information_gain(action, agent, env, current_state)
        uncertainty_reduction = strategy.uncertainty_weight * calculate_expected_uncertainty_reduction(action, agent, env, current_state)
        coverage = strategy.coverage_weight * calculate_coverage_value(targets, agent.belief, 1.0)
        
        total_value = info_gain + uncertainty_reduction + coverage
        
        if total_value > best_value
            best_value = total_value
            best_action = action
        end
    end
    
    return best_action
end

"""
get_possible_targets(agent::Agent, state::GridState, env::SpatialGrid)
Gets possible sensing targets within sensor range
"""
function get_possible_targets(agent::Agent, state::GridState, env::SpatialGrid)
    agent_pos = state.agent_positions[agent.id]
    possible_targets = Vector{Vector{Tuple{Int, Int}}}()
    
    # Get all cells within sensor range
    sensor_range = Int(floor(agent.sensor.range))
    height, width = size(state.event_map)
    
    for dx in -sensor_range:sensor_range
        for dy in -sensor_range:sensor_range
            if dx^2 + dy^2 <= sensor_range^2
                x, y = agent_pos[1] + dx, agent_pos[2] + dy
                if 1 <= x <= width && 1 <= y <= height
                    push!(possible_targets, [(x, y)])
                end
            end
        end
    end
    
    # Also consider combinations of targets (up to max_sensing_targets)
    if env.max_sensing_targets > 1
        # Generate combinations of 2-3 targets
        for i in 1:min(length(possible_targets), 10)  # Limit combinations for efficiency
            for j in (i+1):min(length(possible_targets), 10)
                if j <= length(possible_targets)
                    push!(possible_targets, [possible_targets[i][1], possible_targets[j][1]])
                end
            end
        end
    end
    
    return possible_targets
end

"""
calculate_expected_information_gain(action::SensingAction, agent::Agent, env::SpatialGrid, state::GridState)
Calculates expected information gain for an action
"""
function calculate_expected_information_gain(action::SensingAction, agent::Agent, env::SpatialGrid, state::GridState)
    # Use agent's current belief
    belief = agent.belief.event_probabilities
    
    # Calculate information gain for each target
    total_gain = 0.0
    
    for target in action.target_cells
        x, y = target
        if 1 <= x <= size(belief, 2) && 1 <= y <= size(belief, 1)
            # Calculate entropy reduction
            prior_entropy = calculate_entropy(belief[y, x])
            
            # Expected posterior entropy (simplified)
            # Assume observation reduces uncertainty by 50%
            expected_posterior = belief[y, x] * 0.5 + 0.25
            posterior_entropy = calculate_entropy(expected_posterior)
            
            gain = prior_entropy - posterior_entropy
            total_gain += gain
        end
    end
    
    return total_gain
end

"""
calculate_expected_uncertainty_reduction(action::SensingAction, agent::Agent, env::SpatialGrid, state::GridState)
Calculates expected uncertainty reduction for an action
"""
function calculate_expected_uncertainty_reduction(action::SensingAction, agent::Agent, env::SpatialGrid, state::GridState)
    belief = agent.belief.event_probabilities
    total_reduction = 0.0
    
    for target in action.target_cells
        x, y = target
        if 1 <= x <= size(belief, 2) && 1 <= y <= size(belief, 1)
            current_uncertainty = calculate_entropy(belief[y, x])
            # Expected uncertainty reduction
            expected_reduction = current_uncertainty * 0.5  # Simplified
            total_reduction += expected_reduction
        end
    end
    
    return total_reduction
end

"""
calculate_coverage_value(targets::Vector{Tuple{Int, Int}}, belief::Belief, weight::Float64)
Calculates coverage value for targets
"""
function calculate_coverage_value(targets::Vector{Tuple{Int, Int}}, belief::Belief, weight::Float64)
    # Simple coverage metric: number of targets
    coverage = length(targets) * weight
    
    # Add bonus for covering uncertain areas
    uncertainty_bonus = 0.0
    for target in targets
        x, y = target
        if 1 <= x <= size(belief.uncertainty_map, 2) && 1 <= y <= size(belief.uncertainty_map, 1)
            uncertainty_bonus += belief.uncertainty_map[y, x]
        end
    end
    
    return coverage + uncertainty_bonus
end

"""
calculate_exploration_bonus(targets::Vector{Tuple{Int, Int}}, belief::Belief)
Calculates exploration bonus for targets
"""
function calculate_exploration_bonus(targets::Vector{Tuple{Int, Int}}, belief::Belief)
    bonus = 0.0
    
    for target in targets
        x, y = target
        if 1 <= x <= size(belief.event_probabilities, 2) && 1 <= y <= size(belief.event_probabilities, 1)
            # Bonus for exploring areas with high uncertainty
            uncertainty = belief.uncertainty_map[y, x]
            bonus += uncertainty
        end
    end
    
    return bonus
end

"""
calculate_entropy(probability::Float64)
Calculates entropy for a probability value
"""
function calculate_entropy(probability::Float64)
    if probability <= 0.0 || probability >= 1.0
        return 0.0
    end
    return -(probability * log(probability) + (1 - probability) * log(1 - probability))
end

"""
evaluate_action(action::SensingAction, agent::Agent, env::SpatialGrid, state::GridState)
Evaluates the value of an action
"""
function evaluate_action(action::SensingAction, agent::Agent, env::SpatialGrid, state::GridState)
    # Calculate immediate reward
    next_state_dist = POMDPs.transition(env, state, action)
    next_state = rand(next_state_dist)
    immediate_reward = POMDPs.reward(env, state, action, next_state)
    
    return immediate_reward
end

end # module 
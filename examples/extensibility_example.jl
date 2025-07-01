#!/usr/bin/env julia

"""
Example demonstrating the extensibility of the modular MA-LOMDP system
Shows how to add new event states and planning strategies
"""

using POMDPs
using POMDPTools
using Random

# Include the modular system
include("../src/environment/event_dynamics.jl")
include("../src/environment/spatial_grid.jl")
include("../src/agents/belief_management.jl")
include("../src/agents/planning_strategies.jl")

using .BeliefManagement
using .PlanningModule

# ============================================================================
# EXAMPLE 1: Adding a new 3-state event system
# ============================================================================

"""
3-state event system for wildfire simulation
"""
@enum EventState3 begin
    NO_FIRE_3 = 0
    SMOLDERING_3 = 1
    BURNING_3 = 2
end

"""
DBN transition model for 3-state wildfire events
"""
struct WildfireDBNModel <: TransitionModel
    ignition_rate::Float64      # Rate of new fires starting
    spread_rate::Float64        # Rate of smoldering to burning
    extinction_rate::Float64    # Rate of fires going out
    wind_influence::Float64     # Wind influence on spread
end

"""
Transition probability for wildfire DBN
"""
function transition_probability_dbn(current_state::EventState3, neighbor_states::Vector{EventState3}, model::WildfireDBNModel)
    num_burning_neighbors = count(==(BURNING_3), neighbor_states)
    num_smoldering_neighbors = count(==(SMOLDERING_3), neighbor_states)
    
    if current_state == NO_FIRE_3
        # Probability of ignition from neighbors
        return min(1.0, model.ignition_rate + model.wind_influence * (num_burning_neighbors + 0.5 * num_smoldering_neighbors))
    elseif current_state == SMOLDERING_3
        # Probability of spreading to burning
        return min(1.0, model.spread_rate + model.wind_influence * num_burning_neighbors)
    else  # BURNING_3
        # Probability of extinction
        return model.extinction_rate
    end
end

"""
Update events for wildfire model
"""
function update_events!(model::WildfireDBNModel, event_map::Matrix{EventState3}, rng::AbstractRNG)
    height, width = size(event_map)
    new_map = similar(event_map)
    
    for y in 1:height
        for x in 1:width
            current = event_map[y, x]
            neighbors = get_neighbor_states(event_map, x, y)
            p = transition_probability_dbn(current, neighbors, model)
            
            if rand(rng) < p
                if current == NO_FIRE_3
                    new_map[y, x] = SMOLDERING_3
                elseif current == SMOLDERING_3
                    new_map[y, x] = BURNING_3
                else  # BURNING_3
                    new_map[y, x] = NO_FIRE_3
                end
            else
                new_map[y, x] = current
            end
        end
    end
    
    event_map .= new_map
end

# ============================================================================
# EXAMPLE 2: Adding a new planning strategy
# ============================================================================

"""
Risk-Aware Planning - Plans actions considering risk of events
"""
struct RiskAwarePlanning <: PlanningStrategy
    horizon::Int
    risk_threshold::Float64
    risk_weight::Float64
end

"""
Plan action using risk-aware strategy
"""
function plan_action(strategy::RiskAwarePlanning, agent::Agent, env::SpatialGrid, current_state::GridState)
    possible_targets = get_possible_targets(agent, current_state, env)
    
    best_action = nothing
    best_value = -Inf
    
    for targets in possible_targets
        action = SensingAction(agent.id, targets, false)
        
        # Calculate risk value
        risk_value = calculate_risk_value(targets, agent.belief, strategy.risk_threshold)
        
        # Calculate information gain
        info_gain = calculate_expected_information_gain(action, agent, env, current_state)
        
        # Combine risk and information
        total_value = info_gain + strategy.risk_weight * risk_value
        
        if total_value > best_value
            best_value = total_value
            best_action = action
        end
    end
    
    return best_action
end

"""
Calculate risk value for targets
"""
function calculate_risk_value(targets::Vector{Tuple{Int, Int}}, belief::Belief, threshold::Float64)
    risk_value = 0.0
    
    for target in targets
        x, y = target
        if 1 <= x <= size(belief.event_probabilities, 2) && 1 <= y <= size(belief.event_probabilities, 1)
            # Risk is high if probability of event is above threshold
            event_prob = belief.event_probabilities[y, x]
            if event_prob > threshold
                risk_value += event_prob
            end
        end
    end
    
    return risk_value
end

# ============================================================================
# EXAMPLE 3: Using the extended system
# ============================================================================

function demonstrate_extensibility()
    println("🚀 Demonstrating System Extensibility")
    println("=====================================")
    
    # Test the new 3-state wildfire system
    println("\n1. Testing 3-State Wildfire System:")
    
    wildfire_model = WildfireDBNModel(0.05, 0.3, 0.1, 0.2)
    wildfire_map = fill(NO_FIRE_3, 5, 5)
    wildfire_map[3, 3] = SMOLDERING_3  # Start with a smoldering fire
    
    println("Initial wildfire state: $(count(x -> x != NO_FIRE_3, wildfire_map)) active fires")
    
    # Update wildfire
    rng = Random.GLOBAL_RNG
    update_events!(wildfire_model, wildfire_map, rng)
    
    println("After wildfire update: $(count(x -> x != NO_FIRE_3, wildfire_map)) active fires")
    
    # Show wildfire map
    println("Wildfire map:")
    for y in 1:5
        for x in 1:5
            if wildfire_map[y, x] == NO_FIRE_3
                print(".")
            elseif wildfire_map[y, x] == SMOLDERING_3
                print("S")
            else
                print("B")
            end
        end
        println()
    end
    
    # Test the new risk-aware planning strategy
    println("\n2. Testing Risk-Aware Planning:")
    
    # Create environment and agent
    dynamics = EventDynamics(0.1, 0.05, 0.2, 0.1, 0.3)
    agent = create_agent(1, CircularTrajectory(5, 5, 3.0, 10), RangeLimitedSensor(2.0, π/2, 0.0), 10, 10)
    env = SpatialGrid(10, 10, dynamics, [agent], 3.0, 0.95, 3, 5)
    
    # Get initial state
    initial_dist = POMDPs.initialstate(env)
    initial_state = rand(initial_dist)
    
    # Create risk-aware strategy
    risk_strategy = RiskAwarePlanning(3, 0.7, 2.0)
    
    # Plan action
    action = plan_action(risk_strategy, agent, env, initial_state)
    
    if action !== nothing
        println("Risk-aware action planned: $(action.target_cells)")
        
        # Calculate risk value
        risk_value = calculate_risk_value(action.target_cells, agent.belief, 0.7)
        println("Risk value: $(risk_value)")
    end
    
    println("\n✅ Extensibility demonstration completed!")
    println("\n🎯 What we demonstrated:")
    println("- Added new 3-state wildfire event system")
    println("- Created custom DBN transition model")
    println("- Added new risk-aware planning strategy")
    println("- All integrated seamlessly with existing system")
end

# Run the demonstration
demonstrate_extensibility() 
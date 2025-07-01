#!/usr/bin/env julia

"""
Test script for the modular MA-LOMDP system
Demonstrates different event states, transition models, and planning strategies
"""

using Pkg
Pkg.activate(".")

using POMDPs
using POMDPTools
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Include the modular system files in correct order
include("../src/environment/event_dynamics.jl")
include("../src/environment/spatial_grid.jl")
include("../src/agents/belief_management.jl")
include("../src/agents/planning_strategies.jl")

using .BeliefManagement
using .PlanningModule

function test_2_state_system()
    println("=== Testing 2-State Event System ===")
    
    # Create 2-state DBN model
    dbn_model_2 = DBNTransitionModel2(0.1, 0.05, 0.3)
    
    # Create event map with 2-state events
    event_map_2 = fill(NO_EVENT_2, 5, 5)
    event_map_2[3, 3] = EVENT_PRESENT_2  # Add one event
    
    println("Initial 2-state events: $(count(==(EVENT_PRESENT_2), event_map_2))")
    
    # Update events using 2-state DBN
    rng = Random.GLOBAL_RNG
    update_events!(dbn_model_2, event_map_2, rng)
    
    println("2-state events after DBN update: $(count(==(EVENT_PRESENT_2), event_map_2))")
    
    # Test belief prediction for 2-state
    belief_2 = initialize_belief(5, 5, 0.3)
    neighbor_beliefs = [0.2, 0.4, 0.1, 0.3, 0.5, 0.2, 0.1, 0.4]
    predicted_belief = predict_next_belief_dbn(0.3, neighbor_beliefs, dbn_model_2)
    println("2-state belief prediction: $(predicted_belief)")
end

function test_4_state_system()
    println("\n=== Testing 4-State Event System ===")
    
    # Create 4-state DBN model
    dbn_model_4 = DBNTransitionModel4(0.1, 0.05, 0.2, 0.1, 0.3)
    
    # Create event map with 4-state events
    event_map_4 = fill(NO_EVENT_4, 5, 5)
    event_map_4[3, 3] = EVENT_PRESENT_4  # Add one event
    
    println("Initial 4-state events: $(count(x -> x != NO_EVENT_4, event_map_4))")
    
    # Update events using 4-state DBN
    rng = Random.GLOBAL_RNG
    update_events!(dbn_model_4, event_map_4, rng)
    
    println("4-state events after DBN update: $(count(x -> x != NO_EVENT_4, event_map_4))")
    
    # Show event map
    println("4-state event map after update:")
    for y in 1:5
        for x in 1:5
            if event_map_4[y, x] == NO_EVENT_4
                print(".")
            elseif event_map_4[y, x] == EVENT_PRESENT_4
                print("E")
            elseif event_map_4[y, x] == EVENT_SPREADING_4
                print("S")
            else
                print("D")
            end
        end
        println()
    end
end

function test_planning_strategies()
    println("\n=== Testing Different Planning Strategies ===")
    
    # Create environment with agents
    dynamics = EventDynamics(0.1, 0.05, 0.2, 0.1, 0.3)
    
    # Create agents with different planning strategies
    agent1 = create_agent(1, CircularTrajectory(5, 5, 3.0, 10), RangeLimitedSensor(2.0, π/2, 0.0), 10, 10)
    agent2 = create_agent(2, LinearTrajectory(1, 1, 10, 10, 15), RangeLimitedSensor(2.5, π/3, 0.0), 10, 10)
    
    agents = [agent1, agent2]
    
    env = SpatialGrid(10, 10, dynamics, agents, 3.0, 0.95, 3, 5)
    
    # Get initial state
    initial_dist = POMDPs.initialstate(env)
    initial_state = rand(initial_dist)
    
    # Test different planning strategies
    strategies = [
        InformationGainPlanning(3, 0.1),
        UncertaintyReductionPlanning(3, 0.5),
        CoveragePlanning(3, 1.0),
        MultiObjectivePlanning(3, 0.4, 0.3, 0.3)
    ]
    
    strategy_names = ["Information Gain", "Uncertainty Reduction", "Coverage", "Multi-Objective"]
    
    for (i, strategy) in enumerate(strategies)
        println("\n--- Testing $(strategy_names[i]) Planning ---")
        
        # Plan action for agent 1
        action = plan_action(strategy, agent1, env, initial_state)
        
        if action !== nothing
            println("Planned action: $(action.target_cells)")
            println("Number of targets: $(length(action.target_cells))")
            
            # Evaluate the action
            value = evaluate_action(action, agent1, env, initial_state)
            println("Action value: $(value)")
        else
            println("No action planned")
        end
    end
end

function test_belief_evolution()
    println("\n=== Testing Belief Evolution with DBN ===")
    
    # Create environment
    dynamics = EventDynamics(0.1, 0.05, 0.2, 0.1, 0.3)
    agent = create_agent(1, CircularTrajectory(5, 5, 3.0, 10), RangeLimitedSensor(2.0, π/2, 0.0), 5, 5)
    env = SpatialGrid(5, 5, dynamics, [agent], 3.0, 0.95, 2, 3)
    
    # Get initial belief
    initial_belief = agent.belief
    println("Initial belief mean: $(mean(initial_belief.event_probabilities))")
    println("Initial uncertainty mean: $(mean(initial_belief.uncertainty_map))")
    
    # Predict belief evolution over multiple steps
    predicted_belief = predict_belief_evolution_dbn(initial_belief, env, 5)
    
    println("Predicted belief after 5 steps:")
    println("- Belief mean: $(mean(predicted_belief.event_probabilities))")
    println("- Uncertainty mean: $(mean(predicted_belief.uncertainty_map))")
    println("- Last update: $(predicted_belief.last_update)")
end

function test_modular_event_states()
    println("\n=== Testing Modular Event State System ===")
    
    # Test that we can easily switch between different event state systems
    println("Available event state systems:")
    println("- EventState2: $(EventState2)")
    println("- EventState4: $(EventState4)")
    
    # Test conversion between systems
    println("\nTesting state conversions:")
    
    # Create 2-state event
    event_2 = EVENT_PRESENT_2
    println("2-state event: $(event_2)")
    
    # Create 4-state event
    event_4 = EVENT_PRESENT_4
    println("4-state event: $(event_4)")
    
    # Test transition probabilities
    dbn_2 = DBNTransitionModel2(0.1, 0.05, 0.3)
    dbn_4 = DBNTransitionModel4(0.1, 0.05, 0.2, 0.1, 0.3)
    
    neighbors_2 = [NO_EVENT_2, EVENT_PRESENT_2, NO_EVENT_2]
    neighbors_4 = [NO_EVENT_4, EVENT_PRESENT_4, NO_EVENT_4]
    
    prob_2 = transition_probability_dbn(EVENT_PRESENT_2, neighbors_2, dbn_2)
    prob_4 = transition_probability_dbn(EVENT_PRESENT_4, neighbors_4, dbn_4)
    
    println("2-state transition probability: $(prob_2)")
    println("4-state transition probability: $(prob_4)")
end

function test_legacy_compatibility()
    println("\n=== Testing Legacy Compatibility ===")
    
    # Test that legacy EventDynamics still works
    legacy_dynamics = EventDynamics(0.1, 0.05, 0.2, 0.1, 0.3)
    
    # Convert to new DBN models
    dbn_2 = DBNTransitionModel2(legacy_dynamics)
    dbn_4 = DBNTransitionModel4(legacy_dynamics)
    
    println("Legacy dynamics converted to:")
    println("- DBN 2-state: $(dbn_2)")
    println("- DBN 4-state: $(dbn_4)")
    
    # Test that legacy functions still work with simple event map
    event_map = fill(0, 5, 5)  # Use simple integer array instead of EventState
    event_map[3, 3] = 1  # Set one event
    
    rng = Random.GLOBAL_RNG
    update_events!(legacy_dynamics, event_map, rng)
    
    println("Legacy update successful: $(count(x -> x != 0, event_map)) events")
end

function main()
    println("🧠 Testing Modular MA-LOMDP System")
    println("=====================================")
    
    test_2_state_system()
    test_4_state_system()
    test_planning_strategies()
    test_belief_evolution()
    test_modular_event_states()
    test_legacy_compatibility()
    
    println("\n✅ All modular system tests completed!")
    println("\n🎯 System Features Demonstrated:")
    println("- Multiple event state systems (2-state, 4-state)")
    println("- DBN-based transition models")
    println("- Different planning strategies")
    println("- Belief evolution with DBN")
    println("- Legacy compatibility")
    println("- Easy extensibility for new event states and planning methods")
end

# Run the tests
main() 
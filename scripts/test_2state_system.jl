#!/usr/bin/env julia

"""
Simple test script for the 2-state MA-LOMDP system
Focuses on basic functionality without complex dependencies
"""

using Pkg
Pkg.activate(".")

using POMDPs
using POMDPTools
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Include only the essential files for 2-state system
include("../src/environment/event_dynamics.jl")

function test_2state_dbn()
    println("=== Testing 2-State DBN System ===")
    
    # Create 2-state DBN model
    dbn_model = DBNTransitionModel2(0.1, 0.05, 0.3)
    println("✓ Created DBN model: $(dbn_model)")
    
    # Create event map with 2-state events
    event_map = fill(NO_EVENT_2, 5, 5)
    event_map[3, 3] = EVENT_PRESENT_2  # Add one event
    
    println("Initial 2-state events: $(count(==(EVENT_PRESENT_2), event_map))")
    
    # Update events using 2-state DBN
    rng = Random.GLOBAL_RNG
    update_events!(dbn_model, event_map, rng)
    
    println("2-state events after DBN update: $(count(==(EVENT_PRESENT_2), event_map))")
    
    # Show event map
    println("2-state event map after update:")
    for y in 1:5
        for x in 1:5
            if event_map[y, x] == NO_EVENT_2
                print(".")
            else
                print("E")
            end
        end
        println()
    end
end

function test_2state_transition_probabilities()
    println("\n=== Testing 2-State Transition Probabilities ===")
    
    # Create DBN model
    dbn_model = DBNTransitionModel2(0.1, 0.05, 0.3)
    
    # Test different scenarios
    scenarios = [
        ("No neighbors", [NO_EVENT_2, NO_EVENT_2, NO_EVENT_2]),
        ("One active neighbor", [NO_EVENT_2, EVENT_PRESENT_2, NO_EVENT_2]),
        ("Multiple active neighbors", [EVENT_PRESENT_2, EVENT_PRESENT_2, NO_EVENT_2])
    ]
    
    for (name, neighbors) in scenarios
        # Test transition from NO_EVENT
        prob_no_event = transition_probability_dbn(NO_EVENT_2, neighbors, dbn_model)
        println("$(name) - NO_EVENT → EVENT_PRESENT: $(prob_no_event)")
        
        # Test transition from EVENT_PRESENT
        prob_event = transition_probability_dbn(EVENT_PRESENT_2, neighbors, dbn_model)
        println("$(name) - EVENT_PRESENT → EVENT_PRESENT: $(prob_event)")
        println()
    end
end

function test_2state_belief_prediction()
    println("\n=== Testing 2-State Belief Prediction ===")
    
    # Create DBN model
    dbn_model = DBNTransitionModel2(0.1, 0.05, 0.3)
    
    # Test belief prediction
    current_belief = 0.3  # 30% probability of event
    neighbor_beliefs = [0.2, 0.4, 0.1, 0.3, 0.5, 0.2, 0.1, 0.4]
    
    predicted_belief = predict_next_belief_dbn(current_belief, neighbor_beliefs, dbn_model)
    
    println("Current belief: $(current_belief)")
    println("Neighbor beliefs: $(neighbor_beliefs)")
    println("Predicted belief: $(predicted_belief)")
    println("Change: $(predicted_belief - current_belief)")
end

function test_2state_legacy_compatibility()
    println("\n=== Testing 2-State Legacy Compatibility ===")
    
    # Test that legacy EventDynamics still works
    legacy_dynamics = EventDynamics(0.1, 0.05, 0.2, 0.1, 0.3)
    
    # Convert to new DBN model
    dbn_2 = DBNTransitionModel2(legacy_dynamics)
    
    println("Legacy dynamics: $(legacy_dynamics)")
    println("Converted to DBN 2-state: $(dbn_2)")
    
    # Test that legacy functions still work with simple event map
    event_map = fill(0, 5, 5)  # Use simple integer array
    event_map[3, 3] = 1  # Set one event
    
    rng = Random.GLOBAL_RNG
    update_events!(legacy_dynamics, event_map, rng)
    
    println("Legacy update successful: $(count(x -> x != 0, event_map)) events")
end

function main()
    println("🧠 Testing 2-State MA-LOMDP System")
    println("===================================")
    
    test_2state_dbn()
    test_2state_transition_probabilities()
    test_2state_belief_prediction()
    test_2state_legacy_compatibility()
    
    println("\n✅ All 2-state system tests completed!")
    println("\n🎯 2-State System Features Demonstrated:")
    println("- DBN-based transition model")
    println("- Transition probability calculations")
    println("- Belief prediction using DBN")
    println("- Legacy compatibility")
    println("- Simple and efficient 2-state event system")
end

# Run the tests
main() 
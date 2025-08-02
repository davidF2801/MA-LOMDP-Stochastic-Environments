# Simple test for evolve_no_obs vs evolve_no_obs_fast
# This test directly includes the necessary files without loading the full project

# Add the src directory to the path
push!(LOAD_PATH, "../src")

# Include the necessary modules directly
include("../src/types/Types.jl")
include("../src/agents/Agents.jl")
include("../src/environment/Environment.jl")

using .Types
using .Environment
using .Agents.BeliefManagement

# Import specific functions from Types
import .Types: create_uniform_rsp_maps, RSPParameterMaps

# Import the fast function from BeliefManagement
import .Agents.BeliefManagement: evolve_no_obs_fast

# Create a non-cached version for testing
function evolve_no_obs_fast_no_cache(B::Belief, env; calculate_uncertainty::Bool=true)
    # Create new belief with evolved distributions
    new_distributions = similar(B.event_distributions)
    num_states, height, width = size(new_distributions)
    @assert num_states == 2 "fast version currently supports 2 states"

    # Add some work to verify it's actually running
    total_work = 0.0

    for y in 1:height, x in 1:width
        current_belief = B.event_distributions[:, y, x]
        neighbor_beliefs = BeliefManagement.get_neighbor_beliefs(B, x, y)

        # Get cell-specific parameters
        cell_params = Types.get_cell_rsp_params(env.rsp_params, y, x)

        # Compute expected fraction of active neighbours
        active_probs = [nb[2] for nb in neighbor_beliefs]  # P(EVENT) for each neighbor
        norm_active = isempty(active_probs) ? 0.0 : mean(active_probs)

        # Precompute contagion once
        contagion = 1 - exp(-cell_params.alpha * norm_active)

        # Update distribution for this cell
        # Case current_state = 0 (NO_EVENT)
        p_event_from_no = (1 - exp(-(cell_params.beta0 + cell_params.lambda + contagion)))
        p_no_from_no    = 1 - p_event_from_no

        # Case current_state = 1 (EVENT)
        p_event_from_event = cell_params.delta
        p_no_from_event    = 1 - cell_params.delta

        # Combine with current belief
        new_distributions[2, y, x] = current_belief[1] * p_event_from_no +
                                     current_belief[2] * p_event_from_event
        new_distributions[1, y, x] = current_belief[1] * p_no_from_no +
                                     current_belief[2] * p_no_from_event
        
        # Add some work to verify computation
        total_work += new_distributions[1, y, x] + new_distributions[2, y, x]
    end

    # Normalize distributions
    new_distributions = BeliefManagement.normalize_belief_distributions(new_distributions)

    # Calculate uncertainty map only if requested
    uncertainty_map = calculate_uncertainty ?
        BeliefManagement.calculate_uncertainty_map_from_distributions(new_distributions) :
        similar(B.uncertainty_map)

    evolved_belief = Belief(new_distributions, uncertainty_map, B.last_update + 1, B.history)
    
    # Print work done to verify function is running
    println("    Fast function computed total_work: $total_work")
    
    return evolved_belief
end, evolve_no_obs

# Create a simple test environment
function create_test_env()
    # Create a simple 3x3 environment with minimal parameters
    width, height = 3, 3
    event_dynamics = EventDynamics(0.1, 0.2, 0.3, 0.1, 0.5)  # birth_rate, death_rate, spread_rate, decay_rate, neighbor_influence
    agents = Agent[]  # Empty agents for this test
    sensor_range = 2.0
    discount = 0.95
    initial_events = 1
    max_sensing_targets = 1
    ground_station_pos = (1, 1)
    rsp_params = create_uniform_rsp_maps(3, 3, lambda=0.1, beta0=0.2, alpha=0.5, delta=0.3)
    
    env = SpatialGrid(width, height, event_dynamics, agents, sensor_range, discount, initial_events, max_sensing_targets, ground_station_pos, rsp_params)
    return env
end

# Create a test belief
function create_test_belief()
    # 2-state belief (NO_EVENT=0, EVENT_PRESENT=1)
    event_distributions = zeros(2, 3, 3)
    
    # Set some initial probabilities
    event_distributions[1, :, :] .= 0.8  # 80% probability of NO_EVENT
    event_distributions[2, :, :] .= 0.2  # 20% probability of EVENT_PRESENT
    
    # Normalize
    for x in 1:3, y in 1:3
        total = sum(event_distributions[:, y, x])
        if total > 0
            event_distributions[:, y, x] ./= total
        end
    end
    
    # Create uncertainty map
    uncertainty_map = calculate_uncertainty_map_from_distributions(event_distributions)
    
    return Belief(event_distributions, uncertainty_map, 0, [])
end

# Test function
function test_evolve_functions()
    println("🧪 Testing evolve_no_obs vs evolve_no_obs_fast (10 iterations)")
    
    # Create test environment and belief
    env = create_test_env()
    belief = create_test_belief()
    
    println("Initial belief shape: $(size(belief.event_distributions))")
    println("Initial event probabilities:")
    println(belief.event_distributions[2, :, :])
    
    # Clear cache to ensure fair comparison
    clear_belief_evolution_cache!()
    
    # Test original function - 100 iterations for better timing
    println("\n📊 Testing evolve_no_obs (100 iterations)...")
    start_time = time()
    result_original = deepcopy(belief)
    for i in 1:100
        result_original = evolve_no_obs(result_original, env, calculate_uncertainty=true)
    end
    original_time = time() - start_time
    
    # Clear cache again and create a fresh belief to avoid cache effects
    clear_belief_evolution_cache!()
    belief_fresh = deepcopy(belief)
    
    # Test fast function - 100 iterations (no cache)
    println("📊 Testing evolve_no_obs_fast (100 iterations)...")
    start_time = time()
    result_fast = deepcopy(belief_fresh)
    for i in 1:100
        if i % 10 == 0
            println("  Iteration $i...")
        end
        iter_start = time()
        result_fast = evolve_no_obs_fast(result_fast, env, calculate_uncertainty=true)
        iter_time = time() - iter_start
        if i % 10 == 0
            println("    Iteration $i took: $(round(iter_time * 1000, digits=6)) ms")
        end
    end
    fast_time = time() - start_time
    
    # Compare results
    println("\n🔍 Results comparison:")
    println("Original time: $(round(original_time * 1000, digits=6)) ms")
    println("Fast time: $(round(fast_time * 1000, digits=6)) ms")
    println("Speedup: $(round(original_time / fast_time, digits=2))x")
    
    # Also test with @elapsed for more precision
    println("\n📊 Precise timing with @elapsed:")
    clear_belief_evolution_cache!()
    belief_test = deepcopy(belief)
    original_elapsed = @elapsed for i in 1:100
        belief_test = evolve_no_obs(belief_test, env, calculate_uncertainty=true)
    end
    
    clear_belief_evolution_cache!()
    belief_test = deepcopy(belief)
    fast_elapsed = @elapsed for i in 1:100
        belief_test = evolve_no_obs_fast(belief_test, env, calculate_uncertainty=true)
    end
    
    println("Original @elapsed: $(round(original_elapsed * 1000, digits=6)) ms")
    println("Fast @elapsed: $(round(fast_elapsed * 1000, digits=6)) ms")
    println("Speedup @elapsed: $(round(original_elapsed / fast_elapsed, digits=2))x")
    
    # Compare event distributions
    dist_diff = abs.(result_original.event_distributions - result_fast.event_distributions)
    max_dist_diff = maximum(dist_diff)
    mean_dist_diff = mean(dist_diff)
    
    println("\n📈 Distribution differences:")
    println("Max difference: $(round(max_dist_diff, digits=6))")
    println("Mean difference: $(round(mean_dist_diff, digits=6))")
    
    # Compare uncertainty maps
    uncert_diff = abs.(result_original.uncertainty_map - result_fast.uncertainty_map)
    max_uncert_diff = maximum(uncert_diff)
    mean_uncert_diff = mean(uncert_diff)
    
    println("\n📊 Uncertainty map differences:")
    println("Max difference: $(round(max_uncert_diff, digits=6))")
    println("Mean difference: $(round(mean_uncert_diff, digits=6))")
    
    # Check if results are essentially identical
    tolerance = 1e-06
    identical = max_dist_diff < tolerance && max_uncert_diff < tolerance
    
    if identical
        println("\n✅ SUCCESS: Functions produce identical results!")
    else
        println("\n❌ WARNING: Functions produce different results!")
        println("Original event probs:")
        println(result_original.event_distributions[2, :, :])
        println("Fast event probs:")
        println(result_fast.event_distributions[2, :, :])
    end
    
    return identical, original_time, fast_time
end

# Run the test
println("🚀 Running evolve function comparison test...")
identical, orig_time, fast_time = test_evolve_functions()

println("\n" * "="^50)
println("FINAL SUMMARY")
println("="^50)
println("Functions identical: $(identical ? "✅ YES" : "❌ NO")")
println("Speedup: $(round(orig_time / fast_time, digits=2))x")
println("Original time: $(round(orig_time * 1000, digits=2)) ms")
println("Fast time: $(round(fast_time * 1000, digits=2)) ms") 
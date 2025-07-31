module PolicyTreePlanner

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Infiltrator
using Distributions
using Base: rand
using ..Types

# Constants for MCTS
const MCTS_NUM_SIMULATIONS = 5000  # Increased for better coverage
const MCTS_UCB_CONSTANT = 1.4

"""
standardize_observation_key(obs)
Convert any observation format to a standardized key for policy tree lookup
"""
function standardize_observation_key(obs)
    if obs isa Vector{Tuple{Tuple{Int,Int}, EventState}}
        # Already in the correct format
        return obs
    elseif hasfield(typeof(obs), :sensed_cells) && hasfield(typeof(obs), :event_states)
        # Convert GridObservation to standardized format
        standardized = Vector{Tuple{Tuple{Int,Int}, EventState}}()
        for (i, cell) in enumerate(obs.sensed_cells)
            if i <= length(obs.event_states)
                push!(standardized, (cell, obs.event_states[i]))
            end
        end
        return standardized
    else
        # Fallback: try to convert to the expected format
        error("Unknown observation format: $(typeof(obs))")
    end
end

"""
find_closest_observation_key(obs, available_keys)
Find the closest matching observation key from available keys using progressive widening
"""
function find_closest_observation_key(obs, available_keys)
    if isempty(available_keys)
        return nothing
    end
    
    obs_standardized = standardize_observation_key(obs)
    
    # Simple matching: find exact match first
    for key in available_keys
        if obs_standardized == key
            return key
        end
    end
    
    # Progressive widening: find key with most similar cells
    best_match = nothing
    best_similarity = 0.0
    
    for key in available_keys
        similarity = calculate_observation_similarity(obs_standardized, key)
        if similarity > best_similarity
            best_similarity = similarity
            best_match = key
        end
    end
    
    # Only return if similarity is above threshold
    return best_similarity > 0.5 ? best_match : nothing
end

"""
calculate_observation_similarity(obs1, obs2)
Calculate similarity between two observations based on cell overlap
"""
function calculate_observation_similarity(obs1, obs2)
    cells1 = Set(cell for (cell, _) in obs1)
    cells2 = Set(cell for (cell, _) in obs2)
    
    intersection = length(cells1 ∩ cells2)
    union_size = length(cells1 ∪ cells2)
    
    return union_size > 0 ? intersection / union_size : 0.0
end

"""
validate_policy_completeness(policy_forest, agent_id)
Validate that the policy tree covers expected observation space
"""
function validate_policy_completeness(policy_forest, agent_id)
    total_coverage = 0
    total_policies = length(policy_forest)
    
    for (weight, policy) in policy_forest
        if !isempty(policy)
            total_coverage += weight
        end
    end
    
    coverage_rate = total_coverage / total_policies
    if coverage_rate < 0.8
        @warn "Policy tree coverage is low: $(round(coverage_rate * 100, digits=1))%. Consider increasing MCTS iterations."
    else
        println("✅ Policy tree coverage: $(round(coverage_rate * 100, digits=1))%")
    end
end

"""
visualize_policy_tree(policy_forest, agent_id, max_depth=3)
Visualize the policy tree structure for debugging
"""
function visualize_policy_tree(policy_forest, agent_id, max_depth=3)
    println("🌳 Policy Tree Visualization for Agent $(agent_id)")
    println("=" ^ 50)
    
    for (i, (weight, policy)) in enumerate(policy_forest)
        println("📊 Policy $(i) (weight: $(round(weight, digits=3)))")
        visualize_policy_recursive(policy, "", 0, max_depth)
        println()
    end
end

"""
visualize_policy_recursive(policy, prefix, depth, max_depth)
Recursively visualize policy tree structure
"""
function visualize_policy_recursive(policy, prefix, depth, max_depth)
    if depth >= max_depth
        println("$(prefix)└── [MAX DEPTH REACHED]")
        return
    end
    
    if !isa(policy, Dict) || isempty(policy)
        println("$(prefix)└── [EMPTY POLICY]")
        return
    end
    
    for (action, observation_subtrees) in policy
        action_str = isempty(action.target_cells) ? "WAIT" : "SENSE$(action.target_cells)"
        println("$(prefix)├── Action: $(action_str)")
        
        if isa(observation_subtrees, Dict) && !isempty(observation_subtrees)
            for (obs_key, subtree) in observation_subtrees
                obs_str = string(obs_key)
                if length(obs_str) > 30
                    obs_str = obs_str[1:27] * "..."
                end
                println("$(prefix)│   ├── Obs: $(obs_str)")
                visualize_policy_recursive(subtree, "$(prefix)│   │   ", depth + 1, max_depth)
            end
        else
            println("$(prefix)│   └── [NO SUBTREE]")
        end
    end
end

"""
print_policy_statistics(policy_forest, agent_id)
Print statistics about the policy tree
"""
function print_policy_statistics(policy_forest, agent_id)
    println("📊 Policy Tree Statistics for Agent $(agent_id)")
    println("-" ^ 40)
    
    total_actions = 0
    total_observations = 0
    max_depth = 0
    
    for (weight, policy) in policy_forest
        if !isempty(policy)
            stats = analyze_policy_recursive(policy, 0)
            total_actions += stats.actions * weight
            total_observations += stats.observations * weight
            max_depth = max(max_depth, stats.max_depth)
        end
    end
    
    println("  📈 Total Actions: $(round(total_actions, digits=1))")
    println("  📈 Total Observations: $(round(total_observations, digits=1))")
    println("  📈 Max Depth: $(max_depth)")
    coverage_sum = sum(w for (w, p) in policy_forest if !isempty(p))
    println("  📈 Coverage: $(round(coverage_sum * 100, digits=1))%")
end

"""
analyze_policy_recursive(policy, depth)
Analyze policy tree recursively and return statistics
"""
function analyze_policy_recursive(policy, depth)
    if !isa(policy, Dict) || isempty(policy)
        return (actions=0, observations=0, max_depth=depth)
    end
    
    total_actions = length(policy)
    total_observations = 0
    max_depth = depth
    
    for (action, observation_subtrees) in policy
        if isa(observation_subtrees, Dict)
            total_observations += length(observation_subtrees)
            for (obs_key, subtree) in observation_subtrees
                subtree_stats = analyze_policy_recursive(subtree, depth + 1)
                max_depth = max(max_depth, subtree_stats.max_depth)
            end
        end
    end
    
    return (actions=total_actions, observations=total_observations, max_depth=max_depth)
end
import ..Agents.BeliefManagement: sample_from_belief, evolve_no_obs, collapse_belief_to, enumerate_all_possible_outcomes, merge_equivalent_beliefs, calculate_cell_entropy, clear_belief_evolution_cache!, Belief
import ..Types: check_battery_feasible, simulate_battery_evolution
# Import belief branching function from MacroPlannerAsync
import ..MacroPlannerAsync: precompute_belief_branches
# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..EventState2, ..NO_EVENT_2, ..EVENT_PRESENT_2
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..ComplexTrajectory, ..RangeLimitedSensor, ..EventMap
# Import trajectory functions
import ..Agents.TrajectoryPlanner.get_position_at_time
# Import DBN functions for transition modeling
import ..Environment.EventDynamicsModule.DBNTransitionModel2, ..Environment.EventDynamicsModule.predict_next_belief_dbn

export plan_mcts, extract_policy_forest, π_i, calculate_policy_tree, best_policy_tree, execute_reactive_policy

mutable struct SNode
    belief::Belief
    t::Int
    depth::Int
    p_mass::Float64
    visits::Int
    total_value::Float64
    children::Dict{SensingAction, Any}  # Use Any to avoid circular dependency
end

mutable struct ANode
    action::SensingAction
    visits::Int
    total_value::Float64
    outcomes::Dict{Vector{Tuple{Tuple{Int,Int}, EventState}}, SNode}
end

function calculate_policy_tree(env, initial_beliefs::Vector{Tuple{Belief, Float64}}, agent::Agent, other_policies, gs_state, horizon::Int; n_iterations=5000, c_ucb=1.4)
    clear_belief_evolution_cache!()

    root_nodes = [SNode(belief, gs_state.time_step, 0, p, 0, 0.0, Dict()) for (belief, p) in initial_beliefs]

    for iter in 1:n_iterations
        root = sample(root_nodes, Weights([node.p_mass for node in root_nodes]))
        simulate(env, root, agent, other_policies, gs_state, horizon, c_ucb)
    end

    policy_tree = Dict()
    for root in root_nodes
        policy_tree[root] = extract_policy_from_node(root)
    end

    return policy_tree
end

function simulate(env, node::SNode, agent::Agent, gs_state, horizon, c_ucb)
    # Strict depth limit enforcement
    if node.depth >= horizon
        return 0.0
    end
    
    # Check battery feasibility - if agent can't perform any actions, return low reward
    if agent.battery_level <= 0.0
        println("    🔋 Battery depleted: $(agent.battery_level)")
        return -100.0  # Penalty for running out of battery
    end

    # Step 1: Expand node if not expanded
    if isempty(node.children)
        actions_sequences = generate_action_sequences(agent, env, 1, node.t)
        actions = isempty(actions_sequences) ? SensingAction[] : actions_sequences[1]
        for a in actions
            node.children[a] = ANode(a, 0, 0.0, Dict())
        end
    end

    # Step 2: Select action deterministically to build complete tree
    best_action_node = nothing
    
    # First, try unvisited actions
    for (action, a_node) in node.children
        a_node = a_node::ANode  # Type assertion
        if a_node.visits == 0
            best_action_node = a_node
                break
            end
        end
        
    # If all actions visited, select the one with highest value
    if best_action_node === nothing
        best_value = -Inf
        for (action, a_node) in node.children
            a_node = a_node::ANode  # Type assertion
            if a_node.visits > 0
                value = a_node.total_value / a_node.visits
                if value > best_value
                    best_value = value
                    best_action_node = a_node
                end
            end
        end
    end
    
    if best_action_node === nothing
        println("    ❌ No action node selected - this should not happen!")
        return 0.0
    end
    
    # Step 3: Enumerate all possible joint observation outcomes
    obs_set = [(agent.id, best_action_node.action)]
    for (j, plan) in gs_state.agent_plans
        if j != agent.id && plan !== nothing && gs_state.agent_last_sync[j] <= node.t
            # Check if this agent has a reactive policy
            other_agent = nothing
            for a in env.agents
                if a.id == j
                    other_agent = a
                break
            end
        end
        
            if other_agent !== nothing && other_agent.reactive_policy !== nothing
                # Use reactive policy to get action
                # For now, use empty observation history (can be enhanced later)
                action_j = other_agent.reactive_policy(GridObservation[])
                push!(obs_set, (j, action_j))
            else
                # Fallback to fixed sequence
                plan_timestep = node.t - gs_state.agent_last_sync[j] + 1
                if 1 <= plan_timestep <= length(plan)
                    action_j = plan[plan_timestep]
                    push!(obs_set, (j, action_j))
                end
            end
        end
    end
    
    outcomes = enumerate_all_possible_outcomes(node.belief, obs_set)

    # Step 4: Expand chance nodes if necessary
    if isempty(best_action_node.outcomes)
        for (obs_combo, prob) in outcomes
            b_evolved = evolve_no_obs(node.belief, env)
            b_next = deepcopy(b_evolved)
            for (cell, state) in obs_combo
                b_next = collapse_belief_to(b_next, cell, state)
            end
            
            # Extract LOCAL observations for this agent (this is the key fix!)
            local_obs = Vector{Tuple{Tuple{Int,Int}, EventState}}()
            for (cell, state) in obs_combo
                # Check if this observation is from our agent's action
                if any(cell == target_cell for target_cell in best_action_node.action.target_cells)
                    push!(local_obs, (cell, state))
                end
            end
            
            # Store using LOCAL observations as the key
            best_action_node.outcomes[local_obs] = SNode(b_next, node.t+1, node.depth+1, 
                                                         node.p_mass*prob, 0, 0.0, Dict())
        end
    end
    
    # Step 5: Enumerate all outcomes and simulate recursively
    if !isempty(best_action_node.outcomes)
        # Instead of sampling, enumerate all outcomes to build complete tree
        total_reward = 0.0
        total_prob = 0.0
        
        for (local_obs, child_node) in best_action_node.outcomes
            # Compute reward for this outcome
            immediate_reward = compute_expected_reward([(node.belief, 1.0)], 
                                                       best_action_node.action, 0.0)
            
            # Simulate battery evolution for this action
            old_battery = agent.battery_level
            agent.battery_level = simulate_battery_evolution(agent, best_action_node.action, agent.battery_level)
            
            future_reward = simulate(env, child_node, agent, gs_state, horizon, c_ucb)
            
            # Restore battery level for this simulation
            agent.battery_level = old_battery
            
            # Weight by probability of this outcome
            outcome_reward = immediate_reward + env.discount * future_reward
            total_reward += child_node.p_mass * outcome_reward
            total_prob += child_node.p_mass
        end
        
        # Normalize by total probability
        if total_prob > 0
            total_reward /= total_prob
        end
        
        # Step 6: Backpropagation
        node.visits += 1
        node.total_value += total_reward
        best_action_node.visits += 1
        best_action_node.total_value += total_reward
        
        return total_reward
    else
        # No outcomes available, return immediate reward only
        # Use the UNCCOLLAPSED belief for reward computation (before observation)
        immediate_reward = compute_expected_reward([(node.belief, node.p_mass)], 
                                                   best_action_node.action, 0.0)
        
        # Still need to update visits even if no outcomes
        node.visits += 1
        node.total_value += immediate_reward
        best_action_node.visits += 1
        best_action_node.total_value += immediate_reward
        
        return immediate_reward
        end
    end
    
function extract_policy_from_node(node::SNode)
    if isempty(node.children)
        return Dict()
    end
    
    # Find the best action based on visit counts and values
    best_action = nothing
    best_value = -Inf
    
    for (action, a_node) in node.children
        a_node = a_node::ANode  # Type assertion
        if a_node.visits > 0
            value = a_node.total_value / a_node.visits
            if value > best_value
                best_value = value
                best_action = action
            end
        end
    end
    
    if best_action === nothing
        return Dict()
    end
    
    # Create policy subtree for the best action only
    best_a_node = node.children[best_action]::ANode
    policy_subtree = Dict()
    
    for (local_obs, s_node) in best_a_node.outcomes
        # The local_obs are already filtered to only include this agent's observations
        # Recursively extract policy for this outcome
        subtree_policy = extract_policy_from_node(s_node)
        policy_subtree[local_obs] = subtree_policy
    end
    
    result = Dict(best_action => policy_subtree)
    return result
end

"""
generate_action_sequences(agent, env, C, current_timestep)
Generate all possible action sequences of length C for an agent at current timestep
"""
function generate_action_sequences(agent, env, C::Int, current_timestep::Int=0)
    if C == 0
        return Vector{SensingAction}[]
    end
    
    # Get current position at the current timestep
    current_pos = get_position_at_time(agent.trajectory, current_timestep, agent.phase_offset)
    
    # Get field of regard
    for_cells = get_field_of_regard_at_position(agent, current_pos, env)
    
    # Generate single actions
    actions = SensingAction[]
    
    # Add wait action (always feasible)
    push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))
    
    # Add single-cell sensing actions (check battery)
    for cell in for_cells
        action = SensingAction(agent.id, [cell], false)
        if check_battery_feasible(agent, action, agent.battery_level)
            push!(actions, action)
        end
    end
    
    # Add multi-cell sensing actions (check battery)
    if length(for_cells) > 1 && env.max_sensing_targets > 1
        for subset_size in 2:min(env.max_sensing_targets, length(for_cells))
            for subset in combinations(for_cells, subset_size)
                action = SensingAction(agent.id, collect(subset), false)
                if check_battery_feasible(agent, action, agent.battery_level)
                    push!(actions, action)
                end
            end
        end
    end
    
    return [actions]
end

"""
compute_expected_reward(belief_branches, action, c_obs)
Compute expected reward for an action given belief branches
"""
function compute_expected_reward(belief_branches, action::SensingAction, c_obs::Float64)
    expected_reward = 0.0
    
    for (belief, p_branch) in belief_branches
        if isempty(action.target_cells)
            # Wait action - no reward from sensing
        else
        for cell in action.target_cells
            H_before = calculate_cell_entropy(belief, cell)
                # Simplified: assume perfect observation reduces entropy to 0
                H_after = 0.0
            info_gain = H_before - H_after
                # Weight by event probability to prioritize high-probability cells
                weighted_gain = info_gain
                cell_reward = p_branch * weighted_gain
                expected_reward += cell_reward
            end
        end
        
        if !isempty(action.target_cells)
            cost = p_branch * c_obs
            expected_reward -= cost
        end
    end
    
    return expected_reward
end

"""
sample(collection, weights)
Sample from a collection using weights
"""
function sample(collection, weights)
    if isempty(collection)
        return nothing
    end
    # Convert Weights to Vector for Categorical constructor
    weights_vector = weights.weights
    idx = Random.rand(Categorical(weights_vector))
    return collection[idx]
end

"""
argmax(f, collection)
Find the element that maximizes function f
"""
function argmax(f, collection)
    if isempty(collection)
        return nothing
    end
    
    best = first(collection)
    best_val = f(best)
    
    for item in collection
        val = f(item)
        if val > best_val
            best = item
            best_val = val
        end
    end
    
    return best
end

"""
Weights(weights)
Create a weights object for sampling
"""
struct Weights{T<:Real}
    weights::Vector{T}
end

Base.length(w::Weights) = length(w.weights)
Base.getindex(w::Weights, i) = w.weights[i]

"""
Categorical(weights)
Create a categorical distribution
"""
struct Categorical{T<:Real}
    weights::Vector{T}
end

function Random.rand(cat::Categorical)
    total = sum(cat.weights)
    if total == 0
        return Random.rand(1:length(cat.weights))
    end
    
    r = Random.rand() * total
    cumsum = 0.0
    for (i, w) in enumerate(cat.weights)
        cumsum += w
        if r <= cumsum
            return i
        end
    end
    return length(cat.weights)
end

"""
get_field_of_regard_at_position(agent, position, env)
Get field of regard for agent at position
"""
function get_field_of_regard_at_position(agent::Agent, position::Tuple{Int, Int}, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Check sensor pattern
    if agent.sensor.pattern == :cross
        # Cross-shaped sensor: agent's position and adjacent cells
        ax, ay = position
        for dx in -1:1, dy in -1:1
            nx, ny = ax + dx, ay + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                # Only include cross pattern (not diagonal)
                if (dx == 0 && dy == 0) || (dx == 0 && dy != 0) || (dx != 0 && dy == 0)
                    push!(fov_cells, (nx, ny))
                end
            end
        end
    elseif agent.sensor.pattern == :row_only || agent.sensor.range == 0.0
        # Row-only visibility: agent can only see cells in its current row
        for nx in 1:env.width
            push!(fov_cells, (nx, y))
        end
    else
        # Standard sensor range visibility
        sensor_range = round(Int, agent.sensor.range)
        for dx in -sensor_range:sensor_range
            for dy in -sensor_range:sensor_range
            nx, ny = x + dx, y + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                    # Check if within sensor range
                    distance = sqrt(dx^2 + dy^2)
                    if distance <= agent.sensor.range
                push!(fov_cells, (nx, ny))
                    end
                end
            end
        end
    end
    return fov_cells
end

# Import utility functions
import ..Types: combinations

# Global constants for MCTS
const γ = 0.95  # discount factor
const c_ucb = 1.414  # UCB exploration constant
const depth_limit = 20  # maximum simulation depth

# MCTS Planning Parameters
const MCTS_NUM_SIMULATIONS = 1000  # Number of MCTS simulations
const MCTS_MAX_DEPTH = 10  # Maximum depth for MCTS tree
const MCTS_UCB_CONSTANT = 0.5  # UCB exploration constant (reduced for more exploitation)
const MCTS_DISCOUNT_FACTOR = 0.95  # Discount factor for future rewards

"""
plan_mcts(root_set, N_sim, agent_id, env, gs_state)
Main entry point for MCTS planning (original implementation)
"""
function plan_mcts(root_set::Vector{Tuple{Belief, Float64}}, N_sim::Int, agent_id::Int, env, gs_state)
    values = Float64[]
    t_clean = gs_state.time_step
    
    for (B0, P0) in root_set
        root = SNode(B0, t_clean, 0, P0, 0, 0.0, Dict())
        for sim in 1:N_sim
            simulate_mcts(root, agent_id, env, gs_state)
        end
        push!(values, root.total_value / max(root.visits, 1))
    end
    
    return sum(Pℓ * values[ℓ] for (ℓ, (_, Pℓ)) in enumerate(root_set))
end

"""
simulate_mcts(node::SNode, agent_id::Int, env, gs_state)
MCTS simulation function
"""
function simulate_mcts(node::SNode, agent_id::Int, env, gs_state)
    if node.depth >= MCTS_MAX_DEPTH
        return 0.0
    end
    
    if isempty(node.children)
        # Expand node
        # Find agent by ID
        agent = nothing
        for a in env.agents
            if a.id == agent_id
                agent = a
                break
            end
        end
        
        if agent !== nothing
            actions = generate_action_sequences(agent, env, 1, node.t)[1]
            for a in actions
                node.children[a] = ANode(a, 0, 0.0, Dict())
            end
                end
            end
    
    # Select action using UCB
    best_action = nothing
    best_ucb = -Inf
    
    for (action, a_node) in node.children
        a_node = a_node::ANode  # Type assertion
        if a_node.visits == 0
            best_action = action
            break
        end
        ucb = a_node.total_value / a_node.visits + c_ucb * sqrt(log(node.visits + 1) / a_node.visits)
        if ucb > best_ucb
            best_ucb = ucb
            best_action = action
        end
    end
    
    if best_action === nothing
        return 0.0
    end
    
    a_node = node.children[best_action]::ANode
    
    # Simulate outcome
    immediate_reward = compute_expected_reward([(node.belief, node.p_mass)], best_action, 0.0)
    # Update visits and values
    node.visits += 1
    node.total_value += immediate_reward
    a_node.visits += 1
    a_node.total_value += immediate_reward
    
    return immediate_reward
end

"""
extract_policy_forest(root_set::Vector{SNode})
Extract policy forest from MCTS tree
"""
function extract_policy_forest(root_set::Vector{SNode})
    forest = Vector{Tuple{Float64, Dict}}()
    for root in root_set
        push!(forest, (root.p_mass, extract_policy_from_node(root)))
    end
    return forest
end

"""
create_reactive_policy(policy_forest, agent_id, env, gs_state)
Create a reactive policy function that can handle observation histories
"""
function create_reactive_policy(policy_forest::Vector{Tuple{Float64, Dict}}, agent_id::Int, env, gs_state)
    # Choose the best policy from the forest
    best_entry = argmax(r -> r[1], policy_forest)
    if best_entry === nothing
        # Return a default policy that always waits
        return (obs_history) -> SensingAction(agent_id, Tuple{Int, Int}[], false)
    end
    
    (_, policy) = best_entry
    
    # Create a reactive policy function
    function reactive_policy(obs_history)
        # Navigate through the policy tree based on observation history
        current_tree = policy
        
        for (i, obs) in enumerate(obs_history)
            if current_tree isa Dict && !isempty(current_tree)
                found = false
                
                # The policy tree structure is: Dict{Action => Dict{Observation => Subtree}}
                # We need to find the action that has the observation we're looking for
                for (action, observation_subtrees) in current_tree
                    # Convert observation to a key that can be used to navigate the policy
                    obs_key = convert_observation_to_key(obs, agent_id)
                    
                    if haskey(observation_subtrees, obs_key)
                        current_tree = observation_subtrees[obs_key]
                        found = true
                            break
                        end
                    end
                    
                if !found
                    # Progressive widening: try to find closest observation across all actions
                    all_observation_keys = Set()
                    for (action, observation_subtrees) in current_tree
                        union!(all_observation_keys, keys(observation_subtrees))
                    end
                    
                    obs_key = convert_observation_to_key(obs, agent_id)
                    closest_key = find_closest_observation_key(obs, all_observation_keys)
                    
                    if closest_key !== nothing
                        # Find which action has this closest key
                        for (action, observation_subtrees) in current_tree
                            if haskey(observation_subtrees, closest_key)
                                current_tree = observation_subtrees[closest_key]
                                found = true
                            break
                            end
                        end
                    end
                    
                    if !found
                        return SensingAction(agent_id, Tuple{Int, Int}[], false)  # default wait
                    end
                    end
                else
                return SensingAction(agent_id, Tuple{Int, Int}[], false)
                end
            end
        
        # Return the action at the current node
        if current_tree isa Dict && !isempty(current_tree)
            best_action = first(keys(current_tree))
            return best_action
        else
            return SensingAction(agent_id, Tuple{Int, Int}[], false)
        end
    end
    
    return reactive_policy
end

"""
convert_observation_to_key(obs, agent_id)
Convert an observation to a standardized key for navigating the policy tree
"""
function convert_observation_to_key(obs, agent_id)
    # Use the standardized observation key function
    return standardize_observation_key(obs)
end

"""
execute_reactive_policy(agent, obs_history)
Execute the reactive policy for an agent based on observation history
"""
function execute_reactive_policy(agent, obs_history)
    if agent.reactive_policy === nothing
        # No reactive policy available, return wait action
        return SensingAction(agent.id, Tuple{Int, Int}[], false)
    end
    
    # Execute the reactive policy with the observation history
    return agent.reactive_policy(obs_history)
end

"""
π_i(hist, forest, agent_id::Int, env, gs_state)
Execute policy given history and forest
"""
function π_i(hist, forest::Vector{Tuple{Float64, Dict}}, agent_id::Int, env, gs_state)
    # Choose root with highest probability
    best_entry = argmax(r -> r[1], forest)
    if best_entry === nothing
        return SensingAction(agent_id, Tuple{Int, Int}[], false)
    end
    
    (P_root, policy) = best_entry
    
    # For now, return the first available action (simplified)
    if !isempty(policy)
        return first(keys(policy))
    else
        return SensingAction(agent_id, Tuple{Int, Int}[], false)
    end
end

"""
best_policy_tree(env, belief, agent, C, gs_state; rng=Random.GLOBAL_RNG)
Main entry point for policy tree planning - called by ground station
Returns a reactive policy function that can handle observation histories
"""
function best_policy_tree(env, belief::Belief, agent::Agent, C::Int, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Start timing
    start_time = time()
    
    # Clear belief evolution cache at the start of each planning session
    clear_belief_evolution_cache!()
    
    # Step 1: Precompute root beliefs after asynchronous branching
    root_beliefs = precompute_belief_branches(env, agent, gs_state)[gs_state.time_step]
    
    # Create root nodes for each belief branch
    root_nodes = [SNode(b, gs_state.time_step, 0, p, 0, 0.0, Dict()) 
                  for (b, p) in root_beliefs]
    # Step 2: Build complete policy tree by enumerating all actions
    println("🔄 Building complete policy tree...")
    
    # For each root node, build the complete tree
    for root in root_nodes
        # Build complete tree for this root
        build_complete_tree(env, root, agent, gs_state, C, MCTS_UCB_CONSTANT)
    end
    # Step 3: Extract reactive policy
    policy_forest = extract_policy_forest(root_nodes)
    reactive_policy = create_reactive_policy(policy_forest, agent.id, env, gs_state)
    
    # Validate policy completeness
    validate_policy_completeness(policy_forest, agent.id)
    
    # Visualize the policy tree for debugging
    visualize_policy_tree(policy_forest, agent.id, 4)
    print_policy_statistics(policy_forest, agent.id)
    
    # Store the reactive policy in the agent for future use
    agent.reactive_policy = reactive_policy
    
    # For policy planning, return the reactive policy itself
    # The ground station will handle this differently than action sequences
    
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ Reactive policy tree planning completed in $(round(planning_time, digits=3)) seconds")
    
    return reactive_policy, planning_time
end

"""
build_complete_tree(env, node, agent, gs_state, horizon, c_ucb)
Build a complete policy tree by enumerating all actions and outcomes
"""
function build_complete_tree(env, node::SNode, agent::Agent, gs_state, horizon, c_ucb)
    # Strict depth limit enforcement
    if node.depth >= horizon
        return 0.0
    end
    
    # Check battery feasibility - if agent can't perform any actions, return low reward
    if agent.battery_level <= 0.0
        return -100.0  # Penalty for running out of battery
    end

    # Step 1: Expand node if not expanded
    if isempty(node.children)
        actions_sequences = generate_action_sequences(agent, env, 1, node.t)
        actions = isempty(actions_sequences) ? SensingAction[] : actions_sequences[1]
        for a in actions
            node.children[a] = ANode(a, 0, 0.0, Dict())
            end
        end
    
    # Step 2: Enumerate all actions to build complete tree
    total_reward = 0.0
    total_visits = 0
    
    for (action, a_node) in node.children
        a_node = a_node::ANode  # Type assertion
        
        # Build complete tree for this action
        action_reward = build_complete_tree_for_action(env, node, a_node, agent, gs_state, horizon, c_ucb)
        
        # Update statistics
        a_node.visits += 1
        a_node.total_value += action_reward
        total_reward += action_reward
        total_visits += 1
    end
    
    # Update node statistics
    node.visits += total_visits
    node.total_value += total_reward
    
    return total_reward / max(total_visits, 1)
end

"""
build_complete_tree_for_action(env, node, a_node, agent, gs_state, horizon, c_ucb)
Build complete tree for a specific action by enumerating all outcomes
"""
function build_complete_tree_for_action(env, node::SNode, a_node::ANode, agent::Agent, gs_state, horizon, c_ucb)
    # Step 3: Enumerate all possible joint observation outcomes
    obs_set = [(agent.id, a_node.action)]
    for (j, plan) in gs_state.agent_plans
        if j != agent.id && plan !== nothing && gs_state.agent_last_sync[j] <= node.t
            # Check if this agent has a reactive policy
            other_agent = nothing
            for a in env.agents
                if a.id == j
                    other_agent = a
                    break
                end
            end
            
            if other_agent !== nothing && other_agent.reactive_policy !== nothing
                # Use reactive policy to get action
                action_j = other_agent.reactive_policy(GridObservation[])
                push!(obs_set, (j, action_j))
            else
                # Fallback to fixed sequence
                plan_timestep = node.t - gs_state.agent_last_sync[j] + 1
            if 1 <= plan_timestep <= length(plan)
                    action_j = plan[plan_timestep]
                    push!(obs_set, (j, action_j))
                end
            end
        end
    end
    
    outcomes = enumerate_all_possible_outcomes(node.belief, obs_set)

    # Step 4: Expand all chance nodes
    if isempty(a_node.outcomes)
        for (obs_combo, prob) in outcomes
            b_evolved = evolve_no_obs(node.belief, env)
            b_next = deepcopy(b_evolved)
            for (cell, state) in obs_combo
                b_next = collapse_belief_to(b_next, cell, state)
            end
            
            # Extract LOCAL observations for this agent
            local_obs = Vector{Tuple{Tuple{Int,Int}, EventState}}()
            for (cell, state) in obs_combo
                # Check if this observation is from our agent's action
                if any(cell == target_cell for target_cell in a_node.action.target_cells)
                    push!(local_obs, (cell, state))
                end
            end
            
            # Store using LOCAL observations as the key
            a_node.outcomes[local_obs] = SNode(b_next, node.t+1, node.depth+1, 
                                               node.p_mass*prob, 0, 0.0, Dict())
        end
    end
    
    # Step 5: Enumerate all outcomes and simulate recursively
    if !isempty(a_node.outcomes)
        total_reward = 0.0
        total_prob = 0.0
        
        for (local_obs, child_node) in a_node.outcomes
            # Compute reward for this outcome
            immediate_reward = compute_expected_reward([(node.belief, 1.0)], 
                                                       a_node.action, 0.0)
            
            # Simulate battery evolution for this action
            old_battery = agent.battery_level
            agent.battery_level = simulate_battery_evolution(agent, a_node.action, agent.battery_level)
            
            future_reward = build_complete_tree(env, child_node, agent, gs_state, horizon, c_ucb)
            
            # Restore battery level for this simulation
            agent.battery_level = old_battery
            
            # Weight by probability of this outcome
            outcome_reward = immediate_reward + env.discount * future_reward
            total_reward += child_node.p_mass * outcome_reward
            total_prob += child_node.p_mass
        end
        
        # Normalize by total probability
        if total_prob > 0
            total_reward /= total_prob
        end
        
        return total_reward
    else
        # No outcomes available, return immediate reward only
        immediate_reward = compute_expected_reward([(node.belief, node.p_mass)], 
                                                   a_node.action, 0.0)
        return immediate_reward
    end
end

end # module 

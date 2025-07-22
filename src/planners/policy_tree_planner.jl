module PolicyTreePlanner

using POMDPs
using POMDPTools
using Random
using LinearAlgebra
using Infiltrator
using ..Types
import ..Agents.BeliefManagement: sample_from_belief
# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..EventState2, ..NO_EVENT_2, ..EVENT_PRESENT_2
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..RangeLimitedSensor, ..EventMap
# Import trajectory functions
import ..Agents.TrajectoryPlanner.get_position_at_time
# Import DBN functions for transition modeling
import ..Environment.EventDynamicsModule.DBNTransitionModel2, ..Environment.EventDynamicsModule.predict_next_belief_dbn
# Import belief management functions
import ..Agents.BeliefManagement
import ..Agents.BeliefManagement.predict_belief_evolution_dbn, ..Agents.BeliefManagement.Belief,
       ..Agents.BeliefManagement.calculate_uncertainty_from_distribution, ..Agents.BeliefManagement.predict_belief_rsp,
       ..Agents.BeliefManagement.evolve_no_obs, ..Agents.BeliefManagement.get_neighbor_beliefs,
       ..Agents.BeliefManagement.enumerate_joint_states, ..Agents.BeliefManagement.product,
       ..Agents.BeliefManagement.normalize_belief_distributions, ..Agents.BeliefManagement.collapse_belief_to,
       ..Agents.BeliefManagement.enumerate_all_possible_outcomes, ..Agents.BeliefManagement.merge_equivalent_beliefs,
       ..Agents.BeliefManagement.calculate_cell_entropy, ..Agents.BeliefManagement.get_event_probability,
       ..Agents.BeliefManagement.clear_belief_evolution_cache!, ..Agents.BeliefManagement.get_cache_stats

# Import utility functions from Types module
import ..Types.calculate_entropy_from_distribution, ..Types.calculate_cell_information_gain, ..Types.combinations

export best_policy_tree, calculate_policy_tree_reward

"""
Policy tree node structure
"""
mutable struct PolicyTreeNode
    belief::Belief
    action::SensingAction
    children::Dict{Vector{Tuple{Tuple{Int, Int}, EventState}}, PolicyTreeNode}  # observation -> child node
    value::Float64
    probability::Float64
end

"""
Full tree node structure (includes all agents)
"""
mutable struct FullTreeNode
    belief::Belief
    all_actions::Vector{Tuple{Int, SensingAction}}  # (agent_id, action) pairs
    children::Dict{Vector{Tuple{Tuple{Int, Int}, EventState}}, FullTreeNode}  # all observations -> child node
    value::Float64
    probability::Float64
end

"""
best_policy_tree(env, belief::Belief, agent::Agent, C::Int, gs_state)::PolicyTreeNode
Builds a full tree with all agents and creates a simplified policy tree for the planning agent
"""
function best_policy_tree(env, belief::Belief, agent, C::Int, gs_state; rng::AbstractRNG=Random.GLOBAL_RNG)
    # Start timing
    start_time = time()
    
    # Clear belief evolution cache at the start of each planning session
    clear_belief_evolution_cache!()
    
    println("🔄 Building full policy tree for agent $(agent.id)...")
    
    # Step 1: Pre-compute belief branches up to t_clean (same as macro script)
    B_branches = precompute_belief_branches(env, agent, gs_state)
    
    # Step 2: Build the full tree with all agents
    full_tree = build_full_tree(env, agent, C, B_branches, gs_state)
    @infiltrate
    
    # Step 3: Create simplified policy tree for the planning agent
    policy_tree = create_agent_policy_tree(full_tree, agent.id)
    
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ Policy tree built in $(round(planning_time, digits=3)) seconds")
    
    # Report cache statistics
    cache_stats = get_cache_stats()
    println("📊 Cache statistics: $(cache_stats[:hits]) hits, $(cache_stats[:misses]) misses, $(round(cache_stats[:hit_rate] * 100, digits=1))% hit rate")
    
    return policy_tree, planning_time
end

"""
Build the full tree with all agents' actions and outcomes
"""
function build_full_tree(env, agent, C::Int, B_branches, gs_state)
    tau_i = gs_state.time_step
    
    # Initialize root node
    root_belief = B_branches[tau_i][1][1]  # Take first belief branch
    root_node = FullTreeNode(root_belief, Vector{Tuple{Int, SensingAction}}(), Dict{Vector{Tuple{Tuple{Int, Int}, EventState}}, FullTreeNode}(), 0.0, 1.0)
    
    # Build tree recursively
    build_full_tree_recursive!(root_node, env, agent, C, B_branches, gs_state, tau_i, 1)
    
    return root_node
end

"""
Recursively build the full tree
"""
function build_full_tree_recursive!(node::FullTreeNode, env, agent, C::Int, B_branches, gs_state, current_time::Int, depth::Int)
    if depth > C
        # Leaf node - compute final value
        node.value = compute_leaf_value(node.belief, env)
        return
    end
    
    # Get all agents' actions for this timestep
    all_actions = get_all_agent_actions(env, agent, current_time, gs_state, depth)
    node.all_actions = all_actions
    
    if isempty(all_actions)
        # No actions available - just evolve belief
        next_belief = evolve_no_obs(node.belief, env)
        child_node = FullTreeNode(next_belief, Vector{Tuple{Int, SensingAction}}(), Dict{Vector{Tuple{Tuple{Int, Int}, EventState}}, FullTreeNode}(), 0.0, node.probability)
        build_full_tree_recursive!(child_node, env, agent, C, B_branches, gs_state, current_time + 1, depth + 1)
        node.value = child_node.value
        return
    end
    
    # Get all possible observation outcomes for all agents
    obs_set = [(agent_id, action) for (agent_id, action) in all_actions]
    all_outcomes = enumerate_all_possible_outcomes(node.belief, obs_set)
    
    # Create child nodes for each outcome
    for (observation_combo, probability) in all_outcomes
        # Apply observations to belief
        new_belief = deepcopy(node.belief)
        for (cell, observed_state) in observation_combo
            new_belief = collapse_belief_to(new_belief, cell, observed_state)
        end
        
        # Evolve belief
        next_belief = evolve_no_obs(new_belief, env)
        
        # Create child node
        child_node = FullTreeNode(next_belief, Vector{Tuple{Int, SensingAction}}(), Dict{Vector{Tuple{Tuple{Int, Int}, EventState}}, FullTreeNode}(), 0.0, node.probability * probability)
        
        # Recursively build subtree
        build_full_tree_recursive!(child_node, env, agent, C, B_branches, gs_state, current_time + 1, depth + 1)
        
        # Store child node
        node.children[observation_combo] = child_node
    end
    
    # Compute value as weighted sum of children plus immediate reward
    immediate_reward = compute_immediate_reward(node.belief, all_actions, env)
    expected_future_value = sum(child.value * child.probability for child in values(node.children))
    node.value = immediate_reward + env.discount * expected_future_value
end

"""
Get all agents' actions for a given timestep
"""
function get_all_agent_actions(env, planning_agent, current_time::Int, gs_state, depth::Int)
    all_actions = Vector{Tuple{Int, SensingAction}}()
    
    # Add planning agent's action
    planning_action = get_agent_action(planning_agent, env, current_time, depth, gs_state)
    push!(all_actions, (planning_agent.id, planning_action))
    
    # Add other agents' actions
    for agent in env.agents
        if agent.id != planning_agent.id
            other_action = get_agent_action(agent, env, current_time, depth, gs_state)
            push!(all_actions, (agent.id, other_action))
        end
    end
    
    return all_actions
end

"""
Get action for a specific agent at a given timestep
"""
function get_agent_action(agent, env, current_time::Int, depth::Int, gs_state)
    # Check if agent has a plan
    if haskey(gs_state.agent_plans, agent.id) && gs_state.agent_plans[agent.id] !== nothing
        plan = gs_state.agent_plans[agent.id]
        plan_type = get(gs_state.agent_plan_types, agent.id, :script)
        
        if plan_type == :policy
            # For policy trees, we need to get the action from the policy based on observation history
            # We need to simulate what this agent would have observed up to this point
            agent_obs_history = simulate_agent_observation_history(agent, env, gs_state, current_time, depth)
            return get_action_from_policy_tree(plan, agent_obs_history)
        else
            # For scripts (macro-scripts, random, sweep, etc.), use the plan index
            plan_timestep = depth
            if 1 <= plan_timestep <= length(plan)
                return plan[plan_timestep]
            end
        end
    end
    
    # If no plan, generate action based on trajectory
    trajectory_pos = get_position_at_time(agent.trajectory, current_time - gs_state.time_step)
    for_cells = get_field_of_regard_at_position(agent, trajectory_pos, env)
    
    # For now, return a simple action (could be improved with better heuristics)
    if !isempty(for_cells)
        return SensingAction(agent.id, [for_cells[1]], false)
    else
        return SensingAction(agent.id, Tuple{Int, Int}[], false)
    end
end

"""
Simulate what an agent would have observed up to a given point
"""
function simulate_agent_observation_history(agent, env, gs_state, current_time::Int, depth::Int)
    # This is a simplified simulation - in a full implementation, we'd need to
    # simulate the agent's trajectory and what it would have observed
    # For now, we'll use a basic approach based on the agent's trajectory
    
    obs_history = Vector{GridObservation}()
    
    # Simulate observations for the past few timesteps
    for t in max(0, current_time - depth):(current_time - 1)
        # Get agent's position at this time
        trajectory_pos = get_position_at_time(agent.trajectory, t - gs_state.time_step)
        for_cells = get_field_of_regard_at_position(agent, trajectory_pos, env)
        
        # Create a simple observation (in reality, this would depend on the actual world state)
        if !isempty(for_cells)
            # For now, assume no events observed (could be improved with belief-based sampling)
            event_states = fill(NO_EVENT, length(for_cells))
            obs = GridObservation(for_cells, event_states)
            push!(obs_history, obs)
        end
    end
    
    return obs_history
end

"""
Get action from a policy tree based on observation history
"""
function get_action_from_policy_tree(policy_tree, obs_history::Vector{GridObservation})
    # Traverse the policy tree based on observation history
    current_node = policy_tree
    
    # Follow the tree based on recent observations
    for obs in obs_history
        # Find the child node that matches this observation
        matching_child = nothing
        
        for (child_obs, child_node) in current_node.children
            # Check if this child's observation matches our observation
            if observations_match(child_obs, obs)
                matching_child = child_node
                break
            end
        end
        
        if matching_child !== nothing
            current_node = matching_child
        else
            # No matching child found, stay at current node
            break
        end
    end
    
    # Return the action at the current node
    return current_node.action
end

"""
observations_match(tree_obs::Vector{Tuple{Tuple{Int, Int}, EventState}}, actual_obs::GridObservation)
Check if the tree observation matches the actual observation
"""
function observations_match(tree_obs::Vector{Tuple{Tuple{Int, Int}, EventState}}, actual_obs::GridObservation)
    # Convert actual observation to the same format as tree observations
    actual_obs_formatted = Vector{Tuple{Tuple{Int, Int}, EventState}}()
    
    for (i, cell) in enumerate(actual_obs.sensed_cells)
        if i <= length(actual_obs.event_states)
            push!(actual_obs_formatted, (cell, actual_obs.event_states[i]))
        end
    end
    
    # Check if the observations match
    if length(tree_obs) != length(actual_obs_formatted)
        return false
    end
    
    # Sort both observations to ensure order doesn't matter
    sorted_tree_obs = sort(tree_obs, by = x -> x[1])
    sorted_actual_obs = sort(actual_obs_formatted, by = x -> x[1])
    
    for (tree_obs_item, actual_obs_item) in zip(sorted_tree_obs, sorted_actual_obs)
        if tree_obs_item != actual_obs_item
            return false
        end
    end
    
    return true
end

"""
Create simplified policy tree for the planning agent
"""
function create_agent_policy_tree(full_tree::FullTreeNode, agent_id::Int)
    # Extract only the planning agent's observations and average over other agents' outcomes
    policy_tree = create_agent_policy_tree_recursive(full_tree, agent_id)
    return policy_tree
end

"""
Recursively create simplified policy tree
"""
function create_agent_policy_tree_recursive(full_node::FullTreeNode, agent_id::Int)
    # Find planning agent's action
    planning_action = nothing
    for (aid, action) in full_node.all_actions
        if aid == agent_id
            planning_action = action
            break
        end
    end
    
    if planning_action === nothing
        # No action for this agent at this node
        return PolicyTreeNode(full_node.belief, SensingAction(agent_id, Tuple{Int, Int}[], false), Dict{Vector{Tuple{Tuple{Int, Int}, EventState}}, PolicyTreeNode}(), full_node.value, full_node.probability)
    end
    
    # Group children by planning agent's observations
    agent_obs_groups = Dict{Vector{Tuple{Tuple{Int, Int}, EventState}}, Vector{FullTreeNode}}()
    
    for (full_obs, child_node) in full_node.children
        # Extract planning agent's observations from full observation
        agent_obs = Vector{Tuple{Tuple{Int, Int}, EventState}}()
        for (cell, state) in full_obs
            # Check if this observation belongs to the planning agent
            if any(aid == agent_id && cell in action.target_cells for (aid, action) in full_node.all_actions)
                push!(agent_obs, (cell, state))
            end
        end
        
        # Group by planning agent's observations
        if !haskey(agent_obs_groups, agent_obs)
            agent_obs_groups[agent_obs] = FullTreeNode[]
        end
        push!(agent_obs_groups[agent_obs], child_node)
    end
    
    # Create policy tree children
    policy_children = Dict{Vector{Tuple{Tuple{Int, Int}, EventState}}, PolicyTreeNode}()
    
    for (agent_obs, child_nodes) in agent_obs_groups
        # Average values over all children with this agent observation
        total_prob = sum(node.probability for node in child_nodes)
        avg_value = sum(node.value * node.probability for node in child_nodes) / total_prob
        
        # Recursively create child policy tree node
        child_policy_node = create_agent_policy_tree_recursive(child_nodes[1], agent_id)  # Use first child as representative
        child_policy_node.value = avg_value
        child_policy_node.probability = total_prob
        
        policy_children[agent_obs] = child_policy_node
    end
    
    return PolicyTreeNode(full_node.belief, planning_action, policy_children, full_node.value, full_node.probability)
end

"""
Compute immediate reward for a set of actions
"""
function compute_immediate_reward(belief::Belief, all_actions::Vector{Tuple{Int, SensingAction}}, env)
    total_reward = 0.0
    c_obs = 0.0  # Cost of observation
    
    for (agent_id, action) in all_actions
        for cell in action.target_cells
            H_before = calculate_cell_entropy(belief, cell)
            H_after = 0.0  # Perfect observation
            info_gain = H_before - H_after
            total_reward += info_gain
        end
        
        if !isempty(action.target_cells)
            total_reward -= c_obs
        end
    end
    
    return total_reward
end

"""
Compute leaf node value
"""
function compute_leaf_value(belief::Belief, env)
    # Simple terminal value based on uncertainty reduction
    total_uncertainty = 0.0
    for y in 1:env.height, x in 1:env.width
        total_uncertainty += calculate_cell_entropy(belief, (x, y))
    end
    return -total_uncertainty  # Negative because we want to minimize uncertainty
end

"""
Get field of regard for an agent at a specific position
"""
function get_field_of_regard_at_position(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    
    # Check if we want row-only visibility (sensor range = 0 means row-only)
    if agent.sensor.range == 0.0
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

"""
Pre-compute belief branches (same as macro script)
"""
function precompute_belief_branches(env, agent, gs_state)
    # Step 1: Determine the last time where all observation outcomes are known
    tau_i = gs_state.time_step  # Current sync time of agent i
    tau = gs_state.agent_last_sync  # Last sync times of all agents
    
    # Handle initial case: if any agent hasn't synced yet (last_sync = -1), start from t=0
    other_agent_sync_times = [tau[j] for j in keys(tau) if j != agent.id]
    if any(sync_time == -1 for sync_time in other_agent_sync_times)
        t_clean = 0  # Start from t=0 if any agent hasn't synced yet
        println("  🔄 Initial case: some agents haven't synced yet, starting from t_clean = 0")
    else
        t_clean = minimum(other_agent_sync_times)
        println("  🔄 Using t_clean = $(t_clean) (minimum of other agent sync times)")
    end
    
    # Step 2: Roll forward deterministically from uniform belief to t_clean-1 using known observations
    # Start with uniform belief distribution (we knew nothing at t=0)
    B = initialize_uniform_belief(env)
    for t in 0:(t_clean-1)
        B = evolve_no_obs(B, env)  # Contagion-aware update
        # Apply known observations (perfect observations)
        for (agent_j, action_j) in get_known_observations_at_time(t, gs_state)
            for cell in action_j.target_cells
                if has_known_observation(t, cell, gs_state)
                    observed_value = get_known_observation(t, cell, gs_state)
                    B = collapse_belief_to(B, cell, observed_value)
                end
            end
            end
        end
        
    # Step 3: Initialize branching structure at t_clean
    # At t_clean, we start branching over unknown observations
    B_branches = Dict{Int, Vector{Tuple{Belief, Float64}}}()
    B_branches[t_clean] = [(B, 1.0)]
    
    # Step 4: Create branching windows for each agent j ≠ i
    branch_windows = Vector{Tuple{Int, Int, Int}}()
    for (j, last_sync) in tau
        if j != agent.id
            # Special case: if this agent has a plan and we're at t_clean=0,
            # we need to branch over their scheduled observations starting from t=0
            if t_clean == 0 && haskey(gs_state.agent_plans, j) && gs_state.agent_plans[j] !== nothing
                # Check if this agent has scheduled observations at t=0
                plan = gs_state.agent_plans[j]
                if 1 <= length(plan) && !isempty(plan[1].target_cells)
                    # This agent has a scheduled observation at t=0, branch from t=0
                    push!(branch_windows, (0, tau_i, j))
                end
            else
                # Use t_clean as the starting point to avoid negative timesteps
                t_start = max(t_clean, last_sync)  # We have known observations up to here
                t_end = tau_i        # Branch until current agent i sync
                if t_start < t_end
                    push!(branch_windows, (t_start, t_end, j))
                end
            end
        end
    end
    
    # Sort and merge overlapping windows to avoid double branching
    sort!(branch_windows)
    
    # If we're branching from t=0, make sure we have the belief at t=0
    if t_clean == 0 && any(start_t == 0 for (start_t, _, _) in branch_windows)
        B_branches[0] = [(B, 1.0)]
    end
    
    println("  🔄 Branching windows: $(branch_windows)")
    # Step 5: Forward branch over unknown observations from other agents before tau_i
    for (start_t, end_t, agent_j) in branch_windows
        println("  🔄 Processing branching window: $(start_t) to $(end_t) for agent $(agent_j)")
        for t in start_t:(end_t-1)
            println("    🔄 Processing timestep $(t)")
            new_branches = Vector{Tuple{Belief, Float64}}()
            for (B_cur, p_branch) in B_branches[t]

                B_evolved = evolve_no_obs(B_cur, env)
                
                # Check which observations need branching based on the current window
                obs_set = Vector{Tuple{Int, SensingAction}}()  # Unknown observations for branching
                
                # First, apply known observations from history (deterministic)
                known_obs = get_known_observations_at_time(t, gs_state)
                for (agent_k, action_k) in known_obs
                    if agent_k != agent.id
                        for cell in action_k.target_cells
                            observed_value = get_known_observation(t, cell, gs_state)
                            B_evolved = collapse_belief_to(B_evolved, cell, observed_value)
                        end
                    end
                end
                scheduled_obs = get_scheduled_observations_at_time(t, gs_state)
                println("    🔄 Scheduled observations at t=$(t): $(scheduled_obs)")
                for (agent_k, action_k) in scheduled_obs
                    # Only branch over observations that are not already known
                    # Check if this observation is already known by comparing agent_id and target_cells
                    is_known = false
                    for (known_agent, known_action) in known_obs
                        if agent_k == known_agent && action_k.target_cells == known_action.target_cells
                            is_known = true
                            break
                        end
                    end
                    
                    if !is_known && agent_k != agent.id && agent_k == agent_j
                        push!(obs_set, (agent_k, action_k))
                        println("    🔄 Adding observation from agent $(agent_k): $(action_k)")
                    end
                end
                if !isempty(obs_set)
                    for (observation_combo, probability) in enumerate_all_possible_outcomes(B_evolved, obs_set)
                        # observation_combo = Vector{(cell, observed_state)}
                        B_new = deepcopy(B_evolved)
                        # Apply all observations in the combination together
                        for (cell, observed_state) in observation_combo
                            B_new = collapse_belief_to(B_new, cell, observed_state)
                        end
                        push!(new_branches, (B_new, p_branch * probability))

                    end
                else
                    push!(new_branches, (B_evolved, p_branch))
                end
            end
            B_branches[t + 1] = merge_equivalent_beliefs(new_branches)
        end
    end
    return B_branches
end

"""
Initialize uniform belief distribution
"""
function initialize_uniform_belief(env)
    num_states = 2
    uniform_distribution = fill(1.0/num_states, num_states)
    return BeliefManagement.initialize_belief(env.width, env.height, uniform_distribution)
end

"""
Helper functions for belief branching (same as macro script)
"""
function get_known_observations_at_time(t::Int, gs_state)
    observations = Vector{Tuple{Int, SensingAction}}()
    for (agent_id, obs_history) in gs_state.agent_observation_history
        for (obs_timestep, obs_cell, obs_state) in obs_history
            if obs_timestep == t
                action = SensingAction(agent_id, [obs_cell], false)
                push!(observations, (agent_id, action))
            end
        end
    end
    return observations
end

function has_known_observation(t::Int, cell::Tuple{Int, Int}, gs_state)
    for (agent_id, obs_history) in gs_state.agent_observation_history
        for (obs_timestep, obs_cell, obs_state) in obs_history
            if obs_timestep == t && obs_cell == cell
                return true
            end
        end
    end
    return false
end

function get_known_observation(t::Int, cell::Tuple{Int, Int}, gs_state)
    for (agent_id, obs_history) in gs_state.agent_observation_history
        for (obs_timestep, obs_cell, obs_state) in obs_history
            if obs_timestep == t && obs_cell == cell
                return obs_state
            end
        end
    end
end

"""
Get scheduled observations at a specific time (from agent plans)
"""
function get_scheduled_observations_at_time(t::Int, gs_state)
    observations = Vector{Tuple{Int, SensingAction}}()
    # Check all agents' plans for observations at this time
    for (agent_id, plan) in gs_state.agent_plans
        if plan !== nothing
            # For macro-scripts, check if this timestep has an action
            plan_timestep = (t - gs_state.agent_last_sync[agent_id]) + 1
            if 1 <= plan_timestep <= length(plan)
                action = plan[plan_timestep]
                if !isempty(action.target_cells)
                    push!(observations, (agent_id, action))
                end
            end
        end
    end
    
    return observations
end



end # module 
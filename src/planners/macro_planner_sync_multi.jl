module MacroPlannerSyncMulti

using POMDPs
using POMDPTools
# DISABLED: These packages cause version compatibility issues with Julia 1.12
# using POMDPPolicies
# using POMDPSimulators
# using QMDP
# Import specific POMDPs.jl functions
import POMDPs: action, transition, isterminal
# Note: Using POMDPTools.Deterministic explicitly to avoid conflict with Types.Deterministic
using Random
using LinearAlgebra
using Infiltrator
using Statistics
using Base.Threads
using ..Types
import ..Agents.BeliefManagement: sample_from_belief
import ..Types: check_battery_feasible, simulate_battery_evolution
# Import types from the parent module (Planners)
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..EventState2, ..NO_EVENT_2, ..EVENT_PRESENT_2
import ..Agent, ..SensingAction, ..GridObservation, ..CircularTrajectory, ..LinearTrajectory, ..ComplexTrajectory, ..RangeLimitedSensor, ..EventMap
# Import trajectory functions
import ..Agents.TrajectoryPlanner.get_position_at_time
# Import DBN functions for transition modeling
import ..Environment.EventDynamicsModule.DBNTransitionModel2, ..Environment.EventDynamicsModule.predict_next_belief_dbn
# Import belief management functions
import ..Agents.BeliefManagement
import ..Agents.BeliefManagement.predict_belief_evolution_dbn, ..Agents.BeliefManagement.Belief,
       ..Agents.BeliefManagement.calculate_uncertainty_from_distribution, ..Agents.BeliefManagement.predict_belief_rsp,
       ..Agents.BeliefManagement.evolve_no_obs,..Agents.BeliefManagement.evolve_no_obs_fast, ..Agents.BeliefManagement.get_neighbor_beliefs,
       ..Agents.BeliefManagement.enumerate_joint_states, ..Agents.BeliefManagement.prob_product,
       ..Agents.BeliefManagement.normalize_belief_distributions, ..Agents.BeliefManagement.collapse_belief_to,
       ..Agents.BeliefManagement.enumerate_all_possible_outcomes, ..Agents.BeliefManagement.merge_equivalent_beliefs,
       ..Agents.BeliefManagement.calculate_cell_entropy, ..Agents.BeliefManagement.get_event_probability,
       ..Agents.BeliefManagement.clear_belief_evolution_cache!, ..Agents.BeliefManagement.get_cache_stats,
       ..Agents.BeliefManagement.beliefs_are_equivalent

export best_joint_policy, calculate_joint_reward, configure_reward_weights, set_reward_config_from_main, SyncMultiAgentPOMDP, calculate_sophisticated_reward

# Multi-agent synchronized planning types
struct JointAction
    actions::Vector{SensingAction}  # One action per agent
end

# Equality and hash for JointAction (required for POMDPs.jl)
function Base.:(==)(a1::JointAction, a2::JointAction)
    length(a1.actions) != length(a2.actions) && return false
    for (act1, act2) in zip(a1.actions, a2.actions)
        if act1.agent_id != act2.agent_id || 
           act1.target_cells != act2.target_cells || 
           act1.communicate != act2.communicate
            return false
        end
    end
    return true
end

Base.hash(a::JointAction, h::UInt) = hash([(act.agent_id, act.target_cells, act.communicate) for act in a.actions], h)

struct JointObservation
    observations::Vector{GridObservation}  # One observation per agent
end

# Equality and hash for JointObservation (required for POMDPs.jl)
function Base.:(==)(o1::JointObservation, o2::JointObservation)
    length(o1.observations) != length(o2.observations) && return false
    for (obs1, obs2) in zip(o1.observations, o2.observations)
        if obs1.agent_id != obs2.agent_id || 
           obs1.sensed_cells != obs2.sensed_cells || 
           obs1.event_states != obs2.event_states ||
           obs1.communication_received != obs2.communication_received
            return false
        end
    end
    return true
end

Base.hash(o::JointObservation, h::UInt) = hash([(obs.agent_id, obs.sensed_cells, obs.event_states, obs.communication_received) for obs in o.observations], h)

# Synchronized Multi-Agent POMDP State
struct SyncMultiAgentState
    belief::Belief  # Shared global belief
    agent_positions::Vector{Tuple{Int, Int}}  # Current positions of all agents
    timestep::Int
end

# Synchronized Multi-Agent POMDP Definition
struct SyncMultiAgentPOMDP <: POMDP{SyncMultiAgentState, JointAction, JointObservation}
    env::Any  # Original environment
    agents::Vector{Agent}
    horizon::Int
    discount_factor::Float64
    # Cached action and observation spaces for consistency
    action_space::Vector{JointAction}
    observation_space::Vector{JointObservation}
    state_space::Vector{SyncMultiAgentState}
end

# Constructor with automatic space generation
function SyncMultiAgentPOMDP(env, agents::Vector{Agent}, horizon::Int, discount_factor::Float64)
    # Pre-compute action space
    action_space = enumerate_joint_actions(agents, env)
    
    # Pre-compute observation space (enumerate_joint_observations already returns JointObservation objects)
    observation_space = enumerate_joint_observations(agents)
    
    # Pre-compute state space
    dummy_belief = Belief(
        zeros(2, env.height, env.width), 
        zeros(env.height, env.width), 
        0, 
        []
    )
    state_space = [SyncMultiAgentState(dummy_belief, get_agent_positions(agents, t), t) for t in 0:horizon]
    
    return SyncMultiAgentPOMDP(env, agents, horizon, discount_factor, action_space, observation_space, state_space)
end

# Reward function configuration - these will be set from main.jl
# Note: Not const so they can be reassigned by set_reward_config_from_main()
DEFAULT_ENTROPY_WEIGHT = get(ENV, "ENTROPY_WEIGHT", 1.0)    # w_H: Weight for entropy reduction (coordination)
DEFAULT_VALUE_WEIGHT = get(ENV, "VALUE_WEIGHT", 0.5)        # w_F: Weight for state value (detection priority)
DEFAULT_INFORMATION_STATES = get(ENV, "INFORMATION_STATES", [1, 2])  # I_1: No event, I_2: Event
DEFAULT_STATE_VALUES = get(ENV, "STATE_VALUES", [0.1, 1.0])        # F_1: No event value, F_2: Event value

# POMDP Interface Implementation
POMDPs.discount(pomdp::SyncMultiAgentPOMDP) = pomdp.discount_factor

POMDPs.isterminal(pomdp::SyncMultiAgentPOMDP, s::SyncMultiAgentState) = s.timestep >= pomdp.horizon

POMDPs.actions(pomdp::SyncMultiAgentPOMDP) = pomdp.action_space

POMDPs.states(pomdp::SyncMultiAgentPOMDP) = pomdp.state_space

POMDPs.observations(pomdp::SyncMultiAgentPOMDP) = pomdp.observation_space

function POMDPs.obsindex(pomdp::SyncMultiAgentPOMDP, o::JointObservation)
    idx = findfirst(==(o), pomdp.observation_space)
    return idx === nothing ? 1 : idx
end

# Required for discrete solvers
POMDPs.stateindex(pomdp::SyncMultiAgentPOMDP, s::SyncMultiAgentState) = s.timestep + 1

function POMDPs.actionindex(pomdp::SyncMultiAgentPOMDP, a::JointAction)
    idx = findfirst(==(a), pomdp.action_space)
    if idx === nothing
        @warn "Action not found in action space, returning 1 (wait action)" action=a
        return 1
    end
    return idx
end

function POMDPs.transition(pomdp::SyncMultiAgentPOMDP, s::SyncMultiAgentState, a::JointAction)
    # Evolve belief using DBN
    evolved_belief = evolve_no_obs_fast(s.belief, pomdp.env, calculate_uncertainty=false)
    
    # Update belief with joint observations
    updated_belief = update_belief_with_joint_observations(evolved_belief, a, pomdp.agents, s.agent_positions, pomdp.env)
    
    # Update agent positions
    new_positions = [get_position_at_time(agent.trajectory, s.timestep + 1, agent.phase_offset) for agent in pomdp.agents]
    
    new_state = SyncMultiAgentState(updated_belief, new_positions, s.timestep + 1)
    
    return POMDPTools.Deterministic(new_state)
end

function POMDPs.observation(pomdp::SyncMultiAgentPOMDP, a::JointAction, sp::SyncMultiAgentState)
    # Generate observations based on actions and new state
    observations = Vector{GridObservation}()
    
    for (i, action) in enumerate(a.actions)
        if !isempty(action.target_cells)
            # Simulate observation
            sensed_cells = Tuple{Int, Int}[]
            event_states = EventState[]
            
            for cell in action.target_cells
                x, y = cell
                if 1 <= x <= pomdp.env.width && 1 <= y <= pomdp.env.height
                    push!(sensed_cells, cell)
                    # Perfect observation
                    if pomdp.env.current_state[y, x] == EVENT_PRESENT
                        push!(event_states, EVENT_PRESENT)
                    else
                        push!(event_states, NO_EVENT)
                    end
                end
            end
            
            push!(observations, GridObservation(action.agent_id, sensed_cells, event_states, []))
        else
            # Wait action - no observation
            push!(observations, GridObservation(action.agent_id, Tuple{Int, Int}[], EventState[], []))
        end
    end
    
    return POMDPTools.Deterministic(JointObservation(observations))
end

function POMDPs.reward(pomdp::SyncMultiAgentPOMDP, s::SyncMultiAgentState, a::JointAction)
    return calculate_joint_reward(s, a, pomdp.agents, pomdp.env)
end

# Initial state distribution for the POMDP
function POMDPs.initialstate(pomdp::SyncMultiAgentPOMDP)
    # Create initial state at timestep 0
    dummy_belief = Belief(
        zeros(2, pomdp.env.height, pomdp.env.width), 
        zeros(pomdp.env.height, pomdp.env.width), 
        0, 
        []
    )
    initial_positions = get_agent_positions(pomdp.agents, 0)
    initial_state = SyncMultiAgentState(dummy_belief, initial_positions, 0)
    return POMDPTools.Deterministic(initial_state)
end

"""
best_joint_policy(env, belief::Belief, agents::Vector{Agent}, C::Int, gs_state)
Finds the best joint policy for all agents for a FULL PERIOD.
All agents start at phase 0 and we plan for the entire trajectory period.
"""
function best_joint_policy(env, belief::Belief, agents::Vector{Agent}, C::Int, gs_state; 
                          rng::AbstractRNG=Random.GLOBAL_RNG, solver_type::Symbol=:greedy_lookahead)
    # Start timing
    start_time = time()
    
    # Get the trajectory period (all agents should have same period when synchronized)
    period = agents[1].trajectory.period
    
    println("🔄 Computing centralized joint policy for $(length(agents)) agents for full period $(period)...")
    
    # Use greedy lookahead approach for centralized planning
    joint_policy = Vector{JointAction}()
    
    # Current belief state
    current_belief = deepcopy(belief)
    
    # Plan for the ENTIRE PERIOD starting from phase 0
    for phase in 0:(period-1)
        # Get agent positions at this phase (all agents synchronized at phase)
        agent_positions = [get_position_at_time(agent.trajectory, phase, agent.phase_offset) for agent in agents]
        
        println("  Phase $(phase)/$(period-1): agents at positions $(agent_positions)")
        
        # Get available actions for each agent at their current positions
        agent_action_sets = Vector{Vector{SensingAction}}()
        for (i, agent) in enumerate(agents)
            pos = agent_positions[i]
            available_cells = get_field_of_regard_at_position(agent, pos, env)
            
            println("    Agent $(agent.id) at position $(pos): $(length(available_cells)) cells in FOR")
            
            actions = SensingAction[]
            push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))  # Wait action
            
            for cell in available_cells
                action = SensingAction(agent.id, [cell], false)
                push!(actions, action)
            end
            
            println("    Agent $(agent.id): $(length(actions)) total actions (1 wait + $(length(available_cells)) sensing)")
            push!(agent_action_sets, actions)
        end
        
        total_joint_actions = prod([length(action_set) for action_set in agent_action_sets])
        println("  Evaluating $(total_joint_actions) joint action combinations...")
        
        # Evaluate all joint actions and pick the best
        best_reward = -Inf
        best_joint_action = nothing
        
        for action_combo in Iterators.product(agent_action_sets...)
            joint_action = JointAction(collect(action_combo))
            
            # Calculate immediate reward
            immediate_reward = 0.0
            for action in joint_action.actions
                if !isempty(action.target_cells)
                    for cell in action.target_cells
                        cell_reward = calculate_sophisticated_reward(current_belief, cell)
                        immediate_reward += cell_reward
                    end
                end
            end
            
            if immediate_reward > best_reward
                best_reward = immediate_reward
                best_joint_action = joint_action
            end
        end
        
        # Add best action to policy
        if best_joint_action !== nothing
            println("  Selected joint action with reward $(round(best_reward, digits=3))")
            for (i, action) in enumerate(best_joint_action.actions)
                if !isempty(action.target_cells)
                    println("    Agent $(action.agent_id): sense cells $(action.target_cells)")
                else
                    println("    Agent $(action.agent_id): wait")
                end
            end
            push!(joint_policy, best_joint_action)
            
            # Update belief for next step (simulate observation with most likely states)
            for action in best_joint_action.actions
                if !isempty(action.target_cells)
                    for cell in action.target_cells
                        # Collapse to most likely state
                        x, y = cell
                        cell_dist = current_belief.event_distributions[:, y, x]
                        most_likely_idx = argmax(cell_dist)
                        current_belief.event_distributions[:, y, x] .= 0.0
                        current_belief.event_distributions[most_likely_idx, y, x] = 1.0
                    end
                end
            end
            
            # Evolve belief forward one step
            current_belief = evolve_no_obs_fast(current_belief, env, calculate_uncertainty=false)
        else
            # Fallback: all wait
            wait_actions = [SensingAction(agent.id, Tuple{Int, Int}[], false) for agent in agents]
            push!(joint_policy, JointAction(wait_actions))
        end
    end
    
    # End timing
    end_time = time()
    planning_time = end_time - start_time
    
    println("✅ Centralized joint policy found in $(round(planning_time, digits=3)) seconds")
    
    return joint_policy, planning_time
end

# DISABLED: Requires QMDP package which has version compatibility issues
"""
create_solver(solver_type::Symbol, pomdp::SyncMultiAgentPOMDP)
Creates a POMDPs.jl solver based on the specified type.
"""
# function create_solver(solver_type::Symbol, pomdp::SyncMultiAgentPOMDP)
#     if solver_type == :qmdp
#         # QMDP solver - assumes full observability for planning
#         return QMDPSolver()
#     elseif solver_type == :pbvi
#         # Point-based Value Iteration
#         # TODO: Add PBVI solver when available
#         println("⚠️  PBVI solver not yet implemented, using QMDP instead")
#         return QMDPSolver()
#     elseif solver_type == :sarsop
#         # SARSOP solver
#         # TODO: Add SARSOP solver when available
#         println("⚠️  SARSOP solver not yet implemented, using QMDP instead")
#         return QMDPSolver()
#     elseif solver_type == :incremental_pruning
#         # Incremental Pruning solver
#         # TODO: Add Incremental Pruning solver when available
#         println("⚠️  Incremental Pruning solver not yet implemented, using QMDP instead")
#         return QMDPSolver()
#     else
#         # Default to QMDP
#         println("⚠️  Unknown solver type $(solver_type), using QMDP")
#         return QMDPSolver()
#     end
# end

# DISABLED: Requires POMDPPolicies package which has version compatibility issues
"""
extract_joint_policy_from_pomdp(policy, pomdp::SyncMultiAgentPOMDP, gs_state)
Extracts joint policy from POMDP policy.
"""
# function extract_joint_policy_from_pomdp(policy, pomdp::SyncMultiAgentPOMDP, gs_state)
#     joint_policy = Vector{JointAction}()
#     
#     # Get initial state
#     current_positions = [get_position_at_time(agent.trajectory, gs_state.time_step, agent.phase_offset) for agent in pomdp.agents]
#     current_state = SyncMultiAgentState(gs_state.global_belief, current_positions, gs_state.time_step)
#     
#     # Extract policy for the horizon
#     for h in 1:pomdp.horizon
#         # QMDP policy expects a belief distribution, create a deterministic belief over current state
#         # Create a belief vector with probability 1.0 at current state index
#         belief_vec = zeros(length(pomdp.state_space))
#         state_idx = POMDPs.stateindex(pomdp, current_state)
#         if state_idx >= 1 && state_idx <= length(belief_vec)
#             belief_vec[state_idx] = 1.0
#         else
#             # Fallback: uniform belief
#             belief_vec .= 1.0 / length(belief_vec)
#         end
#         
#         # Get action from policy using the belief vector
#         try
#             joint_action_result = POMDPs.action(policy, belief_vec)
#             
#             if joint_action_result isa JointAction
#                 push!(joint_policy, joint_action_result)
#             else
#                 # Fallback to wait actions
#                 wait_actions = [SensingAction(agent.id, Tuple{Int, Int}[], false) for agent in pomdp.agents]
#                 push!(joint_policy, JointAction(wait_actions))
#             end
#             
#             # Simulate forward to get next state
#             if !isterminal(pomdp, current_state) && joint_action_result isa JointAction
#                 transition_dist = POMDPs.transition(pomdp, current_state, joint_action_result)
#                 current_state = rand(transition_dist)
#             end
#         catch e
#             @warn "Error extracting action from policy, using wait action" exception=e
#             wait_actions = [SensingAction(agent.id, Tuple{Int, Int}[], false) for agent in pomdp.agents]
#             push!(joint_policy, JointAction(wait_actions))
#         end
#     end
#     
#     return joint_policy
# end

"""
get_agent_positions(agents::Vector{Agent}, timestep::Int)
Gets positions of all agents at a given timestep.
"""
function get_agent_positions(agents::Vector{Agent}, timestep::Int)
    return [get_position_at_time(agent.trajectory, timestep, agent.phase_offset) for agent in agents]
end

"""
enumerate_joint_observations(agents::Vector{Agent})
Enumerates all possible joint observations.
"""
function enumerate_joint_observations(agents::Vector{Agent})
    # For simplicity, return empty observations
    # In a full implementation, this would enumerate all possible observation combinations
    return [JointObservation([GridObservation(agent.id, Tuple{Int, Int}[], EventState[], []) for agent in agents])]
end

"""
enumerate_joint_actions(agents::Vector{Agent}, env)
Enumerates all possible joint actions for synchronized agents.
"""
function enumerate_joint_actions(agents::Vector{Agent}, env)
    joint_actions = Vector{JointAction}()
    
    # Get individual action sets for each agent
    agent_action_sets = Vector{Vector{SensingAction}}()
    
    for agent in agents
        # All agents are synchronized (phase 0), so they're at their initial positions
        pos = get_position_at_time(agent.trajectory, 0, agent.phase_offset)
        available_cells = get_field_of_regard_at_position(agent, pos, env)
        
        actions = SensingAction[]
        push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))  # Wait action
        
        for cell in available_cells
            action = SensingAction(agent.id, [cell], false)
            if check_battery_feasible(agent, action, agent.battery_level)
                push!(actions, action)
            end
        end
        
        push!(agent_action_sets, actions)
    end
    
    # Generate all combinations of individual actions using Cartesian product
    for action_combo in Iterators.product(agent_action_sets...)
        push!(joint_actions, JointAction(collect(action_combo)))
    end
    
    return joint_actions
end

"""
calculate_joint_reward(jb::SyncMultiAgentState, joint_action::JointAction, agents::Vector{Agent}, env)
Calculates the reward for a joint action.
"""
function calculate_joint_reward(jb::SyncMultiAgentState, joint_action::JointAction, agents::Vector{Agent}, env)
    total_reward = 0.0
    
    for (i, action) in enumerate(joint_action.actions)
        if !isempty(action.target_cells)
            for cell in action.target_cells
                # Use the sophisticated reward function
                cell_reward = calculate_sophisticated_reward(jb.belief, cell)
                total_reward += cell_reward
            end
        end
    end
    
    return total_reward
end

"""
update_belief_with_joint_observations(belief::Belief, joint_action::JointAction, agents::Vector{Agent}, 
                                    positions::Vector{Tuple{Int, Int}}, env)
Updates belief with observations from all agents.
"""
function update_belief_with_joint_observations(belief::Belief, joint_action::JointAction, agents::Vector{Agent}, 
                                             positions::Vector{Tuple{Int, Int}}, env)
    updated_belief = deepcopy(belief)
    observed_cells = Set{Tuple{Int, Int}}()
    
    for (i, action) in enumerate(joint_action.actions)
        if isempty(action.target_cells)
            continue  # Wait action
        end
        
        agent = agents[i]
        pos = positions[i]
        
        # Get the Field of Regard at this position
        for_cells = get_field_of_regard_at_position(agent, pos, env)
        
        # Simulate observations for this agent's action
        for cell in action.target_cells
            x, y = cell
            if 1 <= x <= env.width && 1 <= y <= env.height && cell in for_cells
                # Only update if this cell hasn't been observed yet (priority to first observer)
                if !(cell in observed_cells)
                    # Get current belief distribution for this cell
                    current_dist = updated_belief.event_distributions[:, y, x]
                    
                    # Simulate perfect observation (most likely state becomes certain)
                    most_likely_state_idx = argmax(current_dist)
                    
                    # Create a more certain distribution around the most likely state
                    new_dist = fill(0.1, length(current_dist))
                    new_dist[most_likely_state_idx] = 0.7
                    
                    # Normalize
                    new_dist ./= sum(new_dist)
                    updated_belief.event_distributions[:, y, x] = new_dist
                    
                    # Mark this cell as observed
                    push!(observed_cells, cell)
                end
            end
        end
    end
    
    return updated_belief
end

"""
calculate_sophisticated_reward(belief::Belief, cell::Tuple{Int, Int})
Calculates sophisticated reward for sensing actions.

R = w_H·(H_prior(b_j) - H_post(b_j)) + w_F·E[F_I_j]

Where:
- w_H: Weight for entropy reduction (inter-agent coordination)
- w_F: Weight for state value (detection priority)
- E[F_I_j]: Expected value of information state I_k under current belief
"""
function calculate_sophisticated_reward(belief::Belief, cell::Tuple{Int, Int})
    # 1. Entropy-based reward: w_H * (H_prior - H_post)
    H_before = calculate_cell_entropy(belief, cell)
    H_after = 0.0  # Simplified: assume perfect observation
    entropy_reward = DEFAULT_ENTROPY_WEIGHT * (H_before - H_after)
    
    # 2. State value reward: w_F * E[F_I_j]
    # Calculate expected value under current belief
    event_prob = get_event_probability(belief, cell)
    no_event_prob = 1.0 - event_prob
    
    # E[F_I_j] = Σ_k p(I_k) * F_k
    expected_value = no_event_prob * DEFAULT_STATE_VALUES[1] + event_prob * DEFAULT_STATE_VALUES[2]
    value_reward = DEFAULT_VALUE_WEIGHT * expected_value
    
    # Total reward for this cell
    return entropy_reward + value_reward
end

"""
get_field_of_regard_at_position(agent::Agent, position::Tuple{Int, Int}, env)
Gets field of regard for an agent at a specific position.
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
    elseif agent.sensor.pattern == :circular
        # Circular sensor: agent's position and all 8 adjacent cells (9-cell pattern)
        for dx in -1:1, dy in -1:1
            nx, ny = x + dx, y + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                push!(fov_cells, (nx, ny))
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

"""
Configure reward function weights
"""
function configure_reward_weights(; w_H::Float64=1.0, w_F::Float64=0.5)
    global DEFAULT_ENTROPY_WEIGHT = w_H
    global DEFAULT_VALUE_WEIGHT = w_F
    
    println("🎯 Joint reward weights configured:")
    println("   • Entropy weight (w_H): $w_H (coordination)")
    println("   • Value weight (w_F): $w_F (detection priority)")
end

"""
Set reward configuration from main.jl constants
"""
function set_reward_config_from_main(entropy_weight::Float64, value_weight::Float64, 
                                   information_states::Vector{Int}, state_values::Vector{Float64})
    global DEFAULT_ENTROPY_WEIGHT = entropy_weight
    global DEFAULT_VALUE_WEIGHT = value_weight
    global DEFAULT_INFORMATION_STATES = information_states
    global DEFAULT_STATE_VALUES = state_values
    
    println("🎯 Joint reward configuration synced from main.jl:")
    println("   • Entropy weight (w_H): $entropy_weight")
    println("   • Value weight (w_F): $value_weight")
    println("   • Information states: $information_states")
    println("   • State values: $state_values")
end

end # module

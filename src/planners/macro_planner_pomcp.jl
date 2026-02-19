module MacroPlannerPOMCP

"""
Literature POMCP (Silver & Veness, NeurIPS 2010): Monte Carlo tree search with
particle belief, UCB action selection, and rollouts only at tree leaves.
Each agent plans independently. Returns an open-loop action sequence of length C
(same interface as other planners). For full online tree reuse, the framework
would need to call the planner every step with the updated belief.
"""

using POMDPs
using POMDPTools
using Random
using Statistics
using ..Types
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..Agent, ..SensingAction, ..GridObservation
import ..Agents.TrajectoryPlanner.get_position_at_time
import ..Agents.BeliefManagement.Belief, ..Agents.BeliefManagement.get_event_probability,
       ..Agents.BeliefManagement.collapse_belief_to, ..Agents.BeliefManagement.evolve_no_obs_fast,
       ..Agents.BeliefManagement.calculate_cell_entropy,
       ..Agents.BeliefManagement.calculate_uncertainty_map_from_distributions

export best_script, POMCPNode, pomcp_simulate!, pomcp_step,
       POMCPOnlinePolicy, init_online_pomcp_policy, pomcp_select_action!, pomcp_update_root!

# State is a grid of event presence: 1 = EVENT_PRESENT, 0 = NO_EVENT
const StateGrid = Matrix{Int}

# Observation key: tuple of event states at action.target_cells (same order)
const ObsKey = Tuple{Vararg{Int}}

"""
MCTS node for POMCP: stores visit count, per-action N and Q, and children (a, obs_key) -> node.
"""
mutable struct POMCPNode
    visits::Int
    action_N::Dict{SensingAction, Int}
    action_Q::Dict{SensingAction, Float64}
    children::Dict{Tuple{SensingAction, ObsKey}, POMCPNode}
    POMCPNode() = new(0, Dict{SensingAction, Int}(), Dict{SensingAction, Float64}(), Dict{Tuple{SensingAction, ObsKey}, POMCPNode}())
end

"""
Online POMCP policy: holds the current root node and hyperparameters.

This enables "literature" POMCP usage:
  - reuse the tree across timesteps
  - select one action per real step
  - then shift the root based on the *actual* observation.
"""
mutable struct POMCPOnlinePolicy
    root::POMCPNode
    gamma::Float64
    c_ucb::Float64
    max_depth::Int
    n_sims::Int
end

"""
Sample a state (grid) from the belief. Each cell is 1 (event) w.p. event_prob, else 0.
"""
function sample_state_from_belief(belief::Belief, env, rng::AbstractRNG)
    height, width = env.height, env.width
    state = zeros(Int, height, width)
    for y in 1:height, x in 1:width
        p = get_event_probability(belief, (x, y))
        state[y, x] = rand(rng) < p ? 1 : 0
    end
    return state
end

"""
Convert belief to N particle states (for POMCP root).
"""
function belief_to_particles(belief::Belief, env, N::Int, rng::AbstractRNG)
    [sample_state_from_belief(belief, env, rng) for _ in 1:N]
end

"""
Observation list from simulate_step is (cell, state) for each cell in action.target_cells.
Return a hashable key: tuple of states in the same order.
"""
function obs_to_key(obs::Vector{Tuple{Tuple{Int, Int}, Int}})
    tuple((ost for (_, ost) in obs)...)
end

"""
Build a belief that is certain at state s (for evolution then sample).
"""
function belief_from_state(state::StateGrid, env)
    height, width = size(state)
    num_states = 2  # NO_EVENT, EVENT_PRESENT
    dist = zeros(Float64, num_states, height, width)
    for y in 1:height, x in 1:width
        if state[y, x] == 1
            dist[2, y, x] = 1.0
        else
            dist[1, y, x] = 1.0
        end
    end
    uncertainty_map = calculate_uncertainty_map_from_distributions(dist)
    return Belief(deepcopy(dist), uncertainty_map, 0, [])
end

"""
Sample next state from current state using env dynamics: b_s = certain(s), evolve(b_s), sample.
"""
function sample_next_state(state::StateGrid, env, rng::AbstractRNG)
    b = belief_from_state(state, env)
    b_next = evolve_no_obs_fast(b, env, calculate_uncertainty=false)
    return sample_state_from_belief(b_next, env, rng)
end

"""
Get available actions for agent at timestep t (absolute time from gs_state).
"""
function get_available_actions(agent::Agent, env, t::Int, gs_state)
    global_timestep = gs_state.time_step + t - 1
    pos = get_position_at_time(agent.trajectory, global_timestep, agent.phase_offset)
    for_cells = get_field_of_regard_at_position(agent, pos, env)
    actions = SensingAction[SensingAction(agent.id, Tuple{Int, Int}[], false)]
    for cell in for_cells
        push!(actions, SensingAction(agent.id, [cell], false))
    end
    return actions
end

function get_field_of_regard_at_position(agent, position, env)
    x, y = position
    fov = Tuple{Int, Int}[]
    if agent.sensor.pattern == :cross
        for dx in -1:1, dy in -1:1
            nx, ny = x + dx, y + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                if (dx == 0 && dy == 0) || (dx == 0 && dy != 0) || (dx != 0 && dy == 0)
                    push!(fov, (nx, ny))
                end
            end
        end
    elseif agent.sensor.pattern == :row_only || (hasproperty(agent.sensor, :range) && agent.sensor.range == 0.0)
        for nx in 1:env.width
            push!(fov, (nx, y))
        end
    else
        r = round(Int, agent.sensor.range)
        for dx in -r:r, dy in -r:r
            nx, ny = x + dx, y + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height && sqrt(dx^2 + dy^2) <= agent.sensor.range
                push!(fov, (nx, ny))
            end
        end
    end
    return fov
end

"""
Simulate one step: (state, action) -> reward, next_state, observation (event states at observed cells).
Observation is a list of (cell, state) for cells in action.target_cells.
"""
function simulate_step(state::StateGrid, action::SensingAction, agent::Agent, env, rng::AbstractRNG)
    # Reward: information-gain style (entropy * event_prob) - use true state for consistency
    r = 0.0
    if !isempty(action.target_cells)
        for cell in action.target_cells
            x, y = cell
            event_here = state[y, x]
            p = Float64(event_here)
            # Use simple reward: 1 if event, 0 else (or could use entropy from belief in caller)
            r += p
        end
    end
    # Observation = true state at observed cells
    obs = Tuple{Tuple{Int, Int}, Int}[]
    for cell in action.target_cells
        x, y = cell
        push!(obs, (cell, state[y, x]))
    end
    # Next state: sample transition
    next_state = sample_next_state(state, env, rng)
    return r, next_state, obs
end

"""
Update belief with observation (cell -> 0 or 1).
"""
function update_belief_with_obs(belief::Belief, cell::Tuple{Int, Int}, observed_state::Int)
    b = deepcopy(belief)
    if observed_state == 1
        collapse_belief_to(b, cell, EVENT_PRESENT)
    else
        collapse_belief_to(b, cell, NO_EVENT)
    end
    return b
end

"""
Rollout from a state (state-only, no belief updates). Fast but inconsistent with belief-based planners.
"""
function rollout_from_state(state::StateGrid, agent::Agent, env, num_steps::Int, gamma::Float64, gs_state, rng::AbstractRNG; start_step::Int=1)
    num_steps <= 0 && return 0.0
    total = 0.0
    s = copy(state)
    for d in 1:num_steps
        t = start_step + d - 1
        actions = get_available_actions(agent, env, t, gs_state)
        isempty(actions) && break
        a = rand(rng, actions)
        r, s_next, _ = simulate_step(s, a, agent, env, rng)
        total += (gamma^(d - 1)) * r
        s = s_next
    end
    return total
end

"""
Rollout from state with full belief updates: maintain belief b, each step sample s~b, take a, get (r,s',obs),
update b with obs and evolve. Slower but consistent with ABBA/PBVI reward and dynamics.
start_step: plan step index (1-based) for the first rollout action.
"""
function rollout_from_state_with_belief_updates(state::StateGrid, agent::Agent, env, num_steps::Int, gamma::Float64, gs_state, rng::AbstractRNG; start_step::Int=1)
    num_steps <= 0 && return 0.0
    b = belief_from_state(state, env)
    total = 0.0
    for d in 1:num_steps
        t = start_step + d - 1
        actions = get_available_actions(agent, env, t, gs_state)
        isempty(actions) && break
        s = sample_state_from_belief(b, env, rng)
        a = rand(rng, actions)
        r, s_next, obs = simulate_step(s, a, agent, env, rng)
        total += (gamma^(d - 1)) * r
        for (cell, ost) in obs
            b = update_belief_with_obs(b, cell, ost)
        end
        b = evolve_no_obs_fast(b, env, calculate_uncertainty=false)
    end
    return total
end

"""
Single rollout from (belief, depth): sample state, take random actions, accumulate discounted reward.
(Legacy / for non-tree rollouts.)
"""
function rollout(belief::Belief, agent::Agent, env, depth::Int, gamma::Float64, gs_state, rng::AbstractRNG)
    depth <= 0 && return 0.0
    state = sample_state_from_belief(belief, env, rng)
    t = 1
    b = deepcopy(belief)
    total = 0.0
    for d in 1:depth
        actions = get_available_actions(agent, env, t, gs_state)
        isempty(actions) && break
        a = rand(rng, actions)
        r, state, obs = simulate_step(state, a, agent, env, rng)
        total += (gamma^(d - 1)) * r
        for (cell, ost) in obs
            b = update_belief_with_obs(b, cell, ost)
        end
        b = evolve_no_obs_fast(b, env, calculate_uncertainty=false)
        t += 1
    end
    return total
end

"""
UCB: argmax_a Q(h,a) + c * sqrt(log(N(h)+1) / (N(h,a)+1)). Untried actions (N(h,a)==0) get +Inf.
"""
function ucb_select_action(node::POMCPNode, actions::Vector{SensingAction}, c_ucb::Float64)
    n_h = node.visits
    best_a = actions[1]
    best_ucb = -Inf
    for a in actions
        n_ha = get(node.action_N, a, 0)
        q_ha = get(node.action_Q, a, 0.0)
        ucb = n_ha == 0 ? Inf : q_ha + c_ucb * sqrt(log(n_h + 1) / (n_ha + 1))
        if ucb > best_ucb
            best_ucb = ucb
            best_a = a
        end
    end
    return best_a
end

"""
Pick an action that has not been tried at this node (N(h,a)==0). If all tried, return nothing.
"""
function pick_untried_action(node::POMCPNode, actions::Vector{SensingAction})
    for a in actions
        if get(node.action_N, a, 0) == 0
            return a
        end
    end
    return nothing
end

"""
POMCP simulation (Silver & Veness): from state s at node, either expand (rollout) or recurse with UCB.
step_offset: plan step index (1-based) for this node (root = step_offset, depth 1 = step_offset+1, ...).
Returns discounted return and backs up into node.
"""
function pomcp_simulate!(s::StateGrid, node::POMCPNode, depth::Int, max_depth::Int,
                         agent::Agent, env, gs_state, gamma::Float64, c_ucb::Float64, rng::AbstractRNG; step_offset::Int=1)
    if depth >= max_depth
        return 0.0
    end
    t = step_offset + depth
    actions = get_available_actions(agent, env, t, gs_state)
    isempty(actions) && return 0.0

    if node.visits == 0
        # Tree leaf: expand by picking a random (untried) action, sample transition, rollout from s' with belief updates
        a = rand(rng, actions)
        r, s_next, obs = simulate_step(s, a, agent, env, rng)
        obs_key = isempty(obs) ? () : obs_to_key(obs)
        R_rollout = rollout_from_state_with_belief_updates(s_next, agent, env, max_depth - depth - 1, gamma, gs_state, rng; start_step=step_offset + depth + 1)
        total = r + gamma * R_rollout
        # Backup
        node.visits = 1
        node.action_N[a] = 1
        node.action_Q[a] = total
        child = POMCPNode()
        node.children[(a, obs_key)] = child
        return total
    end

    # UCB action selection
    a = ucb_select_action(node, actions, c_ucb)
    r, s_next, obs = simulate_step(s, a, agent, env, rng)
    obs_key = isempty(obs) ? () : obs_to_key(obs)
    key = (a, obs_key)

    if haskey(node.children, key)
        R_child = pomcp_simulate!(s_next, node.children[key], depth + 1, max_depth, agent, env, gs_state, gamma, c_ucb, rng; step_offset=step_offset)
    else
        child = POMCPNode()
        node.children[key] = child
        R_child = rollout_from_state_with_belief_updates(s_next, agent, env, max_depth - depth - 1, gamma, gs_state, rng; start_step=step_offset + depth + 1)
    end
    total = r + gamma * R_child

    # Backup
    node.visits += 1
    n_old = get(node.action_N, a, 0)
    q_old = get(node.action_Q, a, 0.0)
    node.action_N[a] = n_old + 1
    node.action_Q[a] = (q_old * n_old + total) / (n_old + 1)
    return total
end

"""
One-step POMCP: particle belief, M simulations, return best action (argmax Q(root,a)).
step_offset: plan step index (1-based) for the root node.
"""
function pomcp_step(particles::Vector{StateGrid}, agent::Agent, env, gs_state, max_depth::Int,
                   gamma::Float64, n_sims::Int, c_ucb::Float64, rng::AbstractRNG; step_offset::Int=1)
    root = POMCPNode()
    for _ in 1:n_sims
        s = rand(rng, particles)
        pomcp_simulate!(s, root, 0, max_depth, agent, env, gs_state, gamma, c_ucb, rng; step_offset=step_offset)
    end
    actions = get_available_actions(agent, env, step_offset, gs_state)
    if isempty(actions)
        return SensingAction(agent.id, Tuple{Int, Int}[], false), root
    end
    best_a = actions[1]
    best_q = -Inf
    for a in actions
        q = get(root.action_Q, a, -Inf)
        if q > best_q
            best_q = q
            best_a = a
        end
    end
    return best_a, root
end

"""
Initialise an online POMCP policy.

The caller is responsible for:
  - providing the current belief at each real timestep,
  - passing the appropriate step_offset (plan index) when selecting actions,
  - calling `pomcp_update_root!` after observing the *actual* observation.
"""
function init_online_pomcp_policy(env; n_sims::Int=200, max_depth::Int=10, c_ucb::Float64=1.4)
    gamma = env.discount
    return POMCPOnlinePolicy(POMCPNode(), gamma, c_ucb, max_depth, n_sims)
end

"""
Run online POMCP from the current root and select the next action.

This function:
  - samples particles from the *current* belief,
  - reuses and grows the existing tree rooted at `policy.root`,
  - returns the argmax-a Q(root,a) according to the updated tree.
"""
function pomcp_select_action!(policy::POMCPOnlinePolicy,
                              belief::Belief,
                              agent::Agent,
                              env,
                              gs_state;
                              step_offset::Int=1,
                              n_particles::Int=100,
                              rng::AbstractRNG=Random.GLOBAL_RNG)

    particles = belief_to_particles(belief, env, n_particles, rng)
    for _ in 1:policy.n_sims
        s = rand(rng, particles)
        pomcp_simulate!(s, policy.root, 0, policy.max_depth,
                        agent, env, gs_state, policy.gamma, policy.c_ucb, rng;
                        step_offset=step_offset)
    end

    actions = get_available_actions(agent, env, step_offset, gs_state)
    if isempty(actions)
        return SensingAction(agent.id, Tuple{Int, Int}[], false)
    end
    best_a = actions[1]
    best_q = -Inf
    for a in actions
        q = get(policy.root.action_Q, a, -Inf)
        if q > best_q
            best_q = q
            best_a = a
        end
    end
    return best_a
end

"""
After executing action `a` in the real environment and receiving an
observation key `obs_key` (tuple of observed event states in action.target_cells
order), shift the tree root to the corresponding child.

If the child does not exist yet (unseen (a,o) pair), the root is reset
to a fresh node, matching literature POMCP behaviour.
"""
function pomcp_update_root!(policy::POMCPOnlinePolicy, a::SensingAction, obs_key::ObsKey)
    key = (a, obs_key)
    if haskey(policy.root.children, key)
        policy.root = policy.root.children[key]
    else
        policy.root = POMCPNode()
    end
    return policy
end

"""
best_script(env, belief, agent, C, other_scripts, gs_state; n_sims=200, n_particles=100, c_ucb=1.4, rng)
Literature POMCP: at each step convert belief to particles, run MCTS with UCB and rollouts at leaves,
pick best action, update belief (sample obs for script mode). Returns (sequence of C actions, planning_time).
"""
function best_script(env, belief::Belief, agent::Agent, C::Int, other_scripts, gs_state;
                     n_sims::Int=200, n_particles::Int=100, c_ucb::Float64=1.4, rng::AbstractRNG=Random.GLOBAL_RNG)
    start_time = time()
    gamma = env.discount
    sequence = SensingAction[]
    b = deepcopy(belief)

    for t in 1:C
        actions = get_available_actions(agent, env, t, gs_state)
        if isempty(actions)
            push!(sequence, SensingAction(agent.id, Tuple{Int, Int}[], false))
            continue
        end
        max_depth = C - t + 1
        particles = belief_to_particles(b, env, n_particles, rng)
        best_a, _ = pomcp_step(particles, agent, env, gs_state, max_depth, gamma, n_sims, c_ucb, rng; step_offset=t)
        push!(sequence, best_a)

        # Update belief for next step (script mode: sample observation from current belief)
        if !isempty(best_a.target_cells)
            for cell in best_a.target_cells
                p = get_event_probability(b, cell)
                obs_state = rand(rng) < p ? 1 : 0
                b = update_belief_with_obs(b, cell, obs_state)
            end
        end
        b = evolve_no_obs_fast(b, env, calculate_uncertainty=false)
    end

    planning_time = time() - start_time
    println("✅ POMCP (literature): sequence for agent $(agent.id) in $(round(planning_time, digits=3)) s (n_sims=$n_sims, n_particles=$n_particles, c_ucb=$c_ucb)")
    return sequence, planning_time
end

end # module

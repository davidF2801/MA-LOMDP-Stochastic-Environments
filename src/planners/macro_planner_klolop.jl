#=
  KL-OLOP: Kullback-Leibler Open-Loop Optimistic Planning (per-agent, individual baseline).
  Based on: Leurent & Maillard, "Practical Open-Loop Optimistic Planning" (2019).
  - Open-loop: commits to a single action sequence per planning phase (no observation feedback).
  - Purely individual: no coordination with other agents; uses same belief-space reward as ABBA.
  - Budget n = M * L (M trajectories of length L); returns most-played sequence.

  Beliefs and replanning: This module does not update beliefs. The caller (ground station) must
  update the global belief with new observations before calling best_script_klolop. Replanning
  happens when an agent syncs: the ground station calls update_global_belief!(..., observations, ...)
  then invokes best_script_klolop(env, gs_state.global_belief, ...), so the plan is always based
  on the latest belief.
=#

module MacroPlannerKLOLOP

using Random
using Statistics
using ..Types
import ..Types: check_battery_feasible
import ..Agents.TrajectoryPlanner.get_position_at_time
# Belief and reward
import ..Agents.BeliefManagement.Belief, ..Agents.BeliefManagement.collapse_belief_to,
       ..Agents.BeliefManagement.evolve_no_obs_fast, ..Agents.BeliefManagement.get_event_probability,
       ..Agents.BeliefManagement.calculate_uncertainty_from_distribution
import ..MacroPlannerPBVI: calculate_sophisticated_reward

export best_script_klolop

# Reward normalization for KL-OLOP (paper assumes r in [0,1])
const R_MIN = -0.5
const R_MAX = 1.5

function _normalize_reward(r::Float64)
    return clamp((r - R_MIN) / (R_MAX - R_MIN + 1e-10), 0.0, 1.0)
end

# Bernoulli KL divergence
function _kl_bernoulli(p::Float64, q::Float64)
    p = clamp(p, 1e-10, 1 - 1e-10)
    q = clamp(q, 1e-10, 1 - 1e-10)
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
end

# Upper confidence bound for mean: max q in [0,1] s.t. T * d_BER(hat_mu, q) <= f
function _kl_upper_mu(hat_mu::Float64, T::Int, f::Float64)
    hat_mu = clamp(hat_mu, 1e-10, 1 - 1e-10)
    if T <= 0
        return 1.0
    end
    rhs = f / T
    if rhs >= _kl_bernoulli(hat_mu, 1.0)
        return 1.0
    end
    lo, hi = hat_mu, 1.0
    for _ in 1:50
        mid = (lo + hi) / 2
        if _kl_bernoulli(hat_mu, mid) <= rhs
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

"""
    get_actions_per_timestep(agent, env, C, gs_state)
Return Vector{Vector{SensingAction}}: for each t in 1:C, list of feasible actions.
"""
function get_actions_per_timestep(agent, env, C::Int, gs_state)
    if C == 0
        return Vector{Vector{SensingAction}}()
    end
    actions_per_timestep = Vector{Vector{SensingAction}}()
    for t in 1:C
        global_timestep = gs_state.time_step + t - 1
        pos = get_position_at_time(agent.trajectory, global_timestep, agent.phase_offset)
        for_cells = get_field_of_regard_at_position(agent, pos, env)
        timestep_actions = SensingAction[]
        push!(timestep_actions, SensingAction(agent.id, Tuple{Int,Int}[], false))
        two_cell_only = env.max_sensing_targets >= 2 && Types.CONTIGUOUS_PAIRS_ONLY[]
        if two_cell_only && length(for_cells) > 1
            for subset in Types.contiguous_pairs(for_cells)
                action = SensingAction(agent.id, collect(subset), false)
                if check_battery_feasible(agent, action, agent.battery_level)
                    push!(timestep_actions, action)
                end
            end
        else
            for cell in for_cells
                action = SensingAction(agent.id, [cell], false)
                if check_battery_feasible(agent, action, agent.battery_level)
                    push!(timestep_actions, action)
                end
            end
            if length(for_cells) > 1 && env.max_sensing_targets > 1
                if Types.CONTIGUOUS_PAIRS_ONLY[]
                    for subset in Types.contiguous_pairs(for_cells)
                        action = SensingAction(agent.id, collect(subset), false)
                        if check_battery_feasible(agent, action, agent.battery_level)
                            push!(timestep_actions, action)
                        end
                    end
                else
                    for subset_size in 2:min(env.max_sensing_targets, length(for_cells))
                        for subset in _combinations(for_cells, subset_size)
                            action = SensingAction(agent.id, collect(subset), false)
                            if check_battery_feasible(agent, action, agent.battery_level)
                                push!(timestep_actions, action)
                            end
                        end
                    end
                end
            end
        end
        push!(actions_per_timestep, timestep_actions)
    end
    return actions_per_timestep
end

function _combinations(arr, k)
    n = length(arr)
    if k > n || k < 0
        return Tuple{Int,Int}[]
    end
    out = []
    function comb(start, chosen)
        if length(chosen) == k
            push!(out, copy(chosen))
            return
        end
        for i in start:(n - k + length(chosen) + 1)
            push!(chosen, arr[i])
            comb(i + 1, chosen)
            pop!(chosen)
        end
    end
    comb(1, [])
    return out
end

function get_field_of_regard_at_position(agent, position, env)
    x, y = position
    fov_cells = Tuple{Int, Int}[]
    if agent.sensor.pattern == :cross
        ax, ay = position
        for dx in -1:1, dy in -1:1
            nx, ny = ax + dx, ay + dy
            if 1 <= nx <= env.width && 1 <= ny <= env.height
                if (dx == 0 && dy == 0) || (dx == 0 && dy != 0) || (dx != 0 && dy == 0)
                    push!(fov_cells, (nx, ny))
                end
            end
        end
    elseif agent.sensor.pattern == :row_only || agent.sensor.range == 0.0
        for nx in 1:env.width
            push!(fov_cells, (nx, y))
        end
    else
        sensor_range = round(Int, agent.sensor.range)
        for dx in -sensor_range:sensor_range
            for dy in -sensor_range:sensor_range
                nx, ny = x + dx, y + dy
                if 1 <= nx <= env.width && 1 <= ny <= env.height
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

function sample_event_state_from(belief::Belief, cell::Tuple{Int, Int})
    p_event = get_event_probability(belief, cell)
    return rand() < p_event ? Types.EVENT_PRESENT : Types.NO_EVENT
end

"""
    rollout_rewards(env, belief, agent, sequence, C, gs_state, rng) -> Vector{Float64}
Purely individual rollout: belief-space reward at each step, then sample obs and evolve.
"""
function rollout_rewards(env, belief::Belief, agent, sequence::Vector{SensingAction}, C::Int, gs_state, rng::AbstractRNG)
    b = deepcopy(belief)
    rewards = Float64[]
    for t in 1:C
        action = sequence[t]
        r_step = 0.0
        if !isempty(action.target_cells)
            for cell in action.target_cells
                r_step += calculate_sophisticated_reward(b, cell)
            end
        end
        push!(rewards, _normalize_reward(r_step))
        if !isempty(action.target_cells)
            for cell in action.target_cells
                state_i = sample_event_state_from(b, cell)
                b = collapse_belief_to(b, cell, state_i)
            end
        end
        b = evolve_no_obs_fast(b, env, calculate_uncertainty = false)
    end
    return rewards
end

function prefix_to_sequence(prefix::Tuple, actions_per_timestep)
    seq = SensingAction[]
    for (t, idx) in enumerate(prefix)
        push!(seq, actions_per_timestep[t][idx])
    end
    return seq
end

"""
    best_script_klolop(env, belief, agent, C, gs_state; rng, budget, gamma)

KL-OLOP (lazy): open-loop optimistic planning for one agent.

- **Belief**: Caller must pass the current global belief *after* updating it with the syncing
  agent's new observations. The ground station does this by calling update_global_belief!()
  before invoking this function when an agent syncs.
- **Replanning**: Invoked only at sync time (by the ground station). Each call produces a
  fresh plan from the given belief; no internal state is kept between syncs.
- **Budget**: n = number of rollouts (M trajectories of length L=C); M = floor(n / L).
- **Returns**: (best_sequence::Vector{SensingAction}, planning_time).
"""
function best_script_klolop(env, belief::Belief, agent, C::Int, gs_state;
                           rng::AbstractRNG = Random.GLOBAL_RNG,
                           budget::Int = 500,
                           gamma::Float64 = 0.95)
    start_time = time()
    if C == 0
        return SensingAction[], time() - start_time
    end

    actions_per_timestep = get_actions_per_timestep(agent, env, C, gs_state)
    if any(isempty(actions_per_timestep[t]) for t in 1:C)
        return SensingAction[], time() - start_time
    end

    L = C
    M = max(1, div(budget, L))
    f_val = 2 * log(max(M, 2)) + 2 * log(max(log(max(M, 2)), 0.5))

    T_map = Dict{Tuple, Int}()
    S_map = Dict{Tuple, Float64}()

    T_plus = Set{Tuple}()
    L_plus = Set{Tuple}()

    root = ()
    T_map[root] = 0
    S_map[root] = 0.0
    push!(T_plus, root)
    push!(L_plus, root)

    for m in 1:M
        B_max = -Inf
        best_leaf = nothing
        for a in L_plus
            h = length(a)
            U_val = 0.0
            for t in 1:h
                prefix_t = a[1:t]
                Ta = get(T_map, prefix_t, 0)
                Sa = get(S_map, prefix_t, 0.0)
                hat_mu = Ta > 0 ? Sa / Ta : 0.5
                U_mu = Ta > 0 ? _kl_upper_mu(hat_mu, Ta, f_val) : 1.0
                U_val += gamma^t * U_mu
            end
            U_val += gamma^(h + 1) / (1 - gamma)
            if U_val > B_max
                B_max = U_val
                best_leaf = a
            end
        end

        if best_leaf === nothing
            best_leaf = root
        end

        if length(best_leaf) < L
            am_prefix = Int[best_leaf...]
            for t in (length(best_leaf) + 1):L
                push!(am_prefix, rand(rng, 1:length(actions_per_timestep[t])))
            end
            am = tuple(am_prefix...)
        else
            am = best_leaf
        end

        seq_am = prefix_to_sequence(am, actions_per_timestep)
        rewards = rollout_rewards(env, belief, agent, seq_am, L, gs_state, rng)

        for t in 1:L
            prefix_t = am[1:t]
            T_map[prefix_t] = get(T_map, prefix_t, 0) + 1
            S_map[prefix_t] = get(S_map, prefix_t, 0.0) + rewards[t]
        end

        for t in 1:L
            K_t = length(actions_per_timestep[t])
            for j in 1:K_t
                new_prefix = t == 1 ? (j,) : tuple(am[1:(t - 1)]..., j)
                push!(T_plus, new_prefix)
            end
            if t == 1
                delete!(L_plus, ())
            else
                delete!(L_plus, am[1:(t - 1)])
            end
        end

        L_plus_new = Set{Tuple}()
        for node in T_plus
            depth = length(node)
            if depth == L
                push!(L_plus_new, node)
            else
                has_child = false
                for next_j in 1:length(actions_per_timestep[depth + 1])
                    child = tuple(node..., next_j)
                    if child in T_plus
                        has_child = true
                        break
                    end
                end
                if !has_child
                    push!(L_plus_new, node)
                end
            end
        end
        L_plus = L_plus_new
    end

    best_prefix = ()
    best_T = -1
    for node in T_plus
        if length(node) == L
            Ta = get(T_map, node, 0)
            if Ta > best_T
                best_T = Ta
                best_prefix = node
            end
        end
    end

    if best_T < 0
        am_first = tuple([rand(rng, 1:length(actions_per_timestep[t])) for t in 1:L]...)
        best_prefix = am_first
    end

    best_sequence = prefix_to_sequence(best_prefix, actions_per_timestep)
    planning_time = time() - start_time
    println("KL-OLOP: best sequence (T=$(get(T_map, best_prefix, 0)) plays) in $(round(planning_time, digits=3)) s")
    return best_sequence, planning_time
end

end # module

#=
  Joint ABBA: extends async ABBA by searching over joint action sequences for all agents
  during the planning window, with peer fixed-interval actions as singletons and dark-period
  peers enumerated. Only agent i's subsequence is returned. Joint space is exact Cartesian
  enumeration (non-empty sensing only), like ABBA's |A|^C but over the joint residual set.

  See task spec: residual joint action space, BranchBeliefs unchanged — dark peers enter
  obs_set like scheduled sensing (Case 2) via explicit actions in the joint candidate.
=#

module MacroPlannerAsyncJoint

using ..Types
import ..Types: check_battery_feasible, simulate_battery_evolution
import ..Agents.TrajectoryPlanner.get_position_at_time
import ..Agents.BeliefManagement: Belief, collapse_belief_to, evolve_no_obs_fast,
       merge_equivalent_beliefs, enumerate_all_possible_outcomes, clear_belief_evolution_cache!,
       get_cache_stats
import ..Types.combinations
using ..MacroPlannerAsync

export best_script_joint

"""
    feasible_actions_nonempty_only(agent, env, t_global, gs_state)

Feasible **sensing** actions at absolute time `t_global` (no empty / wait action).
Same geometry as async ABBA except the no-op is omitted so joint samples are never all-wait by construction.
"""
function feasible_actions_nonempty_only(agent, env, t_global::Int, gs_state)
    pos = get_position_at_time(agent.trajectory, t_global, agent.phase_offset)
    for_cells = MacroPlannerAsync.get_field_of_regard_at_position(agent, pos, env)
    timestep_actions = Types.SensingAction[]
    two_cell_only = env.max_sensing_targets >= 2 && Types.CONTIGUOUS_PAIRS_ONLY[]
    if two_cell_only && length(for_cells) > 1
        for subset in Types.contiguous_pairs(for_cells)
            action = Types.SensingAction(agent.id, collect(subset), false)
            if check_battery_feasible(agent, action, agent.battery_level)
                push!(timestep_actions, action)
            end
        end
    else
        for cell in for_cells
            action = Types.SensingAction(agent.id, [cell], false)
            if check_battery_feasible(agent, action, agent.battery_level)
                push!(timestep_actions, action)
            end
        end
        if length(for_cells) > 1 && env.max_sensing_targets > 1
            if Types.CONTIGUOUS_PAIRS_ONLY[]
                for subset in Types.contiguous_pairs(for_cells)
                    action = Types.SensingAction(agent.id, collect(subset), false)
                    if check_battery_feasible(agent, action, agent.battery_level)
                        push!(timestep_actions, action)
                    end
                end
            else
                for subset_size in 2:min(env.max_sensing_targets, length(for_cells))
                    for subset in combinations(for_cells, subset_size)
                        action = Types.SensingAction(agent.id, collect(subset), false)
                        if check_battery_feasible(agent, action, agent.battery_level)
                            push!(timestep_actions, action)
                        end
                    end
                end
            end
        end
    end
    return timestep_actions
end

"""First non-empty sensing action for a peer when only one choice is needed (e.g. spatially irrelevant dark)."""
function first_nonempty_or_nothing(agent, env, t_global::Int, gs_state)
    acts = feasible_actions_nonempty_only(agent, env, t_global, gs_state)
    return isempty(acts) ? nothing : acts[1]
end

@enum PeerIntervalClass fixed_interval dark_period before_sync outside_window

function peer_interval_class(j::Int, t_global::Int, t_sync_i::Int, H_i::Int, gs_state)
    if t_global < t_sync_i || t_global > t_sync_i + H_i - 1
        return outside_window
    end
    t_sync_j = gs_state.agent_last_sync[j]
    plan_j = get(gs_state.agent_plans, j, nothing)
    H_j = plan_j === nothing ? 0 : length(plan_j)
    if t_global < t_sync_j
        return before_sync
    end
    plan_idx = t_global - t_sync_j + 1
    if 1 <= plan_idx <= H_j
        return fixed_interval
    end
    return dark_period
end

"""
    dark_period_spatially_irrelevant_to_i(env, agent_i, agent_j, t_sync_i, H_i, gs_state)

Heuristic for Proposition 2 / Claim 3: if j's FOV never overlaps i's FOV over dark timesteps in the window,
peer j's dark sensing is collapsed to a single non-empty action (first feasible) when available.
"""
function dark_period_spatially_irrelevant_to_i(env, agent_i, agent_j, t_sync_i::Int, H_i::Int, gs_state)::Bool
    for k in 1:H_i
        t_global = t_sync_i + k - 1
        if peer_interval_class(agent_j.id, t_global, t_sync_i, H_i, gs_state) != dark_period
            continue
        end
        pos_i = get_position_at_time(agent_i.trajectory, t_global, agent_i.phase_offset)
        pos_j = get_position_at_time(agent_j.trajectory, t_global, agent_j.phase_offset)
        cells_i = Set(MacroPlannerAsync.get_field_of_regard_at_position(agent_i, pos_i, env))
        cells_j = Set(MacroPlannerAsync.get_field_of_regard_at_position(agent_j, pos_j, env))
        if !isempty(intersect(cells_i, cells_j))
            return false
        end
    end
    return true
end

"""
Build `per_step[k][agent_id] = Vector{SensingAction}` for k = 1..C.
"""
function build_per_step_joint_action_sets(env, agent_i, C::Int, gs_state;
    skip_spatial_irrelevant_dark::Bool = false)
    t_sync_i = gs_state.time_step
    tau_i = t_sync_i
    all_ids = sort(collect(keys(env.agents)))
    per_step = Vector{Dict{Int, Vector{Types.SensingAction}}}(undef, C)
    for k in 1:C
        t_global = tau_i + k - 1
        d = Dict{Int, Vector{Types.SensingAction}}()
        for aid in all_ids
            ag = env.agents[aid]
            if aid == agent_i.id
                d[aid] = feasible_actions_nonempty_only(ag, env, t_global, gs_state)
                continue
            end
            cls = peer_interval_class(aid, t_global, t_sync_i, C, gs_state)
            if cls == fixed_interval
                plan_j = gs_state.agent_plans[aid]
                H_j = plan_j === nothing ? 0 : length(plan_j)
                plan_idx = t_global - gs_state.agent_last_sync[aid] + 1
                if plan_j !== nothing && 1 <= plan_idx <= H_j
                    # Scheduled peer action (may be wait) — must stay fixed for consistency
                    d[aid] = [plan_j[plan_idx]]
                else
                    d[aid] = feasible_actions_nonempty_only(ag, env, t_global, gs_state)
                end
            elseif cls == before_sync
                d[aid] = feasible_actions_nonempty_only(ag, env, t_global, gs_state)
            elseif cls == dark_period
                if skip_spatial_irrelevant_dark && dark_period_spatially_irrelevant_to_i(env, agent_i, ag, t_sync_i, C, gs_state)
                    one = first_nonempty_or_nothing(ag, env, t_global, gs_state)
                    d[aid] = one === nothing ? Types.SensingAction[] : [one]
                else
                    d[aid] = feasible_actions_nonempty_only(ag, env, t_global, gs_state)
                end
            else
                d[aid] = feasible_actions_nonempty_only(ag, env, t_global, gs_state)
            end
        end
        per_step[k] = d
    end
    return per_step
end

"""
Exact count of joint action sequences: ∏ₖ (∏ₐ |feasible actions at (k,a)|).
"""
function total_joint_sequences_count(per_step::Vector{Dict{Int, Vector{Types.SensingAction}}}, all_ids::Vector{Int})::Int
    total = 1
    for k in eachindex(per_step)
        step_prod = 1
        for aid in all_ids
            step_prod *= length(per_step[k][aid])
        end
        total *= step_prod
    end
    return total
end

"""Build `joint_plan` from one element of the Cartesian product over per-step agent tuples."""
function joint_dict_from_step_tuple(step_seq::Tuple, all_ids::Vector{Int}, C::Int)
    joint = Dict{Int, Vector{Types.SensingAction}}()
    for aid in all_ids
        joint[aid] = Vector{Types.SensingAction}(undef, C)
    end
    for k in 1:C
        tup = step_seq[k]
        for (ii, aid) in enumerate(all_ids)
            joint[aid][k] = tup[ii]
        end
    end
    return joint
end

"""
Evaluate joint plan: same branching as `calculate_macro_script_reward`, but `obs_set` includes
dark-period peer actions from `joint_plan` (scheduled/fixed peers still come from `get_scheduled_observations_at_time`).
"""
function calculate_macro_script_reward_joint(seq_i::Vector{Types.SensingAction},
    joint_plan::Dict{Int, Vector{Types.SensingAction}}, C::Int, env, agent_i, B_branches, gs_state)
    γ = env.discount
    c_obs = 0.0
    tau_i = gs_state.time_step
    R_seq = zeros(length(seq_i))
    B_post = Dict{Int, Vector{Tuple{Belief, Float64}}}()
    B_post[tau_i] = B_branches[tau_i]

    current_battery = agent_i.battery_level
    for k in 1:length(seq_i)
        a_i = seq_i[k]
        t_global = tau_i + k - 1
        current_battery = simulate_battery_evolution(agent_i, a_i, current_battery)
        if !check_battery_feasible(agent_i, a_i, current_battery)
            return -1000.0
        end
    end

    for k in 1:length(seq_i)
        a_i = seq_i[k]
        t_global = tau_i + k - 1
        new_branches = Vector{Tuple{Belief, Float64}}()
        for (B, p_branch) in B_post[t_global]
            obs_map = Dict{Int, Types.SensingAction}()
            for (aid, act) in MacroPlannerAsync.get_scheduled_observations_at_time(t_global, gs_state)
                obs_map[aid] = act
            end
            for aid in keys(joint_plan)
                aid == agent_i.id && continue
                cls = peer_interval_class(aid, t_global, tau_i, C, gs_state)
                if cls == dark_period || cls == before_sync
                    obs_map[aid] = joint_plan[aid][k]
                end
            end
            obs_map[agent_i.id] = a_i
            obs_set = Tuple{Int, Types.SensingAction}[(a, obs_map[a]) for a in sort(collect(keys(obs_map)))]

            all_wait_actions = all(action.target_cells == Tuple{Int, Int}[] for (_, action) in obs_set)
            if !all_wait_actions
                for (observation_combo, probability) in enumerate_all_possible_outcomes(B, obs_set)
                    B_new = deepcopy(B)
                    for (cell, observed_state) in observation_combo
                        B_new = collapse_belief_to(B_new, cell, observed_state)
                    end
                    B_next = evolve_no_obs_fast(B_new, env, calculate_uncertainty = false)
                    push!(new_branches, (B_next, p_branch * probability))
                end
            else
                B_next = evolve_no_obs_fast(B, env, calculate_uncertainty = false)
                push!(new_branches, (B_next, p_branch))
            end
        end
        B_post[t_global + 1] = merge_equivalent_beliefs(new_branches)
        R_seq[k] = MacroPlannerAsync.compute_expected_reward(B_post[t_global], a_i, c_obs)
    end
    return sum((γ^(k - 1)) * R_seq[k] for k in 1:length(seq_i))
end

"""
    best_script_joint(env, belief, agent, C, other_scripts, gs_state; skip_spatial_irrelevant_dark)

Joint ABBA: exact enumeration over the Cartesian product of per-step non-empty joint sensing choices;
return best `agent`'s sequence (same evaluation as ABBA-style macro reward with joint obs_set).
"""
function best_script_joint(env, belief::Belief, agent, C::Int, other_scripts, gs_state;
    skip_spatial_irrelevant_dark::Bool = false)
    start_time = time()
    clear_belief_evolution_cache!()

    if C == 0
        return Types.SensingAction[], time() - start_time
    end

    all_ids = sort(collect(keys(env.agents)))
    per_step = build_per_step_joint_action_sets(env, agent, C, gs_state;
        skip_spatial_irrelevant_dark = skip_spatial_irrelevant_dark)
    for k in 1:C, aid in all_ids
        if isempty(per_step[k][aid])
            println("⚠️ Joint ABBA: no non-empty sensing action for agent $(aid) at local step $(k); cannot plan without empty actions.")
            return Types.SensingAction[], time() - start_time
        end
    end
    n_total = total_joint_sequences_count(per_step, all_ids)
    step_products = Int[]
    for k in 1:C
        p = 1
        for aid in all_ids
            p *= length(per_step[k][aid])
        end
        push!(step_products, p)
    end
    println("🔀 Joint ABBA: exact enumeration of $(n_total) joint action sequences (non-empty sensing only; no wait actions in the enumerated sets)")
    println("   Joint ABBA stepwise joint factors: $(step_products)")

    println("🔄 Pre-computing belief branches for agent $(agent.id) (joint ABBA)...")
    B_branches = MacroPlannerAsync.precompute_belief_branches(env, agent, gs_state)

    best_seq_i = Types.SensingAction[]
    best_value = -Inf
    step_iters = [Iterators.product([per_step[k][aid] for aid in all_ids]...) for k in 1:C]
    joint_iter = Iterators.product(step_iters...)

    for (idx, step_seq) in enumerate(joint_iter)
        joint_plan = joint_dict_from_step_tuple(step_seq, all_ids, C)
        seq_i = joint_plan[agent.id]
        val = calculate_macro_script_reward_joint(seq_i, joint_plan, C, env, agent, B_branches, gs_state)
        if val > best_value
            best_value = val
            best_seq_i = seq_i
        end
        if idx % 100 == 0
            println("  Joint ABBA: $(idx)/$(n_total) sequences, best = $(round(best_value, digits=3))")
        end
    end

    elapsed = time() - start_time
    cache_stats = get_cache_stats()
    println("✅ Joint ABBA best value: $(round(best_value, digits=3)) in $(round(elapsed, digits=3)) s (evaluated $(n_total) joint sequences)")
    println("📊 Cache: $(cache_stats[:hits]) hits, $(cache_stats[:misses]) misses")
    return best_seq_i, elapsed
end

end # module

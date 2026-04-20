#=
    DO-SB-ABBA-α : Decentralized Online SB-ABBA-α with tunable
                   Optimality ↔ Action-Consistency trade-off.

    This is an **online, reactive, decentralized** planner (not a macro-script
    planner), hence the new name. It was historically named "Dec-SB-ABBA" and
    lived in `macro_planner_dec_sb_abba.jl`; the file was renamed to reflect
    what the algorithm actually does.

    Reference intuition (paper):
        Shimron & Indelman, "Towards Optimal Performance and Action
        Consistency Guarantees in Dec-POMDPs with Inconsistent Beliefs
        and Limited Communication", arXiv:2512.20778, 2025.

    Problem setting in this codebase:
    ----------------------------------
    Agents share their observation history through the ground station
    only when they come into contact (τ_i = last sync for agent i).
    Between syncs, each agent observes privately and must act online.
    Trajectories are known and periodic; the *positions* of every
    agent at every future timestep are common knowledge. What is
    *not* known to agent i about agent j (and vice-versa) is:

        1. j's observation outcomes since j's last sync (and i's
           since i's last sync — j doesn't have these).
        2. After the open-loop plan that GS installed in j runs out
           (or when j plans reactively), j's future action choices.

    DO-SB-ABBA-α handles (1) via SB-ABBA-style Monte-Carlo particles
    (sampling plausible world realisations) and handles (2) via a
    light-weight *action predictor* (greedy one-step lookahead under
    i's hypothesised model of j's belief).

    Algorithm sketch (executed online, every timestep, per agent i):
    ----------------------------------------------------------------
    1. Seed N_particles plausible system realisations.
        b_MP^(i)  — i's MPOMDP hypothesis: start from GS snapshot at
                    i's last sync, sample all unshared observations
                    (including j's post-sync obs AND i's post-sync
                    obs treated symmetrically with regard to the
                    hypothesis).  Rolls forward to current time t.
        b_j^(i)   — i's model of j's belief: start from GS snapshot
                    at j's last sync (≤ i's last sync; i knows what
                    GS sent j), apply only observations that i
                    believes j has received, sample the rest.
    2. For each candidate self-action a at time t:
         V_opt(a)  = MC expectation of the H-step discounted return
                     when i plays a and the rest of the team plays
                     their *predicted* actions, started from a
                     b_MP^(i) particle.
         V_cons(a) = min_j MC expectation of the same rollout but
                     started from a b_j^(i) particle (i.e. "from
                     agent j's viewpoint, does a look beneficial?").
    3. Select
           a* = arg max_a  α · V_opt(a) + (1-α) · V_cons(a),
       where α ∈ [0,1] is the user-tunable weight:
           α = 1.0  ⇒ pure MPOMDP-optimality (ignore belief mismatch)
           α = 0.0  ⇒ pure MRAC-consistency (act on what every agent
                      would also consider valuable under *their*
                      information)
           α = 0.5  ⇒ balanced.

    Returns a reactive-policy closure (stored on agent.reactive_policy)
    that is called each timestep, exactly like POMCP and the PBVI
    policy-tree planners already do in this codebase.
=#

module OnlinePlannerDoSBABBA

using Random
using ..Types
import ..Types: check_battery_feasible, Belief
import ..EventState, ..NO_EVENT, ..EVENT_PRESENT
import ..Agent, ..SensingAction, ..GridObservation
import ..Agents.TrajectoryPlanner.get_position_at_time
import ..Agents.BeliefManagement.collapse_belief_to,
       ..Agents.BeliefManagement.evolve_no_obs_fast,
       ..Agents.BeliefManagement.get_event_probability
import ..MacroPlannerPBVI: calculate_sophisticated_reward
import ..MacroPlannerAsync: get_field_of_regard_at_position

export init_do_sb_abba_policy, set_do_sb_abba_params_from_main,
       get_do_sb_abba_params, set_do_consistency_weight_from_main

# =============================================================================
# Hyper-parameters (overridable from the experiment scripts)
# =============================================================================

"""
Parameters for DO-SB-ABBA-α.

* `alpha`        : weight in [0,1] between optimality (1) and consistency (0).
* `N_particles`  : number of Monte-Carlo world realisations sampled each step.
* `H`            : planning horizon used for every rollout. `-1` means use
                   the residual contact horizon of the agent.
* `predict_mode` : how each other agent j's action between t and t+H is
                   predicted when j's installed plan has run out:
                     :greedy_from_bj       - greedy one-step info-gain under b_j^(i)
                     :wait                 - j waits
                     :random               - uniform random feasible action
"""
struct DoSBABBAParams
    alpha::Float64
    N_particles::Int
    H::Int
    predict_mode::Symbol
end

const DO_SB_ABBA_PARAMS = Ref(DoSBABBAParams(0.5, 16, -1, :greedy_from_bj))

"""
    set_do_sb_abba_params_from_main(alpha, N_particles, H, predict_mode)

Override all DO-SB-ABBA hyper-parameters from the main experiment script.
"""
function set_do_sb_abba_params_from_main(alpha::Float64,
                                          N_particles::Int,
                                          H::Int = -1,
                                          predict_mode::Symbol = :greedy_from_bj)
    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0,1]"
    @assert N_particles > 0 "N_particles must be positive"
    DO_SB_ABBA_PARAMS[] = DoSBABBAParams(alpha, N_particles, H, predict_mode)
    println("🎚️  DO-SB-ABBA configured: α=$(alpha), N_particles=$(N_particles), " *
            "H=$(H < 0 ? "auto" : string(H)), predict_mode=:$(predict_mode)")
end

"""
    set_do_consistency_weight_from_main(alpha)

Convenience setter that only changes the α weight (keeps other defaults).
α = 1 → optimality-only, α = 0 → consistency-only.
"""
function set_do_consistency_weight_from_main(alpha::Float64)
    p = DO_SB_ABBA_PARAMS[]
    set_do_sb_abba_params_from_main(alpha, p.N_particles, p.H, p.predict_mode)
end

get_do_sb_abba_params() = DO_SB_ABBA_PARAMS[]

# =============================================================================
# Helpers
# =============================================================================

"""
Sample an EventState from a cell's belief (2-state).
"""
@inline function sample_event_state(belief::Belief, cell::Tuple{Int, Int}, rng::AbstractRNG)
    return rand(rng) < get_event_probability(belief, cell) ? EVENT_PRESENT : NO_EVENT
end

"""
    _feasible_actions_at(agent, position, env)

One-timestep feasible action set at a given position (wait + each FoV cell
+ contiguous 2-cell pairs when enabled). Respects battery.
"""
function _feasible_actions_at(agent::Agent, position::Tuple{Int,Int}, env)
    actions = SensingAction[]
    push!(actions, SensingAction(agent.id, Tuple{Int, Int}[], false))  # wait
    fov = get_field_of_regard_at_position(agent, position, env)

    two_cell_only = env.max_sensing_targets >= 2 && Types.CONTIGUOUS_PAIRS_ONLY[]
    if two_cell_only && length(fov) > 1
        for subset in Types.contiguous_pairs(fov)
            a = SensingAction(agent.id, collect(subset), false)
            if check_battery_feasible(agent, a, agent.battery_level)
                push!(actions, a)
            end
        end
    else
        for cell in fov
            a = SensingAction(agent.id, [cell], false)
            if check_battery_feasible(agent, a, agent.battery_level)
                push!(actions, a)
            end
        end
        if env.max_sensing_targets >= 2 && length(fov) > 1
            for subset in Types.contiguous_pairs(fov)
                a = SensingAction(agent.id, collect(subset), false)
                if check_battery_feasible(agent, a, agent.battery_level)
                    push!(actions, a)
                end
            end
        end
    end
    return actions
end

"""
    _predict_other_action(agent_j, t_global, snapshot_plans, snapshot_syncs,
                          b_j_hyp, env, predict_mode, rng)

Predict the action agent j will take at global time `t_global` from agent i's
viewpoint. Priority:

1. If j has an installed plan *that was known to i at i's last sync* and is
   still valid at `t_global`, use it. i does not see any new plan j may have
   received after i's sync — that is the whole point of the decentralised
   setting.
2. Otherwise, use `predict_mode`:
     :greedy_from_bj — greedy one-step sophisticated-reward maximiser under
                       i's hypothesised model of j's belief (b_j_hyp).
     :wait           — j waits.
     :random         — uniform random feasible action.
"""
function _predict_other_action(agent_j::Agent, t_global::Int,
                               snapshot_plans::Dict, snapshot_syncs::Dict,
                               b_j_hyp::Belief, env, predict_mode::Symbol,
                               rng::AbstractRNG)
    if haskey(snapshot_plans, agent_j.id) &&
       snapshot_plans[agent_j.id] !== nothing
        plan = snapshot_plans[agent_j.id]
        τ_j  = get(snapshot_syncs, agent_j.id, -1)
        if τ_j >= 0
            idx = t_global - τ_j + 1
            if 1 <= idx <= length(plan)
                return plan[idx]
            end
        end
    end

    pos   = get_position_at_time(agent_j.trajectory, t_global, agent_j.phase_offset)
    feas  = _feasible_actions_at(agent_j, pos, env)
    if isempty(feas)
        return SensingAction(agent_j.id, Tuple{Int, Int}[], false)
    end

    if predict_mode == :wait
        return feas[1]
    elseif predict_mode == :random
        return rand(rng, feas)
    else  # :greedy_from_bj
        best_r = -Inf
        best_a = feas[1]
        for a in feas
            r = 0.0
            for cell in a.target_cells
                r += calculate_sophisticated_reward(b_j_hyp, cell)
            end
            if r > best_r
                best_r = r
                best_a = a
            end
        end
        return best_a
    end
end

"""
    _sample_system_realisation(...) → Belief at t_end

Roll `B_start` forward from `t_start` to `t_end`, applying (sampled)
observations of every `observer` whose installed plan scheduled a sensing
action on the way, except those whose id is in `skip_observer_ids`.
"""
function _sample_system_realisation(B_start::Belief, t_start::Int, t_end::Int,
                                    observers::Vector{Agent},
                                    skip_observer_ids::Set{Int},
                                    snapshot_plans::Dict,
                                    snapshot_syncs::Dict,
                                    env, rng::AbstractRNG)
    b = deepcopy(B_start)
    for t in t_start:(t_end-1)
        for j in observers
            j.id in skip_observer_ids && continue
            if !haskey(snapshot_plans, j.id) || snapshot_plans[j.id] === nothing
                continue
            end
            plan = snapshot_plans[j.id]
            τ_j  = get(snapshot_syncs, j.id, -1)
            τ_j < 0 && continue
            idx  = t - τ_j + 1
            if 1 <= idx <= length(plan)
                action = plan[idx]
                for cell in action.target_cells
                    state = sample_event_state(b, cell, rng)
                    b = collapse_belief_to(b, cell, state)
                end
            end
        end
        b = evolve_no_obs_fast(b, env, calculate_uncertainty=false)
    end
    return b
end

"""
    _rollout_value(...) → Float64

Monte-Carlo rollout from belief b0 at time t_now. Self plays `a_self` at the
*first* step and predicted actions thereafter (greedy on its own evolving
belief); other agents play predicted actions at every step. Returns the
discounted sum of sophisticated rewards earned by self over H steps.
"""
function _rollout_value(a_self::SensingAction, b0::Belief, agent_i::Agent,
                        other_agents::Vector{Agent}, t_now::Int, H::Int,
                        snapshot_plans::Dict, snapshot_syncs::Dict,
                        env, predict_mode::Symbol,
                        rng::AbstractRNG)
    γ = env.discount
    b = deepcopy(b0)
    total = 0.0
    a_i = a_self
    for k in 0:(H-1)
        t_k = t_now + k

        r_step = 0.0
        for cell in a_i.target_cells
            r_step += calculate_sophisticated_reward(b, cell)
        end
        total += (γ^k) * r_step

        for cell in a_i.target_cells
            state = sample_event_state(b, cell, rng)
            b = collapse_belief_to(b, cell, state)
        end

        for j in other_agents
            a_j = _predict_other_action(j, t_k, snapshot_plans, snapshot_syncs,
                                        b, env, predict_mode, rng)
            for cell in a_j.target_cells
                state = sample_event_state(b, cell, rng)
                b = collapse_belief_to(b, cell, state)
            end
        end

        b = evolve_no_obs_fast(b, env, calculate_uncertainty=false)

        if k < H - 1
            next_pos = get_position_at_time(agent_i.trajectory, t_k + 1,
                                            agent_i.phase_offset)
            feas = _feasible_actions_at(agent_i, next_pos, env)
            if isempty(feas)
                a_i = SensingAction(agent_i.id, Tuple{Int, Int}[], false)
            else
                best_r = -Inf
                best_a = feas[1]
                for a in feas
                    r = 0.0
                    for cell in a.target_cells
                        r += calculate_sophisticated_reward(b, cell)
                    end
                    if r > best_r
                        best_r = r
                        best_a = a
                    end
                end
                a_i = best_a
            end
        end
    end
    return total
end

# =============================================================================
# Main reactive policy
# =============================================================================

"""
    init_do_sb_abba_policy(env, gs_state, agent_i; rng)

Build a reactive-policy closure implementing DO-SB-ABBA-α for `agent_i`.
The closure is called each execution step with `(local_obs_history, t)`.

All hyper-parameters are read from `DO_SB_ABBA_PARAMS[]`; configure them
from your experiment script via `set_do_sb_abba_params_from_main(...)`.
"""
function init_do_sb_abba_policy(env, gs_state, agent_i::Agent;
                                 rng::AbstractRNG = Random.GLOBAL_RNG)

    B_clean      = deepcopy(gs_state.global_belief)
    τ_i_sync     = gs_state.time_step
    τ_last_syncs = deepcopy(gs_state.agent_last_sync)
    snapshot_plans = deepcopy(gs_state.agent_plans)
    all_agents_sorted = sort(collect(env.agents), by=a->a.id)
    other_agents = [a for a in all_agents_sorted if a.id != agent_i.id]
    t_clean = gs_state.last_clean_time
    if t_clean < 0 && !isempty(τ_last_syncs)
        t_clean = minimum(values(τ_last_syncs))
    end
    if t_clean < 0
        t_clean = 0
    end

    return function reactive_policy(local_obs_history::Vector{GridObservation},
                                    current_time::Int)
        t_plan_start = time()
        params = DO_SB_ABBA_PARAMS[]
        α       = params.alpha
        N       = params.N_particles
        H_req   = params.H
        predmode= params.predict_mode

        C_residual = max(1, agent_i.trajectory.period -
                           (current_time - τ_i_sync) % agent_i.trajectory.period)
        H = H_req > 0 ? min(H_req, C_residual) : C_residual

        pos_i = get_position_at_time(agent_i.trajectory, current_time,
                                     agent_i.phase_offset)
        candidates = _feasible_actions_at(agent_i, pos_i, env)
        if isempty(candidates)
            return SensingAction(agent_i.id, Tuple{Int, Int}[], false)
        end

        b_i_now = agent_i.belief === nothing ? deepcopy(B_clean) :
                                               deepcopy(agent_i.belief)

        j_particles  = Dict{Int, Vector{Belief}}()
        for j in other_agents
            j_particles[j.id] = Belief[]
            for _ in 1:N
                b_j = _sample_system_realisation(
                    B_clean, t_clean, current_time,
                    all_agents_sorted, Set{Int}([agent_i.id]),
                    snapshot_plans, τ_last_syncs, env, rng
                )
                push!(j_particles[j.id], b_j)
            end
        end

        best_a      = candidates[1]
        best_score  = -Inf
        best_Vopt   = 0.0
        best_Vcons  = 0.0
        per_action  = Tuple{SensingAction, Float64, Float64, Float64}[]

        for a in candidates
            V_opt = 0.0
            for _ in 1:N
                V_opt += _rollout_value(a, b_i_now, agent_i, other_agents,
                                        current_time, H,
                                        snapshot_plans, τ_last_syncs,
                                        env, predmode, rng)
            end
            V_opt /= N

            if isempty(other_agents)
                V_cons = V_opt
            else
                V_cons = +Inf
                for j in other_agents
                    Vj = 0.0
                    for b_j in j_particles[j.id]
                        Vj += _rollout_value(a, b_j, agent_i, other_agents,
                                              current_time, H,
                                              snapshot_plans, τ_last_syncs,
                                              env, predmode, rng)
                    end
                    Vj /= N
                    V_cons = min(V_cons, Vj)
                end
            end

            score = α * V_opt + (1.0 - α) * V_cons
            push!(per_action, (a, V_opt, V_cons, score))
            if score > best_score
                best_score = score
                best_a     = a
                best_Vopt  = V_opt
                best_Vcons = V_cons
            end
        end

        step_time = time() - t_plan_start
        if haskey(gs_state.planning_times, agent_i.id)
            push!(gs_state.planning_times[agent_i.id], step_time)
        end
        gs_state.total_planning_time += step_time
        gs_state.num_plans_computed  += 1

        if current_time % 5 == 0
            n_sensing = count(a -> !isempty(a[1].target_cells), per_action)
            println("🧩 DO-SB-ABBA agent $(agent_i.id) t=$(current_time) " *
                    "α=$(α) H=$(H) N=$(N) — chose " *
                    "$(isempty(best_a.target_cells) ? "WAIT" : string(best_a.target_cells)) " *
                    "(V_opt=$(round(best_Vopt,digits=3)), " *
                    "V_cons=$(round(best_Vcons,digits=3)), " *
                    "score=$(round(best_score,digits=3)), " *
                    "$(length(per_action)) candidates, $(n_sensing) sensing)")
        end

        return best_a
    end
end

# -----------------------------------------------------------------------------
# Backward-compat aliases (old Dec-SB-ABBA names). Remove once all call-sites
# have migrated to the `do_sb_abba` naming.
# -----------------------------------------------------------------------------
const init_dec_sb_abba_policy          = init_do_sb_abba_policy
const set_dec_sb_abba_params_from_main = set_do_sb_abba_params_from_main
const get_dec_sb_abba_params           = get_do_sb_abba_params
const set_dec_consistency_weight_from_main = set_do_consistency_weight_from_main
const DEC_SB_ABBA_PARAMS               = DO_SB_ABBA_PARAMS

end # module OnlinePlannerDoSBABBA

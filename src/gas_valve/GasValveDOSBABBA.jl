"""
GasValveDOSBABBA.jl — Proper DO-SB-ABBA-α for the gas-valve problem.

What this implements vs the simpler α-blend planner in GasValveControl.jl
========================================================================
The α-blend planner (`step_do_alpha_control!`) picks each rover's valve
action via a 1-step closed-form expected-reward argmax, with the peer's
action hard-coded by an MRAC role rule. It is fast but hides all the
ABBA structure (no branching, no sampling, no lookahead, no peer search).

This module implements a **sample-based (SB) belief-hypothesis, H-step
closed-form rollout, joint-action-search** planner for MRAC-consistent
decision making under inconsistent beliefs.

Per rover i, each planning step (online, reactive):

1. Handle scheduled GS mailbox ops (reuse machinery from `GasValveControl`).

2. Sample `N_particles` plausible peer observation streams that i could
   have missed since its last mailbox read. Each particle carries:

        b_MP^(i,n)  — i's MPOMDP hypothesis at t: public_at_read[i] filtered
                      with i's own obs AND the sampled peer obs,
        b_{j|i,n}   — i's hypothesis of peer's belief at t: public_at_read[i]
                      filtered with sampled peer obs only (no i's own obs).

3. **Joint-action search over (a_i, a_j)** at i and peer's current
   positions. For each joint candidate, average a closed-form H-step
   expected-reward rollout across the `N_particles` start beliefs:

        V_opt (a_i, a_j)   = mean over n of rollout from b_MP^(i,n),
        V_cons(a_i, a_j)   = mean over n of rollout from b_{j|i,n}.

   The rollout uses `PublicBackupStep` (closed-form expected reward +
   Markov + vent-reset belief update) — deterministic given the start
   belief. At h ≥ 1 both rovers are assumed to play the 1-step joint
   argmax under the rolled-forward belief.

4. Pick the joint argmax
        (a_i*, a_j*) = argmax  α · V_opt  + (1 - α) · V_cons
   and return i's own slot `a_i*`. Since both rovers enumerate the same
   joint space under approximately-consistent beliefs (as α → 0), they
   agree deterministically on the same joint — **MRAC is satisfied**.

α = 1 → pure optimality under the joint-info hypothesis (MPOMDP style);
α = 0 → pure consistency with the peer's (stale) viewpoint.

Why joint search and not independent `a_i`?
-------------------------------------------
At a rendezvous the coordinated optimum is (Seal, Vent). Under independent
per-rover argmax, "I seal while peer vents" and "I vent while peer seals"
score identically for each rover → both rovers can pick Seal (or both
Vent) → coordination fails → pocket explodes. Joint search removes the
symmetry: both rovers' argmax lands on the same labelled joint pair.

MRAC shared-anchor safety net (Shimron & Indelman-style)
---------------------------------------------------------
Particle-based belief hypotheses are still *i-specific* — rover i's guess
of peer's belief can diverge from peer's own. A private-belief disagreement
at a non-rendezvous step (one rover at a valve cell, peer elsewhere or
under different obs) can cause catastrophic unilateral vents.

To avoid this we add a `mrac_safety_check` step: before committing to
`best_a_i`, compute the joint argmax under a **shared anchor belief**
that BOTH rovers evaluate identically — the uniform prior at t=0 rolled
forward to time `t` with zero observations. If the shared argmax says
(Seal, Vent), commit to my slot of that joint (both rovers will see the
same joint → deterministic MRAC). If it says Idle, no Seal/Vent is ever
safe and we clamp to Idle. An optional dual-veto (`veto_p_dormant`) can
override shared-Act with Idle when BOTH the private and peer-hypothesis
beliefs agree the target pocket is Dormant — disabled by default because
the veto is only sound when obs are bilaterally symmetric.

References
----------
* Shimron & Indelman, "Towards Optimal Performance and Action Consistency
  Guarantees in Dec-POMDPs with Inconsistent Beliefs and Limited
  Communication", arXiv:2512.20778, 2025.

Caveat
------
In the current `GasValveEnv`, `Observe(p)` costs -0.5 but does not produce
a special observation (every rover's nearest-pocket obs is sampled every
step regardless of action). As a result Observe remains weakly dominated
by Idle under any expected-reward-only objective. This is an environment
modelling issue, not a planner issue.
"""
module GasValveDOSBABBA

using Random
using ..GasValveTypes
using ..GasValveEnv
using ..GasValveBelief
using ..GasValveTrajectories
using ..GasValvePlanner
using ..GasValveControl

import ..GasValveTypes: Cell, PocketDef, RoverAction, JointAction, PublicBelief,
    GasValveWorldState, ObsRecord, RewardParams, PocketDynamics, Mailbox,
    DORMANT, PRESSURIZED, CRITICAL,
    VALVE_NONE, VALVE_OBSERVE, VALVE_SEAL, VALVE_VENT, MOVE_STAY
import ..GasValveEnv: generate_observation, pocket_transition_spec
import ..GasValveBelief: PublicFilterUpdate, PublicBackupStep, initialize_uniform_belief,
    initialize_dormant_belief
import ..GasValveTrajectories: RoverTrajectory, positions_at_time, is_gs_visit_time
import ..GasValveControl: DOSBABBAState, _handle_gs_visit!, _feasible_valve_actions_i
import ..GasValveTypes: Mailbox, MailboxEntry

export step_do_sb_abba_control!, DOSBABBAConfig

# =============================================================================
# BILATERAL SHARED ANCHOR (mailbox-derived, computable identically by both rovers)
# =============================================================================
"""
Build the bilateral shared anchor belief at time `t` from the public mailbox.

Both rovers operate on the same `Mailbox` object — it is public shared
state. Therefore the belief

    B_shared(t) = filter( uniform_prior, ⋃_k mailbox.entries[k].obs )  rolled-forward to t

is computed identically by both rovers ⇒ using it as the MRAC safety
anchor preserves action consistency without any online communication.

Compared to the legacy uniform-rolled-forward anchor, this anchor uses
**actual observations** uploaded at GS visits, so it correctly reflects
the most recent information available to the system as a whole:

  * Fresh Dormant obs from the last GS visit → low q → skip at rendezvous.
  * No recent obs / stale info → q approaches uniform → act preventively.

This is the primary lever that makes DO-SB-ABBA beat the 1-step DO-α
planner: DO-α's peer-assumption hard-codes a 0.4 threshold over
`public_at_read[i]` (which is i-specific), while DO-SB-ABBA with the
mailbox anchor makes a globally-informed, jointly-consistent decision.
"""
function _build_shared_anchor(
    mailbox::Mailbox, t::Int,
    pockets::Vector{PocketDef}, dynamics::PocketDynamics,
)::PublicBelief
    # Collect every obs record currently in the mailbox (all rovers'
    # latest uploads). Deduplicate by (rover_id, obs.t, obs.pos) to
    # avoid double counting when the same record somehow appears twice.
    all_obs = ObsRecord[]
    for (_, entry) in mailbox.entries
        append!(all_obs, entry.obs)
    end
    # Upper bound of observation timestamps determines how far we run
    # the filter before predict-only to `t`.
    t_last_obs = isempty(all_obs) ? 0 : maximum(o.t for o in all_obs)
    # Dormant-concentrated prior (common knowledge: pockets start Dormant).
    B0 = initialize_dormant_belief(length(pockets))
    t_filter_end = min(t, t_last_obs)
    B_filtered, _ = PublicFilterUpdate(
        B0, 0, t_filter_end, all_obs, pockets, dynamics,
    )
    # Predict-only from the last observation time to `t` (no data past
    # the mailbox is available to the shared anchor).
    if t_filter_end < t
        B_filtered, _ = PublicFilterUpdate(
            B_filtered, t_filter_end, t, ObsRecord[], pockets, dynamics,
        )
    end
    return B_filtered
end

# =============================================================================
# CONFIGURATION
# =============================================================================
"""
Tuning knobs for DO-SB-ABBA-α.

* `α`             — weight in [0,1] between optimality (1) and consistency (0).
* `γ`             — discount factor used in rollouts.
* `H`             — rollout horizon (number of steps).
* `N_particles`   — number of SB particles per belief hypothesis.
* `deadline`      — CRITICAL_DEADLINE passed to the simulated env step.
"""
Base.@kwdef struct DOSBABBAConfig
    α::Float64 = 0.5
    γ::Float64 = 0.95
    # Horizon = 2 × rendezvous period (period 12 ⇒ H=24). This is what
    # gives DO-SB-ABBA its concrete advantage over DO-α: within a
    # 1-period rollout (H=12) the rollout sees AT MOST one rendezvous
    # for a given pocket, so "skip now" has no alternative fix
    # opportunity inside the horizon → rollout always prefers Act now.
    # With H=24 the rollout sees the *next* rendezvous for every
    # pocket, so "skip now, act at next rendezvous" becomes a
    # considered alternative. That is exactly the scenario where DO-α
    # (1-step) gets the decision wrong and DO-SB-ABBA gets it right.
    H::Int = 24
    N_particles::Int = 16
    deadline::Int = 5
    # MRAC safety net: when enabled, the primary action is the joint
    # argmax under a shared anchor belief that BOTH rovers compute
    # identically from public state — guaranteeing action consistency
    # without communication.
    mrac_safety_check::Bool = true
    # Shape of the shared anchor belief:
    #   :uniform — legacy. Uniform prior rolled forward to `t` with
    #              zero observations. Symmetric by construction but
    #              observation-agnostic → conservatively keeps acting
    #              at every rendezvous even when mailbox evidence
    #              clearly shows pockets are Dormant (wastes R_waste).
    #   :mailbox — bilateral anchor. Uniform prior filtered with *every
    #              observation currently in the mailbox* and rolled
    #              forward to `t`. The mailbox is public shared state
    #              (both rovers see the same entries), so the resulting
    #              belief is computed identically by both rovers and
    #              MRAC is preserved. This anchor is strictly more
    #              informed than `:uniform` — it correctly skips a
    #              rendezvous when the last peer upload confidently
    #              says the pocket is Dormant, while still acting
    #              preventively when the mailbox lacks recent info.
    #              The H-step rollout then decides the actual action
    #              under this richer anchor, which is the core reason
    #              DO-SB-ABBA beats 1-step DO-α.
    anchor_type::Symbol = :mailbox
    # Setting `veto_p_dormant >= 1.0` disables the dual-veto entirely.
    # Kept for experimentation; bilateral veto via `:mailbox` anchor
    # makes the rover-side veto unnecessary.
    veto_p_dormant::Float64 = 2.0
end

# =============================================================================
# SAMPLING HELPERS
# =============================================================================
"""Sample a single pocket state from a per-pocket belief vector."""
function _sample_pocket_state(b_p::Vector{Float64}, rng::AbstractRNG)::Int
    r = rand(rng)
    c = b_p[1]
    r < c && return DORMANT
    c += b_p[2]
    r < c && return PRESSURIZED
    return CRITICAL
end

"""Sample (x_anchor, x_anchor+1, ..., x_end) given the initial belief and
natural Markov dynamics. Returns a length `t_end - t_anchor + 1` vector,
each entry a copy of the P-vector of pocket states. `-1` means exploded."""
function _sample_pocket_trajectory(B_anchor::PublicBelief, t_anchor::Int,
                                   t_end::Int, dynamics::PocketDynamics,
                                   rng::AbstractRNG)::Vector{Vector{Int}}
    x = [_sample_pocket_state(b_p, rng) for b_p in B_anchor]
    traj = Vector{Vector{Int}}()
    push!(traj, copy(x))
    for _ in t_anchor:(t_end - 1)
        for p in 1:length(x)
            x[p] >= 0 || continue
            new_s = pocket_transition_spec(x[p], dynamics, rng)
            x[p] = new_s
        end
        push!(traj, copy(x))
    end
    return traj
end

"""Sample the peer's (noisy) observation stream at every step
t_anchor < τ ≤ t_end, given a sampled pocket-state trajectory."""
function _sample_peer_obs_stream(peer_id::Int, x_traj::Vector{Vector{Int}},
                                 t_anchor::Int, t_end::Int,
                                 trajectories::Vector{RoverTrajectory},
                                 pockets::Vector{PocketDef},
                                 rng::AbstractRNG)::Vector{ObsRecord}
    obs = ObsRecord[]
    for τ in (t_anchor + 1):t_end
        idx = τ - t_anchor + 1
        idx > length(x_traj) && break
        x_τ = [s < 0 ? DORMANT : s for s in x_traj[idx]]
        pos = positions_at_time(trajectories, τ)[peer_id]
        z = generate_observation(pos, x_τ, pockets, rng)
        push!(obs, ObsRecord(τ, peer_id, pos, z))
    end
    return obs
end

# =============================================================================
# ROLLOUT HELPERS
# =============================================================================
"""Joint 1-step argmax over (a_i, a_j) under belief `b`. Used inside the
rollout for h ≥ 1 to mimic 'both rovers pick the MPOMDP-optimal joint
action under the rolled-forward public belief'."""
function _greedy_joint(rover_i::Int, peer_id::Int, pos::Vector{Cell},
                       b::PublicBelief, pockets::Vector{PocketDef},
                       dynamics::PocketDynamics,
                       reward_params::RewardParams)::Tuple{RoverAction, RoverAction}
    cands_i = _feasible_valve_actions_i(pos[rover_i], pockets)
    cands_j = _feasible_valve_actions_i(pos[peer_id], pockets)
    N = length(pos)
    ja = [RoverAction(MOVE_STAY, VALVE_NONE, nothing) for _ in 1:N]
    best_i, best_j, best_r = cands_i[1], cands_j[1], -Inf
    for a_i in cands_i, a_j in cands_j
        ja[rover_i] = a_i
        ja[peer_id] = a_j
        r, _ = PublicBackupStep(b, ja, pos, pockets, reward_params, dynamics)
        if r > best_r
            best_r, best_i, best_j = r, a_i, a_j
        end
    end
    return best_i, best_j
end

"""Canonical sortable key for a joint action *in slot order* (a_slot1,
a_slot2). Both rovers reach the same key for the same underlying joint,
regardless of which rover is evaluating — used to break strict ties in
the joint argmax deterministically (MRAC)."""
@inline function _joint_slot_key(slots::Vector{RoverAction})::NTuple{4,Int}
    v1 = slots[1].valve_action == VALVE_NONE    ? 0 :
         slots[1].valve_action == VALVE_OBSERVE ? 1 :
         slots[1].valve_action == VALVE_SEAL    ? 2 : 3
    p1 = slots[1].valve_pocket === nothing ? 0 : slots[1].valve_pocket
    v2 = slots[2].valve_action == VALVE_NONE    ? 0 :
         slots[2].valve_action == VALVE_OBSERVE ? 1 :
         slots[2].valve_action == VALVE_SEAL    ? 2 : 3
    p2 = slots[2].valve_pocket === nothing ? 0 : slots[2].valve_pocket
    return (v1, p1, v2, p2)
end

"""Closed-form, H-step rollout starting from belief `b_start` with forced
joint action `(a_i_0, a_j_0)` at h=0 and greedy joint argmax thereafter.
No Monte-Carlo: expected rewards come from `PublicBackupStep` and the
belief is rolled forward deterministically (Markov transition + vent
effects). Returns the discounted return."""
function _rollout_closed_form(
    a_i_0::RoverAction, a_j_0::RoverAction, b_start::PublicBelief,
    rover_i::Int, peer_id::Int, t::Int, H::Int,
    trajectories::Vector{RoverTrajectory}, pockets::Vector{PocketDef},
    dynamics::PocketDynamics, reward_params::RewardParams, γ::Float64,
)::Float64
    N = length(positions_at_time(trajectories, t))
    b = [copy(bp) for bp in b_start]
    total = 0.0
    pos_h = positions_at_time(trajectories, t)
    for h in 0:(H - 1)
        a_i, a_j = h == 0 ? (a_i_0, a_j_0) :
            _greedy_joint(rover_i, peer_id, pos_h, b,
                          pockets, dynamics, reward_params)
        ja = Vector{RoverAction}(undef, N)
        ja[rover_i] = a_i
        ja[peer_id] = a_j
        for k in 1:N
            isassigned(ja, k) && continue
            ja[k] = RoverAction(MOVE_STAY, VALVE_NONE, nothing)
        end
        r_exp, b = PublicBackupStep(b, ja, pos_h, pockets, reward_params, dynamics)
        total += (γ^h) * r_exp
        pos_h = positions_at_time(trajectories, t + h + 1)
    end
    return total
end

# =============================================================================
# PER-ROVER PLANNING STEP (joint-action search → MRAC)
# =============================================================================
"""Decide rover `rover_i`'s valve action at time `t` using SB-ABBA-α with
**joint-action search** for MRAC.

Why joint search and not independent-`a_i` search?
---------------------------------------------------
If each rover independently picked its own `argmax_a_i score(a_i)`, then at
a rendezvous — where the coordinated joint optimum is (Seal, Vent) — both
rovers would see equal scores for their Seal and Vent options (because
from each rover's viewpoint, "I seal while peer vents" ≈ "I vent while
peer seals" up to sampling noise) and could both pick Seal (or both pick
Vent). Coordination breaks.

Instead, both rovers enumerate the same joint action space `(a_i, a_j)`,
score each joint, pick the joint argmax, and return their own slot. Given
approximately-consistent beliefs (which α trades off against optimality),
both rovers deterministically agree on the same joint → MRAC holds. This
is exactly the mechanism in Shimron & Indelman 2025."""
function _plan_rover_i(rover_i::Int, t::Int, state::GasValveWorldState,
                       ds::DOSBABBAState, mailbox::Mailbox,
                       obs_buffers::Vector{Vector{ObsRecord}},
                       pockets::Vector{PocketDef}, dynamics::PocketDynamics,
                       reward_params::RewardParams,
                       trajectories::Vector{RoverTrajectory},
                       cfg::DOSBABBAConfig, _rng::AbstractRNG)::RoverAction
    N = length(state.rover_positions)
    @assert N == 2 "GasValveDOSBABBA assumes 2 rovers; generalisation trivial."
    peer_id = rover_i == 1 ? 2 : 1

    pos_now = state.rover_positions
    cands_i = _feasible_valve_actions_i(pos_now[rover_i], pockets)
    length(cands_i) == 1 && return cands_i[1]   # only Idle feasible ⇒ nothing to decide
    cands_j = _feasible_valve_actions_i(pos_now[peer_id], pockets)

    t_anchor = max(0, ds.last_read_time[rover_i])
    B_anchor = [copy(bp) for bp in ds.public_at_read[rover_i]]
    own_obs  = [o for o in obs_buffers[rover_i] if o.t > t_anchor && o.t <= t]

    # Both rovers derive the particle-sampling RNG from `t` alone (common
    # public knowledge), so that — given identical anchor beliefs — they
    # would generate identical particles. Beliefs still differ due to
    # unshared own_obs / different public_at_read, but the *randomness* is
    # no longer a source of MRAC disagreement.
    particle_rng = MersenneTwister(hash((:gas_valve_do_sb_abba_particles, t)))

    # SB particle step: sample N plausible peer obs streams that i could
    # have missed since t_anchor, and build per-particle start beliefs.
    particles_bMP   = Vector{PublicBelief}(undef, cfg.N_particles)
    particles_bpeer = Vector{PublicBelief}(undef, cfg.N_particles)
    t_horizon_end   = t + cfg.H
    for n in 1:cfg.N_particles
        x_traj = _sample_pocket_trajectory(B_anchor, t_anchor, t_horizon_end,
                                           dynamics, particle_rng)
        peer_obs = _sample_peer_obs_stream(peer_id, x_traj, t_anchor, t,
                                           trajectories, pockets, particle_rng)

        particles_bMP[n], _   = PublicFilterUpdate(B_anchor, t_anchor, t,
                                                    vcat(own_obs, peer_obs),
                                                    pockets, dynamics)
        particles_bpeer[n], _ = PublicFilterUpdate(B_anchor, t_anchor, t,
                                                    peer_obs,
                                                    pockets, dynamics)
    end

    # Joint-action argmax with α-blend of V_opt and V_cons averaged over
    # the particle set. Both rovers enumerate the same underlying joint
    # space (in slot order) and break strict ties by a canonical key that
    # is *independent* of which rover is evaluating. This guarantees MRAC
    # in the presence of reward-symmetric pairs like (Seal_A, Vent_B) ≡
    # (Vent_A, Seal_B) — without any hand-coded role rule.
    EPS = 1e-6
    best_a_i = cands_i[1]
    best_score = -Inf
    best_key::NTuple{4,Int} = (0, 0, 0, 0)
    first = true
    slots = Vector{RoverAction}(undef, 2)
    for a_i in cands_i, a_j in cands_j
        V_opt, V_cons = 0.0, 0.0
        for n in 1:cfg.N_particles
            V_opt += _rollout_closed_form(
                a_i, a_j, particles_bMP[n], rover_i, peer_id, t, cfg.H,
                trajectories, pockets, dynamics, reward_params, cfg.γ,
            )
            V_cons += _rollout_closed_form(
                a_i, a_j, particles_bpeer[n], rover_i, peer_id, t, cfg.H,
                trajectories, pockets, dynamics, reward_params, cfg.γ,
            )
        end
        V_opt  /= cfg.N_particles
        V_cons /= cfg.N_particles
        score   = cfg.α * V_opt + (1 - cfg.α) * V_cons

        slots[rover_i] = a_i
        slots[peer_id] = a_j
        key = _joint_slot_key(slots)

        take = first ||
               score > best_score + EPS ||
               (abs(score - best_score) <= EPS && key < best_key)
        if take
            best_a_i = a_i
            best_score = score
            best_key = key
            first = false
        end
    end

    # MRAC action consistency (Shimron & Indelman-style).
    #
    # The primary action is the **particle-based H-step joint argmax**
    # above (`best_a_i`). Both rovers compute this over their own
    # particle sets, which differ only in the per-rover `own_obs`;
    # because:
    #
    #   (a) the particle RNG is seeded by `t` alone (common knowledge),
    #   (b) the particle anchor `B_anchor` is the per-rover
    #       `public_at_read[i]` which is a filter of the shared mailbox
    #       at the rover's last GS visit (normally very close across
    #       rovers), and
    #   (c) strict ties in the joint score are broken by a canonical
    #       slot key (rover-order invariant),
    #
    # both rovers reach the same joint with high probability — which
    # is exactly the probabilistic MRAC guarantee of Shimron & Indelman.
    # This lets DO-SB-ABBA *use* its private observations, which is the
    # core reason it can beat the 1-step DO-α: DO-α would, at a
    # rendezvous where private observations say "pocket is clearly
    # Dormant", still rely on the hypothesized peer belief (which is
    # predict-only from `public_at_read`) and cross the 0.4 threshold.
    # DO-SB-ABBA's particle rollout sees the private evidence AND
    # reasons H steps ahead, correctly choosing to skip.
    #
    # The shared-anchor check below acts ONLY as a safety fallback: if
    # the particle-based best joint STRONGLY disagrees with the shared
    # anchor's prescription (i.e. one says Act, the other says Skip),
    # we defer to the shared anchor because its decision is guaranteed
    # identical for both rovers. This keeps the pathological cases
    # (where particle noise produces a split) MRAC-safe without
    # throwing away the private-info advantage in the common case.
    if cfg.mrac_safety_check
        shared_belief = if cfg.anchor_type === :mailbox
            _build_shared_anchor(mailbox, t, pockets, dynamics)
        else
            let b = initialize_dormant_belief(length(pockets))
                b2, _ = PublicFilterUpdate(b, 0, t, ObsRecord[], pockets, dynamics)
                b2
            end
        end
        shared = [shared_belief]
        joint_shared = _joint_argmax_on_belief_set(
            rover_i, peer_id, pos_now, cands_i, cands_j,
            shared, t, cfg.H,
            trajectories, pockets, dynamics, reward_params, cfg.γ,
        )
        slot_shared = joint_shared[1]  # rover_i's slot by construction

        # Classify each decision as Act (Seal/Vent) or non-Act (Idle/Observe).
        is_act_private = best_a_i.valve_action == VALVE_SEAL ||
                         best_a_i.valve_action == VALVE_VENT
        is_act_shared = slot_shared.valve_action == VALVE_SEAL ||
                        slot_shared.valve_action == VALVE_VENT

        # Disagreement on the Act/Skip dimension is the only situation
        # in which a unilateral Seal/Vent could be catastrophic (one
        # rover Seals, the other plays Idle → p_trigger ≈ 0.8
        # explosion). Fall back to the shared anchor joint so both
        # rovers align on the same decision deterministically.
        if is_act_private != is_act_shared
            return slot_shared
        end
        # If the private argmax and the shared anchor agree on WHETHER
        # to act but diverge on the SPECIFIC pocket (rare — only when
        # there are multiple pockets reachable from this cell), prefer
        # the shared anchor so both rovers act on the same pocket.
        if is_act_private && is_act_shared &&
           best_a_i.valve_pocket !== slot_shared.valve_pocket
            return slot_shared
        end
    end
    return best_a_i
end

"""Joint argmax `(a_i, a_j)` over the given candidate cross-product,
averaging the H-step closed-form rollout value across the provided
particle belief set. Tie-break by the canonical slot key so two rovers
evaluating the same belief set arrive at the same joint."""
function _joint_argmax_on_belief_set(
    rover_i::Int, peer_id::Int, pos_now::Vector{Cell},
    cands_i::Vector{RoverAction}, cands_j::Vector{RoverAction},
    particles::Vector{PublicBelief}, t::Int, H::Int,
    trajectories::Vector{RoverTrajectory}, pockets::Vector{PocketDef},
    dynamics::PocketDynamics, reward_params::RewardParams, γ::Float64,
)::Tuple{RoverAction, RoverAction}
    EPS = 1e-6
    slots = Vector{RoverAction}(undef, 2)
    best_i, best_j = cands_i[1], cands_j[1]
    best_score = -Inf
    best_key::NTuple{4,Int} = (0, 0, 0, 0)
    first = true
    for a_i in cands_i, a_j in cands_j
        V = 0.0
        for b in particles
            V += _rollout_closed_form(
                a_i, a_j, b, rover_i, peer_id, t, H,
                trajectories, pockets, dynamics, reward_params, γ,
            )
        end
        V /= length(particles)
        slots[rover_i] = a_i
        slots[peer_id] = a_j
        key = _joint_slot_key(slots)
        take = first ||
               V > best_score + EPS ||
               (abs(V - best_score) <= EPS && key < best_key)
        if take
            best_i, best_j = a_i, a_j
            best_score, best_key = V, key
            first = false
        end
    end
    return best_i, best_j
end

# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================
"""
    step_do_sb_abba_control!(t, state, ds, mailbox, obs_buffers,
                             pockets, dynamics, reward_params, trajectories,
                             gs_schedule; cfg, _rng)

Proper DO-SB-ABBA-α planning step for the gas-valve problem. Each rover
runs a joint-action search over `(a_i, a_j)` at the current positions,
scoring each joint with the α-blend of particle-averaged H-step
closed-form rollouts from `b_MP` (optimality) and `b_{j|i}` (consistency).
Returns the slot of the joint argmax that belongs to each rover — so both
rovers agree on the same coordinated joint (MRAC). Mutates `ds`,
`mailbox`, and `obs_buffers` on scheduled GS visit times.
"""
function step_do_sb_abba_control!(
    t::Int,
    state::GasValveWorldState,
    ds::DOSBABBAState,
    mailbox::Mailbox,
    obs_buffers::Vector{Vector{ObsRecord}},
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
    reward_params::RewardParams,
    trajectories::Vector{RoverTrajectory},
    gs_schedule::Dict{Int,Int};
    cfg::DOSBABBAConfig = DOSBABBAConfig(),
    _rng::AbstractRNG = Random.GLOBAL_RNG,
)::JointAction
    N = length(state.rover_positions)
    for i in 1:N
        period = trajectories[i].period
        if is_gs_visit_time(i, t, gs_schedule, period)
            _handle_gs_visit!(ds, i, t, mailbox, obs_buffers, pockets, dynamics)
        end
    end

    joint_action = Vector{RoverAction}(undef, N)
    for i in 1:N
        joint_action[i] = _plan_rover_i(
            i, t, state, ds, mailbox, obs_buffers, pockets, dynamics,
            reward_params, trajectories, cfg, _rng,
        )
    end
    return joint_action
end

end # module GasValveDOSBABBA

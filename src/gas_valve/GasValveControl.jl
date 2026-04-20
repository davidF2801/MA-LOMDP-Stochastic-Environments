"""
GasValveControl.jl — High-level planners over fixed periodic trajectories
with *asynchronous* ground-station (GS) comms.

All rovers follow fixed periodic trajectories (see `GasValveTrajectories.jl`);
the planners below choose only the **valve action** at each timestep. Motion
is forced by the trajectory schedule.

Communication model
===================
A single **mailbox** at the GS cell stores each rover's last uploaded batch
of observations along with an upload timestamp. Rover `i` can only *read*
or *write* the mailbox at its scheduled GS-visit times (see
`default_gs_visit_schedule`). All other cells — including pass-throughs
of (3,3) — are silent. Peer entries are therefore *stale* by up to one
full period.

Controllers
===========
* `step_public_belief_control!`   — centralized joint planner on shared
  public belief: all rovers' observations are fused into B every timestep,
  then beam search over joint valve actions. Upper-bound reference vs async
  mailbox (where each rover only sees stale peer data at GS visits).

* `step_do_alpha_control!`        — **DO-α**: simple α-blend planner.
  1-step closed-form expected-reward argmax, with the peer's action
  hard-coded by a role rule (lower-id → Sealer, higher-id → Venter).
  Each rover scores a candidate action as
      score(a_i) = (1 - α) · E[r | b_self, (a_i, a_peer)]
                  +  α    · E[r | b_{j|i}, (a_i, a_peer)]
  Fast and cheap; useful as a coordination-only baseline. For proper
  sample-based, multi-step, peer-searching DO-SB-ABBA-α see
  `GasValveDOSBABBA.step_do_sb_abba_control!`.

* `step_random_baseline!`         — random feasible valve action per rover.

Shared infra (also used by `GasValveDOSBABBA`):
  * `DOSBABBAState` — per-rover mailbox snapshot + public-at-read anchor.
  * `initialize_private_beliefs`, `_handle_gs_visit!`,
    `_feasible_valve_actions_i`, `_current_private_belief`,
    `_hypothesized_peer_belief`.
"""

module GasValveControl

using Random
using ..GasValveTypes
using ..GasValveEnv
using ..GasValveBelief
using ..GasValveTrajectories
using ..GasValvePlanner

import ..GasValveTypes: Cell, PocketDef, RoverAction, JointAction, PublicBelief,
    GasValveWorldState, ObsRecord, BroadcastObs, RewardParams, PocketDynamics,
    Mailbox, MailboxEntry, deposit!, peer_entries,
    VALVE_NONE, VALVE_OBSERVE, VALVE_SEAL, VALVE_VENT, MOVE_STAY,
    DORMANT, PRESSURIZED, CRITICAL
import ..GasValveEnv: generate_observation, compute_expected_reward
import ..GasValveBelief: PublicFilterUpdate, PredictOnly, initialize_uniform_belief,
    initialize_dormant_belief, PublicBackupStep, PrivateBeliefRollforward
import ..GasValveTrajectories: RoverTrajectory, positions_at_time,
    default_gs_cell, default_gs_visit_schedule, is_gs_visit_time
import ..GasValvePlanner: NextBroadcastTime, PlanJointOpenLoop_traj,
    generate_joint_actions_fixed, FeasibleValveActions

export step_public_belief_control!,
    step_random_baseline!,
    step_do_alpha_control!,
    build_positions_schedule,
    initialize_private_beliefs,
    DOSBABBAState, DOAlphaConfig,
    _handle_gs_visit!, _current_private_belief,
    _hypothesized_peer_belief, _feasible_valve_actions_i

# -----------------------------------------------------------------------------
# Build the rover-position schedule for the next `H` steps starting at time `t`.
# -----------------------------------------------------------------------------
function build_positions_schedule(
    trajectories::Vector{RoverTrajectory},
    t::Int,
    H::Int
)::Vector{Vector{Cell}}
    return [positions_at_time(trajectories, t + h) for h in 0:H]
end

# =============================================================================
# PUBLIC-BELIEF JOINT CONTROL (centralized: fuse all rovers' obs every step).
# =============================================================================
function _drain_buffers_to_public!(
    t::Int, B_pub::PublicBelief, t_clean::Int,
    obs_buffers::Vector{Vector{ObsRecord}},
    pockets::Vector{PocketDef}, dynamics::PocketDynamics
)::Tuple{PublicBelief, Int}
    O_bc = ObsRecord[]
    for buf in obs_buffers
        append!(O_bc, buf)
    end
    if !isempty(O_bc)
        B_pub, t_clean = PublicFilterUpdate(B_pub, t_clean, t, O_bc, pockets, dynamics)
        for buf in obs_buffers
            empty!(buf)
        end
    end
    return B_pub, t_clean
end

function step_public_belief_control!(
    t::Int,
    state::GasValveWorldState,
    B_pub::PublicBelief,
    t_clean::Int,
    obs_buffers::Vector{Vector{ObsRecord}},
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
    reward_params::RewardParams,
    trajectories::Vector{RoverTrajectory},
    H_max::Int;
    γ::Float64 = 0.95,
    W_beam::Int = 2000,
    _rng::AbstractRNG = Random.GLOBAL_RNG
)::Tuple{JointAction, PublicBelief, Int, Float64}
    # Oracle: always drain all obs into the public belief.
    B_pub, t_clean = _drain_buffers_to_public!(
        t, B_pub, t_clean, obs_buffers, pockets, dynamics
    )

    N = length(state.rover_positions)
    H = max(1, H_max)
    B_tilde = PredictOnly(B_pub, t_clean, t, dynamics)
    schedule = build_positions_schedule(trajectories, t, H)

    Π = PlanJointOpenLoop_traj(
        B_tilde, schedule, H, pockets, dynamics, reward_params;
        W_beam = W_beam, γ = γ
    )
    ja = isempty(Π) ?
        [RoverAction(MOVE_STAY, VALVE_NONE, nothing) for _ in 1:N] :
        Π[1]
    return (ja, B_pub, t_clean, 0.0)
end

# =============================================================================
# DO-α (simple α-blend planner) + shared mailbox/state machinery used by
# the proper DO-SB-ABBA-α planner in `GasValveDOSBABBA`.
#
# State maintained per rover (shared across both planners):
#   b_self[i]        — rover i's private belief (= public-at-last-read rolled
#                      forward through i's own obs).
#   last_read_time[i] — absolute time of rover i's last mailbox read.
#   public_at_read[i] — the public belief rover i *reconstructed* at its
#                      last read (= initial belief updated with every obs
#                      the mailbox contained at that read).
#
# Hypothesized peer belief (i's view of j):
#   last_write_time_of_j_seen_by_i = mailbox entry time i last saw from j.
#   public_at_that_time_of_j       = i's best estimate of j's belief at
#                                    j's own last write (approximation: the
#                                    same mailbox state i saw, roll-forward
#                                    with no new obs).
#
# DO-α action scoring (per rover, per candidate valve action a_i):
#   - Assume peer plays role-consistent action (Sealer/Venter at valve
#     rendezvous, else Idle).
#   - Compute 1-step expected reward under b_self and b_{j|i}.
#   - Blend: score = (1-α)·r_self + α·r_peer_imagined
#   - Break ties by favoring Idle (action consistency prior).
# =============================================================================

"""Per-rover state carried across timesteps by the decentralized planners
(both DO-α and DO-SB-ABBA-α share this mailbox/belief bookkeeping)."""
mutable struct DOSBABBAState
    b_self::Vector{PublicBelief}        # b_self[i]
    public_at_read::Vector{PublicBelief} # public belief i reconstructed at its last read
    last_read_time::Vector{Int}         # absolute time of i's last read (-1 if never)
    # Peer-hypothesis: what i last saw of j (j's obs batch + j's write time).
    peer_seen_obs::Vector{Dict{Int,Vector{ObsRecord}}} # i => {j => obs_of_j_seen_by_i}
    peer_seen_time::Vector{Dict{Int,Int}}              # i => {j => upload_time_of_that_batch}
end

"""Tuning knobs for DO-α (simple 1-step α-blend planner)."""
Base.@kwdef struct DOAlphaConfig
    α::Float64 = 0.5
    γ::Float64 = 0.95
    lookahead::Int = 1       # 1-step argmax
end

"""Initialize a DOSBABBAState for `N` rovers over `P` pockets."""
function initialize_private_beliefs(N::Int, P::Int)::DOSBABBAState
    # Default to Dormant-start prior (common knowledge): at t=0 every
    # pocket is known to be Dormant. The drift-to-risk dynamics
    # (p_arrive > 0) are the only source of uncertainty. This is what
    # makes the H-step lookahead meaningful for DO-SB-ABBA.
    b0 = initialize_dormant_belief(P)
    DOSBABBAState(
        [copy(b0) for _ in 1:N],
        [copy(b0) for _ in 1:N],
        fill(-1, N),
        [Dict{Int,Vector{ObsRecord}}() for _ in 1:N],
        [Dict{Int,Int}() for _ in 1:N],
    )
end

"""
Handle a GS mailbox op for rover `rover_i` at time `t`:
- Upload: deposit contents of `obs_buffers[rover_i]` into the mailbox
  under key `rover_i` (and clear the buffer).
- Download: read every peer entry `j` → update
  `state.peer_seen_obs[rover_i][j]` and `state.peer_seen_time[rover_i][j]`.
- Rebuild the public-at-read belief as (fresh initial belief) filtered
  with every mailbox entry rover i currently knows (including its own
  most-recent upload).
- Re-roll b_self[rover_i] from public-at-read forward with rover_i's
  own obs buffer (now empty — so b_self starts clean, to be updated
  going forward with new obs).
"""
function _handle_gs_visit!(
    ds::DOSBABBAState,
    rover_i::Int,
    t::Int,
    mailbox::Mailbox,
    obs_buffers::Vector{Vector{ObsRecord}},
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
)
    t0 = max(0, ds.last_read_time[rover_i])
    # Collect obs to integrate into rover i's public belief since its last read:
    #  (a) rover i's own staged obs (in obs_buffers[i]),
    #  (b) every peer entry in the mailbox that i has NOT yet seen
    #      (entry.upload_time > last-seen time for that peer).
    # We only keep obs with timestamps in (t0, t] to avoid retroactive updates
    # (the belief roll-forward is forward-only Markov). Peer obs older than
    # t0 would have to be fused retroactively and are dropped — an acceptable
    # approximation consistent with the async-mailbox model.
    range_obs = ObsRecord[]
    for o in obs_buffers[rover_i]
        o.t > t0 && o.t <= t && push!(range_obs, o)
    end
    for (j, entry) in mailbox.entries
        j == rover_i && continue
        last_seen = get(ds.peer_seen_time[rover_i], j, -1)
        entry.upload_time > last_seen || continue
        for o in entry.obs
            o.t > t0 && o.t <= t && push!(range_obs, o)
        end
    end

    # Incremental roll-forward of the public-at-read belief.
    B0 = ds.public_at_read[rover_i]
    B_next, _ = PublicFilterUpdate(B0, t0, t, range_obs, pockets, dynamics)
    ds.public_at_read[rover_i] = B_next

    # UPLOAD after reading (so the deposit does not overwrite the entry we
    # still need to treat as 'incoming' for ourselves).
    deposit!(mailbox, rover_i, t, obs_buffers[rover_i])
    empty!(obs_buffers[rover_i])

    # Snapshot peer entries (for hypothesized-peer-belief construction).
    for (j, entry) in mailbox.entries
        j == rover_i && continue
        ds.peer_seen_obs[rover_i][j] = copy(entry.obs)
        ds.peer_seen_time[rover_i][j] = entry.upload_time
    end

    # b_self resets to public_at_read (no unshared obs yet) — will be
    # rolled forward with future obs on subsequent planning steps.
    ds.b_self[rover_i] = [copy(b) for b in ds.public_at_read[rover_i]]
    ds.last_read_time[rover_i] = t
    return nothing
end

"""
Current b_self[i] at time `t`: public_at_read[i] rolled forward with
rover i's obs_buffers[i] (observations collected since last read).
"""
function _current_private_belief(
    ds::DOSBABBAState, i::Int, t::Int,
    obs_buffers::Vector{Vector{ObsRecord}},
    pockets::Vector{PocketDef}, dynamics::PocketDynamics,
)::PublicBelief
    t0 = max(0, ds.last_read_time[i])
    B0 = ds.public_at_read[i]
    return PrivateBeliefRollforward(B0, t0, t, obs_buffers[i], pockets, dynamics)
end

"""
Hypothesized belief of peer j from i's perspective at time `t`. We use the
simplest consistent approximation:

    B_{j|i}(t) = predict-only forward from (last-known-public-of-j) to t.

where last-known-public-of-j is what i saw of j's mailbox entry at i's
last read, rolled into a public belief along with any self-uploads i has
since made (those are known to i, and i *knows* j will read them at j's
next visit — but hasn't necessarily yet). For simplicity we don't assume
j has read i's writes unless the mailbox-timestamp pair implies so.
"""
function _hypothesized_peer_belief(
    ds::DOSBABBAState, i::Int, j::Int, t::Int,
    pockets::Vector{PocketDef}, dynamics::PocketDynamics,
)::PublicBelief
    # Start from the public belief i believes was shared at j's last write
    # known to i. Conservative choice: use the public-at-read of i (same
    # mutual anchor) — then predict-only forward to t.
    B0 = ds.public_at_read[i]
    t0 = max(0, ds.last_read_time[i])
    return PredictOnly(B0, t0, t, dynamics)
end

"""Role of rover i at pocket p under MRAC tie-break: lower-id → Sealer."""
function _role_at_pocket(rover_i::Int, pocket_id::Int, rover_positions::Vector{Cell},
                          pocket::PocketDef)::Symbol
    # Role is purely deterministic from (rover_i, pocket_id). Valve assignment:
    # lower-id rover → Seal; higher-id rover → Vent.
    # (Swap every *other* pocket to keep things symmetric if desired; here we
    #  keep the simple lower=Sealer rule.)
    return rover_i == 1 ? :Seal : :Vent
end

"""Peer's assumed action at this step, using MRAC roles."""
function _assumed_peer_action(
    rover_i::Int,
    rover_positions::Vector{Cell},
    pockets::Vector{PocketDef},
    b_peer::PublicBelief,
    reward_params::RewardParams,
    dynamics::PocketDynamics,
    γ::Float64,
)::RoverAction
    # Peer id (2-rover case)
    j = rover_i == 1 ? 2 : 1
    pos_j = rover_positions[j]
    # Is peer at a valve?
    for pocket in pockets
        if pos_j == pocket.valve_a || pos_j == pocket.valve_b
            b_p = b_peer[pocket.id]
            # Threshold: expected P(pressurized|critical) > 0.4 → act.
            q = b_p[PRESSURIZED + 1] + b_p[CRITICAL + 1]
            if q > 0.4
                role = _role_at_pocket(j, pocket.id, rover_positions, pocket)
                return RoverAction(MOVE_STAY,
                    role == :Seal ? VALVE_SEAL : VALVE_VENT,
                    pocket.id)
            else
                return RoverAction(MOVE_STAY, VALVE_NONE, nothing)
            end
        end
    end
    return RoverAction(MOVE_STAY, VALVE_NONE, nothing)
end

"""Feasible valve actions for rover i at its current position."""
function _feasible_valve_actions_i(pos_i::Cell, pockets::Vector{PocketDef}
)::Vector{RoverAction}
    out = RoverAction[RoverAction(MOVE_STAY, VALVE_NONE, nothing)]
    for pocket in pockets
        if pos_i == pocket.valve_a || pos_i == pocket.valve_b
            push!(out, RoverAction(MOVE_STAY, VALVE_OBSERVE, pocket.id))
            push!(out, RoverAction(MOVE_STAY, VALVE_SEAL, pocket.id))
            push!(out, RoverAction(MOVE_STAY, VALVE_VENT, pocket.id))
            break
        end
    end
    return out
end

"""
Per-rover independent action selection with α-blend between self-belief
value and hypothesized-peer-belief value. Peer's assumed action comes
from the MRAC role rule + threshold over the *peer's* belief (as seen
by rover i).
"""
function _pick_action_rover_i(
    rover_i::Int,
    rover_positions::Vector{Cell},
    b_self::PublicBelief,
    b_peer::PublicBelief,
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
    reward_params::RewardParams,
    α::Float64,
    γ::Float64,
)::RoverAction
    pos_i = rover_positions[rover_i]
    candidates = _feasible_valve_actions_i(pos_i, pockets)
    # Peer's assumed action comes from rover i's HYPOTHESIZED belief about
    # peer's state (b_peer) — that is the only info i has about what peer
    # will actually do. We use the same a_peer for both evaluation branches
    # because rover i cannot use b_self to predict peer's decision.
    a_peer = _assumed_peer_action(rover_i, rover_positions, pockets,
                                   b_peer, reward_params, dynamics, γ)

    N = length(rover_positions)
    ja = Vector{RoverAction}(undef, N)
    for k in 1:N
        k != rover_i && (ja[k] = a_peer)
    end

    best_a = candidates[1]
    best_score = -Inf
    for a_i in candidates
        ja[rover_i] = a_i
        # r_self : expected reward assuming b_self describes the world.
        r_self, _ = PublicBackupStep(
            b_self, ja, rover_positions, pockets, reward_params, dynamics
        )
        # r_view : expected reward assuming b_peer describes the world
        #          (i.e. the "what the peer sees" hypothesis).
        r_view, _ = PublicBackupStep(
            b_peer, ja, rover_positions, pockets, reward_params, dynamics
        )
        score = (1 - α) * r_self + α * r_view
        # Tiny MRAC prior: prefer Idle on ties (keeps rovers from both venting
        # when reward signal is ambiguous).
        if a_i.valve_action == VALVE_NONE
            score += 1e-6
        end
        if score > best_score
            best_score = score
            best_a = a_i
        end
    end
    return best_a
end

"""
Main decentralized step. Mutates `ds`, `mailbox`, `obs_buffers`.

Arguments:
  t                  : absolute timestep (planning is *at* t, before env step)
  state              : world state (used only for rover positions)
  ds                 : DOSBABBAState (per-rover beliefs and comms snapshots)
  mailbox            : shared GS mailbox (mutated on visits)
  obs_buffers        : per-rover observation buffers since last upload
  pockets            : PocketDef vector
  dynamics           : PocketDynamics
  reward_params      : RewardParams
  trajectories       : rover trajectories (period defines GS schedule)
  gs_schedule        : Dict{rover_id => offset_within_period}
  cfg                : DOAlphaConfig

Returns the joint action at time `t`.
"""
function step_do_alpha_control!(
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
    cfg::DOAlphaConfig = DOAlphaConfig(),
    _rng::AbstractRNG = Random.GLOBAL_RNG,
)::JointAction
    N = length(state.rover_positions)
    # 1) Handle any scheduled GS mailbox ops at time t.
    for i in 1:N
        period = trajectories[i].period
        if is_gs_visit_time(i, t, gs_schedule, period)
            _handle_gs_visit!(ds, i, t, mailbox, obs_buffers, pockets, dynamics)
        end
    end

    # 2) Per-rover independent action selection.
    joint_action = Vector{RoverAction}(undef, N)
    for i in 1:N
        b_self = _current_private_belief(ds, i, t, obs_buffers, pockets, dynamics)
        # 2-rover case: peer = other rover. For N>2 we'd aggregate/minimize.
        j = i == 1 ? 2 : (i == 2 ? 1 : i)
        b_peer = _hypothesized_peer_belief(ds, i, j, t, pockets, dynamics)
        joint_action[i] = _pick_action_rover_i(
            i, state.rover_positions, b_self, b_peer,
            pockets, dynamics, reward_params, cfg.α, cfg.γ,
        )
    end
    return joint_action
end

# =============================================================================
# RANDOM BASELINE
# =============================================================================
function step_random_baseline!(
    state::GasValveWorldState,
    pockets::Vector{PocketDef},
    rng::AbstractRNG
)::JointAction
    N = length(state.rover_positions)
    joint_action = RoverAction[]
    for i in 1:N
        pos = state.rover_positions[i]
        rover_opts = RoverAction[RoverAction(MOVE_STAY, VALVE_NONE, nothing)]
        for pocket in pockets
            if pos == pocket.valve_a || pos == pocket.valve_b
                push!(rover_opts, RoverAction(MOVE_STAY, VALVE_OBSERVE, pocket.id))
                push!(rover_opts, RoverAction(MOVE_STAY, VALVE_SEAL, pocket.id))
                push!(rover_opts, RoverAction(MOVE_STAY, VALVE_VENT, pocket.id))
                break
            end
        end
        push!(joint_action, rand(rng, rover_opts))
    end
    return joint_action
end

end # module

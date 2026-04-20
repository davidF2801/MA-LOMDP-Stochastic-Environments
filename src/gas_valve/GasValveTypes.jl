"""
GasValveTypes.jl - Types for the decentralized Gas-Valve Coordination problem.
Public-Belief Joint Planning with Scheduled Broadcasts.
No modifications to existing codebase; standalone types for this problem.
"""

module GasValveTypes

using Random

# =============================================================================
# POCKET STATE (hidden state per pocket)
# =============================================================================
const DORMANT = 0
const PRESSURIZED = 1
const CRITICAL = 2
const POCKET_STATES = [DORMANT, PRESSURIZED, CRITICAL]
const N_POCKET_STATES = 3

# =============================================================================
# OBSERVATION (noisy reading: Low=0, Med=1, High=2)
# =============================================================================
const OBS_LOW = 0
const OBS_MED = 1
const OBS_HIGH = 2
const OBS_LEVELS = [OBS_LOW, OBS_MED, OBS_HIGH]

# =============================================================================
# MOVE PRIMITIVES
# =============================================================================
const MOVE_UP = :Up
const MOVE_DOWN = :Down
const MOVE_LEFT = :Left
const MOVE_RIGHT = :Right
const MOVE_STAY = :Stay
const MOVE_ACTIONS = [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_STAY]

# =============================================================================
# CELL: (x, y) 1-indexed, x = column, y = row
# =============================================================================
const Cell = Tuple{Int, Int}

# =============================================================================
# POCKET DEFINITION (fixed valves and center; state is in world)
# =============================================================================
struct PocketDef
    valve_a::Cell   # v_p^A
    valve_b::Cell   # v_p^B
    center::Cell    # c_p for observation likelihood / distance
    id::Int
end

# =============================================================================
# VALVE ACTIONS (spec: Idle, Observe(k), Seal(k), Vent(k))
# =============================================================================
const VALVE_NONE = :none
const VALVE_OBSERVE = :Observe
const VALVE_SEAL = :Seal
const VALVE_VENT = :Vent
const VALVE_ACTIONS = [VALVE_NONE, VALVE_OBSERVE, VALVE_SEAL, VALVE_VENT]

# =============================================================================
# ROVER ACTION: move + optional valve action (Observe/Seal/Vent) on pocket when at valve
# =============================================================================
struct RoverAction
    move::Symbol
    valve_action::Symbol       # VALVE_NONE, VALVE_OBSERVE, VALVE_SEAL, VALVE_VENT
    valve_pocket::Union{Int, Nothing}  # pocket id when at valve
end
# Backward-compat: constructor with (move, vent_pocket) treats vent_pocket as Vent(p)
function RoverAction(move::Symbol, vent_pocket::Union{Int, Nothing})
    if vent_pocket === nothing
        return RoverAction(move, VALVE_NONE, nothing)
    end
    RoverAction(move, VALVE_VENT, vent_pocket)
end

function Base.:(==)(a::RoverAction, b::RoverAction)
    a.move == b.move && a.valve_action == b.valve_action && a.valve_pocket == b.valve_pocket
end
function Base.hash(a::RoverAction, h::UInt)
    hash(a.move, hash(a.valve_action, hash(a.valve_pocket, h)))
end

# Legacy alias for code that checks "did this rover open/vent?"
vent_pocket(a::RoverAction) = (a.valve_action == VALVE_VENT && a.valve_pocket !== nothing) ? a.valve_pocket : nothing

# =============================================================================
# JOINT ACTION (one action per rover, ordered by rover index 1..N)
# =============================================================================
const JointAction = Vector{RoverAction}

# =============================================================================
# PUBLIC BELIEF: product over pockets; each b_p is a probability vector of length 3
# B_pub = [b_1, b_2, ..., b_P] where b_p = [P(Dormant), P(Pressurized), P(Critical)]
# =============================================================================
const PocketBelief = Vector{Float64}  # length 3, sums to 1
const PublicBelief = Vector{PocketBelief}  # length P

# =============================================================================
# WORLD STATE (for simulation): pocket states and deadline counters
# =============================================================================
mutable struct GasValveWorldState
    pocket_states::Vector{Int}   # x_p(t) for each pocket, 0/1/2
    deadline_counters::Vector{Int}  # d_p(t) for each pocket, 0..D
    rover_positions::Vector{Cell}   # p_i(t) for each rover
    exploded::Vector{Bool}          # true if pocket p has exploded (terminal)
end

# =============================================================================
# OBSERVATION RECORD (for broadcast): (timestep, rover_id, position, z)
# =============================================================================
struct ObsRecord
    t::Int
    rover_id::Int
    pos::Cell
    z::Int   # 0=Low, 1=Med, 2=High
end

# =============================================================================
# BROADCAST SET: all observations since last broadcast (flat list)
# =============================================================================
const BroadcastObs = Vector{ObsRecord}

# =============================================================================
# ASYNCHRONOUS GS MAILBOX
# One entry per rover. Each entry stores the observations that rover last
# uploaded to the GS, plus the timestamp of that upload. Readers at later
# visits see *stale* info (from the uploader's last visit) — this is the
# decentralized-async model.
# =============================================================================
struct MailboxEntry
    upload_time::Int                  # absolute time the entry was written
    obs::Vector{ObsRecord}            # observations rover uploaded
end

mutable struct Mailbox
    entries::Dict{Int,MailboxEntry}   # rover_id => entry
end

Mailbox() = Mailbox(Dict{Int,MailboxEntry}())

function deposit!(m::Mailbox, rover_id::Int, t::Int, obs::Vector{ObsRecord})
    m.entries[rover_id] = MailboxEntry(t, copy(obs))
    return m
end

"""Return the mailbox entries written by rovers *other than* `rover_id`."""
function peer_entries(m::Mailbox, rover_id::Int)::Vector{MailboxEntry}
    out = MailboxEntry[]
    for (k, v) in m.entries
        k == rover_id && continue
        push!(out, v)
    end
    return out
end

# =============================================================================
# REWARD PARAMETERS (spec: +50 fix, -200 explosion, -5 critical/step, -1 accumulating/step, action costs)
# =============================================================================
struct RewardParams
    R_fix::Float64            # +50 for successful Seal+Vent on critical/accumulating
    R_explosion::Float64       # -200 when x_k transitions to explosion (state 3)
    penalty_critical::Float64  # -5 per timestep when x_k = 2
    penalty_accumulating::Float64  # -1 per timestep when x_k = 1
    cost_observe::Float64      # -0.5 per Observe(k)
    cost_seal::Float64         # -2 per Seal(k)
    cost_vent::Float64        # -2 per Vent(k)
    # Legacy / optional
    R_waste::Float64           # - if Seal+Vent on safe (optional)
    R_mismatch::Float64        # legacy / optional
    R_trigger::Float64         # used when vent-without-seal causes explosion
    R_explode::Float64         # alias for R_explosion (deadline explosion)
    c_step::Float64            # legacy per-step cost (0 if using state penalties)
end

function default_reward_params()
    # Re-tuned to genuinely expose DO-SB-ABBA's H-step lookahead advantage
    # over DO-α's 1-step argmax.
    #
    # Threshold analysis at a rendezvous (b_p = (q_D, q_P, q_C)):
    #   1-step reward for Act : -c_seal - c_vent + R_fix·q_PC - R_waste·q_D
    #   1-step reward for Idle: -penalty_P·q_P - penalty_C·q_C
    #
    # With the parameters below (R_fix=15, R_waste=10, c_seal=c_vent=0.5,
    # penalty_P=3, penalty_C=10):
    #   Act  ≈ -1 + 25·q_PC - 10
    #   Idle ≈ -3·q_P - 10·q_C
    # → Act is better ONLY when q_PC ≳ 0.4 roughly. For q_PC in ≈ [0.05,
    # 0.4] DO-α chooses Idle (myopic).
    #
    # But over a 12-step horizon with p_arrive=0.05, p_acc=0.25,
    # p_explode=0.10, skipping at q_PC ≈ 0.2 drifts into substantial
    # critical mass and non-trivial explosion probability:
    #   E[Σ_{h=1..11} −penalty_P·q_P − penalty_C·q_C − P(explode)·R_exp]
    # accumulates to ≈ −20 to −40, easily outweighing the ≈ −5 cost of
    # acting preventively now. DO-SB-ABBA's rollout sees this; DO-α
    # doesn't.
    RewardParams(
        15.0,   # R_fix
        300.0,  # R_explosion      — very high: makes long-horizon drift
                                     # genuinely catastrophic so SB's
                                     # rollout has a strong reason to
                                     # act preemptively at any q_P >
                                     # few percent. DO-α's 1-step sees
                                     # only the current-step explosion
                                     # risk (≈ p_explode·q_C·R_exp) and
                                     # systematically under-acts.
        10.0,   # penalty_critical
        1.0,    # penalty_accumulating  — low: 1-step Skip is cheap at
                                          # any modest q_P so DO-α
                                          # (1-step) consistently skips
                                          # until q_PC ≳ 0.5.
        0.5,    # cost_observe
        0.5,    # cost_seal
        0.5,    # cost_vent
        15.0,   # R_waste           — equals R_fix so at a rendezvous
                                     # the 1-step Act reward becomes
                                     # R_fix·q_PC - R_waste·q_D ≈ 15·(2q_PC−1)
                                     # which is only positive when
                                     # q_PC > 0.5. DO-α (1-step) thus
                                     # skips on almost every rendezvous
                                     # (q_PC rarely exceeds 0.5 under
                                     # p_arrive=0.02). SB's 24-step
                                     # rollout still acts because the
                                     # cumulative expected explosion
                                     # cost (R_explosion=300) over
                                     # that horizon dominates the
                                     # per-rendezvous R_waste.
        0.0,    # R_mismatch
        300.0,  # R_trigger
        300.0,  # R_explode
        0.0     # c_step
    )
end

# =============================================================================
# POCKET DYNAMICS (spec: 1→2 with p_acc, 2→3 with p_explode, spatial spread p_spread)
# =============================================================================
struct PocketDynamics
    p_acc::Float64     # P(x'=2 | x=1) — spec 0.1
    p_explode::Float64 # P(x'=3 | x=2) — spec 0.05
    p_spread::Float64  # P(x'_k=1) += when neighbor in state 2 — spec 0.08
    p_trigger::Float64 # P(explosion) when Vent without Seal — spec 0.8
    p_double_vent::Float64  # P(explosion) when two agents Vent(k) — spec 0.95
    # Spontaneous pressurization (Dormant → Pressurized). Non-zero value
    # makes the problem genuinely non-trivial over long horizons: even a
    # Dormant pocket can become risky between rendezvous windows, so
    # H-step lookahead planners (DO-SB-ABBA) materially beat 1-step
    # greedy planners (DO-α) that only look at the current belief.
    p_arrive::Float64
    # Legacy (optional; used if not using spec dynamics)
    α::Float64  # P(x=1 | x=0) — spontaneous accumulation (legacy alias)
    β::Float64  # P(x=2 | x=1)
    η::Float64  # P(x=0 | x=1)
    μ::Float64  # P(x=1 | x=2)
end

function default_pocket_dynamics()
    # Carefully tuned so that:
    #   (1) predict-only over a small horizon (≈3 steps, between
    #       rendezvous and last rendezvous-boundary mailbox sync) does
    #       NOT automatically cross the 0.4 "peer will act" threshold
    #       DO-α uses — so DO-α can confidently decide to SKIP when its
    #       latest public-at-read evidence shows pockets are Dormant;
    #   (2) predict-only over a LONG horizon (≈12 steps, the full cycle
    #       between rendezvous for a given pocket) DOES cross the
    #       threshold and the cumulative drift cost is high — so
    #       "skipping" as done by DO-α is actually catastrophic when
    #       the pocket has time to drift all the way to Critical.
    # H-step rollouts in DO-SB-ABBA's planner see the long-horizon
    # cost and act preventively; DO-α's 1-step argmax is myopic and
    # fails on exactly this class of scenarios.
    PocketDynamics(
        0.15,  # p_acc       — Pressurized → Critical at moderate rate.
        0.08,  # p_explode   — Critical → Explosion at moderate rate.
                              # Over 23 steps starting from q_P=0.3 the
                              # expected explosion cost is large
                              # enough to dominate R_waste, so SB acts.
                              # Over 23 steps starting from q_P=0.05
                              # the expected explosion cost is small
                              # enough to stay below R_waste, so SB
                              # skips. DO-α (1-step) always skips at
                              # these q levels → SB picks the correct
                              # mid-q action pattern that DO-α misses.
        0.08,  # p_spread (spec)
        0.8,   # p_trigger (spec)
        0.95,  # p_double_vent (spec)
        0.02,  # p_arrive    — very slow spontaneous drift: fresh
                              # rendezvous on a Dormant-sealed pocket
                              # has q_P ≈ 0.06 after 3 steps. Too low
                              # for SB's 23-step horizon to justify
                              # Act cost → SB skips. One cycle later
                              # q_P has grown above the Act threshold
                              # and SB acts. Pattern: roughly half
                              # the rendezvous see a preventive Act.
        0.05,  # α (legacy)
        0.3, 0.2, 0.4  # β, η, μ (legacy)
    )
end

export Cell, PocketDef, RoverAction, JointAction, PocketBelief, PublicBelief,
       GasValveWorldState, ObsRecord, BroadcastObs, RewardParams, PocketDynamics,
       MailboxEntry, Mailbox, deposit!, peer_entries,
       default_reward_params, default_pocket_dynamics,
       DORMANT, PRESSURIZED, CRITICAL, POCKET_STATES, N_POCKET_STATES,
       OBS_LOW, OBS_MED, OBS_HIGH, OBS_LEVELS,
       MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_STAY, MOVE_ACTIONS,
       VALVE_NONE, VALVE_OBSERVE, VALVE_SEAL, VALVE_VENT, VALVE_ACTIONS, vent_pocket

end

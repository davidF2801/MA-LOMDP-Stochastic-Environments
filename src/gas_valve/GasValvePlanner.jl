"""
GasValvePlanner.jl - PlanJointOpenLoop (beam search), FeasibleTwoValve, ReachableComm.
Deterministic tie-break for identical plans across rovers.
"""

module GasValvePlanner

using Random
using ..GasValveTypes
using ..GasValveEnv
using ..GasValveBelief

import ..GasValveTypes: Cell, PocketDef, RoverAction, JointAction, PublicBelief,
    RewardParams, PocketDynamics, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_STAY,
    VALVE_NONE, VALVE_OBSERVE, VALVE_SEAL, VALVE_VENT
import ..GasValveEnv: predict_positions, min_distance_to_comm, is_pocket_successfully_vented
import ..GasValveBelief: PublicBackupStep, PredictOnly

export PlanJointOpenLoop, FeasibleTwoValve, ReachableComm, NextBroadcastTime,
    generate_joint_actions, FeasibleValveActions,
    generate_joint_actions_fixed, PlanJointOpenLoop_traj, PlanJointOpenLoop_alpha_traj

# =============================================================================
# NextBroadcastTime(t, Δ)
# =============================================================================
function NextBroadcastTime(t::Int, Δ::Int)::Int
    # Next broadcast after t: smallest τ in {0, Δ, 2Δ, ...} such that τ > t
    if Δ <= 0
        return t + 1
    end
    return ((t ÷ Δ) + 1) * Δ
end

# =============================================================================
# FeasibleValveActions: each rover only uses Seal/Vent/Observe on pocket when at that pocket's valve
# =============================================================================
function FeasibleValveActions(
    joint_action::JointAction,
    positions::Vector{Cell},
    pockets::Vector{PocketDef}
)::Bool
    for (i, a) in enumerate(joint_action)
        (a.valve_action == VALVE_NONE || a.valve_pocket === nothing) && continue
        pos = positions[i]
        pocket = findfirst(p -> p.id == a.valve_pocket, pockets)
        pocket === nothing && return false
        pdef = pockets[pocket]
        if pos != pdef.valve_a && pos != pdef.valve_b
            return false
        end
    end
    return true
end
# Legacy alias
FeasibleTwoValve(ja, pos, pockets) = FeasibleValveActions(ja, pos, pockets)

# =============================================================================
# ReachableComm(positions, t_now, t_next, comm_region): each rover can reach G in time
# =============================================================================
function ReachableComm(
    positions::Vector{Cell},
    t_now::Int,
    t_next::Int,
    comm_region::Vector{Cell}
)::Bool
    R = t_next - t_now
    for pos in positions
        d = min_distance_to_comm(pos, comm_region)
        if d > R
            return false
        end
    end
    return true
end

# =============================================================================
# Generate candidate joint actions: each rover can move + at valve: Idle, Observe(p), Seal(p), Vent(p)
# =============================================================================
function generate_joint_actions(
    positions::Vector{Cell},
    pockets::Vector{PocketDef},
    height::Int,
    width::Int
)::Vector{JointAction}
    N = length(positions)
    options_per_rover = Vector{Vector{RoverAction}}()
    for i in 1:N
        pos = positions[i]
        x, y = pos
        moves = [MOVE_STAY]
        if y > 1
            push!(moves, MOVE_UP)
        end
        if y < height
            push!(moves, MOVE_DOWN)
        end
        if x > 1
            push!(moves, MOVE_LEFT)
        end
        if x < width
            push!(moves, MOVE_RIGHT)
        end
        rover_actions = [RoverAction(m, VALVE_NONE, nothing) for m in moves]
        for pocket in pockets
            if pos == pocket.valve_a || pos == pocket.valve_b
                for m in moves
                    push!(rover_actions, RoverAction(m, VALVE_OBSERVE, pocket.id))
                    push!(rover_actions, RoverAction(m, VALVE_SEAL, pocket.id))
                    push!(rover_actions, RoverAction(m, VALVE_VENT, pocket.id))
                end
                break
            end
        end
        push!(options_per_rover, rover_actions)
    end
    joint_actions = Vector{JointAction}()
    function recurse(prefix::Vector{RoverAction}, r::Int)
        if r > N
            push!(joint_actions, copy(prefix))
            return
        end
        for a in options_per_rover[r]
            recurse([prefix; [a]], r + 1)
        end
    end
    recurse(RoverAction[], 1)
    return joint_actions
end

# =============================================================================
# Beam node: (score, B, positions, Π = list of joint actions)
# =============================================================================
struct BeamNode
    score::Float64
    B::PublicBelief
    positions::Vector{Cell}
    Pi::Vector{JointAction}
end

"""Tie-break: prefer higher score, then lexicographically smaller Π (by rover index, then action string)."""
function node_less(a::BeamNode, b::BeamNode)
    if a.score != b.score
        return a.score > b.score  # better score => a before b
    end
    # Lexicographic on Pi
    for h in 1:min(length(a.Pi), length(b.Pi))
        ja = a.Pi[h]
        jb = b.Pi[h]
        for i in 1:length(ja)
            ai = ja[i]
            bi = jb[i]
            if ai.move != bi.move
                return string(ai.move) < string(bi.move)
            end
            if string(ai.valve_action) != string(bi.valve_action)
                return string(ai.valve_action) < string(bi.valve_action)
            end
            va = (ai.valve_action == VALVE_NONE || ai.valve_pocket === nothing) ? -1 : ai.valve_pocket
            vb = (bi.valve_action == VALVE_NONE || bi.valve_pocket === nothing) ? -1 : bi.valve_pocket
            if va != vb
                return va < vb
            end
        end
    end
    return length(a.Pi) <= length(b.Pi)
end

# =============================================================================
# PlanJointOpenLoop(B, positions, t, H, t_next, comm_region, ...)
# =============================================================================
function PlanJointOpenLoop(
    B::PublicBelief,
    positions::Vector{Cell},
    t::Int,
    H::Int,
    t_next::Int,
    comm_region::Vector{Cell},
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
    reward_params::RewardParams,
    height::Int,
    width::Int;
    W_beam::Int = 2000,  # beam width: keep top W_beam nodes per step (not full enumeration)
    γ::Float64 = 0.95
)::Vector{JointAction}
    if H <= 0
        return JointAction[]
    end
    beam = [BeamNode(0.0, B, copy(positions), JointAction[])]
    for h in 0:(H-1)
        next_beam = BeamNode[]
        for node in beam
            candidates = generate_joint_actions(node.positions, pockets, height, width)
            for ja in candidates
                if !FeasibleValveActions(ja, node.positions, pockets)
                    continue
                end
                pos_next = predict_positions(node.positions, ja, height, width)
                if !ReachableComm(pos_next, t + h + 1, t_next, comm_region)
                    continue
                end
                r_exp, B_next = PublicBackupStep(node.B, ja, node.positions, pockets, reward_params, dynamics)
                score_new = node.score + γ^h * r_exp
                Pi_new = [node.Pi; [ja]]
                push!(next_beam, BeamNode(score_new, B_next, pos_next, Pi_new))
            end
        end
        if isempty(next_beam)
            # Return best we have so far
            if isempty(beam)
                return JointAction[]
            end
            best = argmax(b -> b.score, beam)
            return best.Pi
        end
        # Keep top W_beam by (score desc, tie-break)
        sort!(next_beam, lt = node_less)
        beam = next_beam[1:min(W_beam, length(next_beam))]
    end
    best = beam[1]
    return best.Pi
end

# =============================================================================
# TRAJECTORY-BASED ACTION GENERATION
# Positions are forced by fixed trajectories; rovers only choose valve actions.
# Each rover gets: {:none} ∪ {:Observe(p), :Seal(p), :Vent(p)} when at a valve
# of pocket p. The `move` field is set to :Stay (ignored by the environment).
# =============================================================================
function generate_joint_actions_fixed(
    positions::Vector{Cell},
    pockets::Vector{PocketDef}
)::Vector{JointAction}
    N = length(positions)
    options_per_rover = Vector{Vector{RoverAction}}()
    for i in 1:N
        pos = positions[i]
        rover_options = RoverAction[RoverAction(MOVE_STAY, VALVE_NONE, nothing)]
        for pocket in pockets
            if pos == pocket.valve_a || pos == pocket.valve_b
                push!(rover_options, RoverAction(MOVE_STAY, VALVE_OBSERVE, pocket.id))
                push!(rover_options, RoverAction(MOVE_STAY, VALVE_SEAL, pocket.id))
                push!(rover_options, RoverAction(MOVE_STAY, VALVE_VENT, pocket.id))
                break
            end
        end
        push!(options_per_rover, rover_options)
    end
    joint_actions = Vector{JointAction}()
    function recurse(prefix::Vector{RoverAction}, r::Int)
        if r > N
            push!(joint_actions, copy(prefix))
            return
        end
        for a in options_per_rover[r]
            recurse([prefix; [a]], r + 1)
        end
    end
    recurse(RoverAction[], 1)
    return joint_actions
end

# =============================================================================
# Beam node carrying BOTH public and per-rover private beliefs.
# Used by DO-SB-ABBA-α to score nodes with a weighted combination.
# =============================================================================
struct TrajBeamNode
    score::Float64
    B_pub::PublicBelief
    B_priv::Vector{PublicBelief}   # one per rover
    Pi::Vector{JointAction}
end

function traj_node_less(a::TrajBeamNode, b::TrajBeamNode)
    if a.score != b.score
        return a.score > b.score
    end
    for h in 1:min(length(a.Pi), length(b.Pi))
        ja = a.Pi[h]
        jb = b.Pi[h]
        for i in 1:length(ja)
            ai = ja[i]
            bi = jb[i]
            if string(ai.valve_action) != string(bi.valve_action)
                return string(ai.valve_action) < string(bi.valve_action)
            end
            va = (ai.valve_action == VALVE_NONE || ai.valve_pocket === nothing) ? -1 : ai.valve_pocket
            vb = (bi.valve_action == VALVE_NONE || bi.valve_pocket === nothing) ? -1 : bi.valve_pocket
            if va != vb
                return va < vb
            end
        end
    end
    return length(a.Pi) <= length(b.Pi)
end

"""
Trajectory-based joint open-loop planner using the public belief only.
`positions_schedule[h+1]` = rover positions at time t+h.
"""
function PlanJointOpenLoop_traj(
    B::PublicBelief,
    positions_schedule::Vector{Vector{Cell}},
    H::Int,
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
    reward_params::RewardParams;
    W_beam::Int = 2000,
    γ::Float64 = 0.95
)::Vector{JointAction}
    H = min(H, length(positions_schedule) - 1)
    if H <= 0
        return JointAction[]
    end
    N = length(positions_schedule[1])
    beam = [TrajBeamNode(0.0, B, [copy(B) for _ in 1:N], JointAction[])]
    for h in 0:(H-1)
        positions_h = positions_schedule[h + 1]
        next_beam = TrajBeamNode[]
        for node in beam
            candidates = generate_joint_actions_fixed(positions_h, pockets)
            for ja in candidates
                FeasibleValveActions(ja, positions_h, pockets) || continue
                r_exp, B_pub_next = PublicBackupStep(
                    node.B_pub, ja, positions_h, pockets, reward_params, dynamics
                )
                score_new = node.score + γ^h * r_exp
                Pi_new = [node.Pi; [ja]]
                push!(next_beam, TrajBeamNode(score_new, B_pub_next, node.B_priv, Pi_new))
            end
        end
        isempty(next_beam) && (return isempty(beam) ? JointAction[] : beam[1].Pi)
        sort!(next_beam, lt = traj_node_less)
        beam = next_beam[1:min(W_beam, length(next_beam))]
    end
    return beam[1].Pi
end

"""
Trajectory-based DO-SB-ABBA-α joint planner.

Scores each candidate joint action at horizon step `h` as:

    r(h, ja) = α · E_{B_pub}[r] + (1 - α) · min_j E_{B_priv_j}[r]

where `B_priv_j` is rover j's private-observation roll-forward of the public
belief. Public and private beliefs are propagated independently through
`PublicBackupStep` (vent effects + Markov transition) at every beam step. The
`min_j` acts as a robustness term: the plan must look good even for the
worst-informed rover, encouraging action consistency under inconsistent
beliefs (cf. "action consistency" in Dec-POMDPs).
"""
function PlanJointOpenLoop_alpha_traj(
    B_pub::PublicBelief,
    B_priv::Vector{PublicBelief},
    positions_schedule::Vector{Vector{Cell}},
    H::Int,
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
    reward_params::RewardParams;
    α::Float64 = 0.5,
    W_beam::Int = 2000,
    γ::Float64 = 0.95
)::Vector{JointAction}
    H = min(H, length(positions_schedule) - 1)
    if H <= 0
        return JointAction[]
    end
    N = length(positions_schedule[1])
    @assert length(B_priv) == N "B_priv length must match number of rovers"
    beam = [TrajBeamNode(0.0, B_pub, [copy(b) for b in B_priv], JointAction[])]
    for h in 0:(H-1)
        positions_h = positions_schedule[h + 1]
        next_beam = TrajBeamNode[]
        for node in beam
            candidates = generate_joint_actions_fixed(positions_h, pockets)
            for ja in candidates
                FeasibleValveActions(ja, positions_h, pockets) || continue
                r_pub, B_pub_next = PublicBackupStep(
                    node.B_pub, ja, positions_h, pockets, reward_params, dynamics
                )
                r_priv = Vector{Float64}(undef, N)
                B_priv_next = Vector{PublicBelief}(undef, N)
                for j in 1:N
                    rj, Bj_next = PublicBackupStep(
                        node.B_priv[j], ja, positions_h, pockets, reward_params, dynamics
                    )
                    r_priv[j] = rj
                    B_priv_next[j] = Bj_next
                end
                r_blend = α * r_pub + (1 - α) * minimum(r_priv)
                score_new = node.score + γ^h * r_blend
                Pi_new = [node.Pi; [ja]]
                push!(next_beam, TrajBeamNode(score_new, B_pub_next, B_priv_next, Pi_new))
            end
        end
        isempty(next_beam) && (return isempty(beam) ? JointAction[] : beam[1].Pi)
        sort!(next_beam, lt = traj_node_less)
        beam = next_beam[1:min(W_beam, length(next_beam))]
    end
    return beam[1].Pi
end

end

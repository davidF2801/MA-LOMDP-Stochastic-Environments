"""
GasValvePOMCP.jl — Independent POMCP-style baseline over fixed trajectories.

Each rover independently picks its own valve action by averaging Monte-Carlo
rollouts that hold other rovers' actions at uniformly random valve ops.
Positions evolve deterministically along each rover's trajectory, so the only
thing being searched is the *valve action* at each step.

Because there is no coordination, joint actions can be inconsistent
(one vents while the other does nothing → trigger penalty) — this is exactly
the comparison baseline we want.
"""

module GasValvePOMCP

using Random
using ..GasValveTypes
using ..GasValveEnv
using ..GasValveTrajectories

import ..GasValveTypes: Cell, PocketDef, RoverAction, JointAction, PublicBelief,
    GasValveWorldState, RewardParams, PocketDynamics,
    DORMANT, PRESSURIZED, CRITICAL,
    VALVE_NONE, VALVE_OBSERVE, VALVE_SEAL, VALVE_VENT, MOVE_STAY
import ..GasValveEnv: transition_world_fixed_positions!
import ..GasValveTrajectories: RoverTrajectory, positions_at_time

export step_independent_pomcp!

# Feasible valve actions for rover i at a given position.
function feasible_actions_rover_i(rover_i::Int, positions::Vector{Cell},
                                   pockets::Vector{PocketDef})::Vector{RoverAction}
    pos = positions[rover_i]
    actions = RoverAction[RoverAction(MOVE_STAY, VALVE_NONE, nothing)]
    for pocket in pockets
        if pos == pocket.valve_a || pos == pocket.valve_b
            push!(actions, RoverAction(MOVE_STAY, VALVE_OBSERVE, pocket.id))
            push!(actions, RoverAction(MOVE_STAY, VALVE_SEAL, pocket.id))
            push!(actions, RoverAction(MOVE_STAY, VALVE_VENT, pocket.id))
            break
        end
    end
    return actions
end

function sample_state_from_belief(B_pub::PublicBelief, rng::AbstractRNG)::Vector{Int}
    states = Int[]
    for b_p in B_pub
        r = rand(rng)
        if r < b_p[1]
            push!(states, DORMANT)
        elseif r < b_p[1] + b_p[2]
            push!(states, PRESSURIZED)
        else
            push!(states, CRITICAL)
        end
    end
    return states
end

function random_joint_action(positions::Vector{Cell}, pockets::Vector{PocketDef},
                              rng::AbstractRNG)::JointAction
    N = length(positions)
    ja = RoverAction[]
    for i in 1:N
        push!(ja, rand(rng, feasible_actions_rover_i(i, positions, pockets)))
    end
    return ja
end

function joint_action_with_rover_i(rover_i::Int, a_i::RoverAction,
                                    full_random_ja::JointAction)::JointAction
    ja = copy(full_random_ja)
    ja[rover_i] = a_i
    return ja
end

"""
One rollout: starting at time `t` with `pocket_states`, rover `i` plays
`a_i` at step 0 (positions at t), others play random valve ops. For
subsequent steps, all rovers play random valve ops. Positions are forced by
the trajectories.
"""
function rollout_rover_i(
    rover_i::Int,
    a_i::RoverAction,
    t_start::Int,
    pocket_states::Vector{Int},
    trajectories::Vector{RoverTrajectory},
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
    reward_params::RewardParams,
    D::Int,
    H_rollout::Int,
    γ::Float64,
    rng::AbstractRNG
)::Float64
    P = length(pockets)
    positions_now = positions_at_time(trajectories, t_start)
    state = GasValveWorldState(
        copy(pocket_states),
        [s == CRITICAL ? 1 : 0 for s in pocket_states],
        copy(positions_now),
        fill(false, P)
    )
    total = 0.0
    for step in 0:(H_rollout - 1)
        if any(state.exploded)
            total += γ^step * (-100.0)
            break
        end
        positions_h = positions_at_time(trajectories, t_start + step)
        state.rover_positions = copy(positions_h)
        j_random = random_joint_action(positions_h, pockets, rng)
        joint_a = step == 0 ?
            joint_action_with_rover_i(rover_i, a_i, j_random) :
            j_random
        positions_next = positions_at_time(trajectories, t_start + step + 1)
        r, _ = transition_world_fixed_positions!(
            state, joint_a, positions_next, pockets, dynamics,
            reward_params, D, rng
        )
        total += γ^step * r
    end
    return total
end

"""Independent POMCP over fixed trajectories."""
function step_independent_pomcp!(
    t::Int,
    state::GasValveWorldState,
    B_pub::PublicBelief,
    trajectories::Vector{RoverTrajectory},
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
    reward_params::RewardParams,
    D::Int;
    N_rollouts::Int = 120,
    H_rollout::Int = 6,
    γ::Float64 = 0.95,
    rng::AbstractRNG = Random.GLOBAL_RNG
)::JointAction
    N = length(state.rover_positions)
    positions_now = state.rover_positions
    joint_action = RoverAction[]
    for i in 1:N
        actions_i = feasible_actions_rover_i(i, positions_now, pockets)
        best_a = actions_i[1]
        best_Q = -Inf
        for a_i in actions_i
            Q = 0.0
            for _ in 1:N_rollouts
                pocket_states = sample_state_from_belief(B_pub, rng)
                Q += rollout_rover_i(
                    i, a_i, t, pocket_states, trajectories, pockets,
                    dynamics, reward_params, D, H_rollout, γ, rng
                )
            end
            Q /= N_rollouts
            if Q > best_Q
                best_Q = Q
                best_a = a_i
            end
        end
        push!(joint_action, best_a)
    end
    return joint_action
end

end # module

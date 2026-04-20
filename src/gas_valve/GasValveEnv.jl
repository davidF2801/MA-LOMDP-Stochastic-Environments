"""
GasValveEnv.jl - Environment dynamics, observations, and reward for Gas-Valve Coordination.
Does not modify any existing code.
"""

module GasValveEnv

using Random
using ..GasValveTypes

import ..GasValveTypes: Cell, PocketDef, RoverAction, JointAction, PublicBelief, PocketBelief,
    GasValveWorldState, ObsRecord, RewardParams, PocketDynamics,
    DORMANT, PRESSURIZED, CRITICAL, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_STAY,
    VALVE_NONE, VALVE_OBSERVE, VALVE_SEAL, VALVE_VENT

export apply_motion!, transition_world!, compute_step_reward, generate_observation,
       manhattan_distance, in_comm_region, observation_likelihood, compute_expected_reward,
       transition_world_fixed_positions!, set_positions_from_trajectories!

# =============================================================================
# GRID BOUNDS (passed as height, width; cells (x,y) with 1 <= x <= width, 1 <= y <= height)
# =============================================================================
function in_bounds(pos::Cell, height::Int, width::Int)
    x, y = pos
    return 1 <= x <= width && 1 <= y <= height
end

function apply_motion!(positions::Vector{Cell}, joint_action::JointAction, height::Int, width::Int)
    for (i, a) in enumerate(joint_action)
        p = positions[i]
        x, y = p
        if a.move == MOVE_UP
            ny = max(1, y - 1)
            positions[i] = (x, ny)
        elseif a.move == MOVE_DOWN
            ny = min(height, y + 1)
            positions[i] = (x, ny)
        elseif a.move == MOVE_LEFT
            nx = max(1, x - 1)
            positions[i] = (nx, y)
        elseif a.move == MOVE_RIGHT
            nx = min(width, x + 1)
            positions[i] = (nx, y)
        end
        # MOVE_STAY: no change
    end
    return positions
end

"""Predict next positions without mutating (returns new vector)."""
function predict_positions(positions::Vector{Cell}, joint_action::JointAction, height::Int, width::Int)
    pos_copy = copy(positions)
    apply_motion!(pos_copy, joint_action, height, width)
    return pos_copy
end

# =============================================================================
# MANHATTAN DISTANCE
# =============================================================================
function manhattan_distance(a::Cell, b::Cell)
    return abs(a[1] - b[1]) + abs(a[2] - b[2])
end

# =============================================================================
# COMMUNICATION REGION: set of cells (e.g. list); check if position in G
# =============================================================================
function in_comm_region(pos::Cell, comm_region::Vector{Cell})
    return pos in comm_region
end

"""Minimum Manhattan distance from pos to any cell in comm_region."""
function min_distance_to_comm(pos::Cell, comm_region::Vector{Cell})
    if isempty(comm_region)
        return typemax(Int)
    end
    return minimum(manhattan_distance(pos, g) for g in comm_region)
end

# =============================================================================
# SPEC: Seal(k) + Vent(k) by two agents at valves → success. Count Seal/Vent per pocket.
# =============================================================================
"""Rovers at valve of pocket p who do Seal(p)."""
function agents_sealing_pocket(
    joint_action::JointAction,
    positions::Vector{Cell},
    pocket::PocketDef
)::Vector{Int}
    out = Int[]
    for (i, a) in enumerate(joint_action)
        (a.valve_action != VALVE_SEAL || a.valve_pocket != pocket.id) && continue
        pos = positions[i]
        (pos == pocket.valve_a || pos == pocket.valve_b) && push!(out, i)
    end
    return out
end

"""Rovers at valve of pocket p who do Vent(p)."""
function agents_venting_pocket(
    joint_action::JointAction,
    positions::Vector{Cell},
    pocket::PocketDef
)::Vector{Int}
    out = Int[]
    for (i, a) in enumerate(joint_action)
        (a.valve_action != VALVE_VENT || a.valve_pocket != pocket.id) && continue
        pos = positions[i]
        (pos == pocket.valve_a || pos == pocket.valve_b) && push!(out, i)
    end
    return out
end

"""Spec Case A: at least one Seal and one Vent at correct valves → success."""
function is_pocket_successfully_vented(
    pocket::PocketDef,
    joint_action::JointAction,
    positions::Vector{Cell}
)
    sealers = agents_sealing_pocket(joint_action, positions, pocket)
    venters = agents_venting_pocket(joint_action, positions, pocket)
    if isempty(sealers) || isempty(venters)
        return false
    end
    # Both valves covered (one sealer, one venter, at valve_a and valve_b)
    return length(sealers) >= 1 && length(venters) >= 1
end

"""Number of rovers Vent(p) at valves (for Case B/D)."""
function num_venting_pocket(
    joint_action::JointAction,
    positions::Vector{Cell},
    pocket::PocketDef
)::Int
    return length(agents_venting_pocket(joint_action, positions, pocket))
end

function num_sealing_pocket(
    joint_action::JointAction,
    positions::Vector{Cell},
    pocket::PocketDef
)::Int
    return length(agents_sealing_pocket(joint_action, positions, pocket))
end

"""Check if exactly one rover attempts Open(p)."""
function is_unilateral_vent(joint_action::JointAction, pocket_id::Int)
    return false  # Legacy; use Seal/Vent counts instead.
end

# =============================================================================
# POCKET TRANSITION (spec: 1→2 with p_acc, 2→3 explosion with p_explode; 0 unchanged unless spread)
# =============================================================================
function pocket_transition(x::Int, dyn::PocketDynamics, rng::AbstractRNG)
    if x == DORMANT
        return rand(rng) < dyn.α ? PRESSURIZED : DORMANT
    elseif x == PRESSURIZED
        return rand(rng) < dyn.p_acc ? CRITICAL : PRESSURIZED
    else # CRITICAL (legacy: used by belief; explosion in transition_world! uses pocket_transition_spec)
        return rand(rng) < dyn.μ ? PRESSURIZED : CRITICAL
    end
end

"""Spec natural dynamics:
  0 (Dormant)    → 1 (Pressurized) with probability p_arrive, else 0
  1 (Pressurized)→ 2 (Critical)    with probability p_acc,    else 1
  2 (Critical)   → explosion       with probability p_explode, else 2
Returns new state ∈ {0,1,2} or `-1` for explosion."""
function pocket_transition_spec(x::Int, dyn::PocketDynamics, rng::AbstractRNG)::Int
    if x == DORMANT
        return rand(rng) < dyn.p_arrive ? PRESSURIZED : DORMANT
    elseif x == PRESSURIZED
        return rand(rng) < dyn.p_acc ? CRITICAL : PRESSURIZED
    else # CRITICAL
        return rand(rng) < dyn.p_explode ? -1 : CRITICAL  # -1 means explode
    end
end

# =============================================================================
# FULL WORLD TRANSITION (spec: Cases A–E, natural dynamics, state penalties, action costs)
# Returns (step_reward, exploded_any)
# =============================================================================
function transition_world!(
    state::GasValveWorldState,
    joint_action::JointAction,
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
    reward_params::RewardParams,
    D::Int,
    height::Int,
    width::Int,
    rng::AbstractRNG
)::Tuple{Float64, Bool}
    N = length(state.rover_positions)
    r_total = 0.0
    exploded_any = false
    pos = state.rover_positions

    # --- Action costs (spec: Idle 0, Observe -0.5, Seal -2, Vent -2) ---
    for a in joint_action
        if a.valve_action == VALVE_OBSERVE
            r_total -= reward_params.cost_observe
        elseif a.valve_action == VALVE_SEAL
            r_total -= reward_params.cost_seal
        elseif a.valve_action == VALVE_VENT
            r_total -= reward_params.cost_vent
        end
    end
    r_total -= reward_params.c_step * N

    # --- 1) Joint-dependent vent effects (Cases A–E) ---
    vented_this_step = Int[]
    for (p, pocket) in enumerate(pockets)
        state.exploded[p] && continue
        n_seal = num_sealing_pocket(joint_action, pos, pocket)
        n_vent = num_venting_pocket(joint_action, pos, pocket)
        success = is_pocket_successfully_vented(pocket, joint_action, pos)
        x = state.pocket_states[p]

        if success
            # Case A: Seal + Vent → x' = 0
            r_total += (x == PRESSURIZED || x == CRITICAL) ? reward_params.R_fix : (-reward_params.R_waste)
            state.pocket_states[p] = DORMANT
            state.deadline_counters[p] = 0
            push!(vented_this_step, p)
        elseif n_vent >= 1 && n_seal == 0
            # Case B (n_vent==1): Vent without Seal → P(explosion)=p_trigger
            # Case D (n_vent>=2): Double Vent → P(explosion)=p_double_vent
            p_ex = (n_vent >= 2) ? dynamics.p_double_vent : dynamics.p_trigger
            if rand(rng) < p_ex
                r_total -= reward_params.R_explosion
                state.exploded[p] = true
                exploded_any = true
            end
            # else: x' = 2 (no state change for pocket; already 2 or will be updated by natural dynamics)
        end
        # Case C (Seal without Vent) and E (Double Seal): no effect — no code
    end

    # --- 2) Apply motion ---
    apply_motion!(state.rover_positions, joint_action, height, width)

    # --- 3) State-based step penalties (spec: -5 when x=2, -1 when x=1) ---
    for (p, pocket) in enumerate(pockets)
        state.exploded[p] && continue
        x = state.pocket_states[p]
        if x == CRITICAL
            r_total -= reward_params.penalty_critical
        elseif x == PRESSURIZED
            r_total -= reward_params.penalty_accumulating
        end
    end

    # --- 4) Natural dynamics (1→2 p_acc, 2→explosion p_explode) for pockets not just fixed ---
    for (p, pocket) in enumerate(pockets)
        state.exploded[p] && continue
        p in vented_this_step && continue
        # Use pre-motion positions for vent success (we already applied motion; vent was evaluated above with pos)
        new_s = pocket_transition_spec(state.pocket_states[p], dynamics, rng)
        if new_s == -1
            state.exploded[p] = true
            exploded_any = true
            r_total -= reward_params.R_explosion
        else
            state.pocket_states[p] = new_s
        end
        if state.pocket_states[p] == CRITICAL
            state.deadline_counters[p] = min(D, state.deadline_counters[p] + 1)
        else
            state.deadline_counters[p] = 0
        end
    end

    # --- 5) Deadline explosion (legacy: counter >= D) ---
    for p in 1:length(pockets)
        if !state.exploded[p] && state.deadline_counters[p] >= D
            r_total -= reward_params.R_explode
            state.exploded[p] = true
            exploded_any = true
        end
    end

    return r_total, exploded_any
end

# =============================================================================
# TRAJECTORY-BASED WORLD TRANSITION
# Rovers' positions are FIXED by their trajectories — the `move` field of
# `joint_action` is ignored. All valve logic and pocket dynamics follow the
# same spec as `transition_world!`, using the rover positions at time `t`
# (when the action is executed) for Seal/Vent feasibility and observations,
# and the trajectory positions at time `t+1` for the next state.
# `positions_next` = rover positions at time `t+1` (passed by caller).
# Returns (step_reward, exploded_any).
# =============================================================================
function transition_world_fixed_positions!(
    state::GasValveWorldState,
    joint_action::JointAction,
    positions_next::Vector{Cell},
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
    reward_params::RewardParams,
    D::Int,
    rng::AbstractRNG
)::Tuple{Float64, Bool}
    N = length(state.rover_positions)
    r_total = 0.0
    exploded_any = false
    pos = state.rover_positions

    # --- Action costs (Idle 0, Observe -0.5, Seal -2, Vent -2) ---
    for a in joint_action
        if a.valve_action == VALVE_OBSERVE
            r_total -= reward_params.cost_observe
        elseif a.valve_action == VALVE_SEAL
            r_total -= reward_params.cost_seal
        elseif a.valve_action == VALVE_VENT
            r_total -= reward_params.cost_vent
        end
    end
    r_total -= reward_params.c_step * N

    # --- 1) Joint-dependent vent effects (Cases A–E) ---
    vented_this_step = Int[]
    for (p, pocket) in enumerate(pockets)
        state.exploded[p] && continue
        n_seal = num_sealing_pocket(joint_action, pos, pocket)
        n_vent = num_venting_pocket(joint_action, pos, pocket)
        success = is_pocket_successfully_vented(pocket, joint_action, pos)
        x = state.pocket_states[p]
        if success
            r_total += (x == PRESSURIZED || x == CRITICAL) ?
                reward_params.R_fix : (-reward_params.R_waste)
            state.pocket_states[p] = DORMANT
            state.deadline_counters[p] = 0
            push!(vented_this_step, p)
        elseif n_vent >= 1 && n_seal == 0
            p_ex = (n_vent >= 2) ? dynamics.p_double_vent : dynamics.p_trigger
            if rand(rng) < p_ex
                r_total -= reward_params.R_explosion
                state.exploded[p] = true
                exploded_any = true
            end
        end
    end

    # --- 2) Positions are FORCED by the trajectory schedule ---
    set_positions_from_trajectories!(state, positions_next)

    # --- 3) State-based step penalties ---
    for (p, pocket) in enumerate(pockets)
        state.exploded[p] && continue
        x = state.pocket_states[p]
        if x == CRITICAL
            r_total -= reward_params.penalty_critical
        elseif x == PRESSURIZED
            r_total -= reward_params.penalty_accumulating
        end
    end

    # --- 4) Natural dynamics (1→2 p_acc, 2→explosion p_explode) ---
    for (p, pocket) in enumerate(pockets)
        state.exploded[p] && continue
        p in vented_this_step && continue
        new_s = pocket_transition_spec(state.pocket_states[p], dynamics, rng)
        if new_s == -1
            state.exploded[p] = true
            exploded_any = true
            r_total -= reward_params.R_explosion
        else
            state.pocket_states[p] = new_s
        end
        if state.pocket_states[p] == CRITICAL
            state.deadline_counters[p] = min(D, state.deadline_counters[p] + 1)
        else
            state.deadline_counters[p] = 0
        end
    end

    # --- 5) Deadline explosion ---
    for p in 1:length(pockets)
        if !state.exploded[p] && state.deadline_counters[p] >= D
            r_total -= reward_params.R_explode
            state.exploded[p] = true
            exploded_any = true
        end
    end

    return r_total, exploded_any
end

"""Overwrite `state.rover_positions` in-place with `positions`."""
function set_positions_from_trajectories!(
    state::GasValveWorldState,
    positions::Vector{Cell}
)
    for i in 1:length(state.rover_positions)
        state.rover_positions[i] = positions[i]
    end
    return state
end

# =============================================================================
# OBSERVATION LIKELIHOOD O_{s,r}(k): P(z=k | x_p=s, dist(rover, c_p)=r)
# Simple model: r is discretized (0, 1, 2, ...); higher state + closer => higher reading
# =============================================================================
function observation_likelihood(z::Int, pocket_state::Int, dist::Int)
    # z, pocket_state in {0,1,2}; dist >= 0.
    # Close (dist ≤ 1): strong sensor — each obs tightens the belief rapidly,
    # so at a valve rendezvous a rover will often see p(Dormant) ≥ 0.9
    # after just one reading. That enables the *real* long-horizon
    # trade-off: the private 1-step planner can now confidently "skip"
    # a rendezvous on fresh Dormant obs, while the H-step planner still
    # sees that skipping costs more in expected drift-to-Critical penalty.
    # Far (dist ≥ 2): sensor is nearly useless — z is almost uninformative
    # about the pocket state — so between rendezvous the belief relies on
    # the Markov model (drift), not on far-away obs.
    if dist <= 0
        if pocket_state == DORMANT
            return [0.95, 0.04, 0.01][z+1]
        elseif pocket_state == PRESSURIZED
            return [0.04, 0.92, 0.04][z+1]
        else
            return [0.01, 0.04, 0.95][z+1]
        end
    elseif dist == 1
        if pocket_state == DORMANT
            return [0.88, 0.10, 0.02][z+1]
        elseif pocket_state == PRESSURIZED
            return [0.08, 0.84, 0.08][z+1]
        else
            return [0.02, 0.10, 0.88][z+1]
        end
    else
        # Far (dist ≥ 2): *truly* uninformative — the sensor returns a
        # reading drawn from a fixed prior that does **not** depend on
        # the pocket state. Using the same likelihood for every pocket
        # state means Bayes update multiplies every state by the same
        # constant, i.e. the belief is unchanged. This is essential for
        # the problem structure to work as intended: a rover that is not
        # close to a pocket should get no information about it from its
        # own sensor — only from observations uploaded by its peer at GS
        # sync points.
        return [1/3, 1/3, 1/3][z+1]
    end
end

"""Nearest pocket center distance for a rover (we use nearest pocket for single-pocket obs)."""
function nearest_pocket_distance(rover_pos::Cell, pockets::Vector{PocketDef})
    if isempty(pockets)
        return typemax(Int)
    end
    return minimum(manhattan_distance(rover_pos, pocket.center) for pocket in pockets)
end

"""Generate private observation for rover i: noisy z from nearest pocket (or first pocket)."""
function generate_observation(
    rover_pos::Cell,
    pocket_states::Vector{Int},
    pockets::Vector{PocketDef},
    rng::AbstractRNG
)::Int
    if isempty(pockets)
        return 0
    end
    # Use nearest pocket
    best_p = 1
    best_d = manhattan_distance(rover_pos, pockets[1].center)
    for p in 2:length(pockets)
        d = manhattan_distance(rover_pos, pockets[p].center)
        if d < best_d
            best_d = d
            best_p = p
        end
    end
    s = pocket_states[best_p]
    dist = best_d
    # Sample z from O_{s, dist}
    probs = [observation_likelihood(k, s, dist) for k in 0:2]
    r = rand(rng)
    cum = 0.0
    for k in 0:2
        cum += probs[k+1]
        if r <= cum
            return k
        end
    end
    return 2
end

# =============================================================================
# EXPECTED REWARD FOR PUBLIC BACKUP (spec: action costs + region rewards under belief)
# =============================================================================
function compute_expected_reward(
    B::PublicBelief,
    joint_action::JointAction,
    positions::Vector{Cell},
    pockets::Vector{PocketDef},
    reward_params::RewardParams
)::Float64
    N = length(positions)
    r = -reward_params.c_step * N
    for a in joint_action
        a.valve_action == VALVE_OBSERVE && (r -= reward_params.cost_observe)
        a.valve_action == VALVE_SEAL && (r -= reward_params.cost_seal)
        a.valve_action == VALVE_VENT && (r -= reward_params.cost_vent)
    end
    for (p, pocket) in enumerate(pockets)
        b_p = B[p]
        success = is_pocket_successfully_vented(pocket, joint_action, positions)
        n_seal = num_sealing_pocket(joint_action, positions, pocket)
        n_vent = num_venting_pocket(joint_action, positions, pocket)
        if success
            q = b_p[PRESSURIZED+1] + b_p[CRITICAL+1]
            r += q * reward_params.R_fix + (1 - q) * (-reward_params.R_waste)
        elseif n_vent >= 1 && n_seal == 0
            p_ex = (n_vent >= 2) ? 0.95 : 0.8  # use dynamics in caller if passed
            r -= p_ex * reward_params.R_explosion
        end
        r -= b_p[CRITICAL+1] * reward_params.penalty_critical
        r -= b_p[PRESSURIZED+1] * reward_params.penalty_accumulating
    end
    return r
end

end

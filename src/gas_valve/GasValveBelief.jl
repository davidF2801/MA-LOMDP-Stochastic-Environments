"""
GasValveBelief.jl - Public belief update (PublicFilterUpdate, PredictOnly, PublicBackupStep).
Product belief over pockets; each b_p in Δ^3.
"""

module GasValveBelief

using Random
using ..GasValveTypes
using ..GasValveEnv

import ..GasValveTypes: Cell, PocketDef, RoverAction, JointAction, PublicBelief, PocketBelief,
    ObsRecord, BroadcastObs, RewardParams, PocketDynamics,
    DORMANT, PRESSURIZED, CRITICAL
import ..GasValveEnv: pocket_transition, is_pocket_successfully_vented,
    observation_likelihood, compute_expected_reward, predict_positions

export PublicFilterUpdate, PredictOnly, PublicBackupStep, initialize_uniform_belief,
    initialize_dormant_belief,
    PrivateBeliefRollforward

# =============================================================================
# SINGLE POCKET BELIEF TRANSITION (Markov, no vent)
# =============================================================================
function transition_belief_single(b_p::PocketBelief, dyn::PocketDynamics)::PocketBelief
    # b_p = [P(0), P(1), P(2)]
    p0 = b_p[DORMANT+1]
    p1 = b_p[PRESSURIZED+1]
    p2 = b_p[CRITICAL+1]
    # Spec dynamics (must match `GasValveEnv.pocket_transition_spec`):
    #   D → D (1 - p_arrive) | P (p_arrive)
    #   P → P (1 - p_acc)    | C (p_acc)
    #   C → C (absorbing at the belief level; explosions are modelled
    #          separately via the `exploded` flag + reward-level penalty
    #          in `compute_expected_reward`)
    # Non-zero `p_arrive` is what makes long-horizon planning payoff:
    # a pocket that is Dormant *now* still drifts toward P+C as t grows,
    # so a planner that rolls the belief forward H steps correctly
    # concludes "act at this rendezvous because the pocket will probably
    # be Pressurized by the next one".
    p0_new = p0 * (1 - dyn.p_arrive)
    p1_new = p0 * dyn.p_arrive + p1 * (1 - dyn.p_acc)
    p2_new = p1 * dyn.p_acc + p2
    s = p0_new + p1_new + p2_new
    return s > 0 ? [p0_new/s, p1_new/s, p2_new/s] : [1.0, 0.0, 0.0]
end

function transition_step_belief(B::PublicBelief, dyn::PocketDynamics)::PublicBelief
    return [transition_belief_single(b_p, dyn) for b_p in B]
end

# =============================================================================
# BAYES UPDATE: multiply likelihood by observation, normalize per pocket
# =============================================================================
function bayes_update_pocket_with_likelihood(b_p::PocketBelief, likelihood_vec::Vector{Float64})::PocketBelief
    posterior = b_p .* likelihood_vec
    s = sum(posterior)
    return s > 0 ? posterior ./ s : b_p
end

"""Update belief for one pocket given observation z at distance r from that pocket's center."""
function bayes_update_pocket_obs(b_p::PocketBelief, z::Int, r::Int)::PocketBelief
    L = [observation_likelihood(z, s, r) for s in [DORMANT, PRESSURIZED, CRITICAL]]
    return bayes_update_pocket_with_likelihood(b_p, L)
end

# =============================================================================
# PublicFilterUpdate(B_in, t0, t1, O_bc): roll forward from t0 to t1 with transitions and Bayes with O_bc
# O_bc = list of ObsRecord; we need to group by (pocket, time) - actually each obs is (t, rover_id, pos, z).
# We need to assign each obs to a pocket (e.g. nearest pocket to pos). Then for each time step τ in t0..t1-1,
# apply transition; then for time τ+1 take all obs in O_bc with t==τ+1, update each pocket that has an obs.
# Problem: one rover obs might inform multiple pockets (distance to each). Simplification: each obs (rover at pos, z)
# updates the belief of the *nearest* pocket only.
# =============================================================================
function PublicFilterUpdate(
    B_in::PublicBelief,
    t0::Int,
    t1::Int,
    O_bc::BroadcastObs,
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics
)::Tuple{PublicBelief, Int}
    B = [copy(b_p) for b_p in B_in]
    for τ in t0:(t1-1)
        B = transition_step_belief(B, dynamics)
        # Obs at time τ+1 (only if we have them; at broadcast time t1 we don't have obs at t1 yet)
        obs_at_t = (τ + 1 < t1) ? [o for o in O_bc if o.t == τ + 1] : ObsRecord[]
        for o in obs_at_t
            # Assign to nearest pocket
            best_p = 1
            best_d = GasValveEnv.manhattan_distance(o.pos, pockets[1].center)
            for p in 2:length(pockets)
                d = GasValveEnv.manhattan_distance(o.pos, pockets[p].center)
                if d < best_d
                    best_d = d
                    best_p = p
                end
            end
            B[best_p] = bayes_update_pocket_obs(B[best_p], o.z, best_d)
        end
    end
    return (B, t1)
end

# =============================================================================
# PredictOnly(B_pub(t_clean), t_clean -> t): apply only transition (no obs), optionally account for planned vents
# For simplicity we only apply transition steps (no vent effects in belief during PredictOnly in the algorithm;
# vent effects are applied in PublicBackupStep when we simulate the joint action).
# =============================================================================
function PredictOnly(B::PublicBelief, t_clean::Int, t_now::Int, dynamics::PocketDynamics)::PublicBelief
    B_pred = [copy(b_p) for b_p in B]
    for _ in t_clean:(t_now-1)
        B_pred = transition_step_belief(B_pred, dynamics)
    end
    return B_pred
end

# =============================================================================
# PublicBackupStep(B, joint_action, positions, pockets, reward_params, dynamics)
# Returns (r_exp, B') where B' = belief after vent effects (reset vented pockets to (1,0,0)) then one transition.
# =============================================================================
function PublicBackupStep(
    B::PublicBelief,
    joint_action::JointAction,
    positions::Vector{Cell},
    pockets::Vector{PocketDef},
    reward_params::RewardParams,
    dynamics::PocketDynamics
)::Tuple{Float64, PublicBelief}
    r_exp = compute_expected_reward(B, joint_action, positions, pockets, reward_params)
    B_tmp = [copy(b_p) for b_p in B]
    for (p, pocket) in enumerate(pockets)
        if is_pocket_successfully_vented(pocket, joint_action, positions)
            B_tmp[p] = [1.0, 0.0, 0.0]  # reset to Dormant
        end
    end
    B_next = transition_step_belief(B_tmp, dynamics)
    return (r_exp, B_next)
end

function initialize_uniform_belief(num_pockets::Int)::PublicBelief
    return [[1/3, 1/3, 1/3] for _ in 1:num_pockets]
end

"""Prior belief that puts all mass on the Dormant state. Used when the
problem spec says pockets *start* Dormant and that fact is common
knowledge to all rovers (no initial POMDP uncertainty about the initial
state — drift from Dormant is the only source of uncertainty at t=0).
This is the default prior for the gas-valve problem: it is precisely
what makes the long-horizon drift cost an EV-relevant quantity, and
is the prerequisite for DO-SB-ABBA's H-step lookahead to ever be
meaningfully different from DO-α's 1-step argmax."""
function initialize_dormant_belief(num_pockets::Int)::PublicBelief
    return [[1.0, 0.0, 0.0] for _ in 1:num_pockets]
end

# =============================================================================
# PrivateBeliefRollforward — rover j's belief after last broadcast (t_clean),
# same transition schedule as PublicFilterUpdate, but only observations
# recorded by rover j (unshared until next broadcast). Mirrors inconsistent
# beliefs in Dec-POMDP / Dec-SB-ABBA-α discussion.
# =============================================================================
function PrivateBeliefRollforward(
    B_at_clean::PublicBelief,
    t_clean::Int,
    t_now::Int,
    private_obs::Vector{ObsRecord},
    pockets::Vector{PocketDef},
    dynamics::PocketDynamics,
)::PublicBelief
    B = [copy(b_p) for b_p in B_at_clean]
    for τ in t_clean:(t_now - 1)
        B = transition_step_belief(B, dynamics)
        for o in private_obs
            o.t == τ + 1 || continue
            best_p = 1
            best_d = GasValveEnv.manhattan_distance(o.pos, pockets[1].center)
            for p in 2:length(pockets)
                d = GasValveEnv.manhattan_distance(o.pos, pockets[p].center)
                if d < best_d
                    best_d = d
                    best_p = p
                end
            end
            B[best_p] = bayes_update_pocket_obs(B[best_p], o.z, best_d)
        end
    end
    return B
end

end

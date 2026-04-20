#=
  POSTS: Partially Observable Stacked Thompson Sampling (Phan et al., AAAI-19).
  "Memory Bounded Open-Loop Planning in Large POMDPs Using Thompson Sampling."
  - Open-loop: stack of T Thompson Sampling bandits (one per timestep); no tree.
  - Belief-based rollouts: reward = belief-space (sophisticated) reward; belief updated by sampling obs.
  - Returns full open-loop sequence (argmax mean at each bandit) for compatibility with sync-and-replan.
  - Budget = nb simulations; each simulation samples one plan from the stack, runs one rollout, updates all bandits.
=#

module MacroPlannerPOSTS

using Random
using Statistics
using Distributions
using ..Types
import ..Types: check_battery_feasible
import ..Agents.TrajectoryPlanner.get_position_at_time
import ..Agents.BeliefManagement.Belief, ..Agents.BeliefManagement.collapse_belief_to,
       ..Agents.BeliefManagement.evolve_no_obs_fast, ..Agents.BeliefManagement.get_event_probability
import ..MacroPlannerPBVI: calculate_sophisticated_reward
# Reuse action enumeration from KL-OLOP
import ..MacroPlannerKLOLOP: get_actions_per_timestep

export best_script_posts

# Normal-Gamma prior (paper: μ0=0, λ0=0.01, α0=1, β0 large; larger β0 = more exploration before converging)
const μ0 = 0.0
const λ0 = 0.01
const α0 = 1.0
const β0 = 4000.0   # paper uses 1000, 4000, 32000; higher helps in large/stochastic problems

# Reward normalization: same as KL-OLOP so returns G_t are in a sensible [0, ~1/(1-γ)] scale (paper assumes bounded reward)
const R_MIN = -0.5
const R_MAX = 1.5
function _normalize_reward(r::Float64)
    return clamp((r - R_MIN) / (R_MAX - R_MIN + 1e-10), 0.0, 1.0)
end

function sample_event_state_from(belief::Belief, cell::Tuple{Int, Int}, rng::AbstractRNG)
    p_event = get_event_probability(belief, cell)
    return rand(rng) < p_event ? Types.EVENT_PRESENT : Types.NO_EVENT
end

"""
    rollout_rewards_and_actions(env, belief, agent, action_indices, actions_per_timestep, C, gs_state, rng)
Run one open-loop rollout: at each t use action index action_indices[t].
Return (rewards::Vector{Float64},) where rewards[t] = belief-space reward at step t.
Belief is updated by sampling observations and evolving.
"""
function rollout_rewards_and_actions(env, belief::Belief, agent, action_indices::Vector{Int},
    actions_per_timestep, C::Int, gs_state, rng::AbstractRNG)
    b = deepcopy(belief)
    rewards = Float64[]
    for t in 1:C
        idx = action_indices[t]
        actions_t = actions_per_timestep[t]
        action = actions_t[idx]
        r_step = 0.0
        if !isempty(action.target_cells)
            for cell in action.target_cells
                r_step += calculate_sophisticated_reward(b, cell)
            end
        end
        push!(rewards, _normalize_reward(r_step))
        if !isempty(action.target_cells)
            for cell in action.target_cells
                state_i = sample_event_state_from(b, cell, rng)
                b = collapse_belief_to(b, cell, state_i)
            end
        end
        b = evolve_no_obs_fast(b, env, calculate_uncertainty = false)
    end
    return rewards
end

"""
    discounted_returns(rewards, gamma)
G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k} for t in 1:T.
"""
function discounted_returns(rewards::Vector{Float64}, gamma::Float64)
    T = length(rewards)
    G = zeros(T)
    for t in 1:T
        for k in 0:(T - t)
            G[t] += gamma^k * rewards[t + k]
        end
    end
    return G
end

# Bandit: for each action we store n_a, X_a (mean), S_a (sum of squared deviations; variance = S_a/n)
# Posterior Normal-Gamma: μ1 = (λ0*μ0 + n*X)/(λ0+n), λ1 = λ0+n, α1 = α0+n/2,
# β1 = β0 + 0.5*(n*σ² + λ0*n*(X-μ0)²/(λ0+n)), σ² = S_a/n
# Sample τ ~ Gamma(α1, 1/β1), μ ~ Normal(μ1, 1/sqrt(λ1*τ))
function thompson_sample_action(n_a::Vector{Int}, X_a::Vector{Float64}, S_a::Vector{Float64},
    K::Int, rng::AbstractRNG)::Int
    mu_samples = zeros(K)
    for a in 1:K
        n = n_a[a]
        if n == 0
            # Prior: sample τ ~ Gamma(α0, 1/β0), μ ~ N(μ0, 1/sqrt(λ0*τ))
            τ = rand(rng, Gamma(α0, 1 / β0))
            mu_samples[a] = μ0 + randn(rng) / sqrt(λ0 * τ)
        else
            xbar = X_a[a]
            σ2 = max(1e-10, S_a[a] / n)
            μ1 = (λ0 * μ0 + n * xbar) / (λ0 + n)
            λ1 = λ0 + n
            α1 = α0 + n / 2
            β1 = β0 + 0.5 * (n * σ2 + (λ0 * n * (xbar - μ0)^2) / (λ0 + n))
            τ = rand(rng, Gamma(α1, 1 / β1))
            mu_samples[a] = μ1 + randn(rng) / sqrt(λ1 * τ)
        end
    end
    return argmax(mu_samples)
end

function update_bandit!(n_a::Vector{Int}, X_a::Vector{Float64}, S_a::Vector{Float64},
    a::Int, G::Float64)
    n_old = n_a[a]
    x_old = X_a[a]
    n_a[a] = n_old + 1
    n_new = n_a[a]
    X_a[a] = (n_old * x_old + G) / n_new
    # Welford: S = sum of squared deviations from mean; S_new = S_old + (G - x_old)*(G - X_new)
    S_a[a] = (n_old > 0 ? (S_a[a] + (G - x_old) * (G - X_a[a])) : 0.0)
end

"""
    best_script_posts(env, belief, agent, C, gs_state; rng, budget, gamma)

POSTS (Phan et al., AAAI-19): memory-bounded open-loop planning for POMDPs.
- Stack of C Thompson Sampling bandits; each simulation samples one plan, runs one rollout, updates bandits.
- Reward = belief-space (sophisticated) reward per step; returns full open-loop sequence (argmax mean at each t).
"""
function best_script_posts(env, belief::Belief, agent, C::Int, gs_state;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    budget::Int = 10_000,
    gamma::Float64 = 0.95)
    start_time = time()
    if C == 0
        return SensingAction[], time() - start_time
    end

    actions_per_timestep = get_actions_per_timestep(agent, env, C, gs_state)
    if any(isempty(actions_per_timestep[t]) for t in 1:C)
        return SensingAction[], time() - start_time
    end

    K_t = [length(actions_per_timestep[t]) for t in 1:C]
    # Bandit stack: for each t we have n_a, X_a, S_a for each action index 1..K_t[t]
    n_a = [zeros(Int, K_t[t]) for t in 1:C]
    X_a = [zeros(Float64, K_t[t]) for t in 1:C]
    S_a = [zeros(Float64, K_t[t]) for t in 1:C]

    for _ in 1:budget
        # Sample action at each step via Thompson Sampling
        action_indices = Int[]
        for t in 1:C
            a = thompson_sample_action(n_a[t], X_a[t], S_a[t], K_t[t], rng)
            push!(action_indices, a)
        end
        # One rollout
        rewards = rollout_rewards_and_actions(env, belief, agent, action_indices,
            actions_per_timestep, C, gs_state, rng)
        G = discounted_returns(rewards, gamma)
        # Update each bandit t with G[t] for the action chosen at t
        for t in 1:C
            update_bandit!(n_a[t], X_a[t], S_a[t], action_indices[t], G[t])
        end
    end

    # Return open-loop sequence: at each t take action with highest empirical mean.
    # Break ties by: (1) more visits n_a (more confident), (2) prefer sensing over no-op (action 1 is no-op).
    best_sequence = Types.SensingAction[]
    for t in 1:C
        best_a = 1
        best_x = -Inf
        best_n = -1
        best_senses = false
        for a in 1:K_t[t]
            if n_a[t][a] == 0
                continue
            end
            x = X_a[t][a]
            n = n_a[t][a]
            sense_a = !isempty(actions_per_timestep[t][a].target_cells)
            better = (x > best_x) ||
                     (x == best_x && n > best_n) ||
                     (x == best_x && n == best_n && sense_a && !best_senses)
            if better
                best_x = x
                best_n = n
                best_a = a
                best_senses = sense_a
            end
        end
        push!(best_sequence, actions_per_timestep[t][best_a])
    end

    planning_time = time() - start_time
    println("POSTS: best sequence (budget=$(budget)) in $(round(planning_time, digits=3)) s")
    return best_sequence, planning_time
end

end # module

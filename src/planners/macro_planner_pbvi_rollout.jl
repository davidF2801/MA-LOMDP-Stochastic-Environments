#=
  SB-ABBA Rollout (seed-averaged open-loop): same seeding and one-step simulator as PBVI,
  but instead of PBVI sweeps + greedy extraction, we **sample** `N_sequences` candidate
  macro-sequences (uniformly over the per-timestep product space, distinct index tuples)
  and score each by Monte Carlo rollouts from `N_seed` initial (clock, belief) samples at sync.

  For each candidate sequence π_i = (a_0,…,a_{H-1}):
    For each seed n = 1…N_seed: (τ, b) ← sample_system_state_at_τi(…)
      Roll H steps with simulate_one_step (reward + sampled obs + evolve)
      Accumulate ∑_ℓ γ^ℓ r_ℓ
    Average over seeds; pick sequence with highest average.

  Hyperparameters:
  - `N_seed`, `N_particles`, `N_sweeps` from PBVI hyperparams: **only N_seed is used** (rollouts per candidate).
  - `N_sequences` from GroundStation `PBVI_ROLLOUT_N_SEQUENCES`: number of **sampled** candidates.
    Use `N_sequences == -1` for full enumeration (small problems only).
=#

module MacroPlannerPBVIRollout

using Random
using ..Types
import ..Agents.BeliefManagement: Belief, clear_belief_evolution_cache!
import ..MacroPlannerPBVI: sample_system_state_at_τi, simulate_one_step, get_hyperparams
import ..MacroPlannerAsync: get_actions_per_timestep, generate_sequences_from_actions_per_timestep

export best_script

"""
Sample up to `n` distinct macro-sequences by uniform random choice of action index per timestep.
If the Cartesian product has ≤ `n` sequences, returns all sequences exactly once.
"""
function sample_distinct_macro_sequences(actions_per_timestep::Vector{Vector{SensingAction}}, n::Int, rng::AbstractRNG)
    if isempty(actions_per_timestep) || n <= 0
        return Vector{SensingAction}[]
    end
    H = length(actions_per_timestep)
    counts = [length(actions_per_timestep[t]) for t in 1:H]
    if any(iszero, counts)
        return Vector{SensingAction}[]
    end
    # Int `prod` can overflow on large per-step action lists; BigInt keeps the
    # "enumerate all vs sample" branch correct.
    prod_size = foldl(*, counts; init=BigInt(1))
    if prod_size <= BigInt(n)
        return generate_sequences_from_actions_per_timestep(actions_per_timestep)
    end

    sequences = Vector{SensingAction}[]
    seen = Set{Tuple}()
    max_trials = max(100, 10 * n)
    trials = 0
    while length(sequences) < n && trials < max_trials
        trials += 1
        idxs_tuple = Tuple(rand(rng, 1:counts[t]) for t in 1:H)
        if idxs_tuple in seen
            continue
        end
        push!(seen, idxs_tuple)
        push!(sequences, [actions_per_timestep[t][idxs_tuple[t]] for t in 1:H])
    end
    return sequences
end

"""
    best_script(env, belief, agent, C, other_scripts, gs_state; rng, N_sequences)

Same interface as `MacroPlannerPBVI.best_script` for the outer call site.
- Uses `N_seed` from PBVI hyperparameters (number of rollouts averaged per candidate).
- Ignores `N_particles` and `N_sweeps` (no value iteration).
- `N_sequences`: number of **sampled** macro-sequence candidates; `-1` means enumerate all feasible sequences.
"""
function best_script(env, belief::Belief, agent, C::Int, other_scripts, gs_state;
                     rng::AbstractRNG = Random.GLOBAL_RNG,
                     N_sequences::Int = 500)
    start_time = time()
    clear_belief_evolution_cache!()

    B_clean = deepcopy(belief)
    agent_i = agent
    τ_i = gs_state.time_step
    agents_j = [env.agents[j] for j in keys(env.agents) if j != agent.id]
    τ_js_vector = gs_state.agent_last_sync
    H = C
    γ = env.discount

    hyper = get_hyperparams()
    N_seed = hyper.N_seed

    actions_per_timestep = get_actions_per_timestep(agent_i, env, H, gs_state, agent_i.phase_offset)
    if any(isempty, actions_per_timestep)
        return SensingAction[], time() - start_time
    end

    if N_sequences == -1
        candidate_sequences = generate_sequences_from_actions_per_timestep(actions_per_timestep)
        println("🎲 PBVI-Rollout: N_seed=$(N_seed), horizon H=$(H), **full enumeration** ($(length(candidate_sequences)) sequences)")
    else
        n_sample = N_sequences <= 0 ? 500 : N_sequences
        candidate_sequences = sample_distinct_macro_sequences(actions_per_timestep, n_sample, rng)
        println("🎲 PBVI-Rollout: N_seed=$(N_seed), horizon H=$(H), sampled candidates=$(length(candidate_sequences)) (target n=$(n_sample))")
    end

    if isempty(candidate_sequences)
        return SensingAction[], time() - start_time
    end

    # Initial (clock, belief) samples at sync — same start as each PBVI seed trajectory
    seeds = Tuple{Any, Belief}[]
    for _ in 1:N_seed
        τ_clock, b_sys = sample_system_state_at_τi(B_clean, agent_i, τ_i, agents_j, τ_js_vector, env, gs_state)
        push!(seeds, (copy(τ_clock), deepcopy(b_sys)))
    end

    best_sequence = SensingAction[]
    best_value = -Inf
    nseq = length(candidate_sequences)
    progress_step = max(1, Int(cld(nseq, 20)))

    for (i, seq) in enumerate(candidate_sequences)
        acc_seeds = 0.0
        for (τ0, b0) in seeds
            τ_clock = copy(τ0)
            b_sys = deepcopy(b0)
            traj_val = 0.0
            for (ℓ, a) in enumerate(seq)
                r_step, τ_clock, b_sys, _, _ = simulate_one_step(τ_clock, b_sys, a, agent_i, agents_j, env, gs_state)
                traj_val += γ^(ℓ - 1) * r_step
            end
            acc_seeds += traj_val
        end
        avg_val = acc_seeds / N_seed
        if avg_val > best_value
            best_value = avg_val
            best_sequence = seq
        end
        if (i % progress_step == 0) || (i == nseq)
            perc = round(100 * i / nseq; digits = 1)
            println("  PBVI-Rollout progress: $(i)/$(nseq) ($(perc)%), best avg return: $(round(best_value, digits=3))")
        end
    end

    planning_time = time() - start_time
    println("✅ PBVI-Rollout: best avg discounted return $(round(best_value, digits=3)) in $(round(planning_time, digits=3)) s")
    return best_sequence, planning_time
end

end # module MacroPlannerPBVIRollout

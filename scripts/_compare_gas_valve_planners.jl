include(joinpath(@__DIR__, "..", "src", "gas_valve", "GasValve.jl"))
using .GasValve
using .GasValveTypes
using .GasValveDOSBABBA
using .GasValveControl
using .GasValveTrajectories
using .GasValveEnv
using .GasValveBelief
using Random
using Statistics
using Printf

pockets       = [PocketDef((1, 2), (1, 4), (1, 3), 1), PocketDef((5, 2), (5, 4), (5, 3), 2)]
trajectories  = build_default_trajectories()
dynamics      = default_pocket_dynamics()
reward_params = default_reward_params()
gs_schedule   = GasValveTrajectories.default_gs_visit_schedule()

const T_MAX = 120

function simulate(mode::Symbol, seed::Int;
                  sb_cfg::DOSBABBAConfig = DOSBABBAConfig(),
                  alpha_cfg::DOAlphaConfig = DOAlphaConfig())
    rng = MersenneTwister(seed)
    # Start both pockets Dormant so we evaluate the planner's *decisions*,
    # not initial-state luck. Drift-to-risk is the whole point of the
    # problem under the new dynamics (p_arrive > 0).
    state = GasValveWorldState(
        [0, 0],
        zeros(Int, 2),
        positions_at_time(trajectories, 0),
        [false, false],
    )
    ds = GasValveControl.initialize_private_beliefs(2, 2)
    mailbox = Mailbox()
    obs_buffers = [ObsRecord[] for _ in 1:2]
    total_r   = 0.0
    explosions = 0
    n_acts    = 0  # coordinated Seal+Vent pairs
    for t in 0:(T_MAX - 1)
        GasValveEnv.set_positions_from_trajectories!(state,
            positions_at_time(trajectories, t))
        all(state.exploded) && break
        ja = if mode === :do_sb_abba
            GasValveDOSBABBA.step_do_sb_abba_control!(
                t, state, ds, mailbox, obs_buffers, pockets, dynamics, reward_params,
                trajectories, gs_schedule; cfg = sb_cfg, _rng = rng)
        elseif mode === :do_alpha
            GasValveControl.step_do_alpha_control!(
                t, state, ds, mailbox, obs_buffers, pockets, dynamics, reward_params,
                trajectories, gs_schedule; cfg = alpha_cfg, _rng = rng)
        elseif mode === :random
            GasValveControl.step_random_baseline!(state, pockets, rng)
        else
            error("unknown mode $mode")
        end
        v1 = ja[1].valve_action
        v2 = ja[2].valve_action
        if (v1 == :Seal && v2 == :Vent) || (v1 == :Vent && v2 == :Seal)
            n_acts += 1
        end
        exploded_before = copy(state.exploded)
        r, _ = GasValveEnv.transition_world_fixed_positions!(
            state, ja, positions_at_time(trajectories, t + 1),
            pockets, dynamics, reward_params, 5, rng)
        total_r += r
        for p in 1:2
            state.exploded[p] && !exploded_before[p] && (explosions += 1)
        end
        for i in 1:2
            z = GasValveEnv.generate_observation(state.rover_positions[i],
                state.pocket_states, pockets, rng)
            push!(obs_buffers[i], ObsRecord(t, i, state.rover_positions[i], z))
        end
    end
    return (reward = total_r, explosions = explosions, coord_acts = n_acts)
end

println("=== DO-SB-ABBA vs DO-α vs random, new dynamics (p_arrive=$(dynamics.p_arrive)) ===")
seeds = collect(30:89)
rows = []
for seed in seeds
    rs = simulate(:do_sb_abba, seed)
    ra = simulate(:do_alpha, seed)
    rr = simulate(:random, seed)
    push!(rows, (seed, rs, ra, rr))
    @printf "seed %3d  sb=%6.1f (exp=%d, acts=%2d)  α=%6.1f (exp=%d, acts=%2d)  rand=%6.1f (exp=%d)\n" seed rs.reward rs.explosions rs.coord_acts ra.reward ra.explosions ra.coord_acts rr.reward rr.explosions
end

println()
sb_rewards    = [r[2].reward for r in rows]
a_rewards     = [r[3].reward for r in rows]
rand_rewards  = [r[4].reward for r in rows]
sb_exp        = [r[2].explosions for r in rows]
a_exp         = [r[3].explosions for r in rows]
println("DO-SB-ABBA   : mean=$(round(mean(sb_rewards), digits=1))  median=$(round(median(sb_rewards), digits=1))  expl=$(sum(sb_exp))")
println("DO-α         : mean=$(round(mean(a_rewards),  digits=1))  median=$(round(median(a_rewards),  digits=1))  expl=$(sum(a_exp))")
println("random       : mean=$(round(mean(rand_rewards), digits=1))")
println()
wins_sb = sum(sb_rewards .> a_rewards)
println("DO-SB-ABBA beats DO-α on $wins_sb/$(length(seeds)) seeds")

include(joinpath(@__DIR__, "..", "src", "gas_valve", "GasValve.jl"))
using .GasValve
using .GasValveTypes
using .GasValveDOSBABBA
using .GasValveControl
using .GasValveTrajectories
using .GasValveEnv
using .GasValveBelief
using Random, Printf

pockets       = [PocketDef((1, 2), (1, 4), (1, 3), 1), PocketDef((5, 2), (5, 4), (5, 3), 2)]
trajectories  = build_default_trajectories()
dynamics      = default_pocket_dynamics()
reward_params = default_reward_params()
gs_schedule   = GasValveTrajectories.default_gs_visit_schedule()

function run_with(mode::Symbol, seed::Int; verbose_upto::Int = 40)
    rng = MersenneTwister(seed)
    state = GasValveWorldState(
        [0, 0], zeros(Int, 2),
        positions_at_time(trajectories, 0), [false, false])
    ds = GasValveControl.initialize_private_beliefs(2, 2)
    mailbox = Mailbox()
    obs_buffers = [ObsRecord[] for _ in 1:2]
    sb_cfg = DOSBABBAConfig()
    a_cfg = DOAlphaConfig()
    actions = Vector{Tuple{Int,Vector{Int},Vector{Symbol}}}()
    for t in 0:(verbose_upto-1)
        GasValveEnv.set_positions_from_trajectories!(state, positions_at_time(trajectories, t))
        all(state.exploded) && break
        ja = if mode === :sb
            GasValveDOSBABBA.step_do_sb_abba_control!(t, state, ds, mailbox, obs_buffers,
                pockets, dynamics, reward_params, trajectories, gs_schedule;
                cfg = sb_cfg, _rng = rng)
        else
            GasValveControl.step_do_alpha_control!(t, state, ds, mailbox, obs_buffers,
                pockets, dynamics, reward_params, trajectories, gs_schedule;
                cfg = a_cfg, _rng = rng)
        end
        push!(actions, (t, copy(state.pocket_states), [ja[i].valve_action for i in 1:2]))
        GasValveEnv.transition_world_fixed_positions!(state, ja,
            positions_at_time(trajectories, t+1), pockets, dynamics, reward_params, 5, rng)
        for i in 1:2
            z = GasValveEnv.generate_observation(state.rover_positions[i],
                state.pocket_states, pockets, rng)
            push!(obs_buffers[i], ObsRecord(t, i, state.rover_positions[i], z))
        end
    end
    return actions
end

sb_acts = run_with(:sb, 30)
a_acts  = run_with(:a,  30)
println("t  pock     SB-ABBA            DO-α            same?")
for (i, (sb, a)) in enumerate(zip(sb_acts, a_acts))
    same = sb[3] == a[3]
    mark = same ? " " : "X"
    @printf "%3d %s    %-10s %-10s    %s\n" sb[1] string(sb[2]) string(sb[3]) string(a[3]) mark
end
differ = sum(sb[3] != a[3] for (sb, a) in zip(sb_acts, a_acts))
println("\nDiffering actions: $differ / $(length(sb_acts))")

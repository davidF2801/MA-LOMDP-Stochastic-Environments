include(joinpath(@__DIR__, "..", "src", "gas_valve", "GasValve.jl"))
using .GasValve
using .GasValveTypes
using .GasValveDOSBABBA
using .GasValveControl
using .GasValveTrajectories
using .GasValveEnv
using .GasValveBelief
using Random

pockets       = [PocketDef((1, 2), (1, 4), (1, 3), 1), PocketDef((5, 2), (5, 4), (5, 3), 2)]
trajectories  = build_default_trajectories()
dynamics      = default_pocket_dynamics()
reward_params = default_reward_params()
gs_schedule   = GasValveTrajectories.default_gs_visit_schedule()

const T_MAX = 60

function run_both(seed)
    # Two parallel simulations on the SAME world evolution (same pre-simulated
    # state trajectory / obs streams), but each planner computing its own
    # actions. The point is to ask: given the SAME info, would the planners
    # have picked the same action?
    rng_state = MersenneTwister(seed)

    # Shared initial state
    state0 = GasValveWorldState([0, 0], zeros(Int, 2),
                                 positions_at_time(trajectories, 0), [false, false])

    # Duplicate everything
    state_sb = deepcopy(state0); state_a = deepcopy(state0)
    ds_sb = GasValveControl.initialize_private_beliefs(2, 2)
    ds_a  = GasValveControl.initialize_private_beliefs(2, 2)
    mb_sb = Mailbox(); mb_a = Mailbox()
    ob_sb = [ObsRecord[] for _ in 1:2]
    ob_a  = [ObsRecord[] for _ in 1:2]

    diff_count = 0
    for t in 0:(T_MAX - 1)
        GasValveEnv.set_positions_from_trajectories!(state_sb, positions_at_time(trajectories, t))
        GasValveEnv.set_positions_from_trajectories!(state_a , positions_at_time(trajectories, t))
        ja_sb = GasValveDOSBABBA.step_do_sb_abba_control!(
            t, state_sb, ds_sb, mb_sb, ob_sb, pockets, dynamics, reward_params,
            trajectories, gs_schedule; cfg = DOSBABBAConfig(), _rng = MersenneTwister(seed + t))
        ja_a = GasValveControl.step_do_alpha_control!(
            t, state_a, ds_a, mb_a, ob_a, pockets, dynamics, reward_params,
            trajectories, gs_schedule; cfg = DOAlphaConfig(), _rng = MersenneTwister(seed + t))
        for i in 1:2
            if ja_sb[i].valve_action != ja_a[i].valve_action ||
               ja_sb[i].valve_pocket !== ja_a[i].valve_pocket
                diff_count += 1
                println("t=$t rover=$i  SB=($(ja_sb[i].valve_action),$(ja_sb[i].valve_pocket))  α=($(ja_a[i].valve_action),$(ja_a[i].valve_pocket))")
            end
        end
        # Advance both worlds in lockstep with IDENTICAL dynamics RNG so the
        # state trajectory is identical (observations will diverge if actions do).
        rng_step = MersenneTwister(hash((:world_step, seed, t)))
        rng_step2 = MersenneTwister(hash((:world_step, seed, t)))
        GasValveEnv.transition_world_fixed_positions!(state_sb, ja_sb, positions_at_time(trajectories, t + 1), pockets, dynamics, reward_params, 5, rng_step)
        GasValveEnv.transition_world_fixed_positions!(state_a , ja_a , positions_at_time(trajectories, t + 1), pockets, dynamics, reward_params, 5, rng_step2)
        for i in 1:2
            push!(ob_sb[i], ObsRecord(t, i, state_sb.rover_positions[i],
                  GasValveEnv.generate_observation(state_sb.rover_positions[i], state_sb.pocket_states, pockets, MersenneTwister(hash((:obs, seed, t, i))))))
            push!(ob_a[i], ObsRecord(t, i, state_a.rover_positions[i],
                  GasValveEnv.generate_observation(state_a.rover_positions[i], state_a.pocket_states, pockets, MersenneTwister(hash((:obs, seed, t, i))))))
        end
    end
    println("Total diffs for seed $seed: $diff_count")
end

for s in 30:32
    println("\n=== seed $s ===")
    run_both(s)
end

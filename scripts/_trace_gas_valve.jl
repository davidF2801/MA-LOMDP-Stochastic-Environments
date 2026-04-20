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

seed = 30
rng = MersenneTwister(seed)
state = GasValveWorldState([0, 0], zeros(Int, 2), positions_at_time(trajectories, 0), [false, false])
ds = GasValveControl.initialize_private_beliefs(2, 2)
mailbox = Mailbox()
obs_buffers = [ObsRecord[] for _ in 1:2]

for t in 0:25
    GasValveEnv.set_positions_from_trajectories!(state, positions_at_time(trajectories, t))
    # Pre-step: show q values for each rover for each pocket
    # Run DO-α step (trace) but reuse the shared state since we just want to examine beliefs.
    ja = GasValveControl.step_do_alpha_control!(t, state, ds, mailbox, obs_buffers,
        pockets, dynamics, reward_params, trajectories, gs_schedule;
        cfg = DOAlphaConfig(), _rng = rng)
    # Print obs buffer before examining
    if t == 3
        println("t=3 obs_buffers[1]: $([(o.t, o.pos, o.z) for o in obs_buffers[1]])")
        println("t=3 obs_buffers[2]: $([(o.t, o.pos, o.z) for o in obs_buffers[2]])")
        println("t=3 public_at_read[1]: $(ds.public_at_read[1])")
        println("t=3 last_read_time[1]: $(ds.last_read_time[1])")
    end
    # After the step, examine beliefs
    for i in 1:2
        b_self = GasValveControl._current_private_belief(ds, i, t, obs_buffers, pockets, dynamics)
        b_peer = GasValveControl._hypothesized_peer_belief(ds, i, i == 1 ? 2 : 1, t, pockets, dynamics)
        pos = state.rover_positions[i]
        is_rdv = any(pos == p.valve_a || pos == p.valve_b for p in pockets)
        if is_rdv
            for (pidx, p) in enumerate(pockets)
                at_valve = pos == p.valve_a || pos == p.valve_b
                if at_valve
                    q_self = b_self[pidx][2] + b_self[pidx][3]
                    q_peer = b_peer[pidx][2] + b_peer[pidx][3]
                    @printf "t=%2d R%d @ pocket%d (pock_state=%d) : q_self=%.2f q_peer=%.2f  a_i=%s\n" t i pidx state.pocket_states[pidx] q_self q_peer ja[i].valve_action
                end
            end
        end
    end
    exploded_before = copy(state.exploded)
    GasValveEnv.transition_world_fixed_positions!(state, ja, positions_at_time(trajectories, t + 1),
        pockets, dynamics, reward_params, 5, rng)
    for i in 1:2
        z = GasValveEnv.generate_observation(state.rover_positions[i], state.pocket_states, pockets, rng)
        push!(obs_buffers[i], ObsRecord(t, i, state.rover_positions[i], z))
    end
end

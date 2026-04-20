#!/usr/bin/env julia

"""
Main script for the decentralized Gas-Valve Coordination problem with
**asynchronous** ground-station (GS) communication.

Problem summary
===============
* 5x5 grid, two gas pockets, two rovers.
* Fixed periodic trajectories (period 12). **R1 patrols the upper half
  (y ∈ {1,2,3})**, **R2 patrols the lower half (y ∈ {3,4,5})**; they only
  come near each other at pocket rendezvous times. At t ≡ 3 (mod 12) they
  meet at the two valves of pocket 1 (upper-A / lower-B); at t ≡ 9 (mod 12)
  they meet at the two valves of pocket 2. Only the *valve action*
  (Idle / Observe / Seal / Vent) is planned; motion is fixed.
* Single ground-station cell (3,3). R1 reads/writes at t ≡ 0 (mod 12);
  R2 reads/writes at t ≡ 6 (mod 12). Peer information is therefore stale
  by up to one period — true asynchronous Dec-POMDP comms.

Modes
-----
:public_belief     — centralized joint planner on the **shared public belief**
                     B_clean: every timestep all buffered observations are
                     fused into one belief, then a horizon-H beam search picks
                     a consistent joint valve plan. Upper bound on decentralized
                     comms (full shared information each step).
:do_sb_abba        — **DO-SB-ABBA-α** (proper sample-based, multi-step,
                     peer-searching decentralized planner in
                     `GasValveDOSBABBA`). Each rover samples N_particles peer
                     observation streams + pocket trajectories, runs H-step
                     Monte-Carlo rollouts that play peer's best-response to
                     the candidate a_i, and blends V_opt (on b_MP) with
                     V_cons (on b_{j|i}) by α.
:do_alpha          — **DO-α** (simple 1-step α-blend): fast rule-based
                     planner in `GasValveControl`. Closed-form expected-
                     reward argmax with peer action fixed by an MRAC role
                     rule. Useful as a lightweight baseline vs `:do_sb_abba`.
:pomcp_independent — independent POMCP per rover on the current public
                     belief snapshot (no explicit coordination).
:random            — uniformly random valve action per rover.
"""

println("Gas-Valve (async GS, decentralized) starting...")

mkpath("E:/tmp")
ENV["TMPDIR"] = "E:/tmp"
ENV["TEMP"]   = "E:/tmp"
ENV["TMP"]    = "E:/tmp"

using Random
using Statistics
using Plots
using Dates
using DataFrames
using CSV
Plots.gr()

# =============================================================================
# CONFIGURATION
# =============================================================================
const GRID_HEIGHT        = 5
const GRID_WIDTH         = 5
const NUM_ROVERS         = 2
const NUM_POCKETS        = 2
const NUM_STEPS          = 120        # 10 periods of 12
const CRITICAL_DEADLINE  = 5
const H_MAX              = 12         # one period lookahead for :public_belief beam search
const W_BEAM             = 4000
const DISCOUNT           = 0.95
const N_RUNS             = 5

# α knob (shared by :do_sb_abba and :do_alpha):
#   0.0 → only the peer-view value counts (maximally consistency-seeking)
#   1.0 → only the self / joint-info value counts (optimality-seeking)
const DO_ALPHA_WEIGHT      = 0.6

# DO-SB-ABBA-α sampling knobs
const DO_SB_ABBA_H         = 12
const DO_SB_ABBA_PARTICLES = 16

const POMCP_N_ROLLOUTS   = 200
const POMCP_H_ROLLOUT    = 8

const MODES = [:public_belief, :do_sb_abba, :do_alpha, :pomcp_independent, :random]

# =============================================================================
# LOAD GAS VALVE MODULE (standalone)
# =============================================================================
include("../src/gas_valve/GasValve.jl")
using .GasValve
using .GasValveTypes
using .GasValveEnv
using .GasValveTrajectories
using .GasValveBelief
using .GasValvePlanner
using .GasValveControl
using .GasValveDOSBABBA
using .GasValvePOMCP

const GS_CELL          = GasValveTrajectories.default_gs_cell()
const GS_SCHEDULE      = GasValveTrajectories.default_gs_visit_schedule()

# =============================================================================
# PROBLEM INSTANCE
# =============================================================================
function build_pockets()
    # P1 on the left edge, valves on opposite rows (upper & lower);
    # P2 on the right edge, same idea. Pocket center sits between the
    # two valves so the obs model treats both rovers as "far = ~1 cell".
    [
        PocketDef((1, 2), (1, 4), (1, 3), 1),
        PocketDef((5, 2), (5, 4), (5, 3), 2),
    ]
end

build_trajectories() = build_default_trajectories()

function build_initial_state(pockets::Vector{PocketDef},
                              trajectories::Vector{RoverTrajectory};
                              rng = Random.GLOBAL_RNG)
    P = length(pockets)
    pocket_states = [rand(rng) < 0.6 ? 0 : (rand(rng) < 0.5 ? 1 : 2) for _ in 1:P]
    deadline_counters = [s == 2 ? 1 : 0 for s in pocket_states]
    rover_positions = positions_at_time(trajectories, 0)
    return GasValveWorldState(pocket_states, deadline_counters,
                               rover_positions, fill(false, P))
end

# =============================================================================
# SIMULATION
# =============================================================================
function _run_gas_valve(mode::Symbol, num_steps::Int;
                        rng::AbstractRNG, record_states::Bool)
    pockets       = build_pockets()
    trajectories  = build_trajectories()
    dynamics      = default_pocket_dynamics()
    reward_params = default_reward_params()
    state         = build_initial_state(pockets, trajectories; rng = rng)

    # Centralized public belief (used by :public_belief joint planner and by
    # :pomcp_independent as a snapshot).
    B_pub       = initialize_uniform_belief(length(pockets))
    t_clean     = 0
    obs_buffers = [ObsRecord[] for _ in 1:NUM_ROVERS]

    # Decentralized state shared by :do_sb_abba (proper SB-ABBA-α) and
    # :do_alpha (simple 1-step α-blend).
    ds       = GasValveControl.initialize_private_beliefs(NUM_ROVERS, length(pockets))
    mailbox  = Mailbox()
    do_sb_abba_cfg = GasValveDOSBABBA.DOSBABBAConfig(
        α = DO_ALPHA_WEIGHT, γ = DISCOUNT,
        H = DO_SB_ABBA_H, N_particles = DO_SB_ABBA_PARTICLES,
        deadline = CRITICAL_DEADLINE,
    )
    do_alpha_cfg = GasValveControl.DOAlphaConfig(α = DO_ALPHA_WEIGHT, γ = DISCOUNT)

    reward_history    = Float64[]
    action_history    = Vector{JointAction}()
    state_evolution   = GasValveWorldState[]
    plan_time_history = Float64[]  # seconds per planning call

    # Per-step stats
    num_explosions_hist   = Int[]   # cumulative explosions after each step
    num_success_vent_hist = Int[]   # cumulative successful Seal+Vent
    num_seal_hist         = Int[]   # cumulative Seal actions issued
    num_vent_hist         = Int[]   # cumulative Vent actions issued
    num_observe_hist      = Int[]   # cumulative Observe actions issued
    rover_distance_hist   = Int[]   # manhattan distance between rovers (first two)
    pocket_state_hist     = Vector{Vector{Int}}()  # [[s1, s2, ...]] per step

    cum_explosions   = 0
    cum_success_vent = 0
    cum_seal         = 0
    cum_vent         = 0
    cum_observe      = 0

    for t in 0:(num_steps - 1)
        set_positions_from_trajectories!(state, positions_at_time(trajectories, t))

        # Terminal condition: if *every* pocket has exploded we stop early.
        if all(state.exploded)
            break
        end

        joint_action = JointAction()
        plan_time_s = 0.0
        if mode == :public_belief
            plan_time_s = @elapsed begin
                joint_action, B_pub, t_clean, _ = step_public_belief_control!(
                    t, state, B_pub, t_clean, obs_buffers, pockets, dynamics,
                    reward_params, trajectories, H_MAX;
                    γ = DISCOUNT, W_beam = W_BEAM, _rng = rng
                )
            end
        elseif mode == :do_sb_abba
            plan_time_s = @elapsed begin
                joint_action = GasValveDOSBABBA.step_do_sb_abba_control!(
                    t, state, ds, mailbox, obs_buffers, pockets, dynamics,
                    reward_params, trajectories, GS_SCHEDULE;
                    cfg = do_sb_abba_cfg, _rng = rng
                )
            end
        elseif mode == :do_alpha
            plan_time_s = @elapsed begin
                joint_action = GasValveControl.step_do_alpha_control!(
                    t, state, ds, mailbox, obs_buffers, pockets, dynamics,
                    reward_params, trajectories, GS_SCHEDULE;
                    cfg = do_alpha_cfg, _rng = rng
                )
            end
        elseif mode == :pomcp_independent
            plan_time_s = @elapsed begin
                # Snapshot drain of the centralized public belief.
                if !isempty(obs_buffers[1]) || !isempty(obs_buffers[2])
                    O_bc = ObsRecord[]
                    for buf in obs_buffers
                        append!(O_bc, buf)
                    end
                    B_pub, t_clean = GasValveBelief.PublicFilterUpdate(
                        B_pub, t_clean, t, O_bc, pockets, dynamics)
                    for buf in obs_buffers
                        empty!(buf)
                    end
                end
                B_tilde = GasValveBelief.PredictOnly(B_pub, t_clean, t, dynamics)
                joint_action = GasValvePOMCP.step_independent_pomcp!(
                    t, state, B_tilde, trajectories, pockets, dynamics,
                    reward_params, CRITICAL_DEADLINE;
                    N_rollouts = POMCP_N_ROLLOUTS, H_rollout = POMCP_H_ROLLOUT,
                    γ = DISCOUNT, rng = rng
                )
            end
        elseif mode == :random
            plan_time_s = @elapsed begin
                joint_action = step_random_baseline!(state, pockets, rng)
            end
        else
            error("Unknown planning mode: $(mode)")
        end
        push!(plan_time_history, plan_time_s)

        # Count action types and successful vents BEFORE applying the action.
        pos_now = copy(state.rover_positions)
        for a in joint_action
            if a.valve_action == GasValveTypes.VALVE_SEAL
                cum_seal += 1
            elseif a.valve_action == GasValveTypes.VALVE_VENT
                cum_vent += 1
            elseif a.valve_action == GasValveTypes.VALVE_OBSERVE
                cum_observe += 1
            end
        end
        for pocket in pockets
            if GasValveEnv.is_pocket_successfully_vented(pocket, joint_action, pos_now)
                cum_success_vent += 1
            end
        end

        exploded_before = copy(state.exploded)
        positions_next  = positions_at_time(trajectories, t + 1)
        r, _ = transition_world_fixed_positions!(
            state, joint_action, positions_next, pockets, dynamics,
            reward_params, CRITICAL_DEADLINE, rng
        )

        # Count new explosions from this step.
        for p in 1:length(pockets)
            if state.exploded[p] && !exploded_before[p]
                cum_explosions += 1
            end
        end

        push!(reward_history, r)
        push!(action_history, joint_action)
        push!(num_explosions_hist,   cum_explosions)
        push!(num_success_vent_hist, cum_success_vent)
        push!(num_seal_hist,         cum_seal)
        push!(num_vent_hist,         cum_vent)
        push!(num_observe_hist,      cum_observe)
        push!(pocket_state_hist,     copy(state.pocket_states))

        if length(state.rover_positions) >= 2
            p1 = state.rover_positions[1]
            p2 = state.rover_positions[2]
            push!(rover_distance_hist, abs(p1[1] - p2[1]) + abs(p1[2] - p2[2]))
        else
            push!(rover_distance_hist, 0)
        end

        if record_states
            push!(state_evolution, GasValveWorldState(
                copy(state.pocket_states), copy(state.deadline_counters),
                copy(state.rover_positions), copy(state.exploded)))
        end

        # Each rover observes nearest pocket with noise; queued to obs_buffers.
        for i in 1:NUM_ROVERS
            z = GasValveEnv.generate_observation(
                state.rover_positions[i], state.pocket_states, pockets, rng)
            push!(obs_buffers[i], ObsRecord(t, i, state.rover_positions[i], z))
        end
    end

    return (
        total_reward          = sum(reward_history),
        reward_history        = reward_history,
        action_history        = action_history,
        state_evolution       = state_evolution,
        final_state           = state,
        pockets               = pockets,
        trajectories          = trajectories,
        plan_time_history     = plan_time_history,
        num_explosions_hist   = num_explosions_hist,
        num_success_vent_hist = num_success_vent_hist,
        num_seal_hist         = num_seal_hist,
        num_vent_hist         = num_vent_hist,
        num_observe_hist      = num_observe_hist,
        rover_distance_hist   = rover_distance_hist,
        pocket_state_hist     = pocket_state_hist,
        total_explosions      = cum_explosions,
        total_success_vent    = cum_success_vent,
        total_seal            = cum_seal,
        total_vent            = cum_vent,
        total_observe         = cum_observe,
    )
end

run_simulation(mode::Symbol, num_steps::Int; rng::AbstractRNG = Random.GLOBAL_RNG) =
    _run_gas_valve(mode, num_steps; rng = rng, record_states = false)

run_simulation_with_history(mode::Symbol, num_steps::Int;
                             rng::AbstractRNG = Random.GLOBAL_RNG) =
    _run_gas_valve(mode, num_steps; rng = rng, record_states = true)

# =============================================================================
# VISUALIZATION
#   Coordinate system: grid cell (x, y) is drawn at heatmap column=x, row=y.
#   yflip = true → y=1 is at the top of the plot.
#   IMPORTANT: Plots.jl heatmap(xs, ys, Z) expects Z[row=y, col=x]; we build
#   M with exactly that indexing (M[y, x]), so we pass it straight through.
#   (A previous version transposed with permutedims — this produced a mirror
#    image of the world, which is why the trajectories looked nonsensical.)
# =============================================================================
#   0 = empty, 1 = GS cell, 2..4 = pocket state (D/P/C), 5 = exploded,
#   6 = rover 1, 7 = rover 2, 8 = both rovers stacked.
const GV_COLORS = [:white, :deepskyblue, :lightgreen, :orange, :red,
                   :black, :dodgerblue, :purple, :magenta]

const GV_LEGEND_LABELS = (
    "Empty",
    "Ground station (GS)",
    "Pocket: Dormant",
    "Pocket: Pressurized",
    "Pocket: Critical",
    "Pocket: Exploded",
    "Rover 1 (on cell)",
    "Rover 2 (on cell)",
    "Both rovers (same cell)",
)

"""Right-hand legend panel for `visualize_gas_valve_frame` (matches `GV_COLORS` order)."""
function _gas_valve_legend_panel()
    p = plot(;
        framestyle = :none,
        grid = false,
        showaxis = false,
        xticks = :none,
        yticks = :none,
        legend = :topleft,
        legendfontsize = 6,
        legend_title = "Cell colors",
        legend_titlefontsize = 7,
        foreground_color_legend = :white,
        background_color_subplot = :transparent,
        margin = 2Plots.mm,
    )
    for (col, lab) in zip(GV_COLORS, GV_LEGEND_LABELS)
        # Points off-plot so only the legend swatches appear.
        scatter!(p, [-1.0], [-1.0];
            label = lab,
            mc = col,
            ms = 7,
            msw = 0.5,
            msc = :gray45,
            markershape = :square,
        )
    end
    plot!(p; xlims = (0, 1), ylims = (0, 1))
    return p
end

function visualize_gas_valve_frame(
    t::Int,
    state::GasValveWorldState,
    pockets::Vector{PocketDef},
    joint_action::JointAction,
    reward_so_far::Float64;
    title_extra::String = ""
)
    height, width = GRID_HEIGHT, GRID_WIDTH
    M = zeros(Int, height, width)
    gx, gy = GS_CELL
    if 1 <= gx <= width && 1 <= gy <= height
        M[gy, gx] = 1
    end
    for (idx, pocket) in enumerate(pockets)
        if idx <= length(state.pocket_states)
            s = state.pocket_states[idx]
            val = state.exploded[idx] ? 5 : (2 + s)
            for (vx, vy) in [pocket.valve_a, pocket.valve_b, pocket.center]
                if 1 <= vx <= width && 1 <= vy <= height
                    M[vy, vx] = val
                end
            end
        end
    end
    # Paint rovers on top. If both rovers are on the same cell, mark it
    # as "8 = both stacked" so the user can see the overlap explicitly.
    for (i, (ax, ay)) in enumerate(state.rover_positions)
        if 1 <= ax <= width && 1 <= ay <= height
            M[ay, ax] = (M[ay, ax] == 6 || M[ay, ax] == 7) ? 8 : (5 + i)
        end
    end

    ph = heatmap(1:width, 1:height, M;
        aspect_ratio = 1,
        xlabel = "X →",
        ylabel = "Y (1 = top)",
        yflip = true,
        clims = (0, 8),
        color = cgrad(GV_COLORS, 9, categorical = true),
        colorbar = false,
        label = "",
    )

    # Grid lines (half-integer separators) for readability.
    for x in 0.5:1.0:(width + 0.5)
        plot!(ph, [x, x], [0.5, height + 0.5]; color = :gray80, lw = 0.5, label = "")
    end
    for y in 0.5:1.0:(height + 0.5)
        plot!(ph, [0.5, width + 0.5], [y, y]; color = :gray80, lw = 0.5, label = "")
    end

    # GS label.
    annotate!(ph, Float64(gx), Float64(gy) - 0.32, text("GS", 8, :white, :bold))

    # Pocket labels: id + valve A/B tag + current pocket state letter.
    for (idx, pocket) in enumerate(pockets)
        letter = state.exploded[idx] ? "X" :
                 state.pocket_states[idx] == 0 ? "D" :
                 state.pocket_states[idx] == 1 ? "P" : "C"
        (vax, vay) = pocket.valve_a
        (vbx, vby) = pocket.valve_b
        annotate!(ph, Float64(vax), Float64(vay) + 0.35,
                   text("P$(idx).A[$(letter)]", 7, :black))
        annotate!(ph, Float64(vbx), Float64(vby) + 0.35,
                   text("P$(idx).B[$(letter)]", 7, :black))
    end

    # Rover markers + valve action labels.
    for (i, (ax, ay)) in enumerate(state.rover_positions)
        if 1 <= ax <= width && 1 <= ay <= height
            annotate!(ph, Float64(ax), Float64(ay), text("R$i", 14, :white, :bold))
            if i <= length(joint_action)
                a = joint_action[i]
                if a.valve_action != :none && a.valve_pocket !== nothing
                    lbl = a.valve_action == :Seal ? "Seal" :
                          a.valve_action == :Vent ? "Vent" :
                          a.valve_action == :Observe ? "Obs" : string(a.valve_action)
                    annotate!(ph, Float64(ax), Float64(ay) - 0.32,
                               text("$(lbl)(P$(a.valve_pocket))", 7, :yellow, :bold))
                end
            end
        end
    end

    p_leg = _gas_valve_legend_panel()
    return plot(
        ph, p_leg;
        layout = grid(1, 2; widths = (0.74, 0.26)),
        size = (880, 600),
        plot_title = "Gas-Valve t=$t | R=$(round(reward_so_far, digits=1))$title_extra",
        plot_titlefontsize = 11,
        link = :none,
    )
end

# =============================================================================
# ANIMATION + METRICS IO
# =============================================================================
function create_gas_valve_animation(
    state_evolution::Vector{GasValveWorldState},
    action_history::Vector{JointAction},
    reward_history::Vector{Float64},
    pockets::Vector{PocketDef},
    results_dir::String,
    run_number::Int,
    mode::Symbol
)
    anim_dir = joinpath(results_dir, "Run $(run_number)", string(mode), "animations")
    mkpath(anim_dir)
    cum_reward = 0.0
    frames = []
    for step in 1:length(action_history)
        t = step - 1
        cum_reward += (step <= length(reward_history) ? reward_history[step] : 0.0)
        state = step <= length(state_evolution) ? state_evolution[step] : state_evolution[end]
        ja = step <= length(action_history) ? action_history[step] : action_history[end]
        push!(frames, visualize_gas_valve_frame(
            t, state, pockets, ja, cum_reward; title_extra = " | $mode"))
    end
    anim = @animate for f in frames
        plot(f; size = (920, 640))
    end
    gif_path = joinpath(anim_dir, "gas_valve_$(mode)_run$(run_number).gif")
    gif(anim, gif_path, fps = 2.0)
    println("Saved animation: $(basename(gif_path))")
    return gif_path
end

function save_gas_valve_metrics(res, results_dir::String, run_number::Int,
                                 mode::Symbol; alpha::Float64 = NaN)
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(mode), "metrics")
    mkpath(metrics_dir)
    filepath = joinpath(metrics_dir, "performance_metrics_$(mode)_run$(run_number).txt")

    reward_history = res.reward_history
    plan_times_ms  = res.plan_time_history .* 1000.0
    steps_done     = length(reward_history)
    cum_reward     = cumsum(reward_history)

    mean_reward = steps_done > 0 ? Statistics.mean(reward_history) : 0.0
    total_reward = res.total_reward
    mean_plan_ms = steps_done > 0 ? Statistics.mean(plan_times_ms) : 0.0
    std_plan_ms  = steps_done > 1 ? Statistics.std(plan_times_ms)  : 0.0
    max_plan_ms  = steps_done > 0 ? maximum(plan_times_ms)         : 0.0
    total_plan_s = steps_done > 0 ? sum(res.plan_time_history)     : 0.0
    mean_rover_dist = steps_done > 0 ? Statistics.mean(res.rover_distance_hist) : 0.0

    # Terminal state stats
    n_final_exploded = sum(res.final_state.exploded)

    open(filepath, "w") do f
        println(f, "="^60)
        println(f, "GAS-VALVE COORDINATION - PERFORMANCE METRICS")
        println(f, "="^60)
        println(f, "Generated: $(now())")
        println(f, "Mode: $(mode)")
        println(f, "Run: $(run_number)")
        if !isnan(alpha)
            println(f, "DO-SB-ABBA alpha: $(alpha)")
        end
        println(f, "Steps simulated: $(steps_done)")
        println(f)
        println(f, "PERFORMANCE METRICS:")
        println(f, "  Total reward: $(round(total_reward, digits=2))")
        println(f, "  Mean per-step reward: $(round(mean_reward, digits=4))")
        println(f, "  Total successful Seal+Vent operations: $(res.total_success_vent)")
        println(f, "  Total explosions triggered: $(res.total_explosions)")
        println(f, "  Final pockets exploded: $(n_final_exploded) / $(length(res.pockets))")
        println(f, "  Seal actions issued: $(res.total_seal)")
        println(f, "  Vent actions issued: $(res.total_vent)")
        println(f, "  Observe actions issued: $(res.total_observe)")
        println(f, "  Mean rover-rover distance: $(round(mean_rover_dist, digits=3))")
        println(f)
        println(f, "PLANNING TIME STATISTICS:")
        println(f, "  Mean planning time per step: $(round(mean_plan_ms, digits=3)) ms")
        println(f, "  Std  planning time per step: $(round(std_plan_ms,  digits=3)) ms")
        println(f, "  Max  planning time per step: $(round(max_plan_ms,  digits=3)) ms")
        println(f, "  Total planning time: $(round(total_plan_s, digits=3)) seconds")
        println(f, "="^60)
    end
    println("Metrics saved: $(filepath)")
    return filepath
end

function save_reward_log_csv(res, results_dir::String, run_number::Int,
                              mode::Symbol; gamma::Float64 = 1.0)
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(mode), "metrics")
    mkpath(metrics_dir)
    filepath = joinpath(metrics_dir, "reward_log_$(mode)_run$(run_number).csv")

    R     = res.reward_history
    T     = length(R)
    cum   = cumsum(R)
    disc  = Float64[]
    acc   = 0.0
    for (k, r) in enumerate(R)
        acc += gamma^(k - 1) * r
        push!(disc, acc)
    end
    plan_ms = res.plan_time_history .* 1000.0

    rows = Vector{NamedTuple}(undef, T)
    for t in 1:T
        ps = res.pocket_state_hist[t]
        n_exploded = 0
        for (p, s) in enumerate(ps)
            # snapshot from state.exploded is not directly stored, but if a
            # pocket is marked exploded in final_state AND was already exploded
            # at step t, we cannot tell from pocket_state_hist alone; so we
            # approximate n_exploded from cumulative history instead.
        end
        n_crit     = count(x -> x == GasValveTypes.CRITICAL,    ps)
        n_press    = count(x -> x == GasValveTypes.PRESSURIZED, ps)
        n_dorm     = count(x -> x == GasValveTypes.DORMANT,     ps)
        rows[t] = (
            timestep                  = t - 1,
            step_reward               = R[t],
            cumulative_reward         = cum[t],
            cumulative_disc_reward    = disc[t],
            planning_time_ms          = plan_ms[t],
            cumulative_success_vent   = res.num_success_vent_hist[t],
            cumulative_explosions     = res.num_explosions_hist[t],
            cumulative_seal_actions   = res.num_seal_hist[t],
            cumulative_vent_actions   = res.num_vent_hist[t],
            cumulative_observe_actions= res.num_observe_hist[t],
            rover_distance            = res.rover_distance_hist[t],
            n_critical_pockets        = n_crit,
            n_pressurized_pockets     = n_press,
            n_dormant_pockets         = n_dorm,
        )
    end
    CSV.write(filepath, DataFrame(rows))
    println("Reward log saved: $(filepath)")
    return filepath
end

function save_action_history_csv(action_history::Vector{JointAction},
                                  results_dir::String, run_number::Int,
                                  mode::Symbol)
    metrics_dir = joinpath(results_dir, "Run $(run_number)", string(mode), "metrics")
    mkpath(metrics_dir)
    filepath = joinpath(metrics_dir, "action_history_$(mode)_run$(run_number).csv")
    rows = []
    for (t, ja) in enumerate(action_history)
        for (i, a) in enumerate(ja)
            vent_str = (a.valve_action == :none || a.valve_pocket === nothing) ?
                "" : "$(a.valve_action)($(a.valve_pocket))"
            push!(rows, (timestep = t - 1, rover_id = i,
                         move = string(a.move), vent = vent_str))
        end
    end
    CSV.write(filepath, DataFrame(rows))
    println("Actions saved: $(filepath)")
    return filepath
end

# =============================================================================
# MAIN
# =============================================================================
timestamp = replace(string(now()), ":" => "-", "." => "-")
results_base_dir = joinpath(@__DIR__, "..", "results", "gas_valve_run_$(timestamp)")
mkpath(results_base_dir)
println("Results: $(results_base_dir)")
println("GS cell: $GS_CELL | GS schedule: $GS_SCHEDULE | α=$DO_ALPHA_WEIGHT")

for run in 1:N_RUNS
    rng = MersenneTwister(42 + run)
    for mode in MODES
        println("\n" * "="^60)
        println("Run $run / $N_RUNS | Mode: $mode")
        println("="^60)
        res = run_simulation_with_history(mode, NUM_STEPS; rng = copy(rng))
        println("Total reward: $(round(res.total_reward, digits=2))")

        alpha = (mode == :do_sb_abba || mode == :do_alpha) ? DO_ALPHA_WEIGHT : NaN
        save_gas_valve_metrics(res, results_base_dir, run, mode; alpha = alpha)
        save_action_history_csv(res.action_history, results_base_dir, run, mode)
        save_reward_log_csv(res, results_base_dir, run, mode; gamma = DISCOUNT)
        create_gas_valve_animation(res.state_evolution, res.action_history,
                                    res.reward_history, res.pockets,
                                    results_base_dir, run, mode)
    end
end

println("\nGas-Valve simulations completed!")
println("Results in: $(results_base_dir)")

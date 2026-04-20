"""
GasValveTrajectories.jl — Fixed periodic trajectories for the gas-valve rovers,
with **asynchronous** ground-station (GS) visits.

Setup
=====
* Single ground-station cell `GS = (3,3)` (5x5 grid).
* Two pockets, each with valves on *opposite rows* of the grid, so that a
  successful Seal+Vent genuinely requires one rover coming from the upper
  half and one from the lower half:
    P1: valve_A=(1,2) upper,  valve_B=(1,4) lower,  center=(1,3)
    P2: valve_A=(5,2) upper,  valve_B=(5,4) lower,  center=(5,3)
* Period = 12 for both rovers.

Timing (within a period, `t mod 12`):
    t=0   → R1 at GS (3,3)  — mailbox op
    t=3   → rendezvous at P1  (R1 at P1.A=(1,2), R2 at P1.B=(1,4))
    t=6   → R2 at GS (3,3)  — mailbox op
    t=9   → rendezvous at P2  (R1 at P2.A=(5,2), R2 at P2.B=(5,4))

Between rendezvous, **R1 stays in the upper half** (y ∈ {1,2,3}) and
**R2 stays in the lower half** (y ∈ {3,4,5}). They overlap on row y=3 only
to enter/leave their own GS visit, and they **never occupy the same cell
at the same time**. Each rover therefore patrols a genuinely independent
region — the only reason to come back near the peer is the scheduled valve
rendezvous.

GS asynchrony: R1 reads/writes the GS mailbox only at `t ≡ 0 (mod 12)`;
R2 only at `t ≡ 6 (mod 12)`. Any pass-through of (3,3) at other times does
not trigger a mailbox op, so peer information is stale by up to one full
period — the classic asynchronous Dec-POMDP comm model.
"""
module GasValveTrajectories

using ..GasValveTypes
import ..GasValveTypes: Cell, PocketDef

export RoverTrajectory, position_at_time, positions_at_time,
    build_default_trajectories, build_trajectories_v12,
    default_gs_cell, default_gs_visit_schedule, is_gs_visit_time

struct RoverTrajectory
    waypoints::Vector{Cell}
    period::Int
end

"""Position of rover with trajectory `traj` at absolute timestep `t` (>=0)."""
function position_at_time(traj::RoverTrajectory, t::Int)::Cell
    return traj.waypoints[mod(t, traj.period) + 1]
end

"""Positions of all rovers at timestep `t`."""
function positions_at_time(trajs::Vector{RoverTrajectory}, t::Int)::Vector{Cell}
    return [position_at_time(traj, t) for traj in trajs]
end

"""Default ground-station cell (the only place where mailbox ops may happen)."""
default_gs_cell() = (3, 3)

"""
Per-rover scheduled GS-visit offsets (within the period). A rover with
schedule `s` performs a mailbox op at every absolute time `t` such that
`t mod period == s`. Default: R1 at offset 0, R2 at offset 6 (half-period
offset, producing maximal async staleness).
"""
default_gs_visit_schedule() = Dict(1 => 0, 2 => 6)

"""True iff rover `i` is scheduled to do a mailbox op at absolute time `t`."""
function is_gs_visit_time(rover_i::Int, t::Int,
                           schedule::Dict{Int,Int}, period::Int)::Bool
    haskey(schedule, rover_i) || return false
    return mod(t, period) == schedule[rover_i]
end

"""
V12 trajectories: period 12, **asynchronous** GS visits, R1 patrols the
upper half (y ∈ {1,2,3}) and R2 patrols the lower half (y ∈ {3,4,5}).
They only share row y=3, and **never occupy the same cell** at the same
timestep. The only reason to be close to the peer is the valve rendezvous.

    R1 (upper patrol, period 12):
        t=0  (3,3) GS*         t=1  (3,2)            t=2  (2,2)
        t=3  (1,2) P1.A [rdv]  t=4  (2,2)            t=5  (3,2)
        t=6  (4,2)             t=7  (4,1)            t=8  (5,1)
        t=9  (5,2) P2.A [rdv]  t=10 (4,2)            t=11 (3,2)

    R2 (lower patrol, period 12):
        t=0  (3,4)             t=1  (2,4)            t=2  (1,4)
        t=3  (1,4) P1.B [rdv]  t=4  (2,4)            t=5  (3,4)
        t=6  (3,3) GS*         t=7  (3,4)            t=8  (4,4)
        t=9  (5,4) P2.B [rdv]  t=10 (4,4)            t=11 (3,4)

    * Only `t ≡ 0 (mod 12)` is a mailbox op for R1;
      only `t ≡ 6 (mod 12)` is a mailbox op for R2.

Rendezvous:
    * t ≡ 3 (mod 12) → R1 at P1.A=(1,2), R2 at P1.B=(1,4) — Seal+Vent P1.
    * t ≡ 9 (mod 12) → R1 at P2.A=(5,2), R2 at P2.B=(5,4) — Seal+Vent P2.

All consecutive-cell transitions are Manhattan ≤ 1. At every timestep
R1 and R2 occupy distinct cells; R1's y ≤ 3 and R2's y ≥ 3 always.
"""
function build_trajectories_v12()::Vector{RoverTrajectory}
    r1 = [(3, 3), (3, 2), (2, 2), (1, 2), (2, 2), (3, 2),
          (4, 2), (4, 1), (5, 1), (5, 2), (4, 2), (3, 2)]
    r2 = [(3, 4), (2, 4), (1, 4), (1, 4), (2, 4), (3, 4),
          (3, 3), (3, 4), (4, 4), (5, 4), (4, 4), (3, 4)]
    return [RoverTrajectory(r1, 12), RoverTrajectory(r2, 12)]
end

"""Default trajectories used by `main_gas_valve.jl` — period 12, async,
upper/lower patrol split."""
build_default_trajectories() = build_trajectories_v12()

"""
Sanity checks on a trajectory set:
  (i)  every step transition is Manhattan-1 (or 0 = stay);
  (ii) no two rovers ever occupy the same cell at the same time.
Throws an `AssertionError` if violated — the intent is that anyone
modifying `build_trajectories_v12` will immediately know on `using` that
the new design is broken.
"""
function validate_trajectories(trajs::Vector{RoverTrajectory})
    for (i, traj) in enumerate(trajs)
        for k in 0:(traj.period - 1)
            (x1, y1) = position_at_time(traj, k)
            (x2, y2) = position_at_time(traj, k + 1)
            d = abs(x1 - x2) + abs(y1 - y2)
            d <= 1 || error(
                "Rover $(i): illegal jump of $(d) at t=$(k): ($x1,$y1) → ($x2,$y2)"
            )
        end
    end
    if length(trajs) >= 2
        P = maximum(t -> t.period, trajs)
        for t in 0:(P - 1)
            positions = positions_at_time(trajs, t)
            if length(unique(positions)) != length(positions)
                error(
                    "Rover overlap at t=$t: positions = $(positions). " *
                    "Trajectories must keep rovers on distinct cells."
                )
            end
        end
    end
    return true
end

# Validate on module load so a bad edit can't silently ship.
validate_trajectories(build_trajectories_v12())

end # module

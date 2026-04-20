# Gas Valve Problem — Spec vs Implementation

This document compares the **Volatile Gas Neutralization** specification (action space, joint-dependent effects, dynamics, rewards) with the current code in `src/gas_valve/`.

---

## 1. Action Space

| Spec | Current implementation | Gap |
|------|------------------------|-----|
| **A_i(τ) = {Idle} ∪ {Observe(k)} ∪ {Seal(k)} ∪ {Vent(k)}** for k in footprint | **Move** (Up/Down/Left/Right/Stay) + optional **Open(pocket)** when at valve | No **Observe(k)**; no distinction **Seal(k)** vs **Vent(k)**. Only “open pocket” (interpreted as coordinated pair). |
| One action per timestep, only on regions in footprint | One move + at most one vent_pocket per rover | OK (single action per step). |

**Conclusion:** The spec requires **Seal(k)** and **Vent(k)** as distinct actions so that:
- **Correct coordination** = one agent Seal(k), one agent Vent(k) → x'_k = 0.
- **Vent without Seal** → P(explosion) = p_trigger (0.8).
- **Seal without Vent** → no change.
- **Double Vent** → P(explosion) = 0.95.

The current “both open = success” model cannot represent Vent-only, Seal-only, or double-Vent, so it does not implement the full action space.

---

## 2. Action Effects (Joint-Dependent)

| Case | Spec | Current | Gap |
|------|------|---------|-----|
| **A: Correct coordination** | Seal(k) + Vent(k) by two agents (both can access k) → x'_k = 0 with prob 1 | Two rovers Open(p) at valve_a and valve_b → pocket 0 | Conceptually aligned but implemented as single “open” action, not Seal+Vent. |
| **B: Vent without Seal** | ≥1 Vent(k), no Seal(k) → P(x'_k=3)=p_trigger (0.8), P(x'_k=2)=1−p_trigger | Unilateral Open(p) when critical → **deterministic** explosion | Spec is **stochastic** (0.8 / 0.2); current is deterministic. |
| **C: Seal without Vent** | Seal(k) without Vent(k) → x'_k = x_k | Not representable (no Seal-only action) | Missing. |
| **D: Double Vent** | Two agents Vent(k) → P(x'_k=3)=0.95 | Not representable (two “open” = success) | Missing. |
| **E: Double Seal** | No effect | Not representable | Missing. |

---

## 3. Natural Gas Dynamics

| Spec | Current (`pocket_transition`, `PocketDynamics`) | Gap |
|------|--------------------------------------------------|-----|
| **x=1:** P(x'=2) = p_acc (0.1) | x=1: P(2)=β, P(0)=η, P(1)=1−β−η (e.g. β=0.3, η=0.2) | Spec: 1→2 only (p_acc). Current: 1→0 and 1→1 as well. |
| **x=2:** P(x'=3) = p_explode (0.05) | x=2: P(1)=μ, P(2)=1−μ (no explosion in transition); explosion only via **deadline** or unilateral vent | Spec: 2→3 with p_explode. Current: no 2→3 in dynamics; different explosion mechanism. |
| **Spatial spread:** If neighbor in state 2 → P(x'_k=1) += p_spread (0.08) | No spatial spread | Missing. |
| **State 3** = Explosion (absorbing) | `exploded[p]` flag (absorbing) | OK. |

---

## 4. Reward Function

| Spec | Current | Gap |
|------|---------|-----|
| **Successful neutralization** (Seal+Vent on critical): **+50** | R_fix = **10** | Magnitude wrong (spec: +50). |
| **Explosion:** **−200** | R_explosion = 200 (so −200) | OK. |
| **Critical persistence:** **−5 per timestep** when x_k=2 | No per-step penalty for critical | Missing. |
| **Accumulation:** **−1 per timestep** when x_k=1 | No per-step penalty for pressurized | Missing. |
| **Action costs:** Idle 0, Observe −0.5, Seal −2, Vent −2 | Single **c_step** per rover (e.g. 0.1); no Observe/Seal/Vent costs | Action costs not per spec. |

---

## 5. Why This Matters for Coordination

The spec is designed so that:

- **Correct coordination:** +50  
- **Vent without Seal:** E[reward] ≈ 0.8×(−200)+0.2×(0) = **−160**  
- So an agent must reason: “If I Vent, what is the probability another agent Seals?” If that probability is low, Vent is very bad.

The current implementation:

- Uses a single “open” action and “both open = success,” so it does not test **Vent vs Seal** reasoning.
- Makes unilateral vent **deterministically** catastrophic when critical, instead of **stochastic** (p_trigger).
- Uses different dynamics and rewards, so the intended coupling between actions and the advantage of action-consistent decentralized planning are not fully reflected.

---

## 6. Recommended Changes (Summary)

1. **Action space:** Add **Observe(k)**, **Seal(k)**, **Vent(k)** (and keep Idle/Move). Drop single “Open(p)” in favor of Seal/Vent at valves.
2. **Transition:** Implement Cases A–E from the spec; Vent-without-Seal with **stochastic** explosion (p_trigger = 0.8); Double Vent with P(explosion) = 0.95.
3. **Dynamics:** Align with spec: 1→2 with p_acc (0.1); 2→3 with p_explode (0.05); add optional spatial spread (p_spread = 0.08) if regions have neighbors.
4. **Rewards:** R_fix = +50; −5 per step when x=2; −1 per step when x=1; action costs: Observe −0.5, Seal −2, Vent −2.
5. **Planners:** Update feasibility (e.g. FeasibleTwoValve) and backup/expected reward to use Seal/Vent semantics and new rewards/dynamics.

After these changes, the problem will implement the specified action space and joint-dependent effects, making coordination structurally necessary and action inconsistency very costly.

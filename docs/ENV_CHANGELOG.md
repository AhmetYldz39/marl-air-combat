# ENV_CHANGELOG.md

This file records every bump of `ENV_VERSION` and the change that caused it.
**Rule:** bump `ENV_VERSION` on any commit that can alter a trajectory.
Benchmark numbers are only comparable within one `(ENV_VERSION, opponent_version)` pair.

Current version: **1.0.0** — opponent: **heuristic-v1**

---

## 1.0.0 — 2026-09-06

Initial stable release. All six baseline checkpoints evaluated under this version.

### What is frozen at 1.0.0

| Component | Frozen state |
|---|---|
| `DogfightEnv` | phase-4 2v2 normal spawn; max_steps=1000 |
| `AircraftModel` | RK4 dynamics, dt=0.05 s; state[18], action[5] |
| `RewardModel` | 9-term reward + r_fire_ready + r_pursuit + r_smooth; weights as in `configs/config.yaml` |
| `WeaponsModel` | WEZ cone ±30°, 300–8000 m; fire_cooldown=0.5 s |
| `Normalizer` | 50D observation (17 ego + 9 teammate + 12×2 enemies) |
| `MultiHeuristicPolicy` | CRITICAL → EVASION → PURSUIT rule-based logic; tagged `heuristic-v1` |
| `_critical_recovery_blue` | OOB/ground override; _CRIT_MAP_MARGIN=5000 m |

### Key reward weights at 1.0.0

| Weight | Value |
|---|---|
| `w_kill` | 25 |
| `w_wez` | 5 |
| `w_tracking` | 6.0 |
| `w_pursuit` | 2.5 |
| `pursuit_norm_dist` | 10 000 m |
| `w_smooth_ctrl` | 0.05 |
| `w_smooth_throttle` | 0.003 |
| `w_throttle_ctx` | 0.8 |
| `w_survival` | 0.0 |
| `ammo_miss_penalty` | 0.5 (config override; default in code: 2.0) |
| `team_kill_bonus` | 15.0 |

### Provenance note

The QMIX baseline (65.0%) was re-evaluated in August 2026 against a modified
environment (reward weights, OOB recovery, spawn logic changed relative to the
May 2026 run that gave 70.2%). The 65.0% number corresponds to the 1.0.0 env
and checkpoint `qmix_ep2000.pt` (total_steps=1,110,370). The two numbers are
not directly comparable; see `docs/evals/README_evals.md` for the full provenance.

---

## How to bump

1. Make the behavioural change.
2. Update `ENV_VERSION` in `aircombat_bench/envs/dogfight_env.py`.
3. Add an entry here with the date, the new version, and a one-line description
   of what changed and why. Link to the PR or commit.
4. Regenerate `docs/evals/v0.1/` (or a new versioned directory) to confirm
   published numbers still reproduce within their CIs — if not, document the
   delta here.

---

## Comparison rule

> Benchmark numbers from two different `ENV_VERSION` values are **not directly
> comparable**, even if the same checkpoint is used. The version pair
> `(ENV_VERSION, opponent_version)` must match for numbers to be on the same
> scale.

This applies to: reward weights, spawn logic, OOB recovery, WEZ parameters,
observation layout, heuristic behavior, episode length limit.

# SAC HomeBot Route Planner — Project Briefing

## What We're Building

Continuous-action SAC (Soft Actor-Critic, Haarnoja 2018) agent for HomeBot2D navigation.
Sparse 0/1 reward only. Image observation required. HER replay.

**Primary metric:** `Eval/chain_score` — chained task (trash→fridge→human→door→human), greedy eval.
Single-goal reach rate is a useful intermediate signal but not the deployment metric.

---

## Repos

| Repo | Path | Status |
|------|------|--------|
| `sac-homebot-route-planner` | `/home/bobcowher/pythonprojects/sac-homebot-route-planner` | **This repo — new, empty** |
| `q-homebot-route-planner` | `/home/bobcowher/pythonprojects/q-homebot-route-planner` | Working baseline (64% reach, passes chain eval) |
| `homebot-route-planner` | `/home/bobcowher/pythonprojects/homebot-route-planner` | Abandoned — discrete SAC, stalled at 15-20% |

Remote: `git@github.com:bobcowher/sac-homebot-route-planner.git`

---

## Why We're Here

### Q-DQN worked. Discrete SAC didn't.

The Q-DQN repo (`q-homebot-route-planner`) achieved ~64% single-goal reach with Double-DQN + HER.
The discrete SAC repo (`homebot-route-planner`) stalled at 15-20% after ~40 runs.

The root cause of the discrete SAC failure: the max-entropy actor couldn't exploit the value field
the HER critic learned. The only fix was argmaxing the critic directly — which reduces to DQN.
We were writing DQN in SAC's clothing.

### Why continuous SAC now

HomeBot2D supports continuous actions. Continuous SAC (Haarnoja 2018) solves the problems we had:
- **Oscillation handled structurally** — Gaussian policy + entropy bonus, not explicit spin penalties
- **Off-policy** — compatible with HER replay
- **Simpler than Christodoulou discrete** — standard, well-understood, well-validated

TD3 is the credible alternative if entropy tuning proves flaky (deterministic policy + noise, HER-compatible).

---

## Plan

Clone the Q-DQN infrastructure wholesale. Replace the DQN agent with continuous SAC.

**Keep from Q-DQN:**
- `episode_buffer.py` — HER with the done-fix (see below), K=2, 200k buffer
- Chain eval (`run_chain`) + rolling-100 success metric
- Motion state (window=8) + spin metric
- Goal geometry (relative displacement encoding)
- Frame skip (FRAME_SKIP=2)

**Replace:**
- `agent.py` — DQN → continuous SAC (Gaussian actor, double-Q critic, auto-alpha)
- `models/q_model.py` → actor + twin critics (4×512 depth, see below)

**First task before writing code:** Examine `HomeBot2D` continuous action spec — what does the action
look like? `(vx, vy)`? `(angle, speed)`? This sets the actor output dimension.

---

## Critical Lessons (Do Not Regress)

### HER done bug — the biggest single win in the whole project

Hindsight-relabeled successes MUST set `done=True`. The original bug reused the original
transition's `done` (almost always False) → targets bootstrapped past the goal → Q inflated
toward `1/(1-γ) ≈ 100` on ~80% of the buffer. One fix jumped performance 35% → 56%.

**Already fixed in `q-homebot-route-planner/episode_buffer.py`. Do not regress this.**

### Truncation ≠ terminal

Store `term`, not `term or trunc` in the replay buffer. Timeout is not a terminal state.
Training Q→0 at far-from-goal states was the "can't navigate distance" failure mode.

### Buffer depth

K=2 HER + 200k buffer (not K=4 + 100k). At K=4/100k, ~6000 transitions per failed episode
→ only ~16 episodes retained → catastrophic forgetting. 200k → ~66 episodes.

### Chain eval is the real metric

Single-goal `best.pt` (triggered at ep18, first reach) was worthless at greedy eval.
Track rolling-100 success rate and save best on that. Run chain eval every 10 episodes.

### Greedy eval inversion

At peak Q-DQN training: greedy (ε=0) was WORSE than ε=0.1 (43% vs 63%). The 10% random
actions break limit-cycle loops. Deploy policy must not be pure greedy.
For continuous SAC, stochastic sampling at eval time (not mean action) may be the equivalent.

### Goal encoding

Relative goal displacement (not absolute coords) learned faster and higher. Q-DQN Exp 13:
peak smoothed reward 0.94 vs 0.888 prior. Feed the actor displacement vector, not raw coords.

### Network depth

4×512 heads for both actor and critic. 2×256 diverged. Asymmetric (shallow critic, deep actor) regressed hard.

### Random goal tiles

`random_goal_tiles=True` — trains a general-purpose navigator, not a trash-specialist.
Required for chain eval generalization.

### Frame skip + motion history

FRAME_SKIP=2 (action repeat) was a net positive. Motion window=8 net-displacement is required
to detect moving limit cycles — single-step velocity can't see a spin.

### No reward shaping

Sparse 0/1 only. This is a standing constraint. HER IS the curriculum.

---

## Standing Constraints

- No reward shaping / env fixes / hacks — sparse 0/1 reward only
- Image observation required
- Never use `python3 -c "..."` — always write to a `.py` file
- Never install packages to global Python — always use conda env
- Look for `build.sh` first when running/testing
- No `Co-Authored-By` in commit messages
- Never commit CLAUDE.md files
- No file forking — evolve modules, use git for history

---

## Value Funnel Context (Why We Left the Old Repo)

The discrete SAC arc ended with the n=8 run (run 395, branch `nstep-8`). The value funnel
(critic's spatial value field) reached ~200px radius — exactly the HER horizon (50 steps × 4px).
N-step returns (n=8) widened it from 150px → 200px but couldn't push past the HER boundary.
The binding constraint was HER coverage, not propagation speed. Rather than keep tuning
discrete SAC (which had the structural actor problem), we pivoted to continuous.

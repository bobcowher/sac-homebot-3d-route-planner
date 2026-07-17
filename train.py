from agent import Agent
import gymnasium as gym
import homebot3d  # noqa: F401  (side-effect env registration)

# One env.step is one mj_step (0.01 s). At MAX_LIN=1.0 m/s that is ~0.01 m/step,
# so an agent action needs to repeat over several physics steps to cover useful
# ground. skip=10 -> ~0.1 m per agent step (~135 steps to cross the 13.5 m house),
# matching the control granularity the 2D env had. The FrameSkipWrapper renders
# only the final frame of each skip, so a larger skip also means fewer renders.
FRAME_SKIP = 10

# Chain-style training: the base env (same one chain_eval deploys on), not the
# single-goal Goal env. Reward/termination live in the training loop now
# (distance <= REACH_RADIUS, identical to the HER relabel rule). max_steps is a
# generous ceiling; per-leg budgets in agent.train() do the real limiting.
env = gym.make(
    "HomeBot3D-V1",
    render_mode="rgb_array",
    goals=("trash", "drink", "package"),
    width=96,
    height=96,
    n_trash=2,
    max_steps=40000,
    map_name="default",
    random_start=True,
)

if FRAME_SKIP > 1:
    from env_wrappers import FrameSkipWrapper
    env = FrameSkipWrapper(env, skip=FRAME_SKIP)

agent = Agent(
    env=env,
    max_buffer_size=200000,
    goal_layers=2,
    head_layers=4,
    use_motion=True,
    motion_window=8,
    # lr=1e-4 (down from the 3e-4 default): at 3e-4 the policy peaked ~ep75–200
    # then degraded (critic drift under the collapsed-alpha exploit-only actor) —
    # end-of-run chain 0.6/5 vs a peak of 1.4/5. 1e-4 starts slower but does NOT
    # degrade (end ≈ peak) and reaches a higher ceiling (chain 1.6–2.7/5 in the
    # 2nd half, reach up to 0.8). See the bring-up experiments (exp4 vs exp5).
    lr=1e-4,
)

# Long consolidation run. The 600-ep diagnostic phase established that the agent
# learns at lr=1e-4 without degrading and was STILL climbing at ep575 (chain
# 1.6–3.0/5, first full_chain completions at 0.20). This run gives it the episodes
# to push full_chain up. Eval n is bumped for trustworthy trending (the noisy
# n=6/n=10 diagnostic evals hid the 3e-4 degradation): greedy n=30, chain n=20.
# Held-out confirms are re-enabled (n=30, every 250 eps from ep500) so
# best_confirmed.pt captures the true best over the long run.
agent.train(
    episodes=1500,
    batch_size=256,
    eval_interval=100,
    eval_episodes=30,
    chain_eval_interval=50,
    chain_eval_episodes=20,
    goals_per_episode=5,
    her_anneal_start=None,
    confirm_bar=4.2,
    confirm_episodes=30,
    confirm_interval=250,
    confirm_start=500,
    final_eval_episodes=40,
)

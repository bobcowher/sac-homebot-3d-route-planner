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
)

# 3D bring-up phase: short 600-episode diagnostic runs to establish that the
# agent learns at all on HomeBot3D (chain_score climbing above the ~0.1/5 random
# baseline, reach rate rising) before committing to a long consolidation run.
# Eval cadence is kept light so each run turns around fast under MuJoCo rendering:
# greedy reach every 100 eps (n=10), chain every 25 eps (n=6), no held-out
# confirms (confirm_interval=0). FINAL_POLICY/FINAL_CONFIRM measured at n=20.
agent.train(
    episodes=600,
    batch_size=256,
    eval_interval=100,
    eval_episodes=10,
    chain_eval_interval=25,
    chain_eval_episodes=6,
    goals_per_episode=5,
    her_anneal_start=None,
    confirm_interval=0,
    confirm_start=10**9,
    final_eval_episodes=20,
)

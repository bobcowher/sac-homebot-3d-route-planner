import gymnasium as gym


class FrameSkipWrapper(gym.Wrapper):
    """Repeats the chosen action for `skip` steps, accumulating reward.

    In homebot3d every env.step() renders an RGB frame (the expensive part), but
    frame-skip only needs the LAST frame. When the wrapped env exposes the
    render-less physics seam (step_physics + _obs), step the physics `skip` times
    and render once at the end — a ~skitimes cut in renders per agent step. The
    plain per-step path is kept as a fallback for envs without that seam (and for
    the Goal env, whose dict obs is assembled in its own step()).
    """

    def __init__(self, env, skip=2):
        super().__init__(env)
        self._skip = skip
        base = env.unwrapped
        # Fast path only for the plain rgb env: step_physics/_obs exist and the
        # obs is a bare frame (Goal env, which builds a dict in step(), is excluded
        # via its _dict_obs marker).
        self._base = base
        self._fast = (hasattr(base, "step_physics") and hasattr(base, "_obs")
                      and not hasattr(base, "_dict_obs"))

    def step(self, action):
        if not self._fast:
            total_reward = 0.0
            terminated = truncated = False
            info = {}
            for _ in range(self._skip):
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            return obs, total_reward, terminated, truncated, info

        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self._skip):
            reward, terminated, truncated, info = self._base.step_physics(action)
            total_reward += reward
            if terminated or truncated:
                break
        obs = self._base._obs()
        return obs, total_reward, terminated, truncated, info

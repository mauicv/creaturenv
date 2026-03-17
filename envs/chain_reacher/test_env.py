"""Basic checks for ChainReacherEnv."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import envs.chain_reacher  # noqa: F401


def test_chain_reacher_step_shapes() -> None:
    env = gym.make("ChainReacher-v0", n_links=3, n_obs=2, obstacle_seed=123, render_mode=None)
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert "distance_to_target" in info

        action = env.action_space.sample().astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "distance_to_target" in info
    finally:
        env.close()


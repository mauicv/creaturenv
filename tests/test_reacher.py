"""Basic sanity checks for ChainReacherEnv."""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt

from envs.chain_reacher.chain_reacher_env import ChainReacherEnv
from random import randint


def run_sanity_check() -> None:
    env = ChainReacherEnv(
        n_links=1,
        n_obs=0,
        link_length=1.0,
        obstacle_seed=8,
        max_torque=8.0,
        n_lidar_rays=16,
        lidar_range=4.0,
        max_episode_steps=500,
        render_mode="human",
        upright_target=True,
    )
    obs, info = env.reset(seed=randint(0, 1000000))
    env.render()
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert "distance_to_target" in info

    rewards = []

    for _ in range(500):
        action = env.action_space.sample().astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        env.render()
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(reward)
        if terminated or truncated:
            obs, _ = env.reset()
            env.render()
            assert obs.shape == env.observation_space.shape

    # Keep the window open briefly so the simulation is visible after stepping.
    end_time = time.time() + 2.5
    while time.time() < end_time:
        env.render()
        time.sleep(1.0 / 60.0)

    env.close()

    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    run_sanity_check()
    print("Sanity check passed.")


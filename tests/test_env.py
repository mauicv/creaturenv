"""Basic sanity checks for CreatureNavigationEnv."""

from __future__ import annotations

import time

import numpy as np

from creature_env.creature_env import CreatureNavigationEnv


def run_sanity_check() -> None:
    env = CreatureNavigationEnv(
        leg_spec=[1],
        num_obstacles=3,
        arena_size=20.0,
        n_lidar_rays=16,
        lidar_range=10.0,
        obstacle_seed=8,
        render_mode="human",
        max_thrust=5.0,
        fluid_friction=1.2,
    )
    obs, info = env.reset(seed=123)
    env.render()
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert "distance_to_target" in info

    for _ in range(300):
        action = env.action_space.sample().astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
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


if __name__ == "__main__":
    run_sanity_check()
    print("Sanity check passed.")


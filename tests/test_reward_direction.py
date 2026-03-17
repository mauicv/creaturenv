"""Reward-direction checks for SwimmerNavigationEnv."""

from __future__ import annotations

import numpy as np

from envs.swimer.swimmer_env import SwimmerNavigationEnv


def _set_deterministic_single_leg_pose(env: SwimmerNavigationEnv) -> None:
    """Place the swimmer in a controlled pose so reward-direction tests are stable."""
    # Keep the central body fixed at origin with no residual motion.
    env.central_body.position = (0.0, 0.0)
    env.central_body.angle = 0.0
    env.central_body.linearVelocity = (0.0, 0.0)
    env.central_body.angularVelocity = 0.0

    # For leg_spec=[1], put the single tip link on +x, facing +x.
    tip = env.leg_tips[0]
    tip.position = (0.4, 0.0)
    tip.angle = 0.0
    tip.linearVelocity = (0.0, 0.0)
    tip.angularVelocity = 0.0


def test_moving_toward_target_gets_higher_reward_than_moving_away() -> None:
    env = SwimmerNavigationEnv(
        leg_spec=[1],
        num_obstacles=0,
        arena_size=20.0,
        n_lidar_rays=4,
        lidar_range=6.0,
        damping=0.0,
        fluid_friction=0.0,
        max_thrust=8.0,
        render_mode=None,
    )

    try:
        # With tip angle 0, thrust points toward -x in current env logic.
        action = np.asarray([1.0, 0.0], dtype=np.float32)  # [thruster, joint_speed]

        env.reset(seed=123)
        _set_deterministic_single_leg_pose(env)
        env.target_position = np.asarray((-5.0, 0.0), dtype=np.float32)
        env.prev_distance = float(np.linalg.norm(env.target_position - np.asarray((0.0, 0.0), dtype=np.float32)))
        _, reward_toward, _, _, _ = env.step(action)

        env.reset(seed=123)
        _set_deterministic_single_leg_pose(env)
        env.target_position = np.asarray((5.0, 0.0), dtype=np.float32)
        env.prev_distance = float(np.linalg.norm(env.target_position - np.asarray((0.0, 0.0), dtype=np.float32)))
        _, reward_away, _, _, _ = env.step(action)

        assert reward_toward > reward_away
    finally:
        env.close()


"""Run a trained PPO policy in human render mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

import envs.swimer  # noqa: F401  # Registers SwimmerNavigation-v0


@dataclass
class PlayArgs:
    # Fast laptop defaults aligned with TrainArgs in train_ppo.py.
    model_path: Path = field(default_factory=lambda: Path("runs/train_fast_laptop/checkpoints/best_model.zip"))
    episodes: int = 5
    seed: int = 0
    max_episode_steps: int = 300
    deterministic: bool = True

    leg_spec: list[int] = field(default_factory=lambda: [1])
    num_obstacles: int = 0
    arena_size: float = 20.0
    n_lidar_rays: int = 4
    lidar_range: float = 6.0
    obstacle_seed: int | None = 7
    damping: float = 0.1
    fluid_friction: float = 0.8
    max_thrust: float = 6.0
    max_joint_torque: float = 20.0

def main(args: PlayArgs) -> None:
    env = gym.make(
        "SwimmerNavigation-v0",
        leg_spec=args.leg_spec,
        num_obstacles=args.num_obstacles,
        arena_size=args.arena_size,
        n_lidar_rays=args.n_lidar_rays,
        lidar_range=args.lidar_range,
        obstacle_seed=args.obstacle_seed,
        damping=args.damping,
        fluid_friction=args.fluid_friction,
        max_thrust=args.max_thrust,
        max_joint_torque=args.max_joint_torque,
        render_mode="human",
    )
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)

    model = PPO.load(args.model_path)
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            total_reward += float(reward)
        print(f"Episode {episode + 1}: total_reward={total_reward:.3f}")

    env.close()


if __name__ == "__main__":
    args = PlayArgs()
    main(args)


"""Run a trained PPO policy on ChainReacher in human render mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

import envs.chain_reacher  # noqa: F401  # Registers ChainReacher-v0


@dataclass
class PlayArgs:
    # Fast laptop defaults aligned with TrainArgs in train_ppo.py.
    model_path: Path = field(default_factory=lambda: Path("runs/train_fast_laptop/checkpoints/best_model.zip"))
    episodes: int = 5
    seed: int = 0
    max_episode_steps: int = 300
    deterministic: bool = True

    n_links: int = 2
    n_obs: int = 0
    link_length: float = 1.0
    arena_radius: float | None = None
    obstacle_size_range: tuple[float, float] = (0.3, 0.8)
    n_lidar_rays: int = 4
    lidar_range: float | None = None
    obstacle_seed: int | None = 7
    max_torque: float = 8.0
    target_threshold: float = 0.3

def main(args: PlayArgs) -> None:
    env = gym.make(
        "ChainReacher-v0",
        n_links=args.n_links,
        n_obs=args.n_obs,
        link_length=args.link_length,
        arena_radius=args.arena_radius,
        obstacle_size_range=args.obstacle_size_range,
        n_lidar_rays=args.n_lidar_rays,
        lidar_range=args.lidar_range,
        obstacle_seed=args.obstacle_seed,
        max_torque=args.max_torque,
        target_threshold=args.target_threshold,
        max_episode_steps=args.max_episode_steps,
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


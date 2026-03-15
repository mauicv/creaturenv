"""Run a trained PPO policy in human render mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

import creature_env  # noqa: F401  # Registers CreatureNavigation-v0


def parse_leg_spec(value: str) -> list[int]:
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    leg_spec = [int(chunk) for chunk in parts]
    if not leg_spec or any(link_count <= 0 for link_count in leg_spec):
        raise argparse.ArgumentTypeError("leg_spec must be a comma-separated list of positive ints, e.g. 2,1,3")
    return leg_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Play back a trained PPO model.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to .zip model file")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--leg-spec", type=parse_leg_spec, default=[2, 1, 3])
    parser.add_argument("--num-obstacles", type=int, default=3)
    parser.add_argument("--arena-size", type=float, default=20.0)
    parser.add_argument("--n-lidar-rays", type=int, default=16)
    parser.add_argument("--lidar-range", type=float, default=10.0)
    parser.add_argument("--obstacle-seed", type=int, default=7)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--fluid-friction", type=float, default=1.2)
    parser.add_argument("--max-thrust", type=float, default=8.0)
    parser.add_argument("--max-joint-torque", type=float, default=30.0)
    args = parser.parse_args()

    env = gym.make(
        "CreatureNavigation-v0",
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
    main()


"""Train a PPO policy on CreatureNavigationEnv."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import creature_env  # noqa: F401  # Registers CreatureNavigation-v0


def parse_leg_spec(value: str) -> list[int]:
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    leg_spec = [int(chunk) for chunk in parts]
    if not leg_spec or any(link_count <= 0 for link_count in leg_spec):
        raise argparse.ArgumentTypeError("leg_spec must be a comma-separated list of positive ints, e.g. 2,1,3")
    return leg_spec


def make_env(
    *,
    leg_spec: list[int],
    seed: int,
    max_episode_steps: int,
    num_obstacles: int,
    arena_size: float,
    n_lidar_rays: int,
    lidar_range: float,
    obstacle_seed: int | None,
    damping: float,
    fluid_friction: float,
    max_thrust: float,
    max_joint_torque: float,
    render_mode: str | None,
):
    def _init():
        env = gym.make(
            "CreatureNavigation-v0",
            leg_spec=leg_spec,
            num_obstacles=num_obstacles,
            arena_size=arena_size,
            n_lidar_rays=n_lidar_rays,
            lidar_range=lidar_range,
            obstacle_seed=obstacle_seed,
            damping=damping,
            fluid_friction=fluid_friction,
            max_thrust=max_thrust,
            max_joint_torque=max_joint_torque,
            render_mode=render_mode,
        )
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CreatureNavigationEnv.")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="ppo_creature")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--max-episode-steps", type=int, default=1000)

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

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    args = parser.parse_args()

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv(
        [
            make_env(
                leg_spec=args.leg_spec,
                seed=args.seed + idx,
                max_episode_steps=args.max_episode_steps,
                num_obstacles=args.num_obstacles,
                arena_size=args.arena_size,
                n_lidar_rays=args.n_lidar_rays,
                lidar_range=args.lidar_range,
                obstacle_seed=args.obstacle_seed,
                damping=args.damping,
                fluid_friction=args.fluid_friction,
                max_thrust=args.max_thrust,
                max_joint_torque=args.max_joint_torque,
                render_mode=None,
            )
            for idx in range(args.n_envs)
        ]
    )
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv(
        [
            make_env(
                leg_spec=args.leg_spec,
                seed=args.seed + 10_000,
                max_episode_steps=args.max_episode_steps,
                num_obstacles=args.num_obstacles,
                arena_size=args.arena_size,
                n_lidar_rays=args.n_lidar_rays,
                lidar_range=args.lidar_range,
                obstacle_seed=args.obstacle_seed,
                damping=args.damping,
                fluid_friction=args.fluid_friction,
                max_thrust=args.max_thrust,
                max_joint_torque=args.max_joint_torque,
                render_mode=None,
            )
        ]
    )
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoints_dir),
        log_path=str(run_dir),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        seed=args.seed,
        device=args.device,
    )
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback, progress_bar=True)

    final_path = run_dir / "final_model"
    model.save(final_path)
    train_env.close()
    eval_env.close()
    print(f"Saved final model to: {final_path}.zip")
    print(f"Best model (if improved during eval) saved in: {checkpoints_dir}")


if __name__ == "__main__":
    main()


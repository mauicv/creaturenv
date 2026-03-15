"""Train a PPO policy on CreatureNavigationEnv."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import creature_env  # noqa: F401  # Registers CreatureNavigation-v0


@dataclass
class TrainArgs:
    # Fast laptop defaults (Colab-friendly).
    total_timesteps: int = 30_000
    n_envs: int = 2
    seed: int = 0
    device: str = "auto"
    run_name: str = "train_fast_laptop"
    output_dir: Path = field(default_factory=lambda: Path("runs"))
    max_episode_steps: int = 300

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

    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    clip_range: float = 0.2


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


def main(args: TrainArgs) -> None:
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
    args = TrainArgs()
    main(args)


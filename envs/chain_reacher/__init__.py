"""ChainReacher environment package."""

from gymnasium.envs.registration import register

from .chain_reacher_env import ChainReacherEnv

register(
    id="ChainReacher-v0",
    entry_point="envs.chain_reacher.chain_reacher_env:ChainReacherEnv",
    kwargs={"n_links": 2, "n_obs": 0},
)

__all__ = ["ChainReacherEnv"]


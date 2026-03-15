"""Parameterized multi-legged thruster creature environment package."""

from gymnasium.envs.registration import register

from .creature_env import CreatureNavigationEnv

register(
    id="CreatureNavigation-v0",
    entry_point="creature_env.creature_env:CreatureNavigationEnv",
    kwargs={"leg_spec": [2, 1, 3]},
)

__all__ = ["CreatureNavigationEnv"]


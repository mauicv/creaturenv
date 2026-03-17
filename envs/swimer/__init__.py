"""Parameterized multi-legged thruster swimmer environment package."""

from gymnasium.envs.registration import register

from .swimmer_env import SwimmerNavigationEnv

register(
    id="SwimmerNavigation-v0",
    entry_point="envs.swimer.swimmer_env:SwimmerNavigationEnv",
    kwargs={"leg_spec": [2, 1, 3]},
)

__all__ = ["SwimmerNavigationEnv"]


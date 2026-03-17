"""Lidar raycast helpers for ChainReacher."""

from __future__ import annotations

import math

import numpy as np
from Box2D import b2RayCastCallback


def _entity_name(user_data) -> str | None:
    if isinstance(user_data, dict):
        return user_data.get("entity")
    if isinstance(user_data, str):
        return user_data
    return None


class ChainLidarCallback(b2RayCastCallback):
    """Raycast callback that records the closest non-chain hit."""

    def __init__(self) -> None:
        super().__init__()
        self.hit_fraction = 1.0

    def ReportFixture(self, fixture, point, normal, fraction):  # noqa: N802
        entity = _entity_name(fixture.body.userData)
        if entity == "chain":
            return -1.0
        if fraction < self.hit_fraction:
            self.hit_fraction = fraction
        return fraction


def cast_lidar(
    world,
    tip_position: tuple[float, float],
    tip_angle: float,
    n_rays: int,
    max_range: float,
) -> np.ndarray:
    """Cast 360-degree lidar from tip and return hit fractions in [0, 1]."""
    if n_rays <= 0:
        return np.zeros((0,), dtype=np.float32)

    ox, oy = float(tip_position[0]), float(tip_position[1])
    fractions = np.ones((n_rays,), dtype=np.float32)
    for idx in range(n_rays):
        ray_angle = tip_angle + (2.0 * math.pi * idx / n_rays)
        end = (ox + max_range * math.cos(ray_angle), oy + max_range * math.sin(ray_angle))
        callback = ChainLidarCallback()
        world.RayCast(callback, (ox, oy), end)
        fractions[idx] = float(callback.hit_fraction)
    return fractions


"""Lidar raycast utilities."""

from __future__ import annotations

import math

import numpy as np
from Box2D import b2RayCastCallback


class LidarCallback(b2RayCastCallback):
    """Raycast callback that stores only the closest valid hit."""

    def __init__(self, ignored_entity: str = "creature") -> None:
        super().__init__()
        self.ignored_entity = ignored_entity
        self.hit_fraction = 1.0
        self.hit_fixture = None

    def ReportFixture(self, fixture, point, normal, fraction):  # noqa: N802
        user_data = fixture.body.userData or {}
        if user_data.get("entity") == self.ignored_entity:
            return -1.0
        if fraction < self.hit_fraction:
            self.hit_fraction = fraction
            self.hit_fixture = fixture
        return fraction


def cast_lidar(
    world,
    origin: tuple[float, float],
    body_angle: float,
    n_rays: int,
    max_range: float,
) -> np.ndarray:
    """Cast rays in body frame and return normalized hit fractions in [0, 1]."""
    if n_rays <= 0:
        return np.zeros((0,), dtype=np.float32)

    fractions = np.ones((n_rays,), dtype=np.float32)
    ox, oy = float(origin[0]), float(origin[1])

    for idx in range(n_rays):
        ray_angle = body_angle + (2.0 * math.pi * idx / n_rays)
        end = (ox + max_range * math.cos(ray_angle), oy + max_range * math.sin(ray_angle))
        callback = LidarCallback(ignored_entity="creature")
        world.RayCast(callback, (ox, oy), end)
        fractions[idx] = float(callback.hit_fraction)

    return fractions


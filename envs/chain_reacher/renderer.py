"""Pygame renderer for ChainReacher."""

from __future__ import annotations

import math

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None


class ChainReacherRenderer:
    """Renderer supporting human and rgb_array modes."""

    def __init__(
        self,
        *,
        arena_radius: float,
        render_mode: str,
        width: int = 720,
        height: int = 720,
    ) -> None:
        if pygame is None:
            raise ImportError("pygame is required for rendering")
        if render_mode not in {"human", "rgb_array"}:
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.arena_radius = float(arena_radius)
        self.render_mode = render_mode
        self.width = int(width)
        self.height = int(height)
        self.scale = min(self.width, self.height) * 0.46 / max(self.arena_radius, 1e-6)

        pygame.init()
        self._clock = pygame.time.Clock()
        self._surface = pygame.Surface((self.width, self.height))
        self._screen = None
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("ChainReacherEnv")

    def _world_to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        sx = int(self.width * 0.5 + x * self.scale)
        sy = int(self.height * 0.5 - y * self.scale)
        return sx, sy

    def _draw_polygon_body(self, body, color: tuple[int, int, int]) -> None:
        for fixture in body.fixtures:
            shape = fixture.shape
            if not hasattr(shape, "vertices"):
                continue
            vertices = [self._world_to_screen(body.transform * v) for v in shape.vertices]
            if len(vertices) >= 3:
                pygame.draw.polygon(self._surface, color, vertices)

    def render(
        self,
        *,
        world,
        links: list,
        anchor,
        target_position: np.ndarray,
        lidar_fractions: np.ndarray | None = None,
        lidar_range: float = 0.0,
        tip_position: tuple[float, float] | None = None,
        tip_angle: float = 0.0,
    ) -> np.ndarray | None:
        self._surface.fill((20, 20, 24))

        center = (self.width // 2, self.height // 2)
        arena_px_radius = max(2, int(self.arena_radius * self.scale))
        pygame.draw.circle(self._surface, (65, 65, 78), center, arena_px_radius, width=2)

        for body in world.bodies:
            user_data = body.userData or {}
            entity = user_data.get("entity") if isinstance(user_data, dict) else user_data
            if entity == "obstacle":
                self._draw_polygon_body(body, (180, 68, 68))

        target_px = self._world_to_screen((float(target_position[0]), float(target_position[1])))
        pygame.draw.circle(self._surface, (70, 210, 90), target_px, 7)

        for link in links:
            self._draw_polygon_body(link, (72, 130, 225))

        anchor_px = self._world_to_screen((anchor.position[0], anchor.position[1]))
        pygame.draw.circle(self._surface, (140, 140, 150), anchor_px, 4)

        if lidar_fractions is not None and tip_position is not None and len(lidar_fractions) > 0:
            n = len(lidar_fractions)
            origin_px = self._world_to_screen(tip_position)
            for idx, frac in enumerate(lidar_fractions):
                a = tip_angle + (2.0 * math.pi * idx / n)
                dist = float(frac) * lidar_range
                end = (
                    float(tip_position[0] + dist * math.cos(a)),
                    float(tip_position[1] + dist * math.sin(a)),
                )
                pygame.draw.line(self._surface, (210, 205, 105), origin_px, self._world_to_screen(end), width=1)

        if self.render_mode == "human":
            assert self._screen is not None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
            self._screen.blit(self._surface, (0, 0))
            pygame.display.flip()
            self._clock.tick(60)
            return None

        frame = pygame.surfarray.array3d(self._surface)
        return np.transpose(frame, (1, 0, 2))

    def close(self) -> None:
        if pygame is None:
            return
        if self._screen is not None:
            pygame.display.quit()
        pygame.quit()


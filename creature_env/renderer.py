"""Pygame renderer for the creature environment."""

from __future__ import annotations

import math

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover - import error handled at runtime
    pygame = None


class PygameRenderer:
    """Simple world renderer supporting human and rgb_array modes."""

    def __init__(
        self,
        arena_size: float,
        render_mode: str,
        width: int = 720,
        height: int = 720,
    ) -> None:
        if pygame is None:
            raise ImportError("pygame is required for rendering")
        if render_mode not in {"human", "rgb_array"}:
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.arena_size = arena_size
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.scale = min(width, height) / arena_size

        pygame.init()
        self._clock = pygame.time.Clock()
        self._surface = pygame.Surface((self.width, self.height))
        self._screen = None
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("CreatureNavigationEnv")

    def _world_to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        sx = int(self.width * 0.5 + x * self.scale)
        sy = int(self.height * 0.5 - y * self.scale)
        return sx, sy

    def _draw_polygon_body(self, body, color: tuple[int, int, int]) -> None:
        for fixture in body.fixtures:
            shape = fixture.shape
            vertices = [self._world_to_screen(body.transform * v) for v in shape.vertices]
            if len(vertices) >= 3:
                pygame.draw.polygon(self._surface, color, vertices)

    def render(
        self,
        world,
        central_body,
        target_position: np.ndarray,
        lidar_fractions: np.ndarray | None = None,
        lidar_range: float = 0.0,
        leg_tips: list | None = None,
        thruster_levels: np.ndarray | None = None,
        max_thrust: float = 0.0,
    ) -> np.ndarray | None:
        self._surface.fill((18, 18, 24))

        half = self.arena_size * 0.5
        arena_rect = pygame.Rect(0, 0, int(self.arena_size * self.scale), int(self.arena_size * self.scale))
        arena_rect.center = (self.width // 2, self.height // 2)
        pygame.draw.rect(self._surface, (45, 45, 62), arena_rect, width=2)

        for body in world.bodies:
            user_data = body.userData or {}
            if body.type == 0 and user_data.get("entity") == "obstacle":
                self._draw_polygon_body(body, (142, 65, 65))

        target_px = self._world_to_screen((float(target_position[0]), float(target_position[1])))
        pygame.draw.circle(self._surface, (75, 205, 96), target_px, max(4, int(0.25 * self.scale)))

        if lidar_fractions is not None and len(lidar_fractions) > 0:
            origin = central_body.position
            origin_px = self._world_to_screen((origin[0], origin[1]))
            n = len(lidar_fractions)
            for idx, frac in enumerate(lidar_fractions):
                ray_angle = central_body.angle + (2.0 * math.pi * idx / n)
                hit_distance = float(frac) * lidar_range
                end = (
                    origin[0] + hit_distance * math.cos(ray_angle),
                    origin[1] + hit_distance * math.sin(ray_angle),
                )
                end_px = self._world_to_screen(end)
                pygame.draw.line(self._surface, (210, 205, 95), origin_px, end_px, width=1)

        for body in world.bodies:
            user_data = body.userData or {}
            if user_data.get("entity") != "creature":
                continue
            part = user_data.get("part")
            if part == "central":
                radius = int(body.fixtures[0].shape.radius * self.scale)
                pygame.draw.circle(
                    self._surface,
                    (70, 120, 225),
                    self._world_to_screen((body.position[0], body.position[1])),
                    max(radius, 2),
                )
            elif part == "link":
                self._draw_polygon_body(body, (97, 150, 235))

        if leg_tips and thruster_levels is not None:
            for idx, tip in enumerate(leg_tips):
                level = float(thruster_levels[idx]) if idx < len(thruster_levels) else 0.0
                if level <= 1e-6:
                    continue
                angle = tip.angle
                thrust_dir = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
                p0 = np.array([tip.position[0], tip.position[1]], dtype=np.float32)
                # Indicator length scales with force level.
                force_scale = 0.7 + 0.7 * (max_thrust > 0.0)
                p1 = p0 + thrust_dir * level * force_scale
                pygame.draw.line(
                    self._surface,
                    (220, 165, 60),
                    self._world_to_screen(tuple(p0)),
                    self._world_to_screen(tuple(p1)),
                    width=2,
                )

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


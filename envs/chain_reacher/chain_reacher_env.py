"""Gymnasium Box2D ChainReacher environment."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from Box2D import b2ContactListener
from gymnasium import spaces

from .lidar import cast_lidar
from .renderer import ChainReacherRenderer
from .world_builder import build_chain_reacher_world


def _entity_name(user_data) -> str | None:
    if isinstance(user_data, dict):
        return user_data.get("entity")
    if isinstance(user_data, str):
        return user_data
    return None


class ChainObstacleContactListener(b2ContactListener):
    """Tracks chain-obstacle contact state for reward penalties."""

    def __init__(self) -> None:
        super().__init__()
        self._active_contacts: set[tuple[int, int]] = set()
        self._step_contacts = 0

    def begin_step(self) -> None:
        self._step_contacts = 0

    @property
    def n_contacts(self) -> int:
        return len(self._active_contacts)

    @property
    def had_contact_this_step(self) -> bool:
        return self._step_contacts > 0 or self.n_contacts > 0

    def _is_chain_obstacle_contact(self, contact) -> bool:
        entity_a = _entity_name(contact.fixtureA.body.userData)
        entity_b = _entity_name(contact.fixtureB.body.userData)
        entities = {entity_a, entity_b}
        return "chain" in entities and "obstacle" in entities

    def BeginContact(self, contact):  # noqa: N802
        if not self._is_chain_obstacle_contact(contact):
            return
        key = tuple(sorted((id(contact.fixtureA), id(contact.fixtureB))))
        if key not in self._active_contacts:
            self._active_contacts.add(key)
        self._step_contacts += 1

    def EndContact(self, contact):  # noqa: N802
        if not self._is_chain_obstacle_contact(contact):
            return
        key = tuple(sorted((id(contact.fixtureA), id(contact.fixtureB))))
        self._active_contacts.discard(key)


class ChainReacherEnv(gym.Env):
    """Anchored articulated chain that must reach a random target."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        n_links: int,
        n_obs: int,
        link_length: float = 1.0,
        arena_radius: float | None = None,
        obstacle_seed: int | None = None,
        obstacle_size_range: tuple[float, float] = (0.3, 0.8),
        max_torque: float = 7.5,
        n_lidar_rays: int = 16,
        lidar_range: float | None = None,
        target_threshold: float = 0.3,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if n_links < 1:
            raise ValueError("n_links must be >= 1")
        if n_obs < 0:
            raise ValueError("n_obs must be >= 0")
        if link_length <= 0:
            raise ValueError("link_length must be > 0")
        if obstacle_size_range[0] <= 0 or obstacle_size_range[1] < obstacle_size_range[0]:
            raise ValueError("invalid obstacle_size_range")
        if n_lidar_rays < 0:
            raise ValueError("n_lidar_rays must be >= 0")
        if target_threshold <= 0:
            raise ValueError("target_threshold must be > 0")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.n_links = int(n_links)
        self.n_obs = int(n_obs)
        self.link_length = float(link_length)
        self.arena_radius = (
            float(arena_radius) if arena_radius is not None else self.n_links * self.link_length * 1.2
        )
        self.obstacle_seed = obstacle_seed
        self.obstacle_size_range = (float(obstacle_size_range[0]), float(obstacle_size_range[1]))
        # Scale torque with number of links to preserve controllability.
        self.max_torque = float(max_torque) * self.n_links
        self.n_lidar_rays = int(n_lidar_rays)
        self.lidar_range = float(lidar_range) if lidar_range is not None else self.n_links * self.link_length
        self.target_threshold = float(target_threshold)
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode

        self.dt = 1.0 / 60.0
        self.velocity_iterations = 8
        self.position_iterations = 3
        self.max_joint_speed = 6.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_links,), dtype=np.float32)
        # Observation: joint angles, joint speeds, target position, distance-to-target, and lidar.
        obs_dim = 2 * self.n_links + 3 + self.n_lidar_rays
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.world = None
        self.anchor = None
        self.links = []
        self.joints = []
        self.obstacles = []
        self.target_position = np.zeros((2,), dtype=np.float32)
        self._obstacle_specs: list[tuple[float, float, float, float]] = []
        self._fixed_obstacle_specs: list[tuple[float, float, float, float]] | None = None
        self._last_lidar = np.ones((self.n_lidar_rays,), dtype=np.float32)
        self._contact_listener = ChainObstacleContactListener()
        self._renderer = None

        self.elapsed_steps = 0
        self.prev_distance = 0.0
        self.prev_action = np.zeros((self.n_links,), dtype=np.float32)
        self._tip_position = (0.0, 0.0)
        self._tip_angle = 0.0

    def _sample_target(self) -> np.ndarray:
        min_r = self.link_length
        max_r = self.n_links * self.link_length * 0.9
        for _ in range(600):
            angle = float(self.np_random.uniform(-math.pi, math.pi))
            radius = float(self.np_random.uniform(min_r, max_r))
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            if math.hypot(x, y) > (self.arena_radius - 0.2):
                continue
            blocked = False
            for ox, oy, hx, hy in self._obstacle_specs:
                if abs(x - ox) <= (hx + self.target_threshold) and abs(y - oy) <= (hy + self.target_threshold):
                    blocked = True
                    break
            if blocked:
                continue
            return np.asarray((x, y), dtype=np.float32)
        return np.asarray((max_r, 0.0), dtype=np.float32)

    def _tip_state(self) -> tuple[np.ndarray, float]:
        tip_body = self.links[-1]
        tip = tip_body.GetWorldPoint((0.5 * self.link_length, 0.0))
        pos = np.asarray((tip[0], tip[1]), dtype=np.float32)
        return pos, float(tip_body.angle)

    def _is_out_of_arena(self) -> bool:
        for body in self.links:
            x, y = body.position
            if math.hypot(x, y) > self.arena_radius:
                return True
        return False

    def _build_world(self) -> None:
        if self.obstacle_seed is not None:
            if self._fixed_obstacle_specs is None:
                seeded_rng = np.random.default_rng(self.obstacle_seed)
                built = build_chain_reacher_world(
                    n_links=self.n_links,
                    n_obs=self.n_obs,
                    link_length=self.link_length,
                    arena_radius=self.arena_radius,
                    obstacle_size_range=self.obstacle_size_range,
                    max_torque=self.max_torque,
                    obstacle_rng=seeded_rng,
                    target_position=None,
                    obstacle_specs=None,
                )
                self._fixed_obstacle_specs = list(built["obstacle_specs"])
                self.arena_radius = float(built["arena_radius"])
            obstacle_specs = self._fixed_obstacle_specs
        else:
            obstacle_specs = None

        built = build_chain_reacher_world(
            n_links=self.n_links,
            n_obs=self.n_obs,
            link_length=self.link_length,
            arena_radius=self.arena_radius,
            obstacle_size_range=self.obstacle_size_range,
            max_torque=self.max_torque,
            obstacle_rng=self.np_random,
            target_position=None,
            obstacle_specs=obstacle_specs,
        )
        self.world = built["world"]
        self.world.contactListener = self._contact_listener
        self.anchor = built["anchor"]
        self.links = built["links"]
        self.joints = built["joints"]
        self.obstacles = built["obstacles"]
        self._obstacle_specs = list(built["obstacle_specs"])
        self.arena_radius = float(built["arena_radius"])

    def _get_obs(self) -> np.ndarray:
        tip_position, tip_angle = self._tip_state()
        self._tip_position = (float(tip_position[0]), float(tip_position[1]))
        self._tip_angle = tip_angle
        delta = self.target_position - tip_position

        joint_angles = np.zeros((self.n_links,), dtype=np.float32)
        joint_speeds = np.zeros((self.n_links,), dtype=np.float32)
        for idx, joint in enumerate(self.joints):
            joint_angles[idx] = float(joint.angle)
            joint_speeds[idx] = float(joint.speed)

        self._last_lidar = cast_lidar(
            world=self.world,
            tip_position=self._tip_position,
            tip_angle=tip_angle,
            n_rays=self.n_lidar_rays,
            max_range=self.lidar_range,
        ).astype(np.float32)

        distance = float(np.linalg.norm(delta))
        obs = np.concatenate(
            [
                joint_angles,
                joint_speeds,
                self.target_position.astype(np.float32),
                np.asarray([distance], dtype=np.float32),
                self._last_lidar,
            ]
        ).astype(np.float32)
        return obs

    def _get_info(self, distance_to_target: float) -> dict[str, Any]:
        return {
            "distance_to_target": float(distance_to_target),
            "n_contacts": int(self._contact_listener.n_contacts),
            "target_position": self.target_position.copy(),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options

        self._build_world()
        self.target_position = self._sample_target()

        self.elapsed_steps = 0
        obs = self._get_obs()
        self.prev_distance = float(obs[2 * self.n_links + 2])
        self.prev_action.fill(0.0)
        self._contact_listener.begin_step()
        return obs, self._get_info(distance_to_target=self.prev_distance)

    def step(self, action: np.ndarray):
        if self.world is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.n_links,):
            raise ValueError(f"action shape mismatch: expected {(self.n_links,)}, got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        for idx, joint in enumerate(self.joints):
            joint.motorSpeed = float(action[idx] * self.max_joint_speed)

        self._contact_listener.begin_step()
        self.world.Step(self.dt, self.velocity_iterations, self.position_iterations)
        self.elapsed_steps += 1

        obs = self._get_obs()
        distance_to_target = float(obs[2 * self.n_links + 2])
        # Main progress term.
        delta_reward = self.prev_distance - distance_to_target
        # Smooth absolute-distance term gated to be active near the target.
        near_scale = max(self.target_threshold * 2.0, 1e-6)
        near_gate = float(np.exp(-((distance_to_target / near_scale) ** 2)))
        near_target_reward = near_gate * (1.0 / (1.0 + distance_to_target))
        # Smoothness reward: favors smaller action changes between steps.
        action_delta = (action - self.prev_action) * 0.5
        smoothness_reward = 1.0 - float(np.mean(action_delta**2))

        reward = delta_reward + 0.2 * near_target_reward + 0.02 * smoothness_reward

        terminated = False
        if self._is_out_of_arena():
            terminated = True

        truncated = self.elapsed_steps >= self.max_episode_steps
        self.prev_distance = distance_to_target
        self.prev_action = action.copy()
        info = self._get_info(distance_to_target=distance_to_target)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = ChainReacherRenderer(arena_radius=self.arena_radius, render_mode=self.render_mode)
        return self._renderer.render(
            world=self.world,
            links=self.links,
            anchor=self.anchor,
            target_position=self.target_position,
            lidar_fractions=self._last_lidar,
            lidar_range=self.lidar_range,
            tip_position=self._tip_position,
            tip_angle=self._tip_angle,
        )

    def render_from_state(self, observation: np.ndarray):
        """Reconstruct link poses and target position from observation and render."""
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        min_dim = 2 * self.n_links + 2
        if obs.shape[0] < min_dim:
            raise ValueError(f"observation too short: need at least {min_dim}, got {obs.shape[0]}")

        # Ensure a world exists to draw into.
        if self.world is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        angle_slice = obs[: self.n_links]
        target_xy = obs[2 * self.n_links : 2 * self.n_links + 2]

        # Reconstruct link transforms from joint relative angles only.
        anchor_x, anchor_y = 0.0, 0.0
        cumulative_angle = 0.0
        for idx, link in enumerate(self.links):
            cumulative_angle += float(angle_slice[idx])

            ux = math.cos(cumulative_angle)
            uy = math.sin(cumulative_angle)
            center_x = anchor_x + 0.5 * self.link_length * ux
            center_y = anchor_y + 0.5 * self.link_length * uy

            link.position = (center_x, center_y)
            link.angle = cumulative_angle
            link.linearVelocity = (0.0, 0.0)
            link.angularVelocity = 0.0

            anchor_x += self.link_length * ux
            anchor_y += self.link_length * uy

        tip_position, tip_angle = self._tip_state()
        self._tip_position = (float(tip_position[0]), float(tip_position[1]))
        self._tip_angle = float(tip_angle)
        self.target_position = np.asarray((float(target_xy[0]), float(target_xy[1])), dtype=np.float32)
        return self.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


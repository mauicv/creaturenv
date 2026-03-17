"""Gymnasium Box2D environment for a parameterized multi-legged thruster swimmer."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from Box2D import b2ContactListener
from gymnasium import spaces

from .lidar import cast_lidar
from .renderer import PygameRenderer
from .swimmer_builder import build_swimmer


class ObstacleContactListener(b2ContactListener):
    """Tracks whether swimmer-obstacle contact happened in the current step."""

    def __init__(self) -> None:
        super().__init__()
        self.had_obstacle_contact = False

    def begin_step(self) -> None:
        self.had_obstacle_contact = False

    def BeginContact(self, contact):  # noqa: N802
        body_a = contact.fixtureA.body
        body_b = contact.fixtureB.body
        data_a = body_a.userData or {}
        data_b = body_b.userData or {}
        entities = {data_a.get("entity"), data_b.get("entity")}
        if "swimmer" in entities and "obstacle" in entities:
            self.had_obstacle_contact = True


class SwimmerNavigationEnv(gym.Env):
    """2D Box2D navigation task with articulated legs and per-tip thrusters."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        leg_spec: list[int],
        num_obstacles: int = 0,
        arena_size: float = 20.0,
        n_lidar_rays: int = 16,
        lidar_range: float = 10.0,
        obstacle_seed: int | None = None,
        damping: float = 0.1,
        fluid_friction: float = 0.0,
        max_thrust: float = 20.0,
        max_joint_torque: float = 30.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not leg_spec:
            raise ValueError("leg_spec must not be empty")
        if any(link_count <= 0 for link_count in leg_spec):
            raise ValueError("all link counts in leg_spec must be positive")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.leg_spec = list(leg_spec)
        self.num_obstacles = int(num_obstacles)
        self.arena_size = float(arena_size)
        self.n_lidar_rays = int(n_lidar_rays)
        self.lidar_range = float(lidar_range)
        self.obstacle_seed = obstacle_seed
        self.damping = float(damping)
        self.fluid_friction = float(fluid_friction)
        self.max_thrust = float(max_thrust)
        self.max_joint_torque = float(max_joint_torque)
        self.render_mode = render_mode

        self.dt = 1.0 / 60.0
        self.velocity_iterations = 8
        self.position_iterations = 3
        self.joint_speed_limit = 5.0
        self.target_reached_threshold = 0.8
        self.collision_penalty = 1.0
        self.success_bonus = 100.0
        self.last_thruster_action = np.zeros((len(self.leg_spec),), dtype=np.float32)

        self.n_legs = len(self.leg_spec)
        self.n_joints = int(sum(self.leg_spec))
        self.action_dim = self.n_legs + self.n_joints
        self.obs_dim = 9 + 2 * self.n_joints + self.n_lidar_rays

        action_low = np.concatenate(
            [
                np.zeros((self.n_legs,), dtype=np.float32),
                -np.ones((self.n_joints,), dtype=np.float32),
            ]
        )
        action_high = np.concatenate(
            [
                np.ones((self.n_legs,), dtype=np.float32),
                np.ones((self.n_joints,), dtype=np.float32),
            ]
        )
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.world = None
        self.central_body = None
        self.joints = []
        self.leg_tips = []
        self.swimmer_bodies = []
        self.obstacle_bodies = []
        self.target_position = np.zeros((2,), dtype=np.float32)
        self.prev_distance = 0.0
        self.elapsed_steps = 0
        self._obstacle_specs: list[tuple[float, float, float, float]] = []
        self._last_lidar = np.ones((self.n_lidar_rays,), dtype=np.float32)
        self._contact_listener = ObstacleContactListener()
        self._renderer = None

    def _sample_start_pose(self) -> tuple[tuple[float, float], float]:
        start_xy = self.np_random.uniform(-1.5, 1.5, size=(2,))
        start_angle = float(self.np_random.uniform(-math.pi, math.pi))
        return (float(start_xy[0]), float(start_xy[1])), start_angle

    def _circle_clear_of_obstacles(
        self, x: float, y: float, radius: float, obstacle_specs: list[tuple[float, float, float, float]]
    ) -> bool:
        for ox, oy, hx, hy in obstacle_specs:
            dx = abs(x - ox)
            dy = abs(y - oy)
            if dx <= (hx + radius) and dy <= (hy + radius):
                return False
        return True

    def _sample_obstacles(
        self,
        start_position: tuple[float, float],
        obstacle_rng: np.random.Generator,
    ) -> list[tuple[float, float, float, float]]:
        specs: list[tuple[float, float, float, float]] = []
        if self.num_obstacles <= 0:
            return specs

        half = self.arena_size * 0.5
        for _ in range(self.num_obstacles):
            placed = False
            for _attempt in range(300):
                hx = float(obstacle_rng.uniform(0.5, 1.3))
                hy = float(obstacle_rng.uniform(0.5, 1.3))
                x = float(obstacle_rng.uniform(-half + hx + 1.0, half - hx - 1.0))
                y = float(obstacle_rng.uniform(-half + hy + 1.0, half - hy - 1.0))
                if math.dist((x, y), start_position) < 3.0:
                    continue
                overlaps = False
                for ox, oy, ohx, ohy in specs:
                    if abs(x - ox) <= (hx + ohx + 0.7) and abs(y - oy) <= (hy + ohy + 0.7):
                        overlaps = True
                        break
                if overlaps:
                    continue
                specs.append((x, y, hx, hy))
                placed = True
                break
            if not placed:
                break
        return specs

    def _sample_target(self, obstacle_specs: list[tuple[float, float, float, float]], start_position) -> np.ndarray:
        half = self.arena_size * 0.5
        for _attempt in range(500):
            x = float(self.np_random.uniform(-half + 1.0, half - 1.0))
            y = float(self.np_random.uniform(-half + 1.0, half - 1.0))
            if math.dist((x, y), start_position) < 3.0:
                continue
            if not self._circle_clear_of_obstacles(x, y, self.target_reached_threshold + 0.5, obstacle_specs):
                continue
            return np.asarray((x, y), dtype=np.float32)
        return np.asarray((half - 2.0, half - 2.0), dtype=np.float32)

    def _build_world(self, start_position: tuple[float, float], start_angle: float) -> None:
        if self.obstacle_seed is None:
            obstacle_rng = self.np_random
        else:
            obstacle_rng = np.random.default_rng(self.obstacle_seed)
        self._obstacle_specs = self._sample_obstacles(start_position, obstacle_rng)

        built = build_swimmer(
            leg_spec=self.leg_spec,
            start_position=start_position,
            start_angle=start_angle,
            arena_size=self.arena_size,
            damping=self.damping,
            max_joint_torque=self.max_joint_torque,
            obstacle_specs=self._obstacle_specs,
        )
        self.world = built["world"]
        self.world.contactListener = self._contact_listener
        self.central_body = built["central_body"]
        self.joints = built["joints"]
        self.leg_tips = built["leg_tips"]
        self.swimmer_bodies = built["swimmer_bodies"]
        self.obstacle_bodies = built["obstacles"]

    def _body_frame_vector(self, world_vec: np.ndarray) -> np.ndarray:
        c = math.cos(-self.central_body.angle)
        s = math.sin(-self.central_body.angle)
        return np.asarray(
            [
                c * world_vec[0] - s * world_vec[1],
                s * world_vec[0] + c * world_vec[1],
            ],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        pos = np.asarray((self.central_body.position[0], self.central_body.position[1]), dtype=np.float32)
        vel = np.asarray((self.central_body.linearVelocity[0], self.central_body.linearVelocity[1]), dtype=np.float32)
        body_angle = float(self.central_body.angle)
        body_ang_vel = float(self.central_body.angularVelocity)

        to_target_world = self.target_position - pos
        to_target_local = self._body_frame_vector(to_target_world)
        distance_to_target = float(np.linalg.norm(to_target_world))

        joint_features = np.zeros((2 * self.n_joints,), dtype=np.float32)
        for idx, joint in enumerate(self.joints):
            joint_features[2 * idx] = float(joint.angle)
            joint_features[2 * idx + 1] = float(joint.speed)

        self._last_lidar = cast_lidar(
            world=self.world,
            origin=(pos[0], pos[1]),
            body_angle=body_angle,
            n_rays=self.n_lidar_rays,
            max_range=self.lidar_range,
        ).astype(np.float32)

        obs = np.concatenate(
            [
                pos,
                vel,
                np.asarray([body_angle, body_ang_vel], dtype=np.float32),
                to_target_local,
                np.asarray([distance_to_target], dtype=np.float32),
                joint_features,
                self._last_lidar,
            ]
        ).astype(np.float32)
        return obs

    def _compute_reward(self, action: np.ndarray, distance: float) -> float:
        del action
        # "Just move closer" shaping: positive when reducing distance to target.
        return 10.0 * (self.prev_distance - distance)

    def _is_out_of_bounds(self) -> bool:
        half = self.arena_size * 0.5
        x, y = self.central_body.position
        return abs(x) > half or abs(y) > half

    def _get_info(self, distance_to_target: float) -> dict[str, Any]:
        return {
            "distance_to_target": float(distance_to_target),
            "obstacle_contact": bool(self._contact_listener.had_obstacle_contact),
            "target_position": self.target_position.copy(),
        }

    def _apply_medium_resistance(self) -> None:
        """Apply linear and angular drag, approximating fluid-like resistance."""
        if self.fluid_friction <= 0.0:
            return

        # Angular drag is lower than linear drag so links still articulate.
        angular_drag_coeff = 0.25 * self.fluid_friction
        for body in self.swimmer_bodies:
            vx, vy = body.linearVelocity
            linear_drag = (-self.fluid_friction * vx, -self.fluid_friction * vy)
            body.ApplyForceToCenter(linear_drag, wake=True)
            body.ApplyTorque(-angular_drag_coeff * body.angularVelocity, wake=True)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options

        start_position, start_angle = self._sample_start_pose()
        self._build_world(start_position=start_position, start_angle=start_angle)
        self.target_position = self._sample_target(self._obstacle_specs, start_position=start_position)
        self.elapsed_steps = 0
        self.last_thruster_action.fill(0.0)

        initial_distance = float(
            np.linalg.norm(
                self.target_position
                - np.asarray((self.central_body.position[0], self.central_body.position[1]), dtype=np.float32)
            )
        )
        self.prev_distance = initial_distance
        self._contact_listener.begin_step()

        obs = self._get_obs()
        info = self._get_info(distance_to_target=initial_distance)
        return obs, info

    def step(self, action: np.ndarray):
        if self.world is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"action shape mismatch: expected {self.action_dim}, got {action.shape[0]}")

        thruster_action = np.clip(action[: self.n_legs], 0.0, 1.0)
        joint_action = np.clip(action[self.n_legs :], -1.0, 1.0)
        self._contact_listener.begin_step()

        for idx, joint in enumerate(self.joints):
            joint.motorSpeed = float(joint_action[idx] * self.joint_speed_limit)

        for leg_idx, tip_body in enumerate(self.leg_tips):
            thrust_level = float(thruster_action[leg_idx])
            if thrust_level <= 1e-6:
                continue
            angle = float(tip_body.angle)
            # Link local +x is the "pointing outward" axis.
            thrust_direction = np.asarray((-math.cos(angle), -math.sin(angle)), dtype=np.float32)
            force = thrust_direction * (thrust_level * self.max_thrust)
            tip_body.ApplyForceToCenter((float(force[0]), float(force[1])), wake=True)

        self._apply_medium_resistance()
        self.world.Step(self.dt, self.velocity_iterations, self.position_iterations)
        self.elapsed_steps += 1
        self.last_thruster_action = thruster_action.astype(np.float32)

        obs = self._get_obs()
        distance_to_target = float(obs[8])
        reward = self._compute_reward(action=action, distance=distance_to_target)

        terminated = False
        if distance_to_target < self.target_reached_threshold:
            reward += self.success_bonus
            terminated = True
        elif self._is_out_of_bounds():
            terminated = True

        self.prev_distance = distance_to_target
        truncated = False
        info = self._get_info(distance_to_target=distance_to_target)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = PygameRenderer(arena_size=self.arena_size, render_mode=self.render_mode)
        return self._renderer.render(
            world=self.world,
            central_body=self.central_body,
            target_position=self.target_position,
            lidar_fractions=self._last_lidar,
            lidar_range=self.lidar_range,
            leg_tips=self.leg_tips,
            thruster_levels=self.last_thruster_action,
            max_thrust=self.max_thrust,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


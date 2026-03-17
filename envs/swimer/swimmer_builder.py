"""Box2D world and swimmer construction helpers."""

from __future__ import annotations

import math
from typing import Iterable

from Box2D import b2EdgeShape, b2PolygonShape, b2World


def build_swimmer(
    leg_spec: list[int],
    start_position: tuple[float, float],
    start_angle: float,
    arena_size: float,
    damping: float,
    max_joint_torque: float,
    obstacle_specs: Iterable[tuple[float, float, float, float]] = (),
    body_radius: float = 0.5,
    link_length: float = 0.8,
    link_width: float = 0.2,
    body_density: float = 6.0,
    link_density: float = 0.2,
) -> dict:
    """Build a new zero-gravity world with arena, swimmer, and static obstacles."""
    if not leg_spec:
        raise ValueError("leg_spec must contain at least one leg")
    if any(link_count <= 0 for link_count in leg_spec):
        raise ValueError("each leg in leg_spec must have at least one link")

    world = b2World(gravity=(0.0, 0.0), doSleep=True)
    half = arena_size * 0.5

    wall_body = world.CreateStaticBody(userData={"entity": "arena", "part": "wall"})
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    walls = []
    for idx in range(4):
        a = corners[idx]
        b = corners[(idx + 1) % 4]
        wall = wall_body.CreateFixture(
            shape=b2EdgeShape(vertices=[a, b]),
            density=0.0,
            friction=0.2,
            restitution=0.0,
        )
        walls.append(wall)

    central_body = world.CreateDynamicBody(
        position=start_position,
        angle=start_angle,
        linearDamping=damping,
        angularDamping=damping,
        userData={"entity": "swimmer", "part": "central"},
    )
    central_body.CreateCircleFixture(
        radius=body_radius,
        density=body_density,
        friction=0.5,
        restitution=0.0,
    )

    n_legs = len(leg_spec)
    joints = []
    leg_links = []
    leg_tips = []
    swimmer_bodies = [central_body]

    def _create_leg_link(leg_idx: int, link_idx: int, center: tuple[float, float], angle: float):
        link = world.CreateDynamicBody(
            position=center,
            angle=angle,
            linearDamping=damping,
            angularDamping=damping,
            userData={
                "entity": "swimmer",
                "part": "link",
                "leg_index": leg_idx,
                "link_index": link_idx,
            },
        )
        link.CreatePolygonFixture(
            shape=b2PolygonShape(box=(link_length * 0.5, link_width * 0.5)),
            density=link_density,
            friction=0.6,
            restitution=0.0,
        )
        return link

    for leg_idx, link_count in enumerate(leg_spec):
        leg_offset = 2.0 * math.pi * leg_idx / n_legs
        leg_angle = start_angle + leg_offset
        # Anchor offsets passed to GetWorldPoint are in central-body local frame.
        unit_local = (math.cos(leg_offset), math.sin(leg_offset))
        # Link placement/orientation is in world frame.
        unit_world = (math.cos(leg_angle), math.sin(leg_angle))
        links_this_leg = []

        # Handle first link separately: it connects at the central body boundary.
        first_anchor = central_body.GetWorldPoint((body_radius * unit_local[0], body_radius * unit_local[1]))
        first_center = (
            first_anchor[0] + unit_world[0] * (link_length * 0.5),
            first_anchor[1] + unit_world[1] * (link_length * 0.5),
        )
        first_link = _create_leg_link(leg_idx=leg_idx, link_idx=0, center=first_center, angle=leg_angle)
        first_joint = world.CreateRevoluteJoint(
            bodyA=central_body,
            bodyB=first_link,
            anchor=first_anchor,
            enableMotor=True,
            motorSpeed=0.0,
            maxMotorTorque=max_joint_torque,
            enableLimit=True,
            lowerAngle=-math.pi + 0.05,
            upperAngle=math.pi - 0.05,
            collideConnected=False,
        )
        joints.append(first_joint)
        links_this_leg.append(first_link)
        swimmer_bodies.append(first_link)

        prev_body = first_link
        prev_anchor = first_link.GetWorldPoint((link_length * 0.5, 0.0))

        # Subsequent links connect to the previous link tip.
        for link_idx in range(1, link_count):
            center = (
                prev_anchor[0] + unit_world[0] * (link_length * 0.5),
                prev_anchor[1] + unit_world[1] * (link_length * 0.5),
            )
            link = _create_leg_link(leg_idx=leg_idx, link_idx=link_idx, center=center, angle=leg_angle)
            joint = world.CreateRevoluteJoint(
                bodyA=prev_body,
                bodyB=link,
                anchor=prev_anchor,
                enableMotor=True,
                motorSpeed=0.0,
                maxMotorTorque=max_joint_torque,
                enableLimit=True,
                lowerAngle=-math.pi + 0.05,
                upperAngle=math.pi - 0.05,
                collideConnected=False,
            )
            joints.append(joint)
            links_this_leg.append(link)
            swimmer_bodies.append(link)

            prev_body = link
            prev_anchor = link.GetWorldPoint((link_length * 0.5, 0.0))

        leg_links.append(links_this_leg)
        leg_tips.append(links_this_leg[-1])

    obstacle_bodies = []
    for obs_idx, (x, y, hx, hy) in enumerate(obstacle_specs):
        obstacle = world.CreateStaticBody(
            position=(x, y),
            userData={"entity": "obstacle", "index": obs_idx},
        )
        obstacle.CreatePolygonFixture(
            shape=b2PolygonShape(box=(hx, hy)),
            density=0.0,
            friction=0.8,
            restitution=0.0,
        )
        obstacle_bodies.append(obstacle)

    return {
        "world": world,
        "central_body": central_body,
        "joints": joints,
        "leg_links": leg_links,
        "leg_tips": leg_tips,
        "obstacles": obstacle_bodies,
        "swimmer_bodies": swimmer_bodies,
        "wall_body": wall_body,
        "walls": walls,
        "link_length": link_length,
        "link_width": link_width,
        "body_radius": body_radius,
    }


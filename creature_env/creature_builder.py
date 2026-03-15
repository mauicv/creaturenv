"""Box2D world and creature construction helpers."""

from __future__ import annotations

import math
from typing import Iterable

from Box2D import b2EdgeShape, b2PolygonShape, b2World


def build_creature(
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
    """Build a new zero-gravity world with arena, creature, and static obstacles."""
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
        userData={"entity": "creature", "part": "central"},
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
    creature_bodies = [central_body]

    for leg_idx, link_count in enumerate(leg_spec):
        leg_angle = start_angle + (2.0 * math.pi * leg_idx / n_legs)
        unit = (math.cos(leg_angle), math.sin(leg_angle))
        prev_body = central_body
        # First link now anchors at the creature center (not the body perimeter).
        prev_anchor = central_body.GetWorldPoint((0.0, 0.0))
        links_this_leg = []

        for link_idx in range(link_count):
            center = (
                prev_anchor[0] + unit[0] * (link_length * 0.5),
                prev_anchor[1] + unit[1] * (link_length * 0.5),
            )
            link = world.CreateDynamicBody(
                position=center,
                angle=leg_angle,
                linearDamping=damping,
                angularDamping=damping,
                userData={
                    "entity": "creature",
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
            joint = world.CreateRevoluteJoint(
                bodyA=prev_body,
                bodyB=link,
                anchor=prev_anchor,
                enableMotor=True,
                motorSpeed=0.0,
                maxMotorTorque=max_joint_torque,
                enableLimit=False,
                lowerAngle=-math.pi * 0.75,
                upperAngle=math.pi * 0.75,
                collideConnected=False,
            )
            joints.append(joint)
            links_this_leg.append(link)
            creature_bodies.append(link)

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
        "creature_bodies": creature_bodies,
        "wall_body": wall_body,
        "walls": walls,
        "link_length": link_length,
        "link_width": link_width,
        "body_radius": body_radius,
    }


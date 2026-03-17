"""Box2D world construction for ChainReacher."""

from __future__ import annotations

import math
import warnings

from Box2D import b2EdgeShape, b2PolygonShape, b2World


def _segment_intersects_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    cx: float,
    cy: float,
    hx: float,
    hy: float,
) -> bool:
    min_x = min(x0, x1) - hx
    max_x = max(x0, x1) + hx
    min_y = min(y0, y1) - hy
    max_y = max(y0, y1) + hy
    return (min_x <= cx <= max_x) and (min_y <= cy <= max_y)


def _sample_obstacles(
    *,
    rng,
    n_obs: int,
    size_range: tuple[float, float],
    base_arena_radius: float,
    link_length: float,
    n_links: int,
    target_position: tuple[float, float] | None = None,
) -> tuple[list[tuple[float, float, float, float]], float]:
    if n_obs <= 0:
        return [], float(base_arena_radius)

    attempts_per_obstacle = 400
    arena_growth = 1.15
    max_growth_rounds = 5
    current_radius = float(base_arena_radius)

    for _growth_round in range(max_growth_rounds):
        specs: list[tuple[float, float, float, float]] = []
        failed = False
        for _ in range(n_obs):
            placed = False
            for _attempt in range(attempts_per_obstacle):
                half_extent = float(rng.uniform(size_range[0], size_range[1]))
                hx, hy = half_extent, half_extent

                margin = max(hx, hy) + 0.2
                x = float(rng.uniform(-current_radius + margin, current_radius - margin))
                y = float(rng.uniform(-current_radius + margin, current_radius - margin))

                if math.hypot(x, y) > (current_radius - margin):
                    continue
                if math.hypot(x, y) < link_length:
                    continue

                overlaps = False
                for ox, oy, ohx, ohy in specs:
                    if abs(x - ox) <= (hx + ohx + 0.1) and abs(y - oy) <= (hy + ohy + 0.1):
                        overlaps = True
                        break
                if overlaps:
                    continue

                chain_blocked = False
                for seg_idx in range(n_links):
                    x0 = seg_idx * link_length
                    x1 = (seg_idx + 1) * link_length
                    if _segment_intersects_rect(x0, 0.0, x1, 0.0, x, y, hx + 0.12, hy + 0.12):
                        chain_blocked = True
                        break
                if chain_blocked:
                    continue

                if target_position is not None:
                    tx, ty = target_position
                    if abs(x - tx) <= (hx + 0.35) and abs(y - ty) <= (hy + 0.35):
                        continue

                specs.append((x, y, hx, hy))
                placed = True
                break

            if not placed:
                failed = True
                break

        if not failed:
            return specs, current_radius
        current_radius *= arena_growth

    warnings.warn(
        "Obstacle placement was difficult; using partial placement and expanded arena.",
        RuntimeWarning,
        stacklevel=2,
    )
    return specs, current_radius


def build_chain_reacher_world(
    *,
    n_links: int,
    n_obs: int,
    link_length: float,
    arena_radius: float,
    obstacle_size_range: tuple[float, float],
    max_torque: float,
    obstacle_rng,
    target_position: tuple[float, float] | None = None,
    obstacle_specs: list[tuple[float, float, float, float]] | None = None,
) -> dict:
    """Build Box2D world, arena, anchored chain, and obstacles."""
    world = b2World(gravity=(0.0, 0.0), doSleep=True)

    anchor = world.CreateStaticBody(
        position=(0.0, 0.0),
        userData={"entity": "anchor"},
    )
    anchor.CreateCircleFixture(radius=0.06, density=0.0, friction=0.0, restitution=0.0)

    arena_body = world.CreateStaticBody(userData={"entity": "arena"})
    arena_segments = []
    segment_count = 64
    for idx in range(segment_count):
        a0 = 2.0 * math.pi * idx / segment_count
        a1 = 2.0 * math.pi * (idx + 1) / segment_count
        p0 = (arena_radius * math.cos(a0), arena_radius * math.sin(a0))
        p1 = (arena_radius * math.cos(a1), arena_radius * math.sin(a1))
        seg = arena_body.CreateFixture(
            shape=b2EdgeShape(vertices=[p0, p1]),
            density=0.0,
            friction=0.2,
            restitution=0.0,
        )
        arena_segments.append(seg)

    link_width = 0.1
    links = []
    joints = []
    prev_body = anchor
    prev_anchor = (0.0, 0.0)

    for link_idx in range(n_links):
        center = ((link_idx + 0.5) * link_length, 0.0)
        link = world.CreateDynamicBody(
            position=center,
            angle=0.0,
            linearDamping=0.15,
            angularDamping=0.15,
            userData={"entity": "chain", "part": "link", "link_index": link_idx},
        )
        link.CreatePolygonFixture(
            shape=b2PolygonShape(box=(0.5 * link_length, 0.5 * link_width)),
            density=1.0,
            friction=0.6,
            restitution=0.0,
        )

        joint = world.CreateRevoluteJoint(
            bodyA=prev_body,
            bodyB=link,
            anchor=prev_anchor,
            enableMotor=True,
            motorSpeed=0.0,
            maxMotorTorque=max_torque,
            enableLimit=False,
            collideConnected=False,
        )

        joints.append(joint)
        links.append(link)
        prev_body = link
        prev_anchor = (float((link_idx + 1) * link_length), 0.0)

    if obstacle_specs is None:
        specs, grown_radius = _sample_obstacles(
            rng=obstacle_rng,
            n_obs=n_obs,
            size_range=obstacle_size_range,
            base_arena_radius=arena_radius,
            link_length=link_length,
            n_links=n_links,
            target_position=target_position,
        )
        obstacle_specs = specs
        arena_radius = grown_radius
    else:
        obstacle_specs = list(obstacle_specs)

    obstacle_bodies = []
    for obs_idx, (x, y, hx, hy) in enumerate(obstacle_specs):
        obs = world.CreateStaticBody(
            position=(x, y),
            userData={"entity": "obstacle", "index": obs_idx},
        )
        obs.CreatePolygonFixture(
            shape=b2PolygonShape(box=(hx, hy)),
            density=0.0,
            friction=0.8,
            restitution=0.0,
        )
        obstacle_bodies.append(obs)

    return {
        "world": world,
        "anchor": anchor,
        "arena_body": arena_body,
        "arena_segments": arena_segments,
        "links": links,
        "joints": joints,
        "obstacles": obstacle_bodies,
        "obstacle_specs": obstacle_specs,
        "arena_radius": float(arena_radius),
        "link_width": link_width,
    }


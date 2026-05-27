"""Behavior functions — flocking, swarming, orbiting, avoidance."""

from __future__ import annotations

from typing import List, Sequence

from .vector import Vector2D
from .particle import Particle2D


def attract(agent: Particle2D, target: Vector2D, strength: float = 1.0) -> Vector2D:
    """Linear attraction toward a target point."""
    diff = target - agent.position
    dist = diff.magnitude
    if dist == 0:
        return Vector2D()
    return diff.normalized() * strength


def repel(agent: Particle2D, threat: Vector2D, strength: float = 1.0, radius: float = 100.0) -> Vector2D:
    """Inverse-distance repulsion from a point, zero outside radius."""
    diff = agent.position - threat
    dist = diff.magnitude
    if dist >= radius or dist == 0:
        return Vector2D()
    force = diff.normalized() * (strength / dist)
    return force


def align(agent: Particle2D, neighbors: Sequence[Particle2D], radius: float = 50.0) -> Vector2D:
    """Alignment: steer toward average heading of nearby neighbors."""
    if not neighbors:
        return Vector2D()
    total = Vector2D()
    count = 0
    for n in neighbors:
        if n is agent:
            continue
        d = agent.position.distance_to(n.position)
        if 0 < d < radius:
            total = total + n.velocity
            count += 1
    if count == 0:
        return Vector2D()
    avg = total / count
    if avg.magnitude == 0:
        return Vector2D()
    desired = avg.normalized() * agent.max_speed
    steer = desired - agent.velocity
    return steer.limit(agent.max_force)


def cohesion(agent: Particle2D, neighbors: Sequence[Particle2D], radius: float = 50.0) -> Vector2D:
    """Cohesion: steer toward center of mass of nearby neighbors."""
    if not neighbors:
        return Vector2D()
    center = Vector2D()
    count = 0
    for n in neighbors:
        if n is agent:
            continue
        d = agent.position.distance_to(n.position)
        if 0 < d < radius:
            center = center + n.position
            count += 1
    if count == 0:
        return Vector2D()
    center = center / count
    return agent.seek(center)


def separation(agent: Particle2D, neighbors: Sequence[Particle2D], radius: float = 25.0, weight: float = 1.5) -> Vector2D:
    """Separation: steer away from nearby neighbors that are too close."""
    if not neighbors:
        return Vector2D()
    steer = Vector2D()
    count = 0
    for n in neighbors:
        if n is agent:
            continue
        d = agent.position.distance_to(n.position)
        if 0 < d < radius:
            diff = (agent.position - n.position).normalized() / d
            steer = steer + diff
            count += 1
    if count == 0:
        return Vector2D()
    steer = steer / count
    if steer.magnitude > 0:
        steer = steer.normalized() * agent.max_speed - agent.velocity
        steer = steer.limit(agent.max_force)
    return steer * weight


def flocking(
    agent: Particle2D,
    neighbors: Sequence[Particle2D],
    align_weight: float = 1.0,
    cohesion_weight: float = 1.0,
    separation_weight: float = 1.5,
    perception_radius: float = 50.0,
    separation_radius: float = 25.0,
) -> Vector2D:
    """Classic boids flocking: alignment + cohesion + separation."""
    a = align(agent, neighbors, perception_radius) * align_weight
    c = cohesion(agent, neighbors, perception_radius) * cohesion_weight
    s = separation(agent, neighbors, separation_radius) * separation_weight
    return a + c + s


def orbit(agent: Particle2D, center: Vector2D, radius: float = 50.0, speed: float = 1.0) -> Vector2D:
    """Orbital force: tangential + radial spring toward a circular orbit."""
    to_center = center - agent.position
    dist = to_center.magnitude
    if dist == 0:
        return Vector2D()
    # Radial spring: push toward orbit radius
    radial_error = dist - radius
    radial = to_center.normalized() * radial_error * 0.5
    # Tangential: perpendicular to radial
    tangent = Vector2D(-to_center.y, to_center.x).normalized() * speed
    return (radial + tangent).limit(agent.max_force)


def avoidance(
    agent: Particle2D,
    obstacles: Sequence[Vector2D],
    radius: float = 30.0,
    strength: float = 2.0,
) -> Vector2D:
    """Obstacle avoidance: repel from fixed obstacle positions."""
    total = Vector2D()
    for obs in obstacles:
        diff = agent.position - obs
        d = diff.magnitude
        if 0 < d < radius:
            total = total + diff.normalized() * (strength / (d * d))
    return total.limit(agent.max_force)

"""Tests for agent_field.behavior — flocking, orbit, avoidance, etc."""

import math
import pytest
from agent_field.vector import Vector2D
from agent_field.particle import Particle2D
from agent_field.behavior import (
    attract,
    repel,
    align,
    cohesion,
    separation,
    flocking,
    orbit,
    avoidance,
)


class TestAttract:
    def test_attract_toward_target(self):
        p = Particle2D(position=Vector2D(0, 0))
        force = attract(p, Vector2D(10, 0), strength=2.0)
        assert force.x > 0
        assert force.y == pytest.approx(0.0)

    def test_attract_at_target(self):
        p = Particle2D(position=Vector2D(5, 5))
        force = attract(p, Vector2D(5, 5))
        assert force == Vector2D()

    def test_attract_strength(self):
        p = Particle2D(position=Vector2D(0, 0))
        f1 = attract(p, Vector2D(10, 0), strength=1.0)
        f2 = attract(p, Vector2D(10, 0), strength=3.0)
        assert f2.x == pytest.approx(f1.x * 3)


class TestRepel:
    def test_repel_away(self):
        p = Particle2D(position=Vector2D(5, 0))
        force = repel(p, Vector2D(0, 0), strength=1.0, radius=100.0)
        assert force.x > 0

    def test_repel_outside_radius(self):
        p = Particle2D(position=Vector2D(100, 0))
        force = repel(p, Vector2D(0, 0), strength=1.0, radius=10.0)
        assert force == Vector2D()

    def test_repel_at_same_point(self):
        p = Particle2D(position=Vector2D(5, 5))
        force = repel(p, Vector2D(5, 5))
        assert force == Vector2D()


class TestAlign:
    def test_align_with_moving_neighbors(self):
        agent = Particle2D(position=Vector2D(0, 0), max_speed=3.0, max_force=1.0)
        neighbors = [
            Particle2D(position=Vector2D(10, 0), velocity=Vector2D(0, 3), max_speed=3.0, max_force=1.0),
            Particle2D(position=Vector2D(0, 10), velocity=Vector2D(0, 2), max_speed=3.0, max_force=1.0),
        ]
        force = align(agent, neighbors, radius=50.0)
        assert force.y > 0  # should steer upward to match neighbors

    def test_align_no_neighbors(self):
        agent = Particle2D(max_speed=3.0, max_force=1.0)
        assert align(agent, [], radius=50.0) == Vector2D()

    def test_align_out_of_range(self):
        agent = Particle2D(position=Vector2D(0, 0), max_speed=3.0, max_force=1.0)
        neighbors = [
            Particle2D(position=Vector2D(1000, 0), velocity=Vector2D(5, 0), max_speed=3.0, max_force=1.0),
        ]
        assert align(agent, neighbors, radius=50.0) == Vector2D()


class TestCohesion:
    def test_cohesion_toward_center(self):
        agent = Particle2D(position=Vector2D(0, 0), max_speed=3.0, max_force=1.0)
        neighbors = [
            Particle2D(position=Vector2D(20, 0), max_speed=3.0, max_force=1.0),
            Particle2D(position=Vector2D(20, 20), max_speed=3.0, max_force=1.0),
        ]
        force = cohesion(agent, neighbors, radius=50.0)
        assert force.x > 0
        assert force.y > 0

    def test_cohesion_empty(self):
        agent = Particle2D(max_speed=3.0, max_force=1.0)
        assert cohesion(agent, [], radius=50.0) == Vector2D()


class TestSeparation:
    def test_separation_pushes_apart(self):
        agent = Particle2D(position=Vector2D(0, 0), max_speed=3.0, max_force=1.0)
        neighbors = [
            Particle2D(position=Vector2D(1, 0), max_speed=3.0, max_force=1.0),
        ]
        force = separation(agent, neighbors, radius=10.0)
        assert force.x < 0  # pushes left (away from neighbor at right)

    def test_separation_empty(self):
        agent = Particle2D(max_speed=3.0, max_force=1.0)
        assert separation(agent, [], radius=10.0) == Vector2D()


class TestFlocking:
    def test_flocking_returns_vector(self):
        agent = Particle2D(position=Vector2D(0, 0), max_speed=3.0, max_force=1.0)
        neighbors = [
            Particle2D(
                position=Vector2D(10, 0),
                velocity=Vector2D(0, 2),
                max_speed=3.0,
                max_force=1.0,
            ),
        ]
        force = flocking(agent, neighbors, perception_radius=50.0)
        assert isinstance(force, Vector2D)


class TestOrbit:
    def test_orbit_produces_tangent(self):
        agent = Particle2D(position=Vector2D(10, 0), max_speed=3.0, max_force=1.0)
        center = Vector2D(0, 0)
        force = orbit(agent, center, radius=10.0, speed=1.0)
        # Should have tangential component
        assert force.magnitude > 0

    def test_orbit_at_center(self):
        agent = Particle2D(position=Vector2D(0, 0), max_speed=3.0, max_force=1.0)
        force = orbit(agent, Vector2D(0, 0), radius=10.0)
        assert force == Vector2D()


class TestAvoidance:
    def test_avoidance_repels(self):
        agent = Particle2D(position=Vector2D(5, 0), max_speed=3.0, max_force=1.0)
        obstacles = [Vector2D(0, 0)]
        force = avoidance(agent, obstacles, radius=10.0, strength=2.0)
        assert force.x > 0  # push away

    def test_avoidance_no_obstacles(self):
        agent = Particle2D(max_speed=3.0, max_force=1.0)
        assert avoidance(agent, [], radius=10.0) == Vector2D()

    def test_avoidance_distant_obstacle(self):
        agent = Particle2D(position=Vector2D(100, 0), max_speed=3.0, max_force=1.0)
        obstacles = [Vector2D(0, 0)]
        assert avoidance(agent, obstacles, radius=10.0) == Vector2D()

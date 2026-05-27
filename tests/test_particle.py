"""Tests for agent_field.particle — Particle2D and Particle3D."""

import pytest
from agent_field.vector import Vector2D, Vector3D
from agent_field.particle import Particle2D, Particle3D


class TestParticle2D:
    def test_defaults(self):
        p = Particle2D()
        assert p.position == Vector2D()
        assert p.velocity == Vector2D()
        assert p.mass == 1.0

    def test_invalid_mass(self):
        with pytest.raises(ValueError):
            Particle2D(mass=0)
        with pytest.raises(ValueError):
            Particle2D(mass=-1)

    def test_apply_force(self):
        p = Particle2D()
        p.apply_force(Vector2D(2, 0))
        assert p.acceleration.x == pytest.approx(2.0)

    def test_apply_force_with_mass(self):
        p = Particle2D(mass=2.0)
        p.apply_force(Vector2D(4, 0))
        assert p.acceleration.x == pytest.approx(2.0)

    def test_update(self):
        p = Particle2D()
        p.apply_force(Vector2D(1, 0))
        p.update(dt=1.0)
        assert p.velocity.x == pytest.approx(1.0)
        assert p.position.x == pytest.approx(1.0)
        assert p.acceleration.x == pytest.approx(0.0)

    def test_update_resets_acceleration(self):
        p = Particle2D()
        p.apply_force(Vector2D(5, 0))
        p.update()
        assert p.acceleration == Vector2D()

    def test_max_speed(self):
        p = Particle2D(max_speed=2.0)
        p.apply_force(Vector2D(100, 0))
        p.update()
        assert p.velocity.magnitude <= 2.0 + 1e-9

    def test_seek(self):
        p = Particle2D(position=Vector2D(0, 0), max_speed=3.0, max_force=1.0)
        steer = p.seek(Vector2D(10, 0))
        assert steer.x > 0
        assert steer.magnitude <= 1.0 + 1e-9

    def test_seek_at_target(self):
        p = Particle2D(position=Vector2D(5, 5), max_speed=3.0, max_force=1.0)
        steer = p.seek(Vector2D(5, 5))
        assert steer == Vector2D()

    def test_flee(self):
        p = Particle2D(position=Vector2D(5, 0), max_speed=3.0, max_force=1.0)
        steer = p.flee(Vector2D(0, 0), radius=10.0)
        assert steer.x > 0  # moving away from threat

    def test_flee_outside_radius(self):
        p = Particle2D(position=Vector2D(100, 0), max_speed=3.0, max_force=1.0)
        steer = p.flee(Vector2D(0, 0), radius=10.0)
        assert steer == Vector2D()

    def test_flee_same_position(self):
        p = Particle2D(position=Vector2D(5, 5), max_speed=3.0, max_force=1.0)
        steer = p.flee(Vector2D(5, 5))
        assert steer == Vector2D()

    def test_label(self):
        p = Particle2D(label="agent-1")
        assert "agent-1" in repr(p)

    def test_multiple_forces(self):
        p = Particle2D()
        p.apply_force(Vector2D(1, 0))
        p.apply_force(Vector2D(0, 1))
        assert p.acceleration.x == pytest.approx(1.0)
        assert p.acceleration.y == pytest.approx(1.0)


class TestParticle3D:
    def test_defaults(self):
        p = Particle3D()
        assert p.position == Vector3D()
        assert p.mass == 1.0

    def test_invalid_mass(self):
        with pytest.raises(ValueError):
            Particle3D(mass=-0.5)

    def test_apply_force(self):
        p = Particle3D(mass=2.0)
        p.apply_force(Vector3D(4, 0, 0))
        assert p.acceleration.x == pytest.approx(2.0)

    def test_update(self):
        p = Particle3D()
        p.apply_force(Vector3D(0, 0, 3))
        p.update(dt=2.0)
        assert p.velocity.z == pytest.approx(6.0)
        assert p.position.z == pytest.approx(12.0)

    def test_seek(self):
        p = Particle3D(position=Vector3D(0, 0, 0), max_speed=3.0, max_force=1.0)
        steer = p.seek(Vector3D(0, 0, 10))
        assert steer.z > 0

    def test_flee(self):
        p = Particle3D(position=Vector3D(1, 0, 0), max_speed=3.0, max_force=1.0)
        steer = p.flee(Vector3D(0, 0, 0), radius=5.0)
        assert steer.x > 0

"""Tests for agent_field.simulation — Simulation2D."""

import pytest
from agent_field.vector import Vector2D
from agent_field.particle import Particle2D
from agent_field.simulation import Simulation2D
from agent_field.behavior import flocking, attract


class TestSimulation2DCreation:
    def test_empty(self):
        sim = Simulation2D()
        assert len(sim.particles) == 0
        assert sim.step_count == 0

    def test_add_particle(self):
        sim = Simulation2D()
        sim.add_particle(Particle2D())
        assert len(sim.particles) == 1

    def test_remove_particle(self):
        sim = Simulation2D()
        sim.add_particle(Particle2D(label="a"))
        sim.add_particle(Particle2D(label="b"))
        removed = sim.remove_particle(0)
        assert removed.label == "a"
        assert len(sim.particles) == 1


class TestSimulation2DStep:
    def test_step_no_behaviors(self):
        sim = Simulation2D()
        p = Particle2D(velocity=Vector2D(1, 0))
        sim.add_particle(p)
        sim.step()
        assert p.position.x == pytest.approx(1.0)
        assert sim.step_count == 1

    def test_step_with_force(self):
        sim = Simulation2D()
        sim.add_particle(Particle2D(position=Vector2D(0, 0), max_speed=10.0, max_force=5.0))

        def push_right(agent, _):
            return Vector2D(1, 0)

        sim.behaviors.append(push_right)
        sim.step()
        assert sim.particles[0].velocity.x > 0

    def test_multiple_steps(self):
        sim = Simulation2D()
        sim.add_particle(Particle2D(velocity=Vector2D(1, 0)))
        sim.run(5)
        assert sim.step_count == 5
        assert sim.particles[0].position.x == pytest.approx(5.0)


class TestSimulation2DBounds:
    def test_wrap_x(self):
        sim = Simulation2D(bounds=(0, 0, 10, 10))
        p = Particle2D(position=Vector2D(9, 5), velocity=Vector2D(3, 0))
        sim.add_particle(p)
        sim.step()
        assert sim.particles[0].position.x < 5  # wrapped

    def test_wrap_y(self):
        sim = Simulation2D(bounds=(0, 0, 10, 10))
        p = Particle2D(position=Vector2D(5, 1), velocity=Vector2D(0, -3))
        sim.add_particle(p)
        sim.step()
        assert sim.particles[0].position.y > 5  # wrapped up

    def test_no_bounds(self):
        sim = Simulation2D()
        p = Particle2D(position=Vector2D(0, 0), velocity=Vector2D(100, 0))
        sim.add_particle(p)
        sim.step()
        assert sim.particles[0].position.x == pytest.approx(100.0)


class TestSimulation2DAnalysis:
    def test_average_speed_empty(self):
        assert Simulation2D().average_speed() == 0.0

    def test_average_speed(self):
        sim = Simulation2D()
        sim.add_particle(Particle2D(velocity=Vector2D(3, 4)))
        sim.add_particle(Particle2D(velocity=Vector2D(0, 0)))
        assert sim.average_speed() == pytest.approx(2.5)

    def test_center_of_mass_empty(self):
        assert Simulation2D().center_of_mass() == Vector2D()

    def test_center_of_mass(self):
        sim = Simulation2D()
        sim.add_particle(Particle2D(position=Vector2D(0, 0)))
        sim.add_particle(Particle2D(position=Vector2D(10, 0)))
        com = sim.center_of_mass()
        assert com.x == pytest.approx(5.0)

    def test_avg_nn_distance_single(self):
        sim = Simulation2D()
        sim.add_particle(Particle2D())
        assert sim.average_nearest_neighbor_distance() == 0.0

    def test_avg_nn_distance(self):
        sim = Simulation2D()
        sim.add_particle(Particle2D(position=Vector2D(0, 0)))
        sim.add_particle(Particle2D(position=Vector2D(3, 4)))
        assert sim.average_nearest_neighbor_distance() == pytest.approx(5.0)

    def test_repr(self):
        sim = Simulation2D()
        sim.add_particle(Particle2D())
        r = repr(sim)
        assert "Simulation2D" in r


class TestFlockingSimulation:
    def test_flock_converges(self):
        """Particles with flocking behavior should cluster over time."""
        sim = Simulation2D(dt=0.5, bounds=(0, 0, 200, 200))
        for i in range(5):
            angle = i * 2 * 3.14159 / 5
            pos = Vector2D(100 + 40 * math.cos(angle), 100 + 40 * math.sin(angle))
            sim.add_particle(Particle2D(position=pos, max_speed=2.0, max_force=0.5))
        sim.behaviors.append(flocking)

        initial_dist = sim.average_nearest_neighbor_distance()
        sim.run(50)
        final_dist = sim.average_nearest_neighbor_distance()
        # Particles should be closer after flocking
        assert final_dist < initial_dist


import math

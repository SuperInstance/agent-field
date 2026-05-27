"""Simulation engine — timestep integration for collections of particles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .vector import Vector2D
from .particle import Particle2D
from .behavior import flocking


# A behavior function takes (agent, all_particles) → force Vector2D
BehaviorFn = Callable[[Particle2D, Sequence[Particle2D]], Vector2D]


@dataclass
class Simulation2D:
    """2D particle simulation engine.

    Usage:
        sim = Simulation2D()
        for _ in range(10):
            sim.add_particle(Particle2D(max_speed=3.0, max_force=0.5))
        sim.behaviors.append(flocking)
        for _ in range(100):
            sim.step()
    """

    particles: List[Particle2D] = field(default_factory=list)
    behaviors: List[BehaviorFn] = field(default_factory=list)
    dt: float = 1.0
    bounds: Optional[tuple[float, float, float, float]] = None  # (x_min, y_min, x_max, y_max)
    _step_count: int = field(default=0, init=False)

    @property
    def step_count(self) -> int:
        return self._step_count

    def add_particle(self, p: Particle2D) -> None:
        self.particles.append(p)

    def remove_particle(self, index: int) -> Particle2D:
        return self.particles.pop(index)

    def step(self) -> None:
        """Compute forces from all behaviors, then integrate."""
        forces: List[Vector2D] = [Vector2D() for _ in self.particles]

        for behavior in self.behaviors:
            for i, p in enumerate(self.particles):
                f = behavior(p, self.particles)
                forces[i] = forces[i] + f

        for i, p in enumerate(self.particles):
            p.apply_force(forces[i])
            p.update(self.dt)
            if self.bounds is not None:
                self._wrap(p)

        self._step_count += 1

    def run(self, steps: int) -> None:
        """Run simulation for a given number of steps."""
        for _ in range(steps):
            self.step()

    def _wrap(self, p: Particle2D) -> None:
        """Wrap position within bounds (toroidal)."""
        if self.bounds is None:
            return
        x_min, y_min, x_max, y_max = self.bounds
        x, y = p.position.x, p.position.y
        w, h = x_max - x_min, y_max - y_min
        if w <= 0 or h <= 0:
            return
        if x < x_min:
            x += w
        elif x > x_max:
            x -= w
        if y < y_min:
            y += h
        elif y > y_max:
            y -= h
        p.position = Vector2D(x, y)

    def average_speed(self) -> float:
        if not self.particles:
            return 0.0
        return sum(p.velocity.magnitude for p in self.particles) / len(self.particles)

    def center_of_mass(self) -> Vector2D:
        if not self.particles:
            return Vector2D()
        total = Vector2D()
        for p in self.particles:
            total = total + p.position
        return total / len(self.particles)

    def average_nearest_neighbor_distance(self) -> float:
        if len(self.particles) < 2:
            return 0.0
        total = 0.0
        for p in self.particles:
            min_dist = float("inf")
            for q in self.particles:
                if q is p:
                    continue
                d = p.position.distance_to(q.position)
                if d < min_dist:
                    min_dist = d
            total += min_dist
        return total / len(self.particles)

    def __repr__(self) -> str:
        return f"Simulation2D(particles={len(self.particles)}, steps={self._step_count})"

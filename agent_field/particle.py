"""Particle — an agent with position, velocity, mass, and forces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .vector import Vector2D, Vector3D


@dataclass
class Particle2D:
    """A 2D particle with position, velocity, acceleration, and mass."""

    position: Vector2D = field(default_factory=Vector2D)
    velocity: Vector2D = field(default_factory=Vector2D)
    acceleration: Vector2D = field(default_factory=Vector2D)
    mass: float = 1.0
    max_speed: float = float("inf")
    max_force: float = float("inf")
    label: str = ""

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError(f"mass must be positive, got {self.mass}")

    def apply_force(self, force: Vector2D) -> None:
        """Accumulate a force (F = ma → a += F/m)."""
        self.acceleration = self.acceleration + force / self.mass

    def update(self, dt: float = 1.0) -> None:
        """Integrate: velocity += acceleration * dt, position += velocity * dt."""
        self.velocity = (self.velocity + self.acceleration * dt).limit(self.max_speed)
        self.position = self.position + self.velocity * dt
        self.acceleration = Vector2D()

    def seek(self, target: Vector2D) -> Vector2D:
        """Steering force toward a target point."""
        desired = target - self.position
        desired = desired.normalized() * self.max_speed if desired.magnitude > 0 else desired
        steer = desired - self.velocity
        return steer.limit(self.max_force)

    def flee(self, threat: Vector2D, radius: float = float("inf")) -> Vector2D:
        """Steering force away from a threat. Zero if outside radius."""
        diff = self.position - threat
        dist = diff.magnitude
        if dist > radius or dist == 0:
            return Vector2D()
        desired = diff.normalized() * self.max_speed
        steer = desired - self.velocity
        return steer.limit(self.max_force)

    def __repr__(self) -> str:
        lbl = f" {self.label!r}" if self.label else ""
        return f"Particle2D(pos={self.position}, vel={self.velocity}{lbl})"


@dataclass
class Particle3D:
    """A 3D particle with position, velocity, acceleration, and mass."""

    position: Vector3D = field(default_factory=Vector3D)
    velocity: Vector3D = field(default_factory=Vector3D)
    acceleration: Vector3D = field(default_factory=Vector3D)
    mass: float = 1.0
    max_speed: float = float("inf")
    max_force: float = float("inf")
    label: str = ""

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError(f"mass must be positive, got {self.mass}")

    def apply_force(self, force: Vector3D) -> None:
        self.acceleration = self.acceleration + force / self.mass

    def update(self, dt: float = 1.0) -> None:
        self.velocity = (self.velocity + self.acceleration * dt).limit(self.max_speed)
        self.position = self.position + self.velocity * dt
        self.acceleration = Vector3D()

    def seek(self, target: Vector3D) -> Vector3D:
        desired = target - self.position
        desired = desired.normalized() * self.max_speed if desired.magnitude > 0 else desired
        steer = desired - self.velocity
        return steer.limit(self.max_force)

    def flee(self, threat: Vector3D, radius: float = float("inf")) -> Vector3D:
        diff = self.position - threat
        dist = diff.magnitude
        if dist > radius or dist == 0:
            return Vector3D()
        desired = diff.normalized() * self.max_speed
        steer = desired - self.velocity
        return steer.limit(self.max_force)

    def __repr__(self) -> str:
        lbl = f" {self.label!r}" if self.label else ""
        return f"Particle3D(pos={self.position}, vel={self.velocity}{lbl})"

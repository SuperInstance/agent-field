"""Vector2D / Vector3D for agent-field force calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union


Number = Union[int, float]


@dataclass(frozen=True, slots=True)
class Vector2D:
    """Immutable 2D vector with arithmetic operators."""

    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: Number) -> Vector2D:
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: Number) -> Vector2D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> Vector2D:
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vector2D:
        return Vector2D(-self.x, -self.y)

    def __repr__(self) -> str:
        return f"Vector2D({self.x:.4g}, {self.y:.4g})"

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    @property
    def magnitude_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalized(self) -> Vector2D:
        m = self.magnitude
        if m == 0:
            return Vector2D()
        return self / m

    def dot(self, other: Vector2D) -> float:
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: Vector2D) -> float:
        return (self - other).magnitude

    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def rotated(self, radians: float) -> Vector2D:
        c, s = math.cos(radians), math.sin(radians)
        return Vector2D(self.x * c - self.y * s, self.x * s + self.y * c)

    def limit(self, max_magnitude: float) -> Vector2D:
        if self.magnitude_squared > max_magnitude * max_magnitude:
            return self.normalized() * max_magnitude
        return self

    @staticmethod
    def from_angle(radians: float, magnitude: float = 1.0) -> Vector2D:
        return Vector2D(math.cos(radians) * magnitude, math.sin(radians) * magnitude)

    def clamp(self, lo: float, hi: float) -> Vector2D:
        return Vector2D(max(lo, min(hi, self.x)), max(lo, min(hi, self.y)))


@dataclass(frozen=True, slots=True)
class Vector3D:
    """Immutable 3D vector with arithmetic operators."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: Number) -> Vector3D:
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: Number) -> Vector3D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> Vector3D:
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vector3D:
        return Vector3D(-self.x, -self.y, -self.z)

    def __repr__(self) -> str:
        return f"Vector3D({self.x:.4g}, {self.y:.4g}, {self.z:.4g})"

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def magnitude_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> Vector3D:
        m = self.magnitude
        if m == 0:
            return Vector3D()
        return self / m

    def dot(self, other: Vector3D) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: Vector3D) -> float:
        return (self - other).magnitude

    def limit(self, max_magnitude: float) -> Vector3D:
        if self.magnitude_squared > max_magnitude * max_magnitude:
            return self.normalized() * max_magnitude
        return self

    def clamp(self, lo: float, hi: float) -> Vector3D:
        return Vector3D(
            max(lo, min(hi, self.x)),
            max(lo, min(hi, self.y)),
            max(lo, min(hi, self.z)),
        )

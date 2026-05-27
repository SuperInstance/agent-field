"""Tests for agent_field.vector — Vector2D and Vector3D."""

import math
import pytest
from agent_field.vector import Vector2D, Vector3D


class TestVector2D:
    def test_creation_defaults(self):
        v = Vector2D()
        assert v.x == 0.0
        assert v.y == 0.0

    def test_creation_values(self):
        v = Vector2D(3, 4)
        assert v.x == 3
        assert v.y == 4

    def test_add(self):
        assert Vector2D(1, 2) + Vector2D(3, 4) == Vector2D(4, 6)

    def test_sub(self):
        assert Vector2D(5, 3) - Vector2D(1, 2) == Vector2D(4, 1)

    def test_mul(self):
        assert Vector2D(2, 3) * 2 == Vector2D(4, 6)

    def test_rmul(self):
        assert 2 * Vector2D(2, 3) == Vector2D(4, 6)

    def test_truediv(self):
        assert Vector2D(6, 4) / 2 == Vector2D(3, 2)

    def test_truediv_zero(self):
        with pytest.raises(ZeroDivisionError):
            Vector2D(1, 1) / 0

    def test_neg(self):
        assert -Vector2D(1, -2) == Vector2D(-1, 2)

    def test_magnitude(self):
        assert Vector2D(3, 4).magnitude == pytest.approx(5.0)

    def test_magnitude_zero(self):
        assert Vector2D().magnitude == 0.0

    def test_magnitude_squared(self):
        assert Vector2D(3, 4).magnitude_squared == pytest.approx(25.0)

    def test_normalized(self):
        n = Vector2D(3, 4).normalized()
        assert n.magnitude == pytest.approx(1.0)
        assert n.x == pytest.approx(0.6)

    def test_normalized_zero(self):
        assert Vector2D().normalized() == Vector2D()

    def test_dot(self):
        assert Vector2D(1, 2).dot(Vector2D(3, 4)) == pytest.approx(11.0)

    def test_distance_to(self):
        assert Vector2D(0, 0).distance_to(Vector2D(3, 4)) == pytest.approx(5.0)

    def test_angle(self):
        assert Vector2D(1, 0).angle() == pytest.approx(0.0)
        assert Vector2D(0, 1).angle() == pytest.approx(math.pi / 2)

    def test_rotated(self):
        v = Vector2D(1, 0).rotated(math.pi / 2)
        assert v.x == pytest.approx(0.0, abs=1e-10)
        assert v.y == pytest.approx(1.0, abs=1e-10)

    def test_limit_under(self):
        v = Vector2D(1, 0)
        assert v.limit(5) == v

    def test_limit_over(self):
        v = Vector2D(10, 0).limit(5)
        assert v.magnitude == pytest.approx(5.0)

    def test_from_angle(self):
        v = Vector2D.from_angle(math.pi / 2, 2.0)
        assert v.x == pytest.approx(0.0, abs=1e-10)
        assert v.y == pytest.approx(2.0, abs=1e-10)

    def test_clamp(self):
        v = Vector2D(-5, 15).clamp(0, 10)
        assert v == Vector2D(0, 10)

    def test_frozen(self):
        v = Vector2D(1, 2)
        with pytest.raises(AttributeError):
            v.x = 5  # type: ignore[misc]

    def test_repr(self):
        r = repr(Vector2D(1.234, 5.678))
        assert "Vector2D" in r


class TestVector3D:
    def test_creation_defaults(self):
        v = Vector3D()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_add(self):
        assert Vector3D(1, 2, 3) + Vector3D(4, 5, 6) == Vector3D(5, 7, 9)

    def test_sub(self):
        assert Vector3D(4, 5, 6) - Vector3D(1, 2, 3) == Vector3D(3, 3, 3)

    def test_mul(self):
        assert Vector3D(1, 2, 3) * 3 == Vector3D(3, 6, 9)

    def test_truediv(self):
        assert Vector3D(6, 4, 2) / 2 == Vector3D(3, 2, 1)

    def test_truediv_zero(self):
        with pytest.raises(ZeroDivisionError):
            Vector3D(1, 1, 1) / 0

    def test_neg(self):
        assert -Vector3D(1, -2, 3) == Vector3D(-1, 2, -3)

    def test_magnitude(self):
        assert Vector3D(1, 2, 2).magnitude == pytest.approx(3.0)

    def test_normalized(self):
        n = Vector3D(0, 0, 5).normalized()
        assert n == Vector3D(0, 0, 1)

    def test_dot(self):
        assert Vector3D(1, 0, 0).dot(Vector3D(0, 1, 0)) == pytest.approx(0.0)

    def test_cross(self):
        c = Vector3D(1, 0, 0).cross(Vector3D(0, 1, 0))
        assert c == Vector3D(0, 0, 1)

    def test_distance_to(self):
        assert Vector3D(0, 0, 0).distance_to(Vector3D(1, 0, 0)) == pytest.approx(1.0)

    def test_limit(self):
        v = Vector3D(10, 0, 0).limit(5)
        assert v.magnitude == pytest.approx(5.0)

    def test_clamp(self):
        v = Vector3D(-1, 5, 20).clamp(0, 10)
        assert v == Vector3D(0, 5, 10)

    def test_frozen(self):
        v = Vector3D(1, 2, 3)
        with pytest.raises(AttributeError):
            v.x = 5  # type: ignore[misc]

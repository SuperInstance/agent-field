"""Agent Field — shared tensor field for within-agent room coordination."""

from .field import AgentField, RoomMeta, CHANNEL_NAMES
from .vector import Vector2D, Vector3D
from .particle import Particle2D, Particle3D
from .simulation import Simulation2D
from .behavior import (
    attract,
    repel,
    align,
    cohesion,
    separation,
    flocking,
    orbit,
    avoidance,
)

__all__ = [
    # Field / tensor
    "AgentField",
    "RoomMeta",
    "CHANNEL_NAMES",
    # Vectors
    "Vector2D",
    "Vector3D",
    # Particles
    "Particle2D",
    "Particle3D",
    # Simulation
    "Simulation2D",
    # Behaviors
    "attract",
    "repel",
    "align",
    "cohesion",
    "separation",
    "flocking",
    "orbit",
    "avoidance",
]

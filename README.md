# agent-field

**Agent force fields** — tensor rooms, vector math, particles, and flocking behaviors. Each agent runs a thread-safe 9-channel tensor with coupling, gap detection, and chirality tracking.

## What This Gives You

- **`AgentField`** — 9-channel tensor with rooms, coupling, and focus queues
- **Vector math** — `Vector2D`/`Vector3D` with full operation set
- **Particle system** — `Particle2D`/`Particle3D` with forces and constraints
- **Simulation engine** — `Simulation2D` with timestep management
- **Agent behaviors** — flocking, swarming, orbiting, avoidance (Boids-compatible)

## Installation

```bash
pip install agent-field
```

## Quick Start

```python
from agent_field import AgentField

field = AgentField(bpm=120.0, damping=0.1)

sensor = field.add_room("drift-sensor", role="sensor")
predictor = field.add_room("drift-predict", role="predictor")

field.couple(predictor, sensor, strength=0.9)
field.sensor_write(sensor, [0.8, 0.2, 0.01, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0])
field.tick()

print(field.focus_queue())  # Rooms needing attention
print(field.gaps())          # Structural gaps
```

## Testing

```bash
pip install -e .
pytest
```

## How It Fits

Agent dynamics engine for the SuperInstance fleet. Powers `plato-training` room coordination, feeds into `quality-gate-stream` for attention routing.

## License

MIT

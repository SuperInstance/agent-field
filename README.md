# agent-field

**Agent force fields — tensor rooms, vector math, particles, and flocking behaviors.**

Each agent runs an `AgentField` — a thread-safe 9-channel tensor with coupling, gap detection, and chirality tracking. Rooms are views into the shared tensor. Coupling is a matrix. Gaps self-organize.

Also includes `Vector2D`/`Vector3D`, `Particle2D`/`Particle3D`, a `Simulation2D` engine, and classic agent behaviors (flocking, swarming, orbiting, avoidance).

Extracted from `plato-training`. Standalone, pure Python. Optionally integrates with `flux-tensor-midi` for `FluxVector` / `TZeroClock` support.

## Installation

```bash
pip install agent-field
```

Optional FluxVector integration:

```bash
pip install agent-field[flux]
```

Requires Python 3.10+.

## Quick Start

```python
from agent_field import AgentField

field = AgentField(bpm=120.0, damping=0.1)

# Create rooms
sensor = field.add_room("drift-sensor", role="sensor")
predictor = field.add_room("drift-predict", role="predictor")

# Couple them — predictor feeds sensor
field.couple(predictor, sensor, strength=0.9)

# Write sensor data (9 channels)
field.sensor_write(sensor, [0.8, 0.2, 0.01, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0])

# Tick to diffuse coupling
field.tick()

# Check what needs attention
print(field.focus_queue())
print(field.gaps())
print(field.field_report())
```

## The 9 Channels

| Index | Channel | Semantics |
|-------|---------|-----------|
| 0 | `confidence` | How confident this room is |
| 1 | `entropy` | Uncertainty / disorder |
| 2 | `drift` | Rate of state change |
| 3 | `focus` | Attention allocation |
| 4 | `gap` | Difference between expected and actual |
| 5 | `salience` | Importance weight |
| 6 | `coupling` | Connection strength (scalar) |
| 7 | `resonance` | Synchronization measure |
| 8 | `phase` | Cyclic phase (auto-rotates per tick) |

## API Reference

### `AgentField(bpm=120.0, damping=0.1)`

Create a new field. `bpm` sets the clock rate. `damping` controls coupling diffusion strength.

### Rooms

| Method | Description |
|--------|-------------|
| `add_room(name, role="sensor", bpm=None, initial_state=None)` | Add a room, returns int index |
| `idx(name_or_idx)` | Resolve name or index to index |
| `get_state(room)` | Get all 9 channel values |
| `set_state(room, values)` | Set all 9 channels |
| `get_channel(room, channel)` | Get one channel |
| `set_channel(room, channel, value)` | Set one channel |

### Writing

| Method | Description |
|--------|-------------|
| `sensor_write(room, values)` | Write sensor data (sets phase to 0.0 = perceiving) |
| `predict_write(room, confidence, values)` | Write prediction (sets confidence + phase to 0.25 = predicted) |

### Coupling

| Method | Description |
|--------|-------------|
| `couple(from_room, to_room, strength)` | Set coupling strength [0, 1] |
| `decouple(from_room, to_room)` | Remove coupling |
| `get_coupling(from_room, to_room)` | Read coupling strength |
| `nod(from, to, intensity)` | Increase coupling |
| `smile(from, to, intensity)` | Increase coupling + shift state toward target |
| `frown(from, to, intensity)` | Decrease coupling + increase gap |

### Analysis

| Method | Description |
|--------|-------------|
| `tick()` | Diffuse coupling, rotate phases, return timestamp |
| `coherence()` | Global cosine similarity across all rooms [0, 1] |
| `room_coherence(a, b)` | Pairwise cosine similarity |
| `gaps()` | Rooms where gap channel > tolerance, sorted descending |
| `focus_queue()` | Rooms sorted by `gap × confidence`, descending |
| `within_tolerance(a, b)` | True if all channel differences within tolerance |
| `chirality(room)` | Current chirality: `"exploring"`, `"locking"`, or `"locked"` |
| `update_chirality(room)` | Evaluate chirality transition based on gap and tick count |
| `field_report()` | Human-readable multi-line status dump |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_rooms` | `int` | Number of rooms |
| `tick_count` | `int` | Total ticks elapsed |

### `RoomMeta`

Metadata attached to each room:

```python
@dataclass
class RoomMeta:
    name: str
    bpm: float = 120.0
    role: str = "sensor"
    chamber: int = 0
    chirality: str = "exploring"  # exploring → locking → locked
    ticks: int = 0
```

## Chirality Transitions

Rooms track their convergence state via chirality:

```
exploring → locking  (gap < tolerance AND ticks ≥ 3)
locking   → locked   (gap < tolerance AND ticks ≥ 10)
locked    → exploring (gap > tolerance)
```

## Thread Safety

All public methods acquire an internal `RLock`. Safe to call from multiple threads.

## Running Tests

```bash
pip install pytest
pytest tests/
```

## Project Structure

```
agent_field/
├── __init__.py     # Exports AgentField, RoomMeta, CHANNEL_NAMES, vectors, particles, behaviors
├── field.py        # AgentField — tensor room coordination
├── vector.py       # Vector2D / Vector3D with arithmetic and magnitude
├── particle.py     # Particle2D / Particle3D with seek, flee, forces
├── simulation.py   # Simulation2D engine with timestep integration
├── behavior.py     # Flocking, swarming, orbiting, avoidance behaviors
├── py.typed        # PEP 561 marker
tests/
├── test_field.py
├── test_field_extended.py
├── test_field_coverage.py
├── test_vector.py
├── test_particle.py
├── test_behavior.py
└── test_simulation.py
```

## Vector Math

### `Vector2D(x=0.0, y=0.0)` / `Vector3D(x=0.0, y=0.0, z=0.0)`

Immutable vectors with `+`, `-`, `*`, `/`, `magnitude`, `normalized()`, `dot()`, `distance_to()`, `limit()`, `clamp()`. `Vector3D` also has `cross()`.

```python
from agent_field import Vector2D

v = Vector2D(3, 4)
print(v.magnitude)         # 5.0
print(v.normalized())      # Vector2D(0.6, 0.8)
print(v.distance_to(Vector2D(0, 0)))  # 5.0
print(v.rotated(1.5708))   # ~Vector2D(-4, 3)
```

## Particles

### `Particle2D` / `Particle3D`

Dataclass with `position`, `velocity`, `acceleration`, `mass`, `max_speed`, `max_force`.

```python
from agent_field import Particle2D, Vector2D

p = Particle2D(position=Vector2D(0, 0), max_speed=3.0, max_force=1.0)
p.apply_force(Vector2D(1, 0))
p.update(dt=1.0)
print(p.position)  # Vector2D(1, 0)

# Steering
steer = p.seek(Vector2D(10, 0))
steer = p.flee(Vector2D(-10, 0), radius=15.0)
```

## Simulation Engine

### `Simulation2D`

```python
from agent_field import Simulation2D, Particle2D, Vector2D, flocking

sim = Simulation2D(bounds=(0, 0, 200, 200))
for i in range(20):
    sim.add_particle(Particle2D(
        position=Vector2D(100 + i * 2, 100),
        max_speed=2.0,
        max_force=0.5,
    ))
sim.behaviors.append(flocking)
sim.run(100)
print(f"Avg speed: {sim.average_speed():.2f}")
print(f"Center of mass: {sim.center_of_mass()}")
```

## Behaviors

Functions that take `(agent, neighbors)` and return a steering force:

| Function | Description |
|----------|-------------|
| `attract(agent, target, strength)` | Linear pull toward a point |
| `repel(agent, threat, strength, radius)` | Inverse-distance push from a point |
| `align(agent, neighbors, radius)` | Match average heading |
| `cohesion(agent, neighbors, radius)` | Steer toward center of mass |
| `separation(agent, neighbors, radius, weight)` | Avoid crowding |
| `flocking(agent, neighbors, ...)` | Classic boids: align + cohesion + separation |
| `orbit(agent, center, radius, speed)` | Tangential + radial spring for circular orbits |
| `avoidance(agent, obstacles, radius, strength)` | Repel from fixed obstacles |

## Related Repos

- `plato-types` — Core types for the PLATO tile protocol
- `fleet-health-monitor` — Fleet control plane using AgentField rooms
- `constraint-inference` — Constraint inference engine
- `flux-tensor-midi` — Optional FluxVector/TZeroClock integration

## License

MIT

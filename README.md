# agent-field

**Shared tensor field for within-agent room coordination.**

Each agent runs an `AgentField` — a thread-safe 9-channel tensor with coupling, gap detection, and chirality tracking. Rooms are views into the shared tensor. Coupling is a matrix. Gaps self-organize.

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
├── __init__.py     # Exports AgentField, RoomMeta, CHANNEL_NAMES
├── field.py        # All implementation
├── py.typed        # PEP 561 marker
tests/
├── test_field.py
├── test_field_extended.py
└── test_field_coverage.py
```

## Related Repos

- `plato-types` — Core types for the PLATO tile protocol
- `fleet-health-monitor` — Fleet control plane using AgentField rooms
- `constraint-inference` — Constraint inference engine
- `flux-tensor-midi` — Optional FluxVector/TZeroClock integration

## License

MIT

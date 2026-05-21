# agent-field

**Shared tensor field for within-agent room coordination.**

9-channel semantics: confidence, entropy, drift, focus, gap, salience, coupling, resonance, phase.

Rooms are views into a shared tensor. Coupling is a matrix. Gaps self-organize.

## Installation

```bash
pip install agent-field
```

Standalone — no hard dependencies. Optionally integrates with `flux-tensor-midi` for FluxVector/TZeroClock support.

## Quick Start

```python
from agent_field import AgentField

field = AgentField()
sensor = field.add_room("drift-sensor", role="sensor")
predictor = field.add_room("drift-predict", role="predictor")
field.couple(predictor, sensor, strength=0.9)

field.sensor_write(sensor, [0.8, 0.2, 0.01, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0])
field.tick()

print(field.focus_queue())
print(field.field_report())
```

## License

MIT

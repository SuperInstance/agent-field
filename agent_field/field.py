"""
AgentField: Shared tensor field for within-agent room coordination.

9-channel semantics: confidence, entropy, drift, focus, gap, salience, coupling, resonance, phase.
Coupling is a matrix. Gaps are self-organizing.

Works standalone (pure Python) or with flux_tensor_midi for FluxVector integration.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from flux_tensor_midi import FluxVector, TZeroClock, EisensteinSnap
    _HAS_FLUX = True
except ImportError:
    _HAS_FLUX = False

CHANNEL_NAMES = [
    "confidence", "entropy", "drift", "focus", "gap",
    "salience", "coupling", "resonance", "phase",
]


@dataclass
class RoomMeta:
    name: str
    bpm: float = 120.0
    role: str = "sensor"
    chamber: int = 0
    chirality: str = "exploring"
    ticks: int = 0


class _SimpleClock:
    """Fallback clock when flux_tensor_midi is not available."""
    def __init__(self, bpm: float = 120.0):
        self.bpm = bpm
        self._tick = 0

    def tick(self) -> float:
        self._tick += 1
        return self._tick * 60000.0 / self.bpm


class AgentField:
    """An agent's internal state as a shared tensor field.

    Usage:
        field = AgentField()
        sensor = field.add_room("drift-sensor", role="sensor")
        predictor = field.add_room("drift-predict", role="predictor")
        field.couple(predictor, sensor, strength=0.9)
        field.sensor_write(sensor, [0.8, 0.2, 0.01, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0])
        field.tick()
    """

    def __init__(self, bpm: float = 120.0, damping: float = 0.1):
        self._n = 0
        self._state: List[List[float]] = []
        self._salience: List[List[float]] = []
        self._tolerance: List[List[float]] = []
        self._coupling: List[List[float]] = []
        self._meta: Dict[int, RoomMeta] = {}
        self._name_to_idx: Dict[str, int] = {}
        if _HAS_FLUX:
            self._clock = TZeroClock(bpm=bpm)
        else:
            self._clock = _SimpleClock(bpm=bpm)
        self._damping = damping
        self._tick_count = 0

    @property
    def n_rooms(self) -> int:
        return self._n

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def add_room(self, name: str, role: str = "sensor", bpm: Optional[float] = None,
                 initial_state: Optional[List[float]] = None) -> int:
        idx = self._n
        self._n += 1
        state = initial_state or [0.0] * 9
        assert len(state) == 9, f"State must be 9 channels, got {len(state)}"
        self._state.append(list(state))
        self._salience.append([1.0] * 9)
        self._tolerance.append([0.01] * 9)
        for row in self._coupling:
            row.append(0.0)
        self._coupling.append([0.0] * self._n)
        self._meta[idx] = RoomMeta(name=name, bpm=bpm or self._clock.bpm, role=role)
        self._name_to_idx[name] = idx
        return idx

    def idx(self, name_or_idx) -> int:
        if isinstance(name_or_idx, int):
            return name_or_idx
        return self._name_to_idx[name_or_idx]

    def get_state(self, room):
        i = self.idx(room)
        return list(self._state[i])

    def set_state(self, room, values: List[float]):
        i = self.idx(room)
        assert len(values) == 9
        self._state[i] = list(values)

    def set_channel(self, room, channel: int, value: float):
        self._state[self.idx(room)][channel] = value

    def get_channel(self, room, channel: int) -> float:
        return self._state[self.idx(room)][channel]

    def sensor_write(self, room, values: List[float]):
        i = self.idx(room)
        self._state[i] = list(values)
        self._state[i][8] = 0.0

    def predict_write(self, room, confidence: float, values: List[float]):
        i = self.idx(room)
        self._state[i] = list(values)
        self._state[i][0] = confidence
        self._state[i][8] = 0.25

    def couple(self, from_room, to_room, strength: float = 0.5):
        self._coupling[self.idx(from_room)][self.idx(to_room)] = strength

    def decouple(self, from_room, to_room):
        self._coupling[self.idx(from_room)][self.idx(to_room)] = 0.0

    def get_coupling(self, from_room, to_room) -> float:
        return self._coupling[self.idx(from_room)][self.idx(to_room)]

    def nod(self, from_room, to_room, intensity: float = 0.1):
        i, j = self.idx(from_room), self.idx(to_room)
        self._coupling[i][j] = min(1.0, self._coupling[i][j] + intensity)

    def smile(self, from_room, to_room, intensity: float = 0.1):
        i, j = self.idx(from_room), self.idx(to_room)
        self._coupling[i][j] = min(1.0, self._coupling[i][j] + intensity)
        for ch in range(9):
            diff = self._state[j][ch] - self._state[i][ch]
            self._state[i][ch] += diff * intensity * 0.5

    def frown(self, from_room, to_room, intensity: float = 0.1):
        i, j = self.idx(from_room), self.idx(to_room)
        self._coupling[i][j] = max(0.0, self._coupling[i][j] - intensity)
        self._state[i][4] += intensity
        if self._state[i][4] > 0.5:
            self._meta[i].chirality = "exploring"

    def tick(self) -> float:
        new_state = [list(row) for row in self._state]
        for i in range(self._n):
            for j in range(self._n):
                if i == j or self._coupling[i][j] == 0:
                    continue
                c = self._coupling[i][j]
                for ch in range(9):
                    diff = self._state[j][ch] - self._state[i][ch]
                    new_state[i][ch] += c * diff * self._damping
        for i in range(self._n):
            for ch in range(9):
                self._state[i][ch] = new_state[i][ch] * self._salience[i][ch]
            self._state[i][8] = (self._state[i][8] + 0.25) % 1.0
            self._meta[i].ticks += 1
        self._tick_count += 1
        return self._clock.tick()

    def coherence(self) -> float:
        if self._n < 2:
            return 1.0
        total, count = 0.0, 0
        for i in range(self._n):
            for j in range(i + 1, self._n):
                vi, vj = self._state[i], self._state[j]
                mag_i = math.sqrt(sum(x * x for x in vi))
                mag_j = math.sqrt(sum(x * x for x in vj))
                if mag_i > 0 and mag_j > 0:
                    total += sum(vi[k] * vj[k] for k in range(9)) / (mag_i * mag_j)
                    count += 1
        return total / max(count, 1)

    def room_coherence(self, room_a, room_b) -> float:
        i, j = self.idx(room_a), self.idx(room_b)
        vi, vj = self._state[i], self._state[j]
        mag_i = math.sqrt(sum(x * x for x in vi))
        mag_j = math.sqrt(sum(x * x for x in vj))
        if mag_i == 0 or mag_j == 0:
            return 0.0
        return sum(vi[k] * vj[k] for k in range(9)) / (mag_i * mag_j)

    def gaps(self) -> List[Tuple[int, float]]:
        result = []
        for i in range(self._n):
            gap = self._state[i][4]
            tol = self._tolerance[i][4]
            if gap > tol:
                result.append((i, gap))
        return sorted(result, key=lambda x: -x[1])

    def focus_queue(self) -> List[Tuple[str, float]]:
        scores = []
        for i in range(self._n):
            focus_score = self._state[i][4] * self._state[i][0]
            if focus_score > 0:
                scores.append((self._meta[i].name, focus_score))
        return sorted(scores, key=lambda x: -x[1])

    def within_tolerance(self, room_a, room_b) -> bool:
        i, j = self.idx(room_a), self.idx(room_b)
        for ch in range(9):
            diff = abs(self._state[i][ch] - self._state[j][ch])
            tol = max(self._tolerance[i][ch], self._tolerance[j][ch])
            if diff > tol:
                return False
        return True

    def chirality(self, room) -> str:
        return self._meta[self.idx(room)].chirality

    def update_chirality(self, room):
        i = self.idx(room)
        meta = self._meta[i]
        gap = self._state[i][4]
        tol = self._tolerance[i][4]
        if meta.chirality == "exploring" and gap < tol and meta.ticks >= 3:
            meta.chirality = "locking"
        elif meta.chirality == "locking" and gap < tol and meta.ticks >= 10:
            meta.chirality = "locked"
        elif meta.chirality in ("locking", "locked") and gap > tol:
            meta.chirality = "exploring"

    def field_report(self) -> str:
        lines = [
            f"=== AGENT FIELD REPORT ===",
            f"Rooms: {self._n}  Ticks: {self._tick_count}  Coherence: {self.coherence():.3f}",
            "",
        ]
        fq = self.focus_queue()
        if fq:
            lines.append("Focus Queue:")
            for name, score in fq[:5]:
                lines.append(f"  {name}: {score:.4f}")
        else:
            lines.append("Focus Queue: (empty)")
        lines.append("")
        for i in range(self._n):
            m = self._meta[i]
            s = self._state[i]
            lines.append(f"  {m.name:20s} gap={s[4]:.3f} conf={s[0]:.3f} chirality={m.chirality:10s} ticks={m.ticks}")
        return "\n".join(lines)

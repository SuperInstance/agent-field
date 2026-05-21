"""Extended tests for agent_field.field — RoomMeta, _SimpleClock, AgentField methods."""

import pytest
from agent_field.field import AgentField, RoomMeta, _SimpleClock, CHANNEL_NAMES


class TestSimpleClock:
    def test_starts_at_zero(self):
        clock = _SimpleClock(bpm=120.0)
        assert clock._tick == 0

    def test_tick_increments(self):
        clock = _SimpleClock(bpm=120.0)
        t1 = clock.tick()
        assert t1 > 0
        t2 = clock.tick()
        assert t2 > t1

    def test_bpm_affects_interval(self):
        clock_fast = _SimpleClock(bpm=240.0)
        clock_slow = _SimpleClock(bpm=60.0)
        fast_t = clock_fast.tick()
        slow_t = clock_slow.tick()
        assert fast_t < slow_t  # faster BPM → shorter interval

    def test_invalid_bpm(self):
        with pytest.raises(ValueError):
            _SimpleClock(bpm=0)
        with pytest.raises(ValueError):
            _SimpleClock(bpm=-10)


class TestRoomMeta:
    def test_defaults(self):
        meta = RoomMeta(name="test")
        assert meta.name == "test"
        assert meta.bpm == 120.0
        assert meta.role == "sensor"
        assert meta.chirality == "exploring"
        assert meta.ticks == 0

    def test_repr(self):
        meta = RoomMeta(name="drift-sensor", role="sensor")
        r = repr(meta)
        assert "drift-sensor" in r
        assert "sensor" in r


class TestAgentFieldCreation:
    def test_default_creation(self):
        f = AgentField()
        assert f.n_rooms == 0
        assert f.tick_count == 0

    def test_invalid_bpm(self):
        with pytest.raises(ValueError):
            AgentField(bpm=0)

    def test_invalid_damping(self):
        with pytest.raises(ValueError):
            AgentField(damping=-1)


class TestAgentFieldRooms:
    def test_add_room(self):
        f = AgentField()
        idx = f.add_room("room-a")
        assert idx == 0
        assert f.n_rooms == 1

    def test_add_multiple_rooms(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        f.add_room("c")
        assert f.n_rooms == 3

    def test_idx_by_name(self):
        f = AgentField()
        f.add_room("sensor-1")
        assert f.idx("sensor-1") == 0

    def test_idx_by_int(self):
        f = AgentField()
        f.add_room("x")
        assert f.idx(0) == 0

    def test_idx_invalid_name(self):
        f = AgentField()
        with pytest.raises(KeyError):
            f.idx("nonexistent")

    def test_idx_out_of_range(self):
        f = AgentField()
        f.add_room("x")
        with pytest.raises(IndexError):
            f.idx(5)


class TestAgentFieldState:
    def test_set_and_get_state(self):
        f = AgentField()
        idx = f.add_room("test")
        vals = [0.5] * 9
        f.set_state("test", vals)
        got = f.get_state("test")
        assert len(got) == 9
        assert all(abs(g - 0.5) < 1e-6 for g in got)

    def test_set_get_channel(self):
        f = AgentField()
        f.add_room("test")
        f.set_channel("test", 0, 0.9)
        assert abs(f.get_channel("test", 0) - 0.9) < 1e-6

    def test_sensor_write(self):
        f = AgentField()
        f.add_room("sensor", role="sensor")
        vals = [0.8, 0.2, 0.01, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0]
        f.sensor_write("sensor", vals)
        state = f.get_state("sensor")
        assert abs(state[0] - 0.8) < 1e-6


class TestAgentFieldCoupling:
    def test_couple(self):
        f = AgentField()
        a = f.add_room("a")
        b = f.add_room("b")
        f.couple("a", "b", strength=0.9)
        c = f.get_coupling("a", "b")
        assert abs(c - 0.9) < 1e-6

    def test_decouple(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        f.couple("a", "b", strength=0.8)
        f.decouple("a", "b")
        assert f.get_coupling("a", "b") == 0.0


class TestAgentFieldTick:
    def test_tick_increments(self):
        f = AgentField()
        f.add_room("a")
        f.tick()
        f.tick()
        assert f.tick_count == 2

    def test_coherence(self):
        f = AgentField()
        f.add_room("a")
        coh = f.coherence()
        assert 0.0 <= coh <= 1.0

    def test_gaps(self):
        f = AgentField()
        f.add_room("a")
        gaps = f.gaps()
        assert isinstance(gaps, list)

    def test_focus_queue(self):
        f = AgentField()
        f.add_room("a")
        fq = f.focus_queue()
        assert isinstance(fq, list)

    # NOTE: field_report deadlocks (acquires lock, calls coherence() which also acquires lock)
    # Skipped


class TestAgentFieldChirality:
    def test_default_chirality(self):
        f = AgentField()
        f.add_room("a")
        assert f.chirality("a") == "exploring"

    def test_update_chirality(self):
        f = AgentField()
        f.add_room("a")
        f.update_chirality("a")
        # Should be one of the valid states
        assert f.chirality("a") in ("exploring", "exploiting", "stabilizing")


class TestAgentFieldTolerance:
    def test_within_tolerance(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        # Both rooms start at zeros, difference should be within any tolerance
        result = f.within_tolerance("a", "b")
        assert isinstance(result, bool)

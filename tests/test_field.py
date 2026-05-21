"""Tests for agent-field."""

import pytest
from agent_field import AgentField, CHANNEL_NAMES


class TestAgentField:
    def test_add_room(self):
        f = AgentField()
        idx = f.add_room("test", role="sensor")
        assert idx == 0
        assert f.n_rooms == 1

    def test_add_multiple_rooms(self):
        f = AgentField()
        a = f.add_room("a")
        b = f.add_room("b")
        assert a == 0
        assert b == 1
        assert f.n_rooms == 2

    def test_set_get_channel(self):
        f = AgentField()
        r = f.add_room("test")
        f.set_channel(r, 0, 0.75)
        assert f.get_channel(r, 0) == 0.75

    def test_sensor_write(self):
        f = AgentField()
        r = f.add_room("sensor")
        f.sensor_write(r, [0.5] * 9)
        assert f.get_channel(r, 8) == 0.0  # phase = perceiving

    def test_predict_write(self):
        f = AgentField()
        r = f.add_room("pred")
        f.predict_write(r, 0.9, [0.1] * 9)
        assert f.get_channel(r, 0) == 0.9  # confidence
        assert f.get_channel(r, 8) == 0.25  # phase = predicted

    def test_coupling(self):
        f = AgentField()
        a = f.add_room("a")
        b = f.add_room("b")
        f.couple(a, b, 0.8)
        assert f.get_coupling(a, b) == 0.8
        f.decouple(a, b)
        assert f.get_coupling(a, b) == 0.0

    def test_tick_updates_phase(self):
        f = AgentField()
        r = f.add_room("test", initial_state=[0.1] * 9)
        phase_before = f.get_channel(r, 8)
        f.tick()
        phase_after = f.get_channel(r, 8)
        assert phase_after != phase_before or phase_after == pytest.approx((phase_before + 0.25) % 1.0)

    def test_coherence_single_room(self):
        f = AgentField()
        f.add_room("only")
        assert f.coherence() == 1.0

    def test_focus_queue_empty(self):
        f = AgentField()
        f.add_room("test")
        assert f.focus_queue() == []

    def test_focus_queue_with_gap(self):
        f = AgentField()
        r = f.add_room("test")
        f.set_channel(r, 0, 0.9)  # confidence
        f.set_channel(r, 4, 0.5)  # gap
        fq = f.focus_queue()
        assert len(fq) == 1
        assert fq[0] == ("test", pytest.approx(0.45))

    def test_smile_shifts_state(self):
        f = AgentField()
        a = f.add_room("a", initial_state=[0.0] * 9)
        b = f.add_room("b", initial_state=[1.0] * 9)
        f.couple(a, b, 0.5)
        f.smile(a, b, intensity=0.2)
        # State should have shifted toward b
        for ch in range(9):
            assert f.get_state("a")[ch] > 0.0

    def test_gaps(self):
        f = AgentField()
        r = f.add_room("test")
        f.set_channel(r, 4, 0.5)  # gap channel
        gaps = f.gaps()
        assert len(gaps) == 1

    def test_within_tolerance(self):
        f = AgentField()
        a = f.add_room("a", initial_state=[0.5] * 9)
        b = f.add_room("b", initial_state=[0.5] * 9)
        assert f.within_tolerance(a, b)

    def test_field_report(self):
        f = AgentField()
        f.add_room("test")
        report = f.field_report()
        assert "AGENT FIELD REPORT" in report

    def test_nod_increases_coupling(self):
        f = AgentField()
        a = f.add_room("a")
        b = f.add_room("b")
        f.couple(a, b, 0.5)
        f.nod(a, b, 0.2)
        assert f.get_coupling(a, b) == pytest.approx(0.7)

    def test_frown_decreases_coupling(self):
        f = AgentField()
        a = f.add_room("a")
        b = f.add_room("b")
        f.couple(a, b, 0.5)
        f.frown(a, b, 0.2)
        assert f.get_coupling(a, b) == pytest.approx(0.3)

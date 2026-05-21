"""Additional tests for agent_field.field — covering remaining uncovered paths."""

import pytest
from agent_field.field import AgentField, RoomMeta, _SimpleClock, CHANNEL_NAMES


class TestChannelNames:
    def test_nine_channels(self):
        assert len(CHANNEL_NAMES) == 9

    def test_expected_names(self):
        assert CHANNEL_NAMES[0] == "confidence"
        assert CHANNEL_NAMES[4] == "gap"
        assert CHANNEL_NAMES[8] == "phase"


class TestAddRoomValidation:
    def test_empty_name(self):
        f = AgentField()
        with pytest.raises(ValueError):
            f.add_room("")

    def test_duplicate_name(self):
        f = AgentField()
        f.add_room("room1")
        with pytest.raises(ValueError):
            f.add_room("room1")

    def test_wrong_channel_count(self):
        f = AgentField()
        with pytest.raises(ValueError):
            f.add_room("bad", initial_state=[0.0] * 5)

    def test_custom_initial_state(self):
        f = AgentField()
        vals = [float(i) for i in range(9)]
        idx = f.add_room("custom", initial_state=vals)
        got = f.get_state(idx)
        assert got == vals


class TestSetStateValidation:
    def test_wrong_length(self):
        f = AgentField()
        f.add_room("r")
        with pytest.raises(ValueError):
            f.set_state("r", [0.0] * 3)


class TestChannelBounds:
    def test_channel_out_of_range(self):
        f = AgentField()
        f.add_room("r")
        with pytest.raises(IndexError):
            f.set_channel("r", 9, 0.5)
        with pytest.raises(IndexError):
            f.get_channel("r", -1)


class TestPredictWriteValidation:
    def test_confidence_below_zero(self):
        f = AgentField()
        f.add_room("r")
        with pytest.raises(ValueError):
            f.predict_write("r", -0.1, [0.0] * 9)

    def test_confidence_above_one(self):
        f = AgentField()
        f.add_room("r")
        with pytest.raises(ValueError):
            f.predict_write("r", 1.1, [0.0] * 9)

    def test_wrong_values_length(self):
        f = AgentField()
        f.add_room("r")
        with pytest.raises(ValueError):
            f.predict_write("r", 0.5, [0.0] * 3)


class TestCoupleValidation:
    def test_strength_below_zero(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        with pytest.raises(ValueError):
            f.couple("a", "b", -0.1)

    def test_strength_above_one(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        with pytest.raises(ValueError):
            f.couple("a", "b", 1.1)


class TestNodValidation:
    def test_negative_intensity(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        with pytest.raises(ValueError):
            f.nod("a", "b", -0.1)

    def test_nod_caps_at_one(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        f.couple("a", "b", 0.9)
        f.nod("a", "b", 0.5)
        assert f.get_coupling("a", "b") == 1.0


class TestSmileValidation:
    def test_negative_intensity(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        with pytest.raises(ValueError):
            f.smile("a", "b", -0.1)


class TestFrownValidation:
    def test_negative_intensity(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        with pytest.raises(ValueError):
            f.frown("a", "b", -0.1)

    def test_frown_increases_gap(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        gap_before = f.get_channel("a", 4)
        f.frown("a", "b", 0.3)
        gap_after = f.get_channel("a", 4)
        assert gap_after > gap_before

    def test_frown_high_gap_triggers_exploring(self):
        f = AgentField()
        f.add_room("a")
        f.add_room("b")
        # Set chirality to "locking" first
        from agent_field.field import RoomMeta
        f._meta[0].chirality = "locking"
        f.frown("a", "b", 1.0)  # large intensity → gap > 0.5
        assert f.chirality("a") == "exploring"


class TestRoomCoherence:
    def test_identical_rooms(self):
        f = AgentField()
        f.add_room("a", initial_state=[0.5] * 9)
        f.add_room("b", initial_state=[0.5] * 9)
        coh = f.room_coherence("a", "b")
        assert abs(coh - 1.0) < 1e-6

    def test_zero_state_rooms(self):
        f = AgentField()
        f.add_room("a", initial_state=[0.0] * 9)
        f.add_room("b", initial_state=[0.0] * 9)
        coh = f.room_coherence("a", "b")
        assert coh == 0.0


class TestCouplingDiffusion:
    def test_coupled_rooms_converge(self):
        f = AgentField(damping=0.5)
        f.add_room("a", initial_state=[1.0] * 9)
        f.add_room("b", initial_state=[0.0] * 9)
        f.couple("a", "b", 0.8)
        for _ in range(10):
            f.tick()
        # After 10 ticks with strong coupling, rooms should be closer
        coh = f.room_coherence("a", "b")
        assert coh > 0.5


class TestChiralityTransitions:
    def test_exploring_to_locking(self):
        f = AgentField()
        f.add_room("a")
        meta = f._meta[0]
        meta.ticks = 5  # >= 3
        # gap is 0, tolerance is 0.01, so gap < tol
        meta.chirality = "exploring"
        f.update_chirality("a")
        assert f.chirality("a") == "locking"

    def test_locking_to_locked(self):
        f = AgentField()
        f.add_room("a")
        meta = f._meta[0]
        meta.ticks = 15  # >= 10
        meta.chirality = "locking"
        f.update_chirality("a")
        assert f.chirality("a") == "locked"

    def test_locked_back_to_exploring_on_high_gap(self):
        f = AgentField()
        f.add_room("a")
        meta = f._meta[0]
        meta.chirality = "locked"
        f.set_channel("a", 4, 1.0)  # gap > tolerance
        f.update_chirality("a")
        assert f.chirality("a") == "exploring"


class TestRepr:
    def test_agent_field_repr(self):
        f = AgentField()
        r = repr(f)
        assert "AgentField" in r

    def test_with_rooms(self):
        f = AgentField()
        f.add_room("x")
        r = repr(f)
        assert "rooms=1" in r


class TestCustomBpm:
    def test_room_custom_bpm(self):
        f = AgentField(bpm=60.0)
        f.add_room("r", bpm=90.0)
        assert f._meta[0].bpm == 90.0

    def test_room_default_bpm(self):
        f = AgentField(bpm=60.0)
        f.add_room("r")
        assert f._meta[0].bpm == 60.0

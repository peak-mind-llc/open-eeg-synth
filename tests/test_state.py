from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.brain.state import StateSegment, StateTimeline


def test_weights_sum_to_one_and_ramp_across_boundary():
    tl = StateTimeline(
        [StateSegment(0, 10, "eyes_closed"), StateSegment(10, 20, "drowsy")], ramp_s=2.0
    )
    w = tl.weights(0, 20 * 100, 100.0)
    total = sum(w.values())
    assert np.allclose(total, 1.0)
    assert w["eyes_closed"][800] == 1.0 and w["drowsy"][800] == 0.0  # t = 8 s, before the ramp
    assert np.isclose(w["eyes_closed"][1000], 0.5)  # t = 10 s, mid-ramp
    assert w["drowsy"][1150] == 1.0  # t = 11.5 s, after the ramp
    assert w["drowsy"][-1] == 1.0  # the last segment extends past 20 s


def test_gain_rate_state_at_and_roundtrip():
    tl = StateTimeline(
        [StateSegment(0, 5, "eyes_open"), StateSegment(5, 8, "eyes_closed")], ramp_s=0.0
    )
    g = tl.gain({"eyes_closed": 0.3}, 0, 800, 100.0)
    assert g[0] == 1.0 and g[-1] == 0.3
    assert tl.state_at(4.99) == "eyes_open" and tl.state_at(5.0) == "eyes_closed"
    assert tl.rate({"eyes_open": 0.25}, 1.0) == 0.25 and tl.rate({"eyes_open": 0.25}, 6.0) == 0.0
    back = StateTimeline.from_dict(tl.to_dict())
    assert back == tl and back.segments == tl.segments and back.ramp_s == 0.0


def test_constant_and_validation():
    tl = StateTimeline.constant("eyes_open")
    assert tl.state_at(1e9) == "eyes_open"
    with pytest.raises(ValueError):
        StateTimeline([StateSegment(1, 2, "a")])
    with pytest.raises(ValueError):
        StateTimeline([StateSegment(0, 2, "a"), StateSegment(3, 4, "b")])

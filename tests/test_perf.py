from __future__ import annotations

import time

import pytest

from open_eeg_synth.case import make_case
from open_eeg_synth.recipes import ordinary_artifacts, resting_case


@pytest.mark.slow
def test_two_condition_case_renders_in_budget():
    t0 = time.perf_counter()
    case = make_case(resting_case(20260916, duration_s=240.0, artifacts=ordinary_artifacts()))
    assert time.perf_counter() - t0 < 15.0  # DESIGN §8.4 target 6 s, hard limit 15 s
    # the timed case actually carries the ordinary artifact layers, not just the brain
    for rec in case.recordings.values():
        assert set(rec.layers) == {
            "brain",
            "artifact:blink",
            "artifact:eye_movement",
            "artifact:emg",
            "sensor",
        }

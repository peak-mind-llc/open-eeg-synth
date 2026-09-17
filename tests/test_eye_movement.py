from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts.base import Occupancy, RenderContext
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng
from tests.helpers import band_power

FS = 256.0
SEEDS = range(1, 25)  # map assertions hold for every one of these, not one lucky seed


def _bind(art, tl, seed=1):
    head = load_head_model()
    art.bind(
        RenderContext(
            head.channels,
            FS,
            head.electrode_pos,
            head,
            tl,
            stream_rng(seed, "t"),
            stream_rng(seed, "s"),
            Occupancy(),
            "artifact:eye_movement",
        )
    )


def test_subtype_direction_and_lateral_map_over_many_subjects():
    """Every event (not one per state): its subtype follows the state at its onset, and its
    recorded sign/direction matches the rendered deflection - positive at F8 (negative at F7)
    is gaze to the right."""
    tl = StateTimeline([StateSegment(0, 120, "eyes_open"), StateSegment(120, 240, "drowsy")], 0.0)
    f7, f8 = CHANNELS_19.index("F7"), CHANNELS_19.index("F8")
    subtypes, directions = set(), set()
    for seed in SEEDS:
        art = make_artifact("eye_movement")
        _bind(art, tl, seed)
        x = art.render(0, int(240 * FS))
        truth = art.truth()
        assert truth, seed
        for t in truth:
            want = "saccade" if tl.state_at(t.onset_s) == "eyes_open" else "slow_roving"
            assert t.kind == "eye_movement" and t.subtype == want, (seed, t.onset_s)
            sign = t.params["sign"]
            assert sign in (1, -1)
            assert t.params["direction"] == ("right" if sign > 0 else "left")
            a, b = round(t.onset_s * FS), round(t.offset_s * FS)
            first = int(np.flatnonzero(np.abs(x[f8, a:b]) > 1.0)[0])
            assert np.sign(x[f8, a + first]) == sign and np.sign(x[f7, a + first]) == -sign
            assert "F7" in t.channels and "F8" in t.channels
            subtypes.add(t.subtype)
            directions.add(t.params["direction"])
        assert np.corrcoef(x[f7], x[f8])[0, 1] < -0.9, seed
        assert 30 <= np.abs(x[f8, : int(120 * FS)]).max() <= 110, seed
        slow = x[f8, int(120 * FS) :]
        assert band_power(slow, FS, 0.1, 0.8)[0] > 20 * band_power(slow, FS, 3.0, 8.0)[0], seed
    assert subtypes == {"saccade", "slow_roving"} and directions == {"left", "right"}


def test_explicit_empty_rates_mean_never():
    art = make_artifact("eye_movement", rate_by_state={})
    assert art.rate_by_state == {}
    _bind(art, StateTimeline.constant("eyes_open"))
    assert not art.render(0, int(120 * FS)).any() and art.truth() == []

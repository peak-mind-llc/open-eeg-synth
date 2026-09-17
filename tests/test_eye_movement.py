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


def test_subtype_follows_state_and_map_is_lateral():
    tl = StateTimeline([StateSegment(0, 120, "eyes_open"), StateSegment(120, 240, "drowsy")], 0.0)
    art = make_artifact("eye_movement")
    _bind(art, tl)
    x = art.render(0, int(240 * FS))
    truth = art.truth()
    kinds = {round(t.onset_s // 120): t.subtype for t in truth}
    assert kinds.get(0) == "saccade" and kinds.get(1) == "slow_roving"
    f7, f8 = CHANNELS_19.index("F7"), CHANNELS_19.index("F8")
    assert np.corrcoef(x[f7], x[f8])[0, 1] < -0.9
    assert 30 <= np.abs(x[f8, : int(120 * FS)]).max() <= 110
    slow = x[f8, int(120 * FS) :]
    assert band_power(slow, FS, 0.1, 0.8)[0] > 20 * band_power(slow, FS, 3.0, 8.0)[0]
    assert all(
        t.kind == "eye_movement" and "F7" in t.channels and "F8" in t.channels for t in truth
    )

from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts.base import Occupancy, Remedy, RenderContext
from open_eeg_synth.artifacts.blink import Blink
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng

FS = 256.0


def _bind(art, timeline, seed=1):
    head = load_head_model()
    art.bind(
        RenderContext(
            head.channels,
            FS,
            head.electrode_pos,
            head,
            timeline,
            stream_rng(seed, "t"),
            stream_rng(seed, "s"),
            Occupancy(),
            "artifact:blink",
        )
    )


def test_waveform_shape():
    w = Blink.waveform(FS, 0.3)
    assert len(w) == 77 and np.isclose(w.max(), 1.0) and np.argmax(w) < 0.4 * len(w)
    assert w[0] == 0.0 and w[-1] < 0.05


def test_blinks_only_with_eyes_open_and_frontal_positive():
    tl = StateTimeline(
        [StateSegment(0, 60, "eyes_open"), StateSegment(60, 120, "eyes_closed")], 0.0
    )
    art = make_artifact("blink")
    _bind(art, tl)
    x = art.render(0, int(120 * FS))
    truth = art.truth()
    assert 8 <= len(truth) <= 25 and all(t.onset_s < 60 for t in truth)
    fp1, o1 = CHANNELS_19.index("Fp1"), CHANNELS_19.index("O1")
    assert x[fp1].max() > 60 and x[fp1].max() > 4 * np.abs(x[o1]).max()  # |O1| < 0.25 (Task 15)
    t = truth[0]
    assert t.kind == "blink" and t.subtype in ("single", "double") and "Fp1" in t.channels
    assert t.remedies == (Remedy.REMOVE_COMPONENT, Remedy.MASK_SEGMENT)
    assert 60 <= t.peak_uv <= 250
    assert any(t.subtype == "double" for t in art.truth()) or len(truth) < 12

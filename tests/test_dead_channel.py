from __future__ import annotations

import numpy as np

from open_eeg_synth.artifacts.base import Occupancy, Remedy, RenderContext
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.engine import Engine
from open_eeg_synth.seeds import stream_rng, stream_seed
from open_eeg_synth.sensor import SensorNoise

FS = 100.0
CH = ("Fp1", "Fp2", "Cz")


def test_dead_channel_is_flat_and_layer_sum_holds():
    art = make_artifact("dead_channel", channel="fp2", onset_s=2.0, offset_s=6.0)
    assert art.mode == "transform"
    art.bind(
        RenderContext(
            CH,
            FS,
            None,
            None,
            StateTimeline.constant("eyes_open"),
            stream_rng(1, "t"),
            stream_rng(1, "s"),
            Occupancy(),
            "transform:dead_channel",
        )
    )
    eng = Engine(CH, FS, [SensorNoise(3, 10.0, stream_seed(1, "sensor"))], transforms=[art])
    rec = eng.render_all(1000, block=64)
    x = rec.mixed
    assert np.allclose(x, rec.layers["sensor"] + rec.layers["transform:dead_channel"])
    assert x[1, 200:600].std() < 0.5 and x[1, :200].std() > 5.0 and x[0].std() > 5.0
    t = art.truth()[0]
    assert t.kind == "dead_channel" and t.channels == ("Fp2",) and t.onset_s == 2.0
    assert t.remedies == (Remedy.MARK_BAD_CHANNEL, Remedy.INTERPOLATE)

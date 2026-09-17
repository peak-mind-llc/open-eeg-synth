from __future__ import annotations

import math

import numpy as np
import pytest

from open_eeg_synth.artifacts.base import Occupancy, Remedy, RenderContext
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.engine import Engine
from open_eeg_synth.seeds import stream_rng, stream_seed
from open_eeg_synth.sensor import SensorNoise

FS = 100.0
CH = ("Fp1", "Fp2", "Cz")


def _bind(art, fs=FS):
    art.bind(
        RenderContext(
            CH,
            fs,
            None,
            None,
            StateTimeline.constant("eyes_open"),
            stream_rng(1, "t"),
            stream_rng(1, "s"),
            Occupancy(),
            "transform:dead_channel",
        )
    )


def _engine(art):
    return Engine(CH, FS, [SensorNoise(3, 10.0, stream_seed(1, "sensor"))], transforms=[art])


def test_dead_channel_is_flat_and_layer_sum_holds():
    art = make_artifact("dead_channel", channel="fp2", onset_s=2.0, offset_s=6.0)
    assert art.mode == "transform"
    _bind(art)
    rec = _engine(art).render_all(1000, block=64)
    x = rec.mixed
    assert np.allclose(x, rec.layers["sensor"] + rec.layers["transform:dead_channel"])
    assert x[1, 200:600].std() < 0.5 and x[1, :200].std() > 5.0 and x[0].std() > 5.0
    t = art.truth()[0]
    assert t.kind == "dead_channel" and t.channels == ("Fp2",) and t.onset_s == 2.0
    assert t.offset_s == 6.0
    assert t.remedies == (Remedy.MARK_BAD_CHANNEL, Remedy.INTERPOLATE)


def test_truth_times_are_the_actual_sample_boundaries():
    """1.234 s and 2.3456 s at 100 Hz render as samples [123, 235): the truth says 1.23 / 2.35."""
    art = make_artifact("dead_channel", channel="Fp2", onset_s=1.234, offset_s=2.3456)
    _bind(art)
    rec = _engine(art).render_all(400, block=37)
    delta = rec.layers["transform:dead_channel"][1]
    assert np.flatnonzero(delta).tolist() == list(range(123, 235))
    assert rec.mixed[1, 123:235].std() < 0.5 and rec.mixed[1, 235:].std() > 5.0
    t = art.truth()[0]
    assert t.onset_s == 123 / FS and t.offset_s == 235 / FS
    assert art.params()["onset_s"] == 1.234  # the constructor arguments are kept as given
    open_ended = make_artifact("dead_channel", channel="Fp2", onset_s=1.234)
    _bind(open_ended)
    open_ended.render_transform(0, 200, np.zeros((3, 200), np.float32))
    assert open_ended.truth()[0].onset_s == 1.23 and open_ended.truth()[0].offset_s is None


def test_truth_is_listed_only_once_rendering_reaches_the_onset():
    art = make_artifact("dead_channel", channel="Cz", onset_s=1.0, offset_s=2.0)
    _bind(art)
    assert art.truth() == []
    art.render_transform(0, 50, np.zeros((3, 50), np.float32))
    art.render_transform(50, 50, np.zeros((3, 50), np.float32))  # samples [0, 100): not yet
    assert art.truth() == []
    art.render_transform(100, 1, np.zeros((3, 1), np.float32))  # sample 100 is the onset
    assert [t.channels for t in art.truth()] == [("Cz",)]
    _bind(art)  # a rebind starts over
    assert art.truth() == []


@pytest.mark.parametrize(
    ("onset_s", "offset_s"), [(-0.5, 2.0), (2.0, 2.0), (3.0, 2.0), (-1.0, None)]
)
def test_rejects_a_negative_or_empty_span(onset_s, offset_s):
    with pytest.raises(ValueError):
        make_artifact("dead_channel", channel="Fp1", onset_s=onset_s, offset_s=offset_s)


@pytest.mark.parametrize(
    ("onset_s", "offset_s"),
    [
        (0.0, math.inf),
        (1.0, math.inf),
        (0.0, math.nan),
        (math.inf, None),
        (math.inf, math.inf),
        (math.nan, 2.0),
        (-math.inf, 2.0),
    ],
)
def test_rejects_a_non_finite_span(onset_s, offset_s):
    """Both ends must be finite; an infinite offset used to pass and then crash at bind with an
    OverflowError (an open-ended span is offset_s=None)."""
    with pytest.raises(ValueError, match="finite"):
        make_artifact("dead_channel", channel="Fp1", onset_s=onset_s, offset_s=offset_s)


def test_rejects_a_span_that_rounds_to_no_samples():
    art = make_artifact("dead_channel", channel="Fp1", onset_s=1.0, offset_s=1.004)
    with pytest.raises(ValueError):
        _bind(art)  # at 100 Hz both ends round to sample 100


def test_chunking_does_not_change_the_dead_span():
    whole = make_artifact("dead_channel", channel="Fp2", onset_s=1.234)
    _bind(whole)
    chunked = make_artifact("dead_channel", channel="Fp2", onset_s=1.234)
    _bind(chunked)
    a = _engine(whole).render_all(3001, block=4096).mixed
    b = _engine(chunked).render_all(3001, block=7).mixed
    assert np.array_equal(a, b)

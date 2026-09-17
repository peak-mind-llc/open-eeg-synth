from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.artifacts.base import (
    Event,
    EventArtifact,
    Occupancy,
    Remedy,
    RenderContext,
    TruthRecord,
)
from open_eeg_synth.artifacts.registry import ARTIFACTS, make_artifact, register
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.seeds import stream_rng

FS = 100.0
CH = ("A", "B", "C")


@register
class Pulse(EventArtifact):
    kind = "test_pulse"

    def __init__(self, *, rate_by_state=None, min_gap_s=0.0, exclusive=False, length_s=0.5):
        super().__init__(
            rate_by_state=rate_by_state or {"eyes_open": 2.0},
            min_gap_s=min_gap_s,
            exclusive=exclusive,
        )
        self.length_s = length_s

    def make_event(self, onset):
        n = int(self.length_s * self.fs)
        amp = float(self.ctx.rng.uniform(1.0, 2.0))
        truth = TruthRecord(
            "test_pulse",
            None,
            None,
            ("A",),
            onset / self.fs,
            (onset + n) / self.fs,
            amp,
            (Remedy.MASK_SEGMENT,),
            self.layer_name,
            {"amp": amp},
        )
        return Event.from_pattern(onset, np.array([1.0, 0.5, 0.0]), amp * np.ones(n), truth)


def _ctx(timeline, seed=1, occupancy=None, name="artifact:test_pulse"):
    return RenderContext(
        CH,
        FS,
        None,
        None,
        timeline,
        stream_rng(seed, "a"),
        stream_rng(seed, "s"),
        occupancy or Occupancy(),
        name,
    )


def test_registry_and_params():
    assert ARTIFACTS["test_pulse"] is Pulse
    art = make_artifact("test_pulse", length_s=0.2)
    assert art.params() == {
        "rate_by_state": {"eyes_open": 2.0},
        "min_gap_s": 0.0,
        "exclusive": False,
        "length_s": 0.2,
    }
    with pytest.raises(KeyError):
        make_artifact("nope")


def test_rate_follows_state_and_events_carry_truth():
    tl = StateTimeline([StateSegment(0, 30, "eyes_open"), StateSegment(30, 60, "eyes_closed")], 0.0)
    art = Pulse(rate_by_state={"eyes_open": 1.0}, length_s=0.1)
    art.bind(_ctx(tl))
    x = art.render(0, int(60 * FS))
    truth = art.truth()
    onsets = np.array([t.onset_s for t in truth])
    assert 18 <= (onsets < 30).sum() <= 45  # Poisson(30) for 30 s at 1/s; events rarely pushed
    assert (onsets >= 30.5).sum() == 0  # none in the eyes-closed half (a push moves one < 0.1 s)
    assert x.shape == (3, 6000) and x[2].max() == 0.0 and x[1].max() <= 0.5 * x[0].max()
    assert all(
        t.layer == "artifact:test_pulse" and t.remedies == (Remedy.MASK_SEGMENT,) for t in truth
    )
    assert truth[0].to_dict()["remedies"] == ["mask-segment"]


def test_min_gap_and_exclusive_occupancy():
    tl = StateTimeline.constant("eyes_open")
    art = Pulse(rate_by_state={"eyes_open": 0.5}, min_gap_s=1.0, length_s=0.5)
    art.bind(_ctx(tl))
    art.render(0, int(60 * FS))
    on = np.array([t.onset_s for t in art.truth()])
    assert np.all(np.diff(on) >= 1.5 - 1e-9)  # length + gap

    occ = Occupancy()
    a = Pulse(rate_by_state={"eyes_open": 0.4}, exclusive=True, length_s=1.0)
    b = Pulse(rate_by_state={"eyes_open": 0.4}, exclusive=True, length_s=1.0)
    a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:test_pulse#1"))
    b.bind(_ctx(tl, seed=2, occupancy=occ, name="artifact:test_pulse#2"))
    for t0 in range(0, 6000, 500):
        a.render(t0, 500)
        b.render(t0, 500)
    spans = sorted([(t.onset_s, t.offset_s) for t in a.truth() + b.truth()])
    assert all(spans[i][1] <= spans[i + 1][0] + 1e-9 for i in range(len(spans) - 1))


def test_chunk_invariance_of_event_scheduling():
    tl = StateTimeline.constant("eyes_open")

    def run(chunks):
        art = Pulse(rate_by_state={"eyes_open": 1.0})
        art.bind(_ctx(tl, seed=3))
        out, t0 = [], 0
        for n in chunks:
            out.append(art.render(t0, n))
            t0 += n
        return np.concatenate(out, axis=1), [t.onset_s for t in art.truth()]

    whole, t_whole = run([6000])
    chunked, t_chunked = run([13, 400, 1587, 4000])
    assert np.array_equal(whole, chunked) and t_whole == t_chunked

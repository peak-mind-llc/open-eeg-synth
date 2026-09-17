from __future__ import annotations

import numpy as np

from open_eeg_synth.brain.placement import placed_centres, region_centres
from open_eeg_synth.brain.rhythm import BurstGate, Rhythm, RhythmSpec
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.seeds import stream_rng, stream_seed
from tests.helpers import band_power, render_whole_and_chunked

FS = 256.0


def test_placed_centres_mirror_pairs_and_midline():
    head = load_head_model()
    c = placed_centres(head, ("C3", "C4", "Cz"), stream_rng(1, "subject:rhythm:smr"))
    assert len(c) == 3
    assert c[1] == head.mirror_source(c[0])
    assert abs(head.source_pos[c[2], 0]) < 0.012  # near the midline
    assert placed_centres(head, ("C3", "C4", "Cz"), stream_rng(1, "subject:rhythm:smr")) == c


def test_region_centres_posterior_mirrored():
    head = load_head_model()
    c = region_centres(head, "posterior", 4, stream_rng(1, "subject:rhythm:alpha"))
    assert len(c) == 4 and c[2] == head.mirror_source(c[0])
    assert all(head.source_pos[i, 1] < -0.04 for i in c)


def test_rhythm_peaks_at_f0_on_targets_and_is_chunk_invariant():
    head = load_head_model()
    spec = RhythmSpec("smr", f0_hz=13.5, amp_uv=6.0, sites=("C3", "C4", "Cz"))
    centres = placed_centres(head, spec.sites, stream_rng(1, "subject:rhythm:smr"))
    tl = StateTimeline.constant("eyes_closed")

    def make():
        return Rhythm(head, FS, spec, tl, centres, 13.5, stream_seed(1, "eyes_closed:rhythm:smr"))

    whole, chunked = render_whole_and_chunked(make, int(60 * FS), np.random.default_rng(0))
    assert np.allclose(whole, chunked, atol=1e-4)
    c3, o1 = CHANNELS_19.index("C3"), CHANNELS_19.index("O1")
    inband = band_power(whole, FS, 12.5, 14.5)
    around = band_power(whole, FS, 17.0, 19.0)
    assert inband[c3] > 20 * around[c3]
    assert inband[c3] > 3 * inband[o1]
    peak = np.sqrt(2) * whole[c3].std()
    assert 3.0 < peak < 12.0  # amp_uv 6 at the loudest target, envelope 0.25-1.8


def test_state_gain_and_burst_gate():
    head = load_head_model()
    tl = StateTimeline(
        [StateSegment(0, 10, "eyes_closed"), StateSegment(10, 20, "eyes_open")], ramp_s=0.0
    )
    spec = RhythmSpec("alpha", 10.0, 20.0, sites=("O1", "O2"), state_gain={"eyes_open": 0.3})
    centres = placed_centres(head, spec.sites, stream_rng(2, "p"))
    r = Rhythm(head, FS, spec, tl, centres, 10.0, stream_seed(2, "t"))
    x = r.render(0, int(20 * FS))
    o1 = CHANNELS_19.index("O1")
    assert x[o1, : int(10 * FS)].std() > 2.5 * x[o1, int(10 * FS) :].std()

    g = RhythmSpec(
        "b", 6.0, 30.0, sites=("Fz",), burst=BurstGate(on_s=(1, 1), off_s=(2, 2), ramp_s=0.0)
    )
    rb = Rhythm(
        head,
        FS,
        g,
        StateTimeline.constant("eyes_closed"),
        placed_centres(head, g.sites, stream_rng(3, "p")),
        6.0,
        stream_seed(3, "t"),
    )
    y = rb.render(0, int(9 * FS))[CHANNELS_19.index("Fz")]
    active = np.abs(y).reshape(9, int(FS)).max(axis=1) > 1.0
    assert 2 <= active.sum() <= 4  # on for 1 s every 3 s, starting after the first off period


def test_rhythm_spec_roundtrip():
    s = RhythmSpec(
        "theta", 6.0, 7.0, sites=("Fz", "Cz"), state_gain={"drowsy": 2.0}, burst=BurstGate()
    )
    assert RhythmSpec.from_dict(s.to_dict()) == s

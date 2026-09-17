from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.artifacts import patterns
from open_eeg_synth.artifacts.base import Occupancy, Remedy, RenderContext
from open_eeg_synth.artifacts.blink import Blink
from open_eeg_synth.artifacts.registry import make_artifact
from open_eeg_synth.brain.state import StateSegment, StateTimeline
from open_eeg_synth.case import ArtifactSpec, make_case
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import load_head_model
from open_eeg_synth.recipes import resting_case
from open_eeg_synth.seeds import stream_rng

FS = 256.0
SEEDS = range(1, 25)  # map assertions hold for every one of these, not one lucky seed


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


def test_blinks_only_with_eyes_open_frontal_map_over_many_subjects():
    """Per subject: blinks only with eyes open; Fp1 and Fp2 always in the truth channels, O1/O2
    never, |O1|, |O2| < 0.15 on the referenced map; the drawn peak is the peak at the full-head
    loudest channel. Counts and the double-blink share are pooled over subjects (a single 60 s
    subject has a 9 % chance of no double blink at all)."""
    tl = StateTimeline(
        [StateSegment(0, 60, "eyes_open"), StateSegment(60, 120, "eyes_closed")], 0.0
    )
    fp1, fp2 = CHANNELS_19.index("Fp1"), CHANNELS_19.index("Fp2")
    o1, o2 = CHANNELS_19.index("O1"), CHANNELS_19.index("O2")
    n_events = n_double = 0
    worst_o = 0.0
    for seed in SEEDS:
        art = make_artifact("blink")
        _bind(art, tl, seed)
        x = art.render(0, int(120 * FS))
        truth = art.truth()
        assert truth and all(t.onset_s < 60 for t in truth), seed
        assert {"Fp1", "Fp2"} <= set(art.channels) and not {"O1", "O2"} & set(art.channels)
        p = art.pattern
        worst_o = max(worst_o, abs(p[o1]), abs(p[o2]))
        loud = int(np.argmax(np.abs(p)))
        assert loud in (fp1, fp2) and np.isclose(p[loud], 1.0), seed
        assert x[loud].max() == pytest.approx(max(t.peak_uv for t in truth), rel=1e-5)
        assert x[fp1].max() > 60 and x[fp1].max() > 6 * np.abs(x[o1]).max()
        for t in truth:
            assert t.kind == "blink" and t.subtype in ("single", "double")
            assert t.channels == art.channels
            assert t.remedies == (Remedy.REMOVE_COMPONENT, Remedy.MASK_SEGMENT)
            assert 60 <= t.peak_uv <= 250
        n_events += len(truth)
        n_double += sum(t.subtype == "double" for t in truth)
    assert worst_o < 0.15
    # 0.25 / s for 60 s x 24 subjects = 360 expected (Poisson sd 19)
    assert 280 <= n_events <= 440, n_events
    assert 0.08 <= n_double / n_events <= 0.23, n_double / n_events


def test_explicit_empty_rates_mean_never():
    art = make_artifact("blink", rate_by_state={})
    assert art.rate_by_state == {}
    _bind(art, StateTimeline.constant("eyes_open"))
    assert not art.render(0, int(120 * FS)).any() and art.truth() == []


def test_blink_at_o1_is_the_same_on_a_subset_case():
    """Through make_case: a blink's peak at O1 on an O1/O2/T3/T4 case equals its O1 value on
    the 19-channel case of the same subject (the map is not re-normalised over the subset)."""
    arts = (ArtifactSpec("blink", {}),)
    peak = {}
    for label, chs in (("sub", ("O1", "O2", "T3", "T4")), ("full", CHANNELS_19)):
        case = make_case(resting_case(13, duration_s=60.0, artifacts=arts, channels=chs))
        rec = case.recordings["eyes_open"]
        peak[label] = float(np.abs(rec.layers["artifact:blink"][rec.channels.index("O1")]).max())
    assert peak["full"] > 0.1
    assert peak["sub"] == pytest.approx(peak["full"], rel=0.01)


def test_truth_channels_are_those_at_or_above_0_3_of_the_full_head_peak(monkeypatch):
    """With the subject jitter fixed at z = 0 the referenced map reads Fp1 1.000, F3 0.331,
    F2 0.307, F1 0.293, Fz 0.287 and C3 0.078. The truth lists exactly the channels at or above
    0.3: F1, just under, is left out (a threshold of 0.29 would list it, one of 0.31 or 0.5
    would drop F2 or F3)."""
    real = patterns.empirical
    calls = []

    def at_z0(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **{**kwargs, "z": 0.0})

    monkeypatch.setattr(patterns, "empirical", at_z0)
    chs = ("Fp1", "F3", "F2", "F1", "Fz", "C3")
    art = make_artifact("blink")
    art.bind(
        RenderContext(
            chs,
            FS,
            None,
            None,
            StateTimeline.constant("eyes_open"),
            stream_rng(1, "t"),
            stream_rng(1, "s"),
            Occupancy(),
            "artifact:blink",
        )
    )
    assert len(calls) == 1 and calls[0]["rng"] is not None
    assert np.allclose(art.pattern, [1.0, 0.3309, 0.3070, 0.2932, 0.2871, 0.0781], atol=0.0005)
    assert art.channels == ("Fp1", "F3", "F2")
    art.render(0, int(60 * FS))
    truth = art.truth()
    assert truth and all(t.channels == ("Fp1", "F3", "F2") for t in truth)

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


def _random_partition(rng, total, one_sample_prob=0.3, max_chunk=400):
    """Partitions of `total` samples, deliberately including some 1-sample chunks."""
    parts, t = [], 0
    while t < total:
        n = 1 if rng.random() < one_sample_prob else int(rng.integers(1, max_chunk))
        n = min(n, total - t)
        parts.append(n)
        t += n
    return parts


def test_two_exclusive_artifacts_are_chunk_and_call_order_invariant():
    """Two exclusive plug-ins sharing an Occupancy: whole vs >=20 random partitions (incl.
    1-sample chunks) give bit-identical samples and truth, and their spans never overlap
    (ruling P9/§8.2: the shared scheduler, not a per-artifact one, must be chunk-invariant)."""
    tl = StateTimeline.constant("eyes_open")

    def run_pair(chunks, seed_a=1, seed_b=2, rate=0.4, length_s=1.0):
        occ = Occupancy()
        a = Pulse(rate_by_state={"eyes_open": rate}, exclusive=True, length_s=length_s)
        b = Pulse(rate_by_state={"eyes_open": rate}, exclusive=True, length_s=length_s)
        a.bind(_ctx(tl, seed=seed_a, occupancy=occ, name="artifact:test_pulse#1"))
        b.bind(_ctx(tl, seed=seed_b, occupancy=occ, name="artifact:test_pulse#2"))
        xa, xb, t0 = [], [], 0
        for n in chunks:
            xa.append(a.render(t0, n))
            xb.append(b.render(t0, n))
            t0 += n
        return (
            np.concatenate(xa, axis=1),
            np.concatenate(xb, axis=1),
            [t.to_dict() for t in a.truth()],
            [t.to_dict() for t in b.truth()],
        )

    n_total = 6000
    whole_a, whole_b, whole_ta, whole_tb = run_pair([n_total])
    assert whole_ta and whole_tb  # the scenario actually exercises overlap avoidance

    for trial in range(20):
        chunks = _random_partition(np.random.default_rng(trial), n_total)
        ca, cb, tca, tcb = run_pair(chunks)
        assert np.array_equal(whole_a, ca), trial
        assert np.array_equal(whole_b, cb), trial
        assert whole_ta == tca, trial
        assert whole_tb == tcb, trial

    spans = sorted((t["onset_s"], t["offset_s"]) for t in whole_ta + whole_tb)
    assert all(spans[i][1] <= spans[i + 1][0] + 1e-9 for i in range(len(spans) - 1))


def test_ioi_coefficient_of_variation_is_poisson_not_a_fixed_grid():
    """Inter-onset-interval spread/mean ~ 1 (exponential); a fixed candidate grid gives ~0."""
    tl = StateTimeline.constant("eyes_open")
    art = Pulse(rate_by_state={"eyes_open": 2.0}, length_s=0.01)
    art.bind(_ctx(tl, seed=99))
    art.render(0, int(3000 * FS))
    onsets = np.array([t.onset_s for t in art.truth()])
    gaps = np.diff(onsets)
    assert len(gaps) > 1000  # enough draws for a stable estimate
    cv = gaps.std() / gaps.mean()
    assert 0.85 <= cv <= 1.15


def test_two_state_rate_gives_thinned_counts_at_rate_max_equals_max():
    """rate_by_state = {1.0, 0.25}: ~30 events in the 1.0 half, ~7.5 in the 0.25 half over
    30 s each - proof both that thinning actually thins (not accept-if-rate>0) and that
    rate_max is the max over states (not e.g. the min)."""
    tl = StateTimeline([StateSegment(0, 30, "eyes_open"), StateSegment(30, 60, "drowsy")], 0.0)
    opens, drowsies = [], []
    for seed in range(30):
        art = Pulse(rate_by_state={"eyes_open": 1.0, "drowsy": 0.25}, length_s=0.05)
        art.bind(_ctx(tl, seed=seed + 1000))
        art.render(0, int(60 * FS))
        onsets = np.array([t.onset_s for t in art.truth()])
        opens.append(int((onsets < 30).sum()))
        drowsies.append(int(((onsets >= 30) & (onsets < 60)).sum()))
    assert 25 <= np.mean(opens) <= 35  # nominal 30
    assert 4 <= np.mean(drowsies) <= 12  # nominal 7.5


def test_busy_fraction_nominal_rate_holds_under_push_not_drop():
    """rate 0.5/s, length 0.5 s, gap 1.0 s -> 75% busy; the count over a long render still
    matches the nominal rate x duration because pushed candidates are kept, not discarded."""
    tl = StateTimeline.constant("eyes_open")
    art = Pulse(rate_by_state={"eyes_open": 0.5}, min_gap_s=1.0, length_s=0.5)
    art.bind(_ctx(tl, seed=11))
    duration_s = 3600.0
    art.render(0, int(duration_s * FS))
    nominal = 0.5 * duration_s
    assert abs(len(art.truth()) - nominal) < 0.1 * nominal


def test_truth_after_partial_render_excludes_events_pushed_past_its_end():
    """A dense schedule with a long refractory gap pushes some events well past the render
    window; truth() must not report an onset beyond what was actually rendered, even though
    the artifact's internal schedule may already have committed it."""
    tl = StateTimeline.constant("eyes_open")
    art = Pulse(rate_by_state={"eyes_open": 3.0}, min_gap_s=0.5, length_s=0.5)
    art.bind(_ctx(tl, seed=5))
    rendered_n = 1000
    art.render(0, rendered_n)
    reported = art.truth()
    assert reported
    assert max(t.onset_s for t in reported) * FS < rendered_n
    # the internal schedule ran ahead of the render window (proving something was excluded)
    assert len(art._truth) > len(reported)
    assert max(art._truth_onset) >= rendered_n


def test_next_free_resolves_chained_spans_regardless_of_list_order():
    """A candidate that collides with one span may, after being pushed past it, collide with
    another; next_free must keep resolving (multi-pass), not stop after one linear scan."""
    occ = Occupancy()
    for start, end in [(20, 30), (0, 10), (10, 20)]:  # deliberately out of chronological order
        occ.add(start, end)
    assert occ.next_free(5, 3) == 30


def test_exclusive_push_handles_a_truth_record_with_no_offset():
    """An exclusive artifact whose truth has offset_s=None must not crash when pushed, and the
    pushed truth's onset_s must reflect the final (post-push) onset."""

    class NoOffset(EventArtifact):
        kind = "test_no_offset"

        def __init__(self, *, rate_by_state, exclusive=True, length_s=1.0):
            super().__init__(rate_by_state=rate_by_state, exclusive=exclusive)
            self.length_s = length_s

        def make_event(self, onset):
            n = int(self.length_s * self.fs)
            truth = TruthRecord(
                "no_offset",
                None,
                None,
                ("A",),
                onset / self.fs,
                None,
                1.0,
                (Remedy.LEAVE,),
                self.layer_name,
                {},
            )
            return Event.from_pattern(onset, np.array([1.0]), np.ones(n), truth)

    tl = StateTimeline.constant("eyes_open")
    occ = Occupancy()
    occ.add(0, 1000)  # pre-occupy so the first candidate is forced to push
    art = NoOffset(rate_by_state={"eyes_open": 50.0})
    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:no_offset"))
    art.render(0, 2000)
    truth = art.truth()
    assert truth
    assert truth[0].offset_s is None
    assert truth[0].onset_s * FS >= 1000


def test_bind_resets_scheduler_state_for_reuse():
    tl = StateTimeline.constant("eyes_open")
    art = Pulse(rate_by_state={"eyes_open": 2.0}, length_s=0.1)
    art.bind(_ctx(tl, seed=1))
    art.render(0, 1000)
    assert art.truth()  # something was scheduled against the first context

    art.bind(_ctx(tl, seed=2))  # rebind the same instance, as if reused for a new condition
    assert art.truth() == []
    assert art._pending == []


def test_register_rejects_a_class_that_does_not_define_its_own_kind():
    class NoKind(EventArtifact):
        def make_event(self, onset):
            raise NotImplementedError

    with pytest.raises(TypeError, match="kind"):
        register(NoKind)


def test_discover_skips_a_broken_entry_point_without_blocking_the_rest(monkeypatch):
    from open_eeg_synth.artifacts import registry

    class Good(EventArtifact):
        kind = "test_good_third_party"

        def make_event(self, onset):
            raise NotImplementedError

    class BrokenEP:
        name = "broken"

        def load(self):
            raise ImportError("broken third-party plug-in")

    class GoodEP:
        name = "good"

        def load(self):
            return Good

    monkeypatch.setattr(registry, "entry_points", lambda group: [BrokenEP(), GoodEP()])
    monkeypatch.setattr(registry, "_discovered", False)
    try:
        with pytest.warns(UserWarning, match="broken"):
            registry.discover()
        assert "test_good_third_party" in ARTIFACTS
        assert registry._discovered is True
    finally:
        ARTIFACTS.pop("test_good_third_party", None)


def test_truth_record_to_dict_round_trips_without_rounding():
    r = TruthRecord(
        "x", "sub", "left", ("A", "B"), 1 / 256 * 3, 2 / 256, 2.0, (Remedy.NOTCH,), "l", {"a": [1]}
    )
    assert TruthRecord.from_dict(r.to_dict()) == r
    r_no_offset = TruthRecord("x", None, None, (), 1.5, None, 1.0, (), "l")
    assert TruthRecord.from_dict(r_no_offset.to_dict()) == r_no_offset


def test_truth_record_to_dict_is_json_safe_for_numpy_scalars():
    import json

    r = TruthRecord("x", None, None, (), np.float32(1.5), None, np.float32(1.0), (), "l")
    json.dumps(r.to_dict())  # must not raise TypeError on numpy scalar types


def test_event_rejects_a_non_2d_block():
    truth = TruthRecord("x", None, None, (), 0.0, None, 1.0, (), "l")
    with pytest.raises(ValueError):
        Event(0, np.zeros(5), truth)  # 1-D, not (n_ch, L)


@pytest.fixture(autouse=True, scope="module")
def _unregister_test_plugins():
    yield
    ARTIFACTS.pop("test_pulse", None)
    ARTIFACTS.pop("test_no_offset", None)

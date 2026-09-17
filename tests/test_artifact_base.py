from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from open_eeg_synth.artifacts.base import (
    Event,
    EventArtifact,
    Occupancy,
    Remedy,
    RenderContext,
    TransformArtifact,
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
    with pytest.raises(ValueError, match="'nope'.*known.*test_pulse"):
        make_artifact("nope")


def test_transform_artifact_params_without_its_own_init_returns_empty_dict():
    """A `TransformArtifact` subclass that declares no `__init__` of its own inherits
    `object.__init__(self, /, *args, **kwargs)`: `_ParamsMixin.params()` used to walk that
    signature's `*args`/`**kwargs` entries and call `getattr(self, "args")`, raising
    AttributeError instead of reporting "no constructor parameters" as `{}`."""

    class Bare(TransformArtifact):
        kind = "test_bare_transform"

        def render_transform(self, t0, n, mix):
            return mix

        def truth(self):
            return []

    assert Bare().params() == {}


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
    (DESIGN §8.2: the shared scheduler, not a per-artifact one, must be chunk-invariant)."""
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
    monkeypatch.setattr(registry, "_failed_entry_points", {})
    try:
        with pytest.warns(UserWarning, match="broken"):
            registry.discover()
        assert "test_good_third_party" in ARTIFACTS
        assert registry._discovered is True
    finally:
        ARTIFACTS.pop("test_good_third_party", None)


def test_discover_warns_on_every_call_and_names_the_failure_in_the_unknown_kind_error(
    monkeypatch,
):
    """A broken entry point is warned about on every discover() call (not only the first), and
    make_artifact's error for an unrelated unknown kind names it too."""
    from open_eeg_synth.artifacts import registry

    class BrokenEP:
        name = "still_broken"

        def load(self):
            raise ImportError("still broken")

    monkeypatch.setattr(registry, "entry_points", lambda group: [BrokenEP()])
    monkeypatch.setattr(registry, "_discovered", False)
    monkeypatch.setattr(registry, "_failed_entry_points", {})

    with pytest.warns(UserWarning, match="still_broken"):
        registry.discover()  # first call: attempts loading, records and warns
    with pytest.warns(UserWarning, match="still_broken"):
        registry.discover()  # second call: does not re-attempt, but warns again

    with pytest.raises(ValueError, match="still_broken") as exc_info:
        with pytest.warns(UserWarning, match="still_broken"):
            registry.make_artifact("nope")
    assert "still broken" in str(exc_info.value)  # the recorded error text itself


def test_discover_flag_is_set_only_after_the_whole_attempt_not_before(monkeypatch):
    """The "done" flag must reflect having actually finished attempting every entry point, not
    just having started: if listing the entry points itself blows up partway through (not an
    individual entry's own load() failing, which is already handled per-entry), a later call
    should retry rather than silently treat discovery as complete."""
    from open_eeg_synth.artifacts import registry

    class GoodEP:
        name = "good"

        def load(self):
            class Good(EventArtifact):
                kind = "test_good_partial"

                def make_event(self, onset):
                    raise NotImplementedError

            return Good

    def bad_entry_points(group):
        yield GoodEP()
        raise RuntimeError("entry_points() itself failed partway through")

    monkeypatch.setattr(registry, "entry_points", bad_entry_points)
    monkeypatch.setattr(registry, "_discovered", False)
    monkeypatch.setattr(registry, "_failed_entry_points", {})
    try:
        with pytest.raises(RuntimeError):
            registry.discover()
        assert registry._discovered is False
    finally:
        ARTIFACTS.pop("test_good_partial", None)


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


def test_exclusive_push_rebuilds_the_event_at_its_final_onset():
    """A pushed event must be rebuilt via make_event at its final onset, not just have its truth
    patched: anything the plug-in derives from `onset` (state, params, ...) must reflect where
    the event actually landed, not the pre-push candidate."""

    class StateAware(EventArtifact):
        kind = "test_state_aware"

        def __init__(self, *, rate_by_state, exclusive=True, length_s=0.3):
            super().__init__(rate_by_state=rate_by_state, exclusive=exclusive)
            self.length_s = length_s

        def make_event(self, onset):
            n = int(self.length_s * self.fs)
            state = self.ctx.timeline.state_at(onset / self.fs)
            truth = TruthRecord(
                "state_aware",
                state,
                None,
                ("A",),
                onset / self.fs,
                (onset + n) / self.fs,
                1.0,
                (Remedy.LEAVE,),
                self.layer_name,
                {"built_onset": onset, "state": state},
            )
            return Event.from_pattern(onset, np.array([1.0]), np.ones(n), truth)

    tl = StateTimeline([StateSegment(0, 10, "eyes_open"), StateSegment(10, 60, "drowsy")], 0.0)
    occ = Occupancy()
    occ.add(900, 1100)  # busy span straddling the state boundary at sample 1000
    art = StateAware(rate_by_state={"eyes_open": 1.0, "drowsy": 1.0})
    art.bind(_ctx(tl, seed=4, occupancy=occ, name="artifact:test_state_aware"))
    art._next = (8.73, True)  # candidate at sample 873 (still eyes_open), pushed past [900,1100)
    art._consume_next()
    ev = art._pending[0]
    assert ev.onset == 1100  # pushed to the end of the busy span
    assert ev.truth.subtype == "drowsy"  # reflects the state AT the final onset, not at 873
    assert ev.truth.params["built_onset"] == 1100  # make_event really ran again there
    assert ev.truth.onset_s == 1100 / FS


def test_min_gap_after_an_exclusive_push_is_measured_from_the_pushed_event_end():
    """The refractory gap after a pushed event must be measured from where it actually ends,
    not from the pre-push candidate's would-be end (a wrong implementation might compute
    `_earliest = onset + block.shape[1] + gap` using the stale pre-push `onset`)."""
    tl = StateTimeline.constant("eyes_open")
    occ = Occupancy()
    occ.add(0, 500)  # forces a push well past the candidate's own onset
    art = Pulse(rate_by_state={"eyes_open": 1.0}, min_gap_s=2.0, exclusive=True, length_s=0.3)
    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:test_pulse"))
    art._next = (0.10, True)  # candidate at sample 10, well inside [0, 500)
    art._consume_next()
    assert len(art._truth) == 1
    pushed_onset = art._truth_onset[0]
    assert pushed_onset >= 500
    ev_end = pushed_onset + round(0.3 * FS)  # fixed length_s, so this equals ev.end
    assert art._earliest == ev_end + round(2.0 * FS)


def test_truth_boundary_uses_the_stored_integer_onset_not_a_recomputed_float():
    """At fs=100, `29 / fs * fs` evaluates to 28.999999999999996, just under 29. truth() must
    still exclude an event whose onset is exactly at the render boundary, which filtering on the
    recomputed float would not."""
    tl = StateTimeline.constant("eyes_open")
    art = Pulse(rate_by_state={"eyes_open": 1.0}, length_s=0.05)
    art.bind(_ctx(tl, seed=1))
    art._next = (0.10, True)  # accepted at sample 10, pushed by the refractory gap to 29
    art._earliest = 29
    art.render(0, 29)
    assert art.truth() == []
    assert 0.29 * FS < 29  # the float pitfall this guards against


def test_rebind_to_the_same_occupancy_after_it_has_advanced_works():
    """Reusing a plug-in instance (rebinding it to the same Occupancy it was already on, after
    that occupancy has scheduled events) must actually work, not silently produce nothing."""
    tl = StateTimeline.constant("eyes_open")
    occ = Occupancy()
    art = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.2)
    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))
    art.render(0, 3000)
    assert art.truth()

    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))  # rebind, same occupancy
    x = art.render(0, 3000)
    assert art.truth()
    assert np.any(x)


def test_binding_a_new_artifact_into_an_already_advanced_occupancy_raises():
    tl = StateTimeline.constant("eyes_open")
    occ = Occupancy()
    a = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.2)
    a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))
    a.render(0, 3000)

    b = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.2)
    with pytest.raises(ValueError, match="already advanced"):
        b.bind(_ctx(tl, seed=2, occupancy=occ, name="artifact:b"))


def test_rebind_to_a_new_occupancy_removes_it_from_the_old_ones_list():
    tl = StateTimeline.constant("eyes_open")
    occ1, occ2 = Occupancy(), Occupancy()
    art = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.2)
    art.bind(_ctx(tl, seed=1, occupancy=occ1, name="artifact:a"))
    assert art in occ1._exclusive

    art.bind(_ctx(tl, seed=1, occupancy=occ2, name="artifact:a"))
    assert art not in occ1._exclusive
    assert art in occ2._exclusive


def test_truth_before_bind_returns_empty_list():
    class Bare(EventArtifact):
        kind = "test_bare"

        def make_event(self, onset):
            raise NotImplementedError

    art = Bare(rate_by_state={"eyes_open": 1.0})
    assert art.truth() == []


def test_make_event_must_return_an_event_at_the_given_onset():
    """A plug-in whose Event starts somewhere other than the onset it was given must raise,
    naming the plug-in's kind - not spin the exclusive rebuild loop forever colliding with
    itself at the same sample every time."""

    class PreRoll(EventArtifact):
        kind = "test_preroll"

        def make_event(self, onset):
            n = 10
            truth = TruthRecord(
                "preroll",
                None,
                None,
                ("A",),
                (onset - 5) / self.fs,
                None,
                1.0,
                (Remedy.LEAVE,),
                self.layer_name,
                {},
            )
            return Event.from_pattern(onset - 5, np.array([1.0]), np.ones(n), truth)

    tl = StateTimeline.constant("eyes_open")
    art = PreRoll(rate_by_state={"eyes_open": 1.0}, exclusive=True)
    occ = Occupancy()
    # The busy span is where the old loop spun; the check now fires on the FIRST build (sample
    # 150), before any push, so this test never reaches the rebuild path. The push path has its
    # own test below.
    occ.add(100, 200)
    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:test_preroll"))
    art._next = (1.5, True)  # candidate at sample 150, inside the busy span
    with pytest.raises(ValueError, match=r"test_preroll.*onset=150.*at 145"):
        art._consume_next()


def test_onset_check_also_fires_on_a_rebuild_after_a_push():
    """A plug-in whose first build is correct but whose rebuild at a pushed onset returns a
    shifted Event must raise on that rebuild (the push path), naming the pushed onset."""

    class ShiftOnRebuild(EventArtifact):
        kind = "test_shift_on_rebuild"

        def make_event(self, onset):
            start = onset if onset < 200 else onset + 3  # wrong only where the push lands
            truth = TruthRecord(
                "shift", None, None, ("A",), start / self.fs, None, 1.0, (Remedy.LEAVE,), "x", {}
            )
            return Event.from_pattern(start, np.array([1.0]), np.ones(10), truth)

    tl = StateTimeline.constant("eyes_open")
    art = ShiftOnRebuild(rate_by_state={"eyes_open": 1.0}, exclusive=True)
    occ = Occupancy()
    occ.add(100, 200)
    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:test_shift_on_rebuild"))
    assert art.make_event(150).onset == 150  # the first build alone is fine
    art._next = (1.5, True)  # candidate at sample 150, pushed to 200 by the busy span
    with pytest.raises(ValueError, match=r"test_shift_on_rebuild.*onset=200.*at 203"):
        art._consume_next()
    assert art._truth == [] and occ.spans == [(100, 200)]  # nothing was committed


def test_rebind_to_the_same_occupancy_matches_a_fresh_bind():
    """A same-occupancy rebind must drop the artifact's own pre-rebind spans - otherwise its new
    events get pushed around by its own stale ones, and the result depends on how many times it
    was rebound rather than matching a fresh instance with the same seed."""
    tl = StateTimeline.constant("eyes_open")
    occ = Occupancy()
    art = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.2)
    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))
    art.render(0, 3000)
    assert art.truth()

    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))  # rebind, same occupancy
    x_rebind = art.render(0, 3000)
    truth_rebind = [t.to_dict() for t in art.truth()]

    fresh = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.2)
    fresh.bind(_ctx(tl, seed=1, occupancy=Occupancy(), name="artifact:a"))
    x_fresh = fresh.render(0, 3000)
    truth_fresh = [t.to_dict() for t in fresh.truth()]

    assert np.array_equal(x_rebind, x_fresh)
    assert truth_rebind == truth_fresh


def test_refused_bind_leaves_the_artifact_rendering_exactly_as_before():
    """A refused bind (joining an Occupancy that has already advanced) must not half-migrate the
    artifact (switch its ctx, drop it from the old occupancy) before the join is validated - it
    must go on rendering exactly as it would have if the failed bind() call had never happened
    at all."""
    tl = StateTimeline.constant("eyes_open")
    occ1, occ2 = Occupancy(), Occupancy()
    a = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.2)
    a.bind(_ctx(tl, seed=1, occupancy=occ1, name="artifact:a"))
    b = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.2)
    b.bind(_ctx(tl, seed=2, occupancy=occ2, name="artifact:b"))
    b.render(0, 1000)  # advances occ2

    with pytest.raises(ValueError):
        a.bind(_ctx(tl, seed=1, occupancy=occ2, name="artifact:a"))

    assert a.ctx.occupancy is occ1
    assert a in occ1._exclusive
    assert a not in occ2._exclusive

    x = a.render(0, 3000)
    assert np.any(x)
    assert a.truth()

    control = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.2)
    control.bind(_ctx(tl, seed=1, occupancy=Occupancy(), name="artifact:a"))
    xc = control.render(0, 3000)
    assert np.array_equal(x, xc)
    assert [t.to_dict() for t in a.truth()] == [t.to_dict() for t in control.truth()]


def test_rebuild_loop_rechecks_after_a_rebuild_that_changes_length():
    """A rebuild that lands somewhere the event's own natural length is different (here:
    longer) must have that new placement re-checked against occupancy too - rebuilding once and
    trusting the result, without re-checking, can still leave an overlap."""

    class Grow(EventArtifact):
        kind = "test_grow"

        def make_event(self, onset):
            n = 50 if onset < 200 else 100  # the pushed rebuild is longer
            truth = TruthRecord(
                "grow",
                None,
                None,
                ("A",),
                onset / self.fs,
                (onset + n) / self.fs,
                1.0,
                (Remedy.LEAVE,),
                self.layer_name,
                {},
            )
            return Event.from_pattern(onset, np.array([1.0]), np.ones(n), truth)

    tl = StateTimeline.constant("eyes_open")
    occ = Occupancy()
    occ.add(100, 200)
    occ.add(260, 400)  # the grown rebuild (landing at 200, ending 300) still collides with this
    art = Grow(rate_by_state={"eyes_open": 1.0}, exclusive=True)
    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:test_grow"))
    art._next = (1.5, True)  # candidate at sample 150, inside [100, 200)
    art._consume_next()
    assert art._truth_onset == [400]  # pushed past BOTH busy spans, not just the first
    assert art._pending[0].end == 500
    spans = sorted(occ.spans)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_exclusive_tie_break_goes_to_the_first_registered_artifact():
    """When two exclusive artifacts' next candidates land at the exact same sample, the tie must
    resolve to whichever was registered (bound) first - deterministic, independent of anything
    else."""
    tl = StateTimeline.constant("eyes_open")
    occ = Occupancy()
    a = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.5)
    b = Pulse(rate_by_state={"eyes_open": 1.0}, exclusive=True, length_s=0.5)
    a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))  # registered first
    b.bind(_ctx(tl, seed=2, occupancy=occ, name="artifact:b"))
    a._next = (1.0, True)
    b._next = (1.0, True)
    a._rate_max = b._rate_max = 1e-9  # keep any further candidates far away
    occ.advance_exclusive(101)
    assert a._truth_onset == [100]  # a claims the tied sample
    assert b._truth_onset == [150]  # b is pushed past a's [100, 150) claim


def _run_long_stream(occ, cfgs, seeds, names, seconds=600.0, chunk_s=1.0):
    """Bind `cfgs` (Pulse kwargs) as exclusive artifacts sharing `occ`, render `seconds` of
    simulated stream in `chunk_s` chunks, and return (artifacts, max span count seen)."""
    arts = [Pulse(**c, exclusive=True) for c in cfgs]
    for art, seed, name in zip(arts, seeds, names, strict=True):
        art.bind(_ctx(StateTimeline.constant("eyes_open"), seed=seed, occupancy=occ, name=name))
    max_spans = 0
    n = int(chunk_s * FS)
    for t0 in range(0, int(seconds * FS), n):
        for art in arts:
            art.render(t0, n)
        max_spans = max(max_spans, len(occ.spans))
    return arts, max_spans


def test_occupancy_prunes_spans_behind_the_watermark_over_a_long_stream():
    """A long-running stream keeps scheduling exclusive events forever; without pruning,
    `Occupancy.spans` grows by one entry per committed event and never shrinks, so `next_free`'s
    scan gets slower and slower the longer the stream runs. A span that ends at or before every
    registered exclusive artifact's own earliest possible next onset can never overlap a future
    event, so it is safe to drop. `spans` should stay small no matter how long the stream runs,
    scheduling must stay collision-free, and it must match scheduling with pruning switched off
    bit for bit - pruning is a memory optimisation, not a second scheduler."""
    cfgs = [dict(rate_by_state={"eyes_open": 5.0}, min_gap_s=0.05, length_s=0.05)] * 2
    seeds, names = (11, 12), ("artifact:a", "artifact:b")

    occ_pruned = Occupancy()
    arts_pruned, max_spans = _run_long_stream(occ_pruned, cfgs, seeds, names)
    occ_control = Occupancy()
    occ_control._prune_stale_spans = lambda: None  # pruning switched off
    arts_control, _ = _run_long_stream(occ_control, cfgs, seeds, names)

    truth = [t.to_dict() for a in arts_pruned for t in a.truth()]
    truth_control = [t.to_dict() for a in arts_control for t in a.truth()]
    assert len(truth) > 2000  # the scheduler really did keep going the whole time
    assert max_spans < 20  # never grows past a handful of not-yet-safe-to-drop spans
    assert truth == truth_control  # pruning never changes a scheduling decision
    offsets = sorted((t.onset_s, t.offset_s) for t in [tr for a in arts_pruned for tr in a.truth()])
    assert all(offsets[i][1] <= offsets[i + 1][0] + 1e-9 for i in range(len(offsets) - 1))


def test_pruning_stays_bounded_with_a_rare_exclusive_peer():
    """A quiet-but-live peer (rate 0.005/s) commits an event only rarely, so its own `_earliest`
    sits at whatever sample its last (long-ago) commit ended at. Flooring the watermark on
    `_earliest` alone would let this peer hold the whole Occupancy's pruning back to that stale
    value; the fix also floors on the peer's own cached next candidate (`_next`), which keeps
    advancing even while unconsumed, so pruning keeps working regardless. Seed 10 (rather than
    the busy peer's own seed 1) makes it fire early rather than not at all over the 600 s run."""
    cfgs = [
        dict(rate_by_state={"eyes_open": 2.0}, length_s=0.1),
        dict(rate_by_state={"eyes_open": 0.005}, length_s=0.1),
    ]
    seeds, names = (1, 10), ("artifact:a", "artifact:r")

    occ_pruned = Occupancy()
    arts_pruned, max_spans = _run_long_stream(occ_pruned, cfgs, seeds, names)
    occ_control = Occupancy()
    occ_control._prune_stale_spans = lambda: None
    arts_control, _ = _run_long_stream(occ_control, cfgs, seeds, names)

    truth = [t.to_dict() for a in arts_pruned for t in a.truth()]
    truth_control = [t.to_dict() for a in arts_control for t in a.truth()]
    assert len(arts_pruned[1].truth()) > 0  # the rare peer really did fire, and fired early
    assert arts_pruned[1].truth()[0].onset_s < 30.0
    assert max_spans < 20  # bounded despite the rare peer (see docstring)
    assert truth == truth_control


def test_pruning_stays_bounded_with_a_peer_gated_off_by_the_timeline():
    """A peer whose only nonzero rate applies to a state ("drowsy") this constant-eyes_open
    timeline never visits still has `_rate_max > 0` (so it is not excluded outright): it keeps
    drawing and rejecting candidates forever and never commits, so its `_earliest` sits at 0 for
    the whole stream. Flooring on `_earliest` alone would hold the shared watermark at 0 forever;
    flooring on the peer's own cached `_next` candidate (which tracks close to "now" even though
    every draw is rejected) keeps pruning working."""
    cfgs = [
        dict(rate_by_state={"eyes_open": 2.0}, length_s=0.1),
        dict(rate_by_state={"drowsy": 1.0}, length_s=0.1),
    ]
    seeds, names = (1, 2), ("artifact:a", "artifact:g")

    occ_pruned = Occupancy()
    arts_pruned, max_spans = _run_long_stream(occ_pruned, cfgs, seeds, names)
    occ_control = Occupancy()
    occ_control._prune_stale_spans = lambda: None
    arts_control, _ = _run_long_stream(occ_control, cfgs, seeds, names)

    truth = [t.to_dict() for a in arts_pruned for t in a.truth()]
    truth_control = [t.to_dict() for a in arts_control for t in a.truth()]
    assert arts_pruned[1].truth() == []  # the gated peer really never fires
    assert max_spans < 20  # bounded despite the permanently-gated peer
    assert truth == truth_control


def _dataclass_peer_cls(register_kwargs=None, **dataclass_kwargs):
    """Build (and register) a fresh `@dataclass` `EventArtifact` subclass: dataclass's generated
    `__init__` runs first and sets only the declared fields, so `__post_init__` calls
    `EventArtifact.__init__` itself to finish construction the way a hand-written plug-in would.
    """

    @dataclass(**dataclass_kwargs)
    class DataclassPeer(EventArtifact):
        kind = (register_kwargs or {}).get("kind", "test_dataclass_peer")
        rate: float = 2.0
        length_s: float = 0.3

        def __post_init__(self) -> None:
            EventArtifact.__init__(self, rate_by_state={"eyes_open": self.rate}, exclusive=True)

        def make_event(self, onset):
            n = int(self.length_s * self.fs)
            truth = TruthRecord(
                self.kind,
                None,
                None,
                ("A",),
                onset / self.fs,
                (onset + n) / self.fs,
                1.0,
                (Remedy.MASK_SEGMENT,),
                self.layer_name,
                {},
            )
            return Event.from_pattern(onset, np.array([1.0, 0.0, 0.0]), np.ones(n), truth)

    DataclassPeer.kind = (register_kwargs or {}).get("kind", "test_dataclass_peer")
    return register(DataclassPeer)


def test_pruning_a_dataclass_plugins_span_does_not_crash():
    """Occupancy._pruned_owners must key owners by identity, not put them in a plain set: a
    mutable `@dataclass` plug-in gets `__eq__` (the dataclass default) without `__hash__` (which
    a non-frozen dataclass sets to None precisely because it defined `__eq__`), so adding one to
    a `set` raised ``TypeError: unhashable type`` on this Occupancy's very first prune."""
    # Different field values, so the two instances are not value-equal to each other: this lands
    # squarely on `_prune_stale_spans`, which is where a set-typed `_pruned_owners` would raise.
    Peer = _dataclass_peer_cls({"kind": "test_dataclass_peer_unhashable"})
    try:
        tl = StateTimeline.constant("eyes_open")
        occ = Occupancy()
        a = Peer(rate=2.0, length_s=0.3)
        b = Peer(rate=2.5, length_s=0.31)
        a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))
        b.bind(_ctx(tl, seed=2, occupancy=occ, name="artifact:b"))
        for t0 in range(0, 6000, 100):
            a.render(t0, 100)
            b.render(t0, 100)
        assert occ._pruned_owners  # pruning ran to completion without raising
    finally:
        ARTIFACTS.pop("test_dataclass_peer_unhashable", None)


def test_pruning_keys_owners_by_identity_not_by_equality():
    """A value-equal-but-distinct peer must not be treated as "the same owner" when
    `add_exclusive` decides whether a same-occupancy rebind is still safe: pruning must be keyed
    on which *object* had a span discarded, not on its current field values. `b` never itself
    binds to `occ` (test_value_equal_plugins_are_separate_participants covers two value-equal
    participants); it only ever owns one span there, exactly as a real peer's already-pruned
    history would look from `add_exclusive`'s point of view."""
    Peer = _dataclass_peer_cls({"kind": "test_dataclass_peer_hashable"}, unsafe_hash=True)
    try:
        tl = StateTimeline.constant("eyes_open")
        occ = Occupancy()
        a = Peer(rate=2.0, length_s=0.3)
        b = Peer(rate=2.0, length_s=0.3)  # equal to a, but a different object
        assert a == b and a is not b

        a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))
        occ.add(0, 10, owner=b)
        for t0 in range(0, 6000, 100):
            a.render(t0, 100)  # advances a's own floor well past b's span
        assert any(oid != id(a) for oid in occ._pruned_owners)  # b's span really got pruned

        with pytest.raises(ValueError, match="pruning"):
            a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))
    finally:
        ARTIFACTS.pop("test_dataclass_peer_hashable", None)


def test_value_equal_plugins_are_separate_participants():
    """The Occupancy registers exclusive participants by identity (DESIGN §5.2), so two
    value-equal dataclass plug-ins both join, both schedule and never overlap, and rebinding one
    elsewhere removes that one only."""
    Peer = _dataclass_peer_cls({"kind": "test_dataclass_peer_equal"})
    try:
        tl = StateTimeline.constant("eyes_open")
        occ, other = Occupancy(), Occupancy()
        a, b = Peer(rate=2.0, length_s=0.3), Peer(rate=2.0, length_s=0.3)
        assert a == b and a is not b
        a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))
        b.bind(_ctx(tl, seed=2, occupancy=occ, name="artifact:b"))
        assert len(occ._exclusive) == 2
        for t0 in range(0, 3000, 100):
            a.render(t0, 100)
            b.render(t0, 100)
        assert a.truth() and b.truth()
        spans = sorted(
            (round(t.onset_s * FS), round(t.offset_s * FS)) for t in a.truth() + b.truth()
        )
        assert all(end <= nxt for (_, end), (nxt, _) in zip(spans, spans[1:], strict=False))
        b.bind(_ctx(tl, seed=2, occupancy=other, name="artifact:b"))
        assert len(occ._exclusive) == 1 and occ._exclusive[0] is a
    finally:
        ARTIFACTS.pop("test_dataclass_peer_equal", None)


def test_same_occupancy_rebind_after_pruning_a_peers_span_is_refused():
    """A same-occupancy rebind resets the rebinding artifact's own scheduler to sample 0 and
    drops only its own spans (`bind`'s docstring) - it relies on every *other* participant's
    already-committed spans still being in `occ.spans` to avoid re-colliding with their history.
    Once pruning has discarded a peer's span, that history is gone, and a naive rebind can
    schedule straight through it, overlapping the peer. The occupancy must instead refuse the
    rebind."""
    tl = StateTimeline.constant("eyes_open")
    occ = Occupancy()
    a = Pulse(rate_by_state={"eyes_open": 2.0}, exclusive=True, length_s=0.3)
    b = Pulse(rate_by_state={"eyes_open": 2.0}, exclusive=True, length_s=0.3)
    a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))
    b.bind(_ctx(tl, seed=2, occupancy=occ, name="artifact:b"))
    for t0 in range(0, 6000, 100):
        a.render(t0, 100)
        b.render(t0, 100)
    assert any(oid != id(a) for oid in occ._pruned_owners)  # really did prune a peer's span

    with pytest.raises(ValueError, match="pruning"):
        a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))


def test_same_occupancy_rebind_stays_collision_free_with_pruning_off():
    """Control for the refusal above: with pruning switched off, nothing was ever forgotten, so
    the pre-existing same-occupancy rebind path (drop only the rebinding artifact's own spans,
    replay from sample 0) is exactly as safe as it always was."""
    tl = StateTimeline.constant("eyes_open")
    occ = Occupancy()
    occ._prune_stale_spans = lambda: None  # pruning switched off
    a = Pulse(rate_by_state={"eyes_open": 2.0}, exclusive=True, length_s=0.3)
    b = Pulse(rate_by_state={"eyes_open": 2.0}, exclusive=True, length_s=0.3)
    a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))
    b.bind(_ctx(tl, seed=2, occupancy=occ, name="artifact:b"))
    for t0 in range(0, 6000, 100):
        a.render(t0, 100)
        b.render(t0, 100)

    a.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))  # allowed: nothing was pruned
    for t0 in range(0, 6000, 100):
        a.render(t0, 100)

    spans = sorted((t.onset_s, t.offset_s) for t in a.truth() + b.truth())
    assert all(spans[i][1] <= spans[i + 1][0] + 1e-9 for i in range(len(spans) - 1))


def test_single_artifact_rebind_still_works_after_pruning():
    """A same-occupancy rebind stays unrestricted when there is no peer to lose history about:
    every span pruning discards on a one-artifact Occupancy is that same artifact's own, so
    `add_exclusive`'s refusal never triggers and rebinding keeps working."""
    tl = StateTimeline.constant("eyes_open")
    occ = Occupancy()
    art = Pulse(rate_by_state={"eyes_open": 5.0}, exclusive=True, min_gap_s=0.05, length_s=0.05)
    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))
    for t0 in range(0, 6000, 100):
        art.render(t0, 100)
    assert occ._pruned_owners  # pruning actually ran

    art.bind(_ctx(tl, seed=1, occupancy=occ, name="artifact:a"))  # same occupancy, no exception
    x = art.render(0, 3000)
    assert np.any(x)
    assert art.truth()


@pytest.fixture(autouse=True, scope="module")
def _unregister_test_plugins():
    yield
    ARTIFACTS.pop("test_pulse", None)
    ARTIFACTS.pop("test_no_offset", None)
    ARTIFACTS.pop("test_state_aware", None)
    ARTIFACTS.pop("test_bare", None)
    ARTIFACTS.pop("test_preroll", None)
    ARTIFACTS.pop("test_grow", None)
    ARTIFACTS.pop("test_shift_on_rebuild", None)

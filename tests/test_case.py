from __future__ import annotations

import dataclasses
import json
import math
import warnings

import numpy as np
import pytest

from open_eeg_synth.artifacts import (
    Event,
    EventArtifact,
    Remedy,
    TransformArtifact,
    TruthRecord,
    patterns,
)
from open_eeg_synth.artifacts.registry import ARTIFACTS, register
from open_eeg_synth.brain.plants import (
    FocalSlow,
    LateralImbalance,
    PeakShift,
    ReducedRhythm,
    RhythmicBursts,
    WidespreadExcess,
)
from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.case import (
    ArtifactSpec,
    CaseSpec,
    ConditionSpec,
    SensorSpec,
    case_id_for,
    compiled_rhythms,
    make_case,
    make_engine,
    make_subject,
)
from open_eeg_synth.channels import CHANNELS_19
from open_eeg_synth.headmodel import HeadModel
from open_eeg_synth.recipes import resting_case
from open_eeg_synth.seeds import stream_rng
from tests.helpers import band_power


def test_spec_roundtrip_digest_and_case_id():
    spec = resting_case(123, duration_s=8.0, artifacts=())
    assert CaseSpec.from_dict(spec.to_dict()) == spec
    assert spec.digest() == CaseSpec.from_dict(spec.to_dict()).digest()
    case = make_case(spec)
    assert case.case_id.startswith("synth-") and len(case.case_id) == 14
    assert set(case.recordings) == {"eyes_closed", "eyes_open"}
    rec = case.recordings["eyes_closed"]
    assert rec.channels == CHANNELS_19 and rec.n_samples == 8 * 256
    assert set(rec.layers) == {"brain", "sensor"}
    assert np.allclose(rec.mixed, rec.layers["brain"] + rec.layers["sensor"])


def test_subject_is_shared_across_conditions_and_seed_dependent():
    a = make_subject(resting_case(5, duration_s=1.0, artifacts=()))
    b = make_subject(resting_case(5, duration_s=1.0, artifacts=()))
    c = make_subject(resting_case(6, duration_s=1.0, artifacts=()))
    assert a.placements == b.placements and a.f0_hz == b.f0_hz
    assert a.head.perturbation == b.head.perturbation
    assert a.placements != c.placements
    case = make_case(resting_case(5, duration_s=8.0, artifacts=()))
    o1 = CHANNELS_19.index("O1")
    ec = band_power(case.recordings["eyes_closed"].mixed, 256.0, 8, 13)[o1]
    eo = band_power(case.recordings["eyes_open"].mixed, 256.0, 8, 13)[o1]
    assert eo < 0.5 * ec


def test_unperturbed_head_when_asked():
    spec = resting_case(7, duration_s=1.0, artifacts=())
    spec = CaseSpec.from_dict({**spec.to_dict(), "perturb_head": False})
    assert make_subject(spec).head.perturbation is None


def test_make_subject_warns_once_per_process_about_a_broken_entry_point(monkeypatch):
    """``discover()`` itself re-warns about a broken entry point on every call, by design (kept
    as-is - see registry's own test_discover_warns_on_every_call_and_names_the_failure_in_the_
    keyerror); but ``make_subject`` calling it once per case must not spam that same warning for
    every case built in this process afterwards - only the first case should surface it."""
    import open_eeg_synth.case as case_module
    from open_eeg_synth.artifacts import registry

    class BrokenEP:
        name = "make_subject_still_broken"

        def load(self):
            raise ImportError("broken for the make_subject warn-once test")

    monkeypatch.setattr(registry, "entry_points", lambda group: [BrokenEP()])
    monkeypatch.setattr(registry, "_discovered", False)
    monkeypatch.setattr(registry, "_failed_entry_points", {})
    monkeypatch.setattr(case_module, "_warned_broken_entry_points", set())

    spec = resting_case(1, duration_s=1.0, artifacts=())
    with pytest.warns(UserWarning, match="make_subject_still_broken"):
        make_subject(spec)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_subject(spec)
    assert caught == []


class _Tick(EventArtifact):
    """A flat, all-channel pulse used only to exercise make_engine's artifact binding."""

    kind = "test_case_tick"

    def __init__(self, *, amp_uv=3.0):
        super().__init__(rate_by_state={"eyes_closed": 2.0, "eyes_open": 2.0})
        self.amp_uv = amp_uv

    def make_event(self, onset):
        n = int(0.1 * self.fs)
        truth = TruthRecord(
            self.kind,
            None,
            None,
            (),
            onset / self.fs,
            (onset + n) / self.fs,
            self.amp_uv,
            (Remedy.MASK_SEGMENT,),
            self.layer_name,
        )
        return Event.from_pattern(onset, np.ones(self.n_ch), self.amp_uv * np.ones(n), truth)


class _Flat(TransformArtifact):
    kind = "test_case_flat"

    def __init__(self):
        pass

    def render_transform(self, t0, n, mix):
        return np.zeros_like(mix)

    def truth(self):
        return []


@pytest.fixture
def _test_kinds():
    register(_Tick)
    register(_Flat)
    yield
    ARTIFACTS.pop("test_case_tick", None)
    ARTIFACTS.pop("test_case_flat", None)


def test_artifacts_are_bound_named_per_design_and_summed(_test_kinds):
    arts = (
        ArtifactSpec("test_case_tick"),
        ArtifactSpec("test_case_flat"),
        ArtifactSpec("test_case_tick", {"amp_uv": 5.0}),
    )
    spec = resting_case(9, duration_s=4.0, artifacts=arts)
    assert CaseSpec.from_dict(spec.to_dict()) == spec
    case = make_case(spec)
    # neither test kind declares jitter_pattern (no scalp map of its own to jitter), so neither
    # draws or records one
    assert case.subject.pattern_jitter == {}
    rec = case.recordings["eyes_open"]
    ticks = ["artifact:test_case_tick#1", "artifact:test_case_tick#2"]
    assert list(rec.layers) == ["brain", *ticks, "sensor", "transform:test_case_flat"]
    assert np.allclose(rec.mixed, sum(rec.layers.values()), atol=1e-4)
    assert {t.layer for t in rec.truth} == set(ticks)
    assert {t.peak_uv for t in rec.truth if t.layer == ticks[1]} == {5.0}
    # each instance draws from its own <condition>:artifact:<kind>:<i> stream
    assert not np.array_equal(rec.layers[ticks[0]] / 3.0, rec.layers[ticks[1]] / 5.0)


def test_smoothing_matrix_is_computed_once_per_subject(monkeypatch):
    calls = []
    real = HeadModel.smoothed_mixing

    def counting(self, scale_mm):
        calls.append(scale_mm)
        return real(self, scale_mm)

    monkeypatch.setattr(HeadModel, "smoothed_mixing", counting)
    case = make_case(resting_case(11, duration_s=1.0, artifacts=()))
    assert len(case.recordings) == 2 and len(calls) == 1


def test_channel_subset_records_rows_of_the_same_brain():
    """Patches live on the full head, so a 4-channel case of a seed is 4 rows of its 19-channel
    case's brain, not a different brain placed on a 4-channel head."""
    labels = ("O1", "O2", "T3", "T4")
    full = make_case(resting_case(13, duration_s=4.0, artifacts=()))
    sub = make_case(resting_case(13, duration_s=4.0, artifacts=(), channels=labels))
    rows = [CHANNELS_19.index(c) for c in labels]
    assert sub.subject.rows == rows
    for cond in ("eyes_closed", "eyes_open"):
        rec = sub.recordings[cond]
        assert rec.channels == labels and rec.layers["sensor"].shape == (4, 4 * 256)
        assert np.array_equal(rec.layers["brain"], full.recordings[cond].layers["brain"][rows])


def test_drowsy_segment_is_inserted_into_eyes_closed_only():
    spec = resting_case(3, duration_s=10.0, drowsy_from_s=6.0, artifacts=())
    ec, eo = spec.conditions
    assert [(s.t0_s, s.t1_s, s.state) for s in ec.timeline.segments] == [
        (0.0, 6.0, "eyes_closed"),
        (6.0, 10.0, "drowsy"),
    ]
    assert [s.state for s in eo.timeline.segments] == ["eyes_open"]
    assert CaseSpec.from_dict(spec.to_dict()) == spec


def _json_round_trip(spec: CaseSpec) -> CaseSpec:
    return CaseSpec.from_dict(json.loads(json.dumps(spec.to_dict(), allow_nan=False)))


def test_identity_is_stable_across_numeric_types_and_json():
    """Equal specs serialise identically: ints and numpy scalars are normalised on construction,
    so digest and case_id agree before and after a strict-JSON round trip (DESIGN §7.2)."""
    base = resting_case(1, duration_s=8.0, artifacts=())
    bg = dataclasses.replace(base.brain.background, rms_uv=int(base.brain.background.rms_uv))
    variants = [
        resting_case(1, duration_s=8, artifacts=()),
        resting_case(np.int64(1), duration_s=np.float32(8.0), fs=256, artifacts=()),
        dataclasses.replace(
            base, seed=np.int64(1), fs=np.int64(256), sensor=SensorSpec(white_uv=np.float32(1.5))
        ),
        dataclasses.replace(base, brain=dataclasses.replace(base.brain, background=bg)),
    ]
    for v in variants:
        assert v == base
        assert v.digest() == base.digest() and case_id_for(v) == case_id_for(base)
        back = _json_round_trip(v)
        assert back == base and back.digest() == base.digest()
        assert type(back.seed) is int and type(back.fs) is float
    assert CaseSpec(seed=np.int64(3)).digest() == CaseSpec(seed=3).digest()
    with_params = dataclasses.replace(
        base, artifacts=(ArtifactSpec("x", {"amp": np.float32(2.5), "sides": ("l", "r")}),)
    )
    back = _json_round_trip(with_params)
    assert back == with_params and back.digest() == with_params.digest()


def test_artifact_streams_are_indexed_within_their_kind(_test_kinds):
    """<condition>:artifact:<kind>:<i> counts instances of the same kind only (DESIGN §8.1), so
    putting an artifact of another kind first leaves the existing layers' draws unchanged."""
    ticks = (ArtifactSpec("test_case_tick"), ArtifactSpec("test_case_tick", {"amp_uv": 5.0}))
    a = make_case(resting_case(9, duration_s=8.0, artifacts=ticks))
    b = make_case(
        resting_case(9, duration_s=8.0, artifacts=(ArtifactSpec("test_case_flat"), *ticks))
    )
    for cond in ("eyes_closed", "eyes_open"):
        for name in ("artifact:test_case_tick#1", "artifact:test_case_tick#2"):
            la, lb = a.recordings[cond].layers[name], b.recordings[cond].layers[name]
            assert np.abs(la).max() > 0 and np.array_equal(la, lb)


def test_condition_names_must_be_unique():
    tl = StateTimeline.constant("eyes_open", 1.0)
    with pytest.raises(ValueError, match="duplicate condition"):
        CaseSpec(seed=1, conditions=(ConditionSpec("a", 1.0, tl), ConditionSpec("a", 1.0, tl)))


def test_loose_spec_values_that_used_to_pass_silently_are_now_rejected():
    """A wrong type used to pass silently: bool(...) coerced any string to True/False, float(...)
    parsed a numeric string, and bool is an int subclass so it sailed through operator.index."""
    with pytest.raises(TypeError, match="perturb_head"):
        CaseSpec(seed=1, perturb_head="false")  # a truthy string, not the bool False
    with pytest.raises(TypeError, match="fs"):
        CaseSpec(seed=1, fs="256")
    with pytest.raises(TypeError, match="seed"):
        CaseSpec(seed=True)  # bool is not a seed
    tl = StateTimeline.constant("eyes_open", 1.0)
    with pytest.raises(TypeError, match="duration_s"):
        ConditionSpec("a", "5.0", tl)  # non-numeric duration
    with pytest.raises(TypeError, match="white_uv"):
        SensorSpec(white_uv="1.5")


def test_int_is_still_accepted_for_a_float_field():
    """The tightened type check must not stop an int from filling a float-annotated field: only
    the previously-silent wrong *types* (str, bool-into-int/float) are newly rejected."""
    a = CaseSpec(seed=1, fs=256)
    b = CaseSpec(seed=1, fs=256.0)
    assert a == b and a.digest() == b.digest() and type(a.fs) is float
    assert SensorSpec(white_uv=2).white_uv == 2.0


def test_infinite_durations_survive_strict_json():
    spec = CaseSpec(
        seed=1,
        conditions=(ConditionSpec("stream", math.inf, StateTimeline.constant("eyes_open")),),
    )
    back = _json_round_trip(spec)
    assert back == spec and back.digest() == spec.digest()
    assert back.conditions[0].duration_s == math.inf
    assert back.conditions[0].timeline.segments[0].t1_s == math.inf


def test_make_case_raises_a_clear_error_for_a_non_finite_condition_duration():
    """make_case/make_recording render a whole condition up front, so a condition whose
    duration_s is not finite used to blow up with an opaque OverflowError (round(inf * fs)
    cannot convert float infinity to integer) instead of naming the condition. StreamSource's
    own condition is deliberately unbounded (duration_s=inf) but never hits this path, since it
    renders chunk by chunk rather than a whole condition at once."""
    spec = CaseSpec(
        seed=1,
        conditions=(ConditionSpec("stream", math.inf, StateTimeline.constant("eyes_open")),),
    )
    with pytest.raises(ValueError, match="stream"):
        make_case(spec)


def test_stream_source_keeps_working_with_its_unbounded_condition():
    from open_eeg_synth.stream import StreamSource

    src = StreamSource(["O1", "O2"], 100.0, seed=1)
    assert math.isinf(src.spec.conditions[0].duration_s)
    chunk = src.next_chunk(50)
    assert chunk.shape == (2, 50)


def test_patch_offsets_and_lags_are_subject_properties():
    """Per-patch centre-frequency offsets and lags belong to the subject (DESIGN §8.1
    subject:rhythm:<name>): identical in every condition, seed-dependent, recorded."""
    spec = resting_case(5, duration_s=1.0, artifacts=())
    subj = make_subject(spec)
    engines = [make_engine(spec, subj, c) for c in spec.conditions]
    for r in spec.brain.rhythms:
        ec, eo = (
            next(p for p in e.layers[0].parts if p.name == f"brain.rhythm:{r.name}")
            for e in engines
        )
        assert ec.lags == eo.lags and [o.f0 for o in ec.own] == [o.f0 for o in eo.own]
        assert list(ec.f0_offsets_hz) == subj.patch_f0_offsets_hz[r.name]
        assert list(ec.lags_ms) == subj.patch_lags_ms[r.name]
        assert len(subj.patch_lags_ms[r.name]) == len(subj.placements[r.name])
        assert all(0.0 <= lag <= r.lag_ms for lag in subj.patch_lags_ms[r.name])
    alpha = subj.patch_f0_offsets_hz["alpha"]
    assert len(set(alpha)) == len(alpha) and max(abs(v) for v in alpha) > 0.05
    again = make_subject(spec)
    assert again.patch_f0_offsets_hz == subj.patch_f0_offsets_hz
    assert again.patch_lags_ms == subj.patch_lags_ms
    other = make_subject(resting_case(6, duration_s=1.0, artifacts=()))
    assert other.patch_f0_offsets_hz["alpha"] != alpha
    d = subj.to_dict()
    assert d["patch_f0_offsets_hz"] == subj.patch_f0_offsets_hz
    assert d["patch_lags_ms"] == subj.patch_lags_ms
    json.dumps(d, allow_nan=False)


def test_whole_case_is_chunk_invariant(_test_kinds):
    """Every layer of a two-condition case with artifacts and a drowsy segment renders the same
    for random block partitions as in one pass (DESIGN §8.2)."""
    arts = (
        ArtifactSpec("test_case_tick"),
        ArtifactSpec("test_case_flat"),
        ArtifactSpec("test_case_tick", {"amp_uv": 5.0}),
    )
    spec = resting_case(21, duration_s=20.0, artifacts=arts, drowsy_from_s=10.0)
    case = make_case(spec)
    rng = np.random.default_rng(0)
    for cond in spec.conditions:
        whole = case.recordings[cond.name]
        n_total = whole.n_samples
        for _ in range(2):
            eng = make_engine(spec, case.subject, cond)
            parts: dict[str, list[np.ndarray]] = {}
            t0 = 0
            while t0 < n_total:
                n = int(min(n_total - t0, rng.integers(1, 700)))
                for k, v in eng.render(t0, n).layers.items():
                    parts.setdefault(k, []).append(v)
                t0 += n
            assert list(parts) == list(whole.layers)
            for k, blocks in parts.items():
                assert np.allclose(np.concatenate(blocks, axis=1), whole.layers[k], atol=1e-4)
            assert eng.truth() == whole.truth


def test_plant_rhythms_are_subject_properties_too():
    """A plant's own rhythm draws its per-patch offsets/lags once per subject too (DESIGN §8.1
    subject:rhythm:<name>), identical in every condition — and adding a plant must not disturb
    any base rhythm's own subject draws. Checks that the plant rhythm's own subject draws behave
    like a base rhythm's, not just that its name shows up."""
    plants = (RhythmicBursts(("Fz", "F3", "F4")), FocalSlow("T3"))
    base = resting_case(9, duration_s=1.0, artifacts=())
    planted = resting_case(9, duration_s=1.0, artifacts=(), plants=plants)
    subj_base, subj = make_subject(base), make_subject(planted)
    for k in subj_base.placements:  # every base rhythm's subject draws are untouched by the plants
        assert subj_base.placements[k] == subj.placements[k]
        assert subj_base.f0_hz[k] == subj.f0_hz[k]
        assert subj_base.patch_f0_offsets_hz[k] == subj.patch_f0_offsets_hz[k]
        assert subj_base.patch_lags_ms[k] == subj.patch_lags_ms[k]

    engines = [make_engine(planted, subj, c) for c in planted.conditions]
    plant_rhythms = [r for r in compiled_rhythms(planted) if r.name.startswith("plant")]
    assert plant_rhythms  # sanity: the spec above does compile plant rhythms
    for r in plant_rhythms:
        ec, eo = (
            next(p for p in e.layers[0].parts if p.name == f"brain.rhythm:{r.name}")
            for e in engines
        )
        assert ec.lags == eo.lags and [o.f0 for o in ec.own] == [o.f0 for o in eo.own]
        assert list(ec.f0_offsets_hz) == subj.patch_f0_offsets_hz[r.name]
        assert list(ec.lags_ms) == subj.patch_lags_ms[r.name]
        assert len(subj.patch_lags_ms[r.name]) == len(subj.placements[r.name])


def test_bare_case_spec_defaults_follow_design_and_render():
    """DESIGN §7.2: CaseSpec()'s brain/artifacts/conditions defaults are recipes.resting_brain(),
    recipes.ordinary_artifacts() and eyes_closed/eyes_open 240 s constant timelines, via lazy
    default factories (case.py's existing ``_default_brain`` pattern) that avoid the
    case<->recipes import cycle. A bare spec (seed only) is exactly what resting_case(seed)
    builds too."""
    from open_eeg_synth.recipes import ordinary_artifacts, resting_brain

    spec = CaseSpec(seed=1)
    assert spec.brain == resting_brain()
    assert spec.artifacts == ordinary_artifacts()
    assert [c.name for c in spec.conditions] == ["eyes_closed", "eyes_open"]
    assert [c.duration_s for c in spec.conditions] == [240.0, 240.0]
    assert [c.timeline.segments[0].state for c in spec.conditions] == ["eyes_closed", "eyes_open"]
    assert all(len(c.timeline.segments) == 1 for c in spec.conditions)  # constant timelines
    assert spec == resting_case(1)
    assert spec.digest() == resting_case(1).digest()

    # a bare spec renders both conditions with the ordinary artifacts (short override: fast test;
    # tests/test_perf.py exercises the full 240 s x2 default duration under the perf budget)
    short = dataclasses.replace(
        spec,
        conditions=tuple(
            dataclasses.replace(c, duration_s=2.0, timeline=StateTimeline.constant(c.name, 2.0))
            for c in spec.conditions
        ),
    )
    case = make_case(short)
    assert set(case.recordings) == {"eyes_closed", "eyes_open"}
    for rec in case.recordings.values():
        assert set(rec.layers) == {
            "brain",
            "artifact:blink",
            "artifact:eye_movement",
            "artifact:emg",
            "sensor",
        }


def test_channel_aliases_canonicalise_on_construction():
    """T7/T8/P7/P8 (case-insensitive) canonicalise to T3/T4/T5/T6 on CaseSpec construction, so
    the two spellings give the same spec, digest and case id."""
    aliased = tuple({"T3": "t7", "T4": "T8", "T5": "p7", "T6": "P8"}.get(c, c) for c in CHANNELS_19)
    a = resting_case(15, duration_s=1.0, artifacts=(), channels=aliased)
    b = resting_case(15, duration_s=1.0, artifacts=(), channels=CHANNELS_19)
    assert a.channels == CHANNELS_19 == b.channels
    assert a == b
    assert a.digest() == b.digest()
    assert case_id_for(a) == case_id_for(b)


def test_channels_canonicalise_against_the_selected_head_model(monkeypatch):
    """Channel canonicalisation must consult the channel list of the head model the spec actually
    selects, not a hard-wired CHANNELS_19: a future head model with its own, different channel
    set must not be refused before it is even looked at."""
    import open_eeg_synth.case as case_module

    monkeypatch.setattr(case_module, "head_model_channels", lambda name: ("Cz", "Fp1"))
    spec = CaseSpec(
        seed=1,
        head_model="a-future-head-model",
        channels=("cz", "FP1"),
        conditions=(),
        artifacts=(),
    )
    assert spec.channels == ("Cz", "Fp1")


def test_duplicate_channels_after_canonicalisation_raise():
    """("T3", "t7") must not silently become ("T3", "T3")."""
    with pytest.raises(ValueError, match="duplicate"):
        resting_case(1, duration_s=1.0, artifacts=(), channels=("T3", "t7"))


def test_pattern_jitter_is_recorded_only_for_kinds_that_apply_it():
    """Only blink and eye movement draw an empirical, per-subject-jittered scalp map (jaw EMG's
    map is a fixed analytic Gaussian; dead channel has no map at all), so a subject's
    pattern_jitter must record exactly the kinds whose rendered signal actually depends on one —
    clipped to [-1, 1] exactly as artifacts.patterns.empirical applies it. One subject (the
    default ordinary artifact set: blink, eye movement, emg) covers every assertion below -
    building a second just for the "emg draws no jitter" check would double this test's cost for
    no extra coverage."""
    spec = resting_case(7, duration_s=1.0)
    subject = make_subject(spec)
    raw = float(stream_rng(spec.seed, "subject:artifact:blink").normal(0.0, 0.6))
    assert raw < -1.0  # sanity: seed 7's raw blink draw is outside [-1, 1] (about -1.011)
    assert subject.pattern_jitter["blink"] == -1.0
    assert set(subject.pattern_jitter) == {"blink", "eye_movement"}  # not emg

    # the bound artifact's own map is exactly what the recorded, clipped z produces
    eng = make_engine(spec, subject, spec.conditions[0])
    blink = next(lay for lay in eng.layers if getattr(lay, "kind", None) == "blink")
    head = subject.head.subset(spec.channels)
    expected = patterns.empirical(
        "blink", spec.channels, z=subject.pattern_jitter["blink"], electrode_pos=head.electrode_pos
    )
    assert np.array_equal(blink.pattern, expected)


def test_case_with_plants_is_chunk_invariant():
    """A case using all six planted-pattern primitives together, with a drowsy segment, renders
    the same for random block partitions as in one pass (DESIGN §8.2)."""
    plants = (
        FocalSlow("F7", state_gain={"eyes_open": 0.0}),
        RhythmicBursts(("Fz", "Cz"), burst_s=(1, 2), gap_s=(2, 4)),
        LateralImbalance("alpha", "left", 0.5),
        WidespreadExcess("theta", 2.0, extra_sites=("Pz",)),
        PeakShift("alpha", -1.0),
        ReducedRhythm("smr", 0.3),
    )
    spec = resting_case(11, duration_s=20.0, artifacts=(), plants=plants, drowsy_from_s=8.0)
    case = make_case(spec)
    rng = np.random.default_rng(0)
    for cond in spec.conditions:
        whole = case.recordings[cond.name]
        n_total = whole.n_samples
        eng = make_engine(spec, case.subject, cond)
        parts, t0 = [], 0
        while t0 < n_total:
            n = int(min(n_total - t0, rng.integers(1, 700)))
            parts.append(eng.render(t0, n).mixed)
            t0 += n
        assert np.allclose(np.concatenate(parts, axis=1), whole.mixed, atol=1e-4)

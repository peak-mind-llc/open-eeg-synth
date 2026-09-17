from __future__ import annotations

import math
import time

import numpy as np
import pytest

from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.case import CaseSpec, ConditionSpec, SensorSpec
from open_eeg_synth.markers import MarkerSchedule
from open_eeg_synth.stream import StreamSource
from tests.helpers import band_power

Q21 = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
    "HR",
]
DRAGON = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
]


def _kurt(x):
    x = x - x.mean()
    return float(np.mean(x**4) / np.mean(x**2) ** 2 - 3.0)


def test_validation():
    with pytest.raises(ValueError):
        StreamSource([], 256.0)
    with pytest.raises(ValueError):
        StreamSource(["O1"], 0.0)
    s = StreamSource(["O1"], 256.0, seed=1)
    with pytest.raises(ValueError):
        s.next_chunk(0)


def test_validation_rejects_non_finite_srate():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            StreamSource(["O1"], bad, seed=1)


def test_duplicate_eeg_labels_are_rejected_but_heart_labels_may_repeat():
    """CaseSpec rejects duplicate channels after alias canonicalisation, and StreamSource's own
    channels feed straight into one, so two EEG rows for the same electrode now raise - a literal
    repeat or two labels that alias to the same one (T7 canonicalises to T3). Heart labels are
    excluded from that channel set before it ever reaches CaseSpec (several HR rows commonly
    carry the same ECG), so they may still repeat."""
    with pytest.raises(ValueError, match="duplicate"):
        StreamSource(["O1", "O1"], 256.0, seed=1)
    with pytest.raises(ValueError, match="duplicate"):
        StreamSource(["T3", "T7"], 256.0, seed=1)

    src = StreamSource(["O1", "O2", "HR", "HR"], 256.0, seed=1)
    chunk = src.next_chunk(32)
    assert chunk.shape == (4, 32)
    assert np.array_equal(chunk[2], chunk[3])  # both HR rows carry the same ECG


def test_labels_the_head_model_lacks_carry_sensor_noise_only():
    """A label that is neither a heart label nor a head-model channel (10-10 sites, ear
    references, anything else) is an unmodelled row with sensor noise only, as the classic
    synthesizer accepts any label (DESIGN §7.3). The modelled rows do not change: the Fp1 row
    equals a ["Fp1", "HR"] source's with the same seed. A label spelled as a head-model channel is
    still modelled, and two labels naming one modelled channel still raise."""
    labels = ["Fp1", "Fpz", "Oz", "A1", "X1", "HR", "ECG"]
    with pytest.warns(UserWarning, match="Fpz, Oz, A1, X1") as caught:
        src = StreamSource(labels, 256.0, seed=11, sensor=SensorSpec(white_uv=4.0))
    assert len([w for w in caught if "unmodelled" in str(w.message)]) == 1
    assert src.unmodelled_labels == ("Fpz", "Oz", "A1", "X1")
    ref = StreamSource(["Fp1", "HR"], 256.0, seed=11, sensor=SensorSpec(white_uv=4.0))
    assert ref.unmodelled_labels == ()
    x = np.concatenate([src.next_chunk(512) for _ in range(8)], axis=1)
    y = np.concatenate([ref.next_chunk(512) for _ in range(8)], axis=1)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    assert np.array_equal(x[0], y[0])
    assert np.array_equal(x[5], y[1]) and np.array_equal(x[6], y[1])
    noise = x[1:5].astype(float)
    assert np.allclose(noise.std(axis=1), 4.0, rtol=0.06)
    assert np.abs(np.corrcoef(noise)[np.triu_indices(4, 1)]).max() < 0.1

    rng = np.random.default_rng(20260917)
    with pytest.warns(UserWarning, match="unmodelled"):
        again = StreamSource(labels, 256.0, seed=11, sensor=SensorSpec(white_uv=4.0))
    parts, t = [], 0
    while t < x.shape[1]:
        n = min(int(rng.integers(1, 700)), x.shape[1] - t)
        parts.append(again.next_chunk(n))
        t += n
    assert np.allclose(np.concatenate(parts, axis=1), x, atol=1e-4)

    with pytest.warns(UserWarning, match="unmodelled"):
        StreamSource(["X1"], 256.0, seed=1)  # nothing modelled at all still streams
    with pytest.raises(ValueError, match="duplicate"):
        StreamSource(["O1", "o1", "X1"], 256.0, seed=1)


def test_chunk_shape_dtype_scale_and_determinism():
    a = StreamSource(Q21, 256.0, seed=7)
    b = StreamSource(Q21, 256.0, seed=7)
    c = a.next_chunk(32)
    assert c.shape == (20, 32) and c.dtype == np.float32 and np.isfinite(c).all()
    assert np.allclose(c, b.next_chunk(32)) and np.allclose(a.next_chunk(32), b.next_chunk(32))
    x = np.concatenate([a.next_chunk(256) for _ in range(8)], axis=1)
    assert 5.0 < x[8].std() < 300.0


@pytest.fixture(scope="module")
def fresh_q21_pair():
    """Two independently seed=None-drawn Q21 @ 256 Hz sources, built once for the whole module
    and never advanced (no next_chunk/due_markers calls): construction (~0.5 s - head
    perturbation plus mixing-matrix smoothing) dominates this file's runtime, so the one check
    that only needs to read `.seed` off a pair of fresh instances shares them instead of paying
    for its own pair."""
    return StreamSource(Q21, 256.0), StreamSource(Q21, 256.0)


def test_fresh_seeds_are_unique(fresh_q21_pair):
    a, b = fresh_q21_pair
    assert a.seed != b.seed


def test_chunk_size_does_not_change_the_signal():
    """DESIGN §8.2: the signal must not depend on how it was chunked. Q21 (unlike DRAGON) carries
    an HR row, so this also exercises HeartSource's own chunk continuity; random partition sizes
    (a fixed seed, so the test stays deterministic) exercise more than one arbitrary split."""
    rng = np.random.default_rng(20260917)
    total = 1000
    parts, t = [], 0
    while t < total:
        n = min(int(rng.integers(1, 97)), total - t)
        parts.append(n)
        t += n
    a = StreamSource(Q21, 500.0, seed=3)
    b = StreamSource(Q21, 500.0, seed=3)
    x = a.next_chunk(total)
    y = np.concatenate([b.next_chunk(n) for n in parts], axis=1)
    assert np.allclose(x, y, atol=1e-4)


def test_pink_slope_posterior_alpha_and_ecg_only_on_hr():
    # Eyes closed and artifact-free: the layered default (eyes open + blinks) legitimately has more
    # frontal 8-12 Hz power than occipital. Posterior dominance is judged after average
    # referencing (the lead field's native reference carries a common component, DESIGN §2.3)
    # and as a median over subjects, because alpha varies by subject as much as it does in life.
    ratios = []
    for seed in (3, 4, 5):
        s = StreamSource(
            Q21, 256.0, seed=seed, timeline=StateTimeline.constant("eyes_closed"), artifacts=()
        )
        x = np.concatenate([s.next_chunk(256) for _ in range(16)], axis=1)
        cz = x[17]
        assert band_power(cz, 256.0, 2, 6)[0] > band_power(cz, 256.0, 30, 50)[0]
        eeg = x[:19] - x[:19].mean(axis=0, keepdims=True)
        a = band_power(eeg, 256.0, 8, 12)
        ratios.append((a[8] + a[9]) / (a[0] + a[1]))
        hr = x[19]
        assert hr.max() > 120.0 and _kurt(hr) > 3.0 and _kurt(x[8]) < 1.5
    assert np.median(ratios) > 2.0  # occipital alpha well above frontal


def test_arbitrary_montages_and_rates():
    for labels, sr in ((["O1", "O2", "T3", "T4"], 250.0), (DRAGON, 500.0)):
        c = StreamSource(labels, sr, seed=5).next_chunk(64)
        assert c.shape == (len(labels), 64) and np.isfinite(c).all()


def test_markers_and_truth():
    s = StreamSource(["Cz"], 100.0, seed=1, markers=MarkerSchedule(kind="periodic", period_s=1.0))
    got = []
    for _ in range(250):
        got += [lbl for _, lbl in s.due_markers(100)]
    assert got.count("standard") >= 2
    s2 = StreamSource(
        ["Cz"], 100.0, seed=2, markers=MarkerSchedule(kind="oddball", period_s=0.5, p_target=0.3)
    )
    labels = []
    for _ in range(400):
        labels += [lbl for _, lbl in s2.due_markers(100)]
    assert "target" in labels and "standard" in labels
    assert StreamSource(["Cz"], 256.0, seed=1).due_markers(64) == []
    s3 = StreamSource(Q21, 256.0, seed=9)
    for _ in range(int(60 * 256 / 32)):
        s3.next_chunk(32)
    assert any(t.kind == "blink" for t in s3.truth)


def test_next_chunk_is_fast_enough_for_a_live_mock_amplifier():
    """A mock amplifier calls next_chunk 8-10 times a second, so each call has roughly
    100-125 ms to run in. 32 samples at 256 Hz is 125 ms of signal; construction (~0.5 s, the
    head perturbation and mixing-matrix smoothing) happens once up front and is excluded here."""
    s = StreamSource(Q21, 256.0, seed=13)
    n_calls = 200
    t0 = time.perf_counter()
    for _ in range(n_calls):
        s.next_chunk(32)
    elapsed = time.perf_counter() - t0
    assert elapsed / n_calls < 0.025


def test_seed_none_reproduces_its_first_chunks_truth_and_markers_from_its_own_seed():
    """A StreamSource built with seed=None draws and exposes a fresh seed (.seed); a second
    StreamSource built with that seed must reproduce not just the samples but the scheduled truth
    and due markers too, over several chunks, so a recording application can log `.seed` and
    replay a whole session, not just its waveform. Oddball markers draw from the marker rng (a
    periodic schedule never does, so it could not have caught a marker-replay mismatch), and the
    span is long enough (30 s) that the default artifacts' truth is essentially never empty -
    both are asserted non-empty, not just equal to each other, so this cannot silently degenerate
    into comparing two empty lists."""
    ms = MarkerSchedule(kind="oddball", period_s=0.5)
    a = StreamSource(Q21, 256.0, markers=ms)
    b = StreamSource(Q21, 256.0, seed=a.seed, markers=ms)
    chunks_a, chunks_b, markers_a, markers_b = [], [], [], []
    for _ in range(5):
        chunks_a.append(a.next_chunk(1536))
        markers_a += a.due_markers(1536)
        chunks_b.append(b.next_chunk(1536))
        markers_b += b.due_markers(1536)
    assert np.array_equal(np.concatenate(chunks_a, axis=1), np.concatenate(chunks_b, axis=1))
    assert markers_a == markers_b and len(markers_a) > 0
    assert [t.to_dict() for t in a.truth] == [t.to_dict() for t in b.truth]
    assert len(a.truth) > 0


def test_unbounded_stream_condition_serialises_and_digests():
    """StreamSource builds its ConditionSpec with duration_s=math.inf (an endless stream has no
    fixed length). CaseSpec.to_dict/from_dict and .digest() already turn an infinite duration
    into JSON null and back (case.ConditionSpec.to_dict, _canon.inf_to_json/inf_from_json), so no
    change was needed for this: a case built with an unbounded condition still serialises,
    round-trips and digests like any other case. StreamSource itself never asks for a case id or
    writes a case file, so this is a locking test for the plumbing it depends on, not new
    behaviour StreamSource adds."""
    spec = CaseSpec(
        seed=1,
        fs=256.0,
        channels=("O1",),
        conditions=(ConditionSpec("stream", math.inf, StateTimeline.constant("eyes_open")),),
    )
    d = spec.to_dict()
    assert d["conditions"][0]["duration_s"] is None
    assert spec.digest()  # json.dumps(..., allow_nan=False) must not raise on the inf duration
    assert CaseSpec.from_dict(d) == spec

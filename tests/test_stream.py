import math
import time

import numpy as np
import pytest

from open_eeg_synth.brain.state import StateTimeline
from open_eeg_synth.case import CaseSpec, ConditionSpec
from open_eeg_synth.channels import UnknownChannelError
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
    with pytest.raises(UnknownChannelError):
        StreamSource(["O1", "XYZ"], 256.0)
    s = StreamSource(["O1"], 256.0, seed=1)
    with pytest.raises(ValueError):
        s.next_chunk(0)


def test_chunk_shape_dtype_scale_and_determinism():
    a = StreamSource(Q21, 256.0, seed=7)
    b = StreamSource(Q21, 256.0, seed=7)
    c = a.next_chunk(32)
    assert c.shape == (20, 32) and c.dtype == np.float32 and np.isfinite(c).all()
    assert np.allclose(c, b.next_chunk(32)) and np.allclose(a.next_chunk(32), b.next_chunk(32))
    x = np.concatenate([a.next_chunk(256) for _ in range(8)], axis=1)
    assert 5.0 < x[8].std() < 300.0
    assert StreamSource(Q21, 256.0).seed != StreamSource(Q21, 256.0).seed  # fresh seeds


def test_chunk_size_does_not_change_the_signal():
    a = StreamSource(DRAGON, 500.0, seed=3)
    b = StreamSource(DRAGON, 500.0, seed=3)
    x = a.next_chunk(1000)
    y = np.concatenate([b.next_chunk(50) for _ in range(20)], axis=1)
    assert np.allclose(x, y, atol=1e-3)


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
    """S1: a mock amplifier calls next_chunk 8-10 times a second, so each call has roughly
    100-125 ms to run in. 32 samples at 256 Hz is 125 ms of signal; construction (~0.5 s, the
    head perturbation and mixing-matrix smoothing) happens once up front and is excluded here."""
    s = StreamSource(Q21, 256.0, seed=13)
    n_calls = 200
    t0 = time.perf_counter()
    for _ in range(n_calls):
        s.next_chunk(32)
    elapsed = time.perf_counter() - t0
    assert elapsed / n_calls < 0.025


def test_seed_none_reproduces_its_first_chunk_from_its_own_seed():
    """S4: a StreamSource built with seed=None draws and exposes a fresh seed (.seed); a second
    StreamSource built with that seed must reproduce the first one's first chunk exactly, so a
    recording application can log `.seed` and replay a session byte for byte."""
    a = StreamSource(Q21, 256.0)
    first = a.next_chunk(64)
    b = StreamSource(Q21, 256.0, seed=a.seed)
    assert np.array_equal(first, b.next_chunk(64))


def test_unbounded_stream_condition_serialises_and_digests():
    """S3: StreamSource builds its ConditionSpec with duration_s=math.inf (an endless stream has
    no fixed length). CaseSpec.to_dict/from_dict and .digest() already turn an infinite duration
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

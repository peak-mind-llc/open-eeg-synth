"""Behaviour tests for the classic engine, moved from Coherence Recorder."""

from __future__ import annotations

import numpy as np
import pytest

from open_eeg_synth.classic.synth import MarkerSchedule, RealisticEEGSynthesizer, alpha_weight

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


def _psd_band(x, srate, lo, hi):
    f = np.fft.rfftfreq(x.shape[-1], d=1.0 / srate)
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    m = (f >= lo) & (f < hi)
    return p[..., m].sum(axis=-1)


def test_alpha_weight_posterior_gt_frontal():
    assert alpha_weight("O1") > alpha_weight("Fp1")
    assert alpha_weight("Pz") > alpha_weight("Fz")
    assert alpha_weight("zzz") == pytest.approx(0.40)  # default for unknown


def test_validation():
    with pytest.raises(ValueError):
        RealisticEEGSynthesizer([], 256.0)
    with pytest.raises(ValueError):
        RealisticEEGSynthesizer(["O1"], 0.0)
    s = RealisticEEGSynthesizer(["O1"], 256.0)
    with pytest.raises(ValueError):
        s.next_chunk(0)


def test_chunk_shape_dtype_and_scale():
    s = RealisticEEGSynthesizer(Q21, 256.0, seed=1)
    c = s.next_chunk(256)
    assert c.shape == (len(Q21), 256)
    assert c.dtype == np.float32
    assert np.isfinite(c).all()
    # EEG channels are tens of µV, not absurd
    assert 5.0 < float(np.sqrt(np.mean(c[8] ** 2))) < 300.0  # O1


def test_determinism_under_seed():
    a = RealisticEEGSynthesizer(Q21, 256.0, seed=7)
    b = RealisticEEGSynthesizer(Q21, 256.0, seed=7)
    assert np.allclose(a.next_chunk(128), b.next_chunk(128))
    assert np.allclose(a.next_chunk(128), b.next_chunk(128))  # phase-continuous + reproducible


def test_pink_1f_slope():
    s = RealisticEEGSynthesizer(["Cz"], 256.0, seed=2)
    x = np.concatenate([s.next_chunk(256)[0] for _ in range(8)])  # ~8 s
    low = _psd_band(x, 256.0, 2, 6)
    high = _psd_band(x, 256.0, 30, 50)
    assert low > high  # 1/f: more power at low freq


def test_alpha_posterior_dominant():
    # The synth is calibrated with strong (97%) common-mode coupling, matching
    # raw unreferenced Q21 data — so the shared 1/f floor dominates the raw
    # 8-12 Hz band power on every channel and the raw occ/front ratio is only
    # ~2x (NOT >3x). The physically correct test of posterior-dominant alpha is
    # the *alpha-band excess*: 8-12 Hz power above the immediately-lower 5-8 Hz
    # 1/f baseline. Occipital channels show a clear alpha bump above that floor;
    # frontal channels (alpha weight 0.15) show none.
    s = RealisticEEGSynthesizer(Q21, 256.0, seed=3)
    x = np.concatenate([s.next_chunk(256) for _ in range(16)], axis=1)  # ~16 s
    a_occ = _psd_band(x[8], 256.0, 8, 12) + _psd_band(x[9], 256.0, 8, 12)  # O1+O2
    a_front = _psd_band(x[0], 256.0, 8, 12) + _psd_band(x[1], 256.0, 8, 12)  # Fp1+Fp2
    assert a_occ > a_front  # occipital alpha-band power exceeds frontal
    base_occ = _psd_band(x[8], 256.0, 5, 8) + _psd_band(x[9], 256.0, 5, 8)
    base_front = _psd_band(x[0], 256.0, 5, 8) + _psd_band(x[1], 256.0, 5, 8)
    occ_alpha_excess = a_occ - base_occ
    front_alpha_excess = a_front - base_front
    assert occ_alpha_excess > 0.0  # posterior alpha bump present
    assert occ_alpha_excess > 3.0 * front_alpha_excess  # and posterior-dominant


def test_ecg_only_on_hr_channel():
    # The HR channel is replaced wholesale by synthesized ECG (≈200 µV R-wave,
    # but only ~49 µV RMS — it's spiky). EEG channels carry no ECG: their power
    # comes from 65 µV-RMS pink noise, whose natural peak excursions over ~8 s
    # legitimately exceed 200 µV, so an absolute peak threshold can't separate
    # them. The discriminating feature is excess kurtosis: the sparse ECG
    # R-waves make the HR channel strongly leptokurtic vs broadband-noise EEG.
    s = RealisticEEGSynthesizer(Q21, 256.0, seed=4)
    x = np.concatenate([s.next_chunk(256) for _ in range(8)], axis=1)
    hr = x[19]
    o1 = x[8]
    assert np.max(np.abs(hr)) > 120.0  # HR carries the ≈200 µV R-wave

    def _excess_kurtosis(sig):
        sig = sig - np.mean(sig)
        var = np.mean(sig**2)
        return float(np.mean(sig**4) / (var * var) - 3.0)

    # The ECG's brief, sparse R-waves make the HR channel strongly leptokurtic
    # (excess kurtosis ≈ 5-6), while broadband 1/f EEG noise is near-Gaussian
    # (excess kurtosis ≈ 0). This separates them robustly where an absolute
    # peak threshold cannot.
    assert _excess_kurtosis(hr) > 3.0  # spiky ECG
    assert _excess_kurtosis(o1) < 1.0  # Gaussian-ish EEG, no ECG


def test_arbitrary_montages():
    for labels, sr in [
        (["O1", "O2", "T3", "T4"], 250.0),  # BrainBit subset
        (
            [
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
            ],
            500.0,
        ),
    ]:  # DragonEEG
        s = RealisticEEGSynthesizer(labels, sr, seed=5)
        c = s.next_chunk(64)
        assert c.shape == (len(labels), 64)
        assert np.isfinite(c).all()


def test_markers_none_default():
    s = RealisticEEGSynthesizer(["Cz"], 256.0, seed=1)
    assert s.due_markers(64) == []  # default schedule = none


def test_markers_periodic():
    s = RealisticEEGSynthesizer(
        ["Cz"], 100.0, seed=1, markers=MarkerSchedule(kind="periodic", period_s=1.0)
    )
    got = []
    for _ in range(250):  # 2.5 s at 100 samples/chunk
        got += [lbl for _, lbl in s.due_markers(100)]
    assert got.count("standard") >= 2  # ~one per second


def test_markers_oddball_targets_present():
    s = RealisticEEGSynthesizer(
        ["Cz"], 100.0, seed=2, markers=MarkerSchedule(kind="oddball", period_s=0.5, p_target=0.3)
    )
    labels = []
    for _ in range(400):
        labels += [lbl for _, lbl in s.due_markers(100)]
    assert "target" in labels and "standard" in labels


def test_marker_schedule_period_s_zero_raises():
    with pytest.raises(ValueError, match="period_s must be positive"):
        MarkerSchedule(kind="periodic", period_s=0)


def test_next_chunk_oversized_raises():
    s = RealisticEEGSynthesizer(["Cz"], 256.0, seed=1)
    with pytest.raises(ValueError, match="exceeds the"):
        s.next_chunk(10_000_000)
